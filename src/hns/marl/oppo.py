"""OPPO-MC, the launderer learner (SPEC §9.3 options, §10; adopted 2026-11-12, fifth Sol consult).

On-policy PPO over the launderer's options, with complete-world Monte Carlo returns:

- :class:`OptionPolicy` is the env's launderer policy. In flight it always proceeds. An idle launderer with funds
  and no wait outstanding decides (``hns.models.policy``): wait for a duration, or launch a mode with its recipient,
  mules and change. On a **scheduled** world (the first ``CURRICULUM_WORLDS`` of training, and the matched
  evaluation) the gate is exogenous instead: each launderer launches once per episode, at its first idle sub-step
  on or after a sub-step drawn uniformly from [0, ``LAUNCH_LAST``] where a launch is feasible, and otherwise
  waits; only the launch factors come from the actor. Every decision is stored with its observation (the exact
  one the actor saw), masks, sampled action, log-probability and the critic's value.
- :class:`Rollout` chains a launderer's decisions. ``R_j`` adds up every reward the launderer gets from its decision
  ``j`` up to its next one, discounted per sub-step (waiting, hops in flight, the holding cost); ``G_j = R_j +
  gamma ** (t_j+1 - t_j) G_j+1``, and world end is the only terminal. A decision that the env turned into a no-op
  stays the decision that was sampled. :meth:`Rollout.check` recomputes each chain's first return from the raw
  per-sub-step rewards.
- :class:`OppoLearner` holds the actor and the critic (no shared parameters) and updates once per world, after
  every chain is closed: the critic ``V = m v`` (``m`` the wealth scale of ``hns.marl.learner``) on
  ``0.5 (v - G / m)^2``; the actor on the clipped surrogate of the complete action's probability ratio with the
  advantage ``G - V`` divided by the world's RMS advantage, plus 0.01 times the joint entropy. Four passes each
  (the actor's up to four), minibatch 128 (micro-batched), Adam 3e-4, clip norm 1.0. After every actor Adam step
  the joint, launch-conditional and duration-conditional KL from the actor that collected the world are measured
  over the world's decisions; above 0.02 nats that step is undone, its Adam state included, the steps before it
  are kept, and the actor stops for that world (amended 2026-11-12 before the first block: the guard was per pass,
  CONSULT6.md). The ratio quantiles, clip fraction, entropy and gradient norms in the statistics cover every
  attempted step, an undone one included.

Before any update, the stored log-probabilities are recomputed from the stored observations (``ratio_dev_max``,
the largest difference: the probability ratios start at one) and so are the values (``value_dev_max``).

Not implemented here (SPEC §10, before the live pilot): the per-world acting copies of the live game's roles, and
NMAI's static option launderers.
"""

from __future__ import annotations

import copy
import math
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch

from hns.marl.learner import WEALTH_REF, MarlConfig, wealth_scale
from hns.marl.obs import D_CAND, D_W_AGENT, L_CTRL, ObsBatch, collate, launderer_obs
from hns.models import policy as P
from hns.sim.entities import GO, WAIT

CURRICULUM_WORLDS = 5
LAUNCH_LAST = 43                 # a scheduled launch's sub-step is uniform on [0, 43] (as hns.marl.capability's)
EPOCHS, MINIBATCH, MICRO = 4, 128, 64
LR, CLIP_RATIO, KL_MAX, ENT_COEF, GRAD_CLIP, ADV_MIN = 3e-4, 0.2, 0.02, 0.01, 1.0, 1e-3
RETURN_TOL = 1e-6                # the physical-return check, in reward units


def _t(a, device) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(a)).to(device)


# ---------------------------------------------------------------------------- decisions and their chains
@dataclass
class Row:
    agent: int
    t: int                       # the trainer's step at the decision
    tw: int                      # sub-steps since the world's start
    obs: tuple                   # the observation the actor saw (``ObsBatch.split`` form)
    masks: np.ndarray            # (MASK_WIDTH,) bool
    gate_free: bool              # false: a scheduled launch (the gate was exogenous)
    action: np.ndarray           # (6,) in policy.ACTION order
    logp: float                  # log-probability of the effective action when sampled
    v: float                     # the critic's V = m v when sampled
    m: float                     # the wealth scale m(o)
    R: float = 0.0
    delta: int = 0               # sub-steps to the next decision (to world end for the last)
    done: bool = False
    noop: bool = False           # the env could not execute the sampled launch


class Rollout:
    """One world's decisions of every launderer, chained per launderer (module docstring)."""

    def __init__(self, gamma: float):
        self.gamma, self.rows, self.pending, self.closed = gamma, [], {}, False
        self.raw: dict[int, list[tuple[int, float]]] = defaultdict(list)    # agent -> (step, reward), for check()

    def add(self, row: Row) -> None:
        if self.closed:
            raise RuntimeError("a closed rollout takes no decision")
        j = self.pending.get(row.agent)
        if j is not None:
            prev = self.rows[j]
            if row.t <= prev.t:
                raise RuntimeError(f"agent {row.agent}: decisions out of order")
            prev.delta = row.t - prev.t
        self.pending[row.agent] = len(self.rows)
        self.rows.append(row)

    def observe(self, agent: int, t: int, r: float, noop: bool = False) -> None:
        """The reward of ``agent``'s transition at step ``t``: it goes to the agent's current decision."""
        self.raw[agent].append((t, float(r)))
        j = self.pending.get(agent)
        if j is None:
            return                                   # before the agent's first decision of the world
        row = self.rows[j]
        row.R += self.gamma ** (t - row.t) * r
        if t == row.t and noop:
            row.noop = True

    def close(self, t_end: int) -> None:
        """World end: every pending decision is terminal, its rewards run to step ``t_end`` (exclusive)."""
        for j in self.pending.values():
            r = self.rows[j]
            r.delta, r.done = t_end - r.t, True
        self.pending.clear()
        self.closed = True

    def returns(self) -> np.ndarray:
        if not self.closed:
            raise RuntimeError("returns of an open rollout")
        G = np.zeros(len(self.rows))
        nxt: dict[int, float] = {}
        for j in range(len(self.rows) - 1, -1, -1):
            r = self.rows[j]
            G[j] = r.R + (0.0 if r.done else self.gamma ** r.delta * nxt[r.agent])
            nxt[r.agent] = G[j]
        return G

    def check(self, G: np.ndarray) -> float:
        """The largest difference between each launderer's first return and its discounted rewards from its first
        decision to world end, recomputed from the raw rewards; raises above ``RETURN_TOL``."""
        first: dict[int, int] = {}
        for j, r in enumerate(self.rows):
            first.setdefault(r.agent, j)
        worst = 0.0
        for a, j in first.items():
            t0 = self.rows[j].t
            direct = sum(self.gamma ** (t - t0) * r for t, r in self.raw[a] if t >= t0)
            worst = max(worst, abs(direct - G[j]))
        if worst > RETURN_TOL:
            raise RuntimeError(f"a decision chain's return differs from its rewards by {worst:.3g}")
        return worst


# ---------------------------------------------------------------------------- the env's launderer policy
def option_masks(st, sub: int, substeps: int) -> np.ndarray:
    """The option masks of an idle launderer state (``hns.sim.env.SimEnv.launderer_state``) at sub-step ``sub``."""
    m = np.zeros(P.MASK_WIDTH, bool)
    go = bool(st.masks["initiate"][GO])
    m[P.SLICES["gate"]] = (True, go)
    m[P.SLICES["dur"]] = P.duration_mask(sub, substeps)
    m[P.SLICES["mode"]] = np.outer(st.masks["amount"], st.masks["strategy"]).ravel() & go
    m[P.SLICES["rcpt"]] = st.masks["recipient"]
    m[P.SLICES["mules"]] = st.masks["mules"]
    m[P.SLICES["change"]] = st.masks["change"]
    return m


def env_action(st, a: np.ndarray) -> dict:
    """The env's action dict for an option action row (a wait sends ``initiate = 0`` and feasible placeholders)."""
    out = {b: int(np.flatnonzero(st.masks[b])[0]) for b in st.branches}
    if a[0] == P.WAIT:
        return out | {"initiate": WAIT}
    amount, strategy = P.split_mode(int(a[2]))
    return out | {"initiate": GO, "amount": amount, "strategy": strategy, "recipient": int(a[3]), "mules": int(a[4]),
                  "change": int(a[5])}


class OptionPolicy:
    """The launderers' policy under options (module docstring). ``tr``: the trainer (its featurizer, readout, view
    seeds, step counter and config); ``rng``: this policy's action samples; ``sched_rng``: the launch schedule of a
    scheduled world; ``greedy``: every factor's most likely option (a reported evaluation only).
    :meth:`begin_world` starts a world's rollout; the env's transitions reach it through :meth:`observe`."""

    def __init__(self, tr, learner: "OppoLearner", rng: np.random.Generator, sched_rng: np.random.Generator,
                 greedy: bool = False, scheduled: bool = False):
        self.tr, self.L, self.rng, self.sched_rng, self.greedy = tr, learner, rng, sched_rng, greedy
        self.begin_world(scheduled)

    def begin_world(self, scheduled: bool) -> None:
        self.scheduled, self.rollout = scheduled, Rollout(self.L.cfg.gamma)
        self.wait_until: dict[int, int] = {}
        self.ep, self.when, self.launched = None, {}, set()
        self.waits = np.zeros(len(P.DURATIONS), np.int64)        # effective wait choices, by original duration

    def finish(self, t_end: int) -> Rollout:
        self.rollout.close(t_end)
        return self.rollout

    def observe(self, agent: int, t: int, r: float, noop: bool) -> None:
        self.rollout.observe(agent, t, r, noop)

    def act(self, env, states: dict) -> dict:
        import hns.marl.train as mt                     # the loop's episode length (patched in tests)

        t, out, decide = self.tr.step_no, {}, []
        if self.scheduled and env.episode != self.ep:
            self.ep, self.launched = env.episode, set()
            self.when = {x.id: int(self.sched_rng.integers(0, LAUNCH_LAST + 1)) for x in env.launderers}
        for i, st in states.items():
            if len(st.branches) == 1:                    # in flight: always proceed
                out[i] = {"initiate": GO}
            elif self.scheduled:
                if i not in self.launched and env.sub >= self.when[i] and st.masks["initiate"][GO]:
                    self.launched.add(i)
                    decide.append(i)
                else:
                    out[i] = env_action(st, np.zeros(len(P.ACTION), np.int64))
            elif self.wait_until.get(i, -1) > t:
                out[i] = env_action(st, np.zeros(len(P.ACTION), np.int64))
            else:
                decide.append(i)
        if not decide:
            return out
        tr = self.tr
        ob = launderer_obs(env, tr.fz, decide, tr.gvec, tr.view_seed(1), tr.cfg.k2)
        obs = ob.split()
        masks = np.stack([option_masks(states[i], env.sub, mt.SUBSTEPS) for i in decide])
        gate_free = np.full(len(decide), not self.scheduled)
        tab, v, m = self.L.evaluate(collate(obs), masks)
        a = P.sample(tab, gate_free, self.rng, self.greedy)
        lp = self.L.logp(tab, a, gate_free)
        for j, i in enumerate(decide):
            self.rollout.add(Row(i, t, t - tr.world_step, obs[j], masks[j], bool(gate_free[j]), a[j], float(lp[j]),
                                 float(v[j]), float(m[j])))
            if a[j, 0] == P.WAIT:
                self.wait_until[i] = t + P.effective_duration(int(a[j, 1]), env.sub, mt.SUBSTEPS)
                self.waits[a[j, 1]] += 1
            else:
                self.wait_until.pop(i, None)
            out[i] = env_action(states[i], a[j])
        return out


# ---------------------------------------------------------------------------- the learner
class OppoLearner:
    """The launderers' actor and critic and their world-end update (module docstring)."""

    def __init__(self, cfg: MarlConfig, device):
        self.cfg, self.device = cfg, device
        self.actor = P.OptionActor(17, D_W_AGENT, L_CTRL, D_CAND).to(device)
        self.critic = P.ValueNet(17, D_W_AGENT, L_CTRL).to(device)
        self.critic.trunk.load_state_dict(self.actor.trunk.state_dict())    # the same start, then independent
        self.opt_a = torch.optim.Adam(self.actor.parameters(), lr=LR)
        self.opt_c = torch.optim.Adam(self.critic.parameters(), lr=LR)
        self.mb_rng = np.random.default_rng([cfg.seed, 13])                 # minibatch order, its own stream
        self.n_updates = 0                                                   # attempted actor optimizer steps
        self.stats: dict = {}

    # --- forward passes
    def _forward(self, ob: ObsBatch, masks: np.ndarray):
        g, f = ob.tensors(self.device)
        logits = self.actor(g, f["x_tx"], f["x_w"], f["x_c"], _t(ob.cand, self.device))
        return P.log_tables(logits, _t(masks, self.device))

    def _value(self, ob: ObsBatch) -> torch.Tensor:
        g, f = ob.tensors(self.device)
        return self.critic(g, f["x_tx"], f["x_w"], f["x_c"])

    @torch.no_grad()
    def evaluate(self, ob: ObsBatch, masks: np.ndarray):
        """For acting: numpy log-probability tables, ``V = m v`` and ``m`` of a batch of decisions. A non-finite
        probability or value raises."""
        self.actor.eval()
        self.critic.eval()
        tab = self._forward(ob, masks)
        m = wealth_scale(ob.x_c)
        v = self._value(ob).cpu().numpy() * m
        tab = {k: x.cpu().numpy() for k, x in tab.items()}
        feas = masks[:, P.SLICES["gate"]]
        if not (np.isfinite(tab["gate"][feas]).all() and np.isfinite(v).all()):
            raise FloatingPointError("launderer: non-finite probability or value at a decision")
        return tab, v, m

    def logp(self, tab: dict[str, np.ndarray], a: np.ndarray, gate_free: np.ndarray) -> np.ndarray:
        t = {k: torch.from_numpy(x) for k, x in tab.items()}
        lp = P.logp_taken(t, torch.from_numpy(a), torch.from_numpy(gate_free)).numpy()
        if not np.isfinite(lp).all():
            raise FloatingPointError("launderer: a sampled action has zero probability")
        return lp

    # --- the world-end update
    def _chunks(self, idx: np.ndarray, size: int):
        for s in range(0, len(idx), size):
            yield idx[s:s + size]

    def _batch(self, rows: list[Row], idx: np.ndarray):
        d = self.device
        return (collate([rows[i].obs for i in idx]), np.stack([rows[i].masks for i in idx]),
                _t(np.stack([rows[i].action for i in idx]), d), _t(np.array([rows[i].gate_free for i in idx]), d))

    @torch.no_grad()
    def _tables(self, rows: list[Row]):
        """Every decision's log-probability tables, sampled log-probability and value under the current networks."""
        self.actor.eval()
        self.critic.eval()
        tabs, lp, v = [], [], []
        for idx in self._chunks(np.arange(len(rows)), MICRO):
            ob, masks, act, gf = self._batch(rows, idx)
            t = self._forward(ob, masks)
            tabs.append(t)
            lp.append(P.logp_taken(t, act, gf).cpu().numpy())
            v.append(self._value(ob).cpu().numpy() * wealth_scale(ob.x_c))
        return tabs, np.concatenate(lp), np.concatenate(v)

    @torch.no_grad()
    def _kl(self, rows: list[Row], old: list) -> tuple[float, float, float]:
        """Mean old-to-new KL: joint over every decision, launch-conditional over those where a launch was feasible,
        duration-conditional over those with more than one duration."""
        self.actor.eval()
        s = np.zeros(3)
        n = np.zeros(3)
        for k, idx in enumerate(self._chunks(np.arange(len(rows)), MICRO)):
            ob, masks, act, gf = self._batch(rows, idx)
            joint, launch, dur = P.kl_parts(old[k], self._forward(ob, masks), gf)
            mk = _t(masks, self.device)
            sel = (torch.ones_like(joint, dtype=torch.bool), mk[:, P.SLICES["gate"]][:, P.LAUNCH],
                   mk[:, P.SLICES["dur"]].sum(1) > 1)
            for i, (x, m) in enumerate(zip((joint, launch, dur), sel)):
                s[i] += float(x[m].sum())
                n[i] += float(m.sum())
        kl = s / np.maximum(n, 1)
        if not np.isfinite(kl).all():
            raise FloatingPointError("launderer: non-finite KL")
        return tuple(float(x) for x in kl)

    def learn(self, ro: Rollout) -> dict:
        """The update after a world (module docstring); returns its statistics, also kept for :meth:`pop_stats`."""
        rows = ro.rows
        n = len(rows)
        G = ro.returns()
        st = {"n": n, "return_check": ro.check(G) if n else 0.0}
        if n == 0:                                              # a world with no decision updates nothing
            self.stats = st
            return st
        m = np.array([r.m for r in rows])
        v_old = np.array([r.v for r in rows])
        lp_old = np.array([r.logp for r in rows])
        old, lp_now, v_now = self._tables(rows)
        st["ratio_dev_max"] = float(np.abs(lp_now - lp_old).max())
        st["value_dev_max"] = float(np.abs(v_now - v_old).max())
        A = G - v_old
        s_a = max(ADV_MIN, float(np.sqrt(np.mean(A ** 2))))
        adv = A / s_a
        dev = self.device
        lp_old_t, adv_t = _t(lp_old.astype(np.float32), dev), _t(adv.astype(np.float32), dev)
        # actor: up to EPOCHS passes; the KL guard after every Adam step undoes the step that crosses it (its Adam
        # state included), keeps the steps before it, and stops the actor for this world
        kls, accepted, rejected, passes, ratios, clipped, ents, gnorm = [], 0, 0, 0, [], 0, [], []
        stop = False
        for _ in range(EPOCHS):
            for mb in self._chunks(self.mb_rng.permutation(n), MINIBATCH):
                snap = (copy.deepcopy(self.actor.state_dict()), copy.deepcopy(self.opt_a.state_dict()))
                self.actor.train()
                self.opt_a.zero_grad(set_to_none=True)
                for idx in self._chunks(mb, MICRO):
                    ob, masks, act, gf = self._batch(rows, idx)
                    t = self._forward(ob, masks)
                    i_t = _t(idx, dev)
                    ratio = torch.exp(P.logp_taken(t, act, gf) - lp_old_t[i_t])
                    a_ = adv_t[i_t]
                    surr = torch.minimum(ratio * a_, ratio.clamp(1 - CLIP_RATIO, 1 + CLIP_RATIO) * a_)
                    h = P.entropy(t, gf)
                    loss = -(surr.sum() + ENT_COEF * h.sum()) / len(mb)
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"launderer: non-finite actor loss at update {self.n_updates}")
                    loss.backward()
                    ratios.append(ratio.detach().cpu().numpy())
                    clipped += int(((ratio - 1).abs() > CLIP_RATIO).sum())
                    ents.append(h.detach().cpu().numpy())
                g = float(torch.nn.utils.clip_grad_norm_(self.actor.parameters(), GRAD_CLIP))
                if not math.isfinite(g):
                    raise FloatingPointError(f"launderer: non-finite actor gradient at update {self.n_updates}")
                gnorm.append(g)
                self.opt_a.step()
                self.n_updates += 1
                kl = self._kl(rows, old)                    # against the actor that collected the world
                kls.append(kl)
                if max(kl) > KL_MAX:
                    self.actor.load_state_dict(snap[0])
                    self.opt_a.load_state_dict(snap[1])
                    rejected, stop = 1, True
                    break
                accepted += 1
            if stop:
                break
            passes += 1
        # critic: EPOCHS passes on 0.5 (v - G / m)^2
        y = _t((G / m).astype(np.float32), dev)
        closs, cgnorm = [], []
        self.critic.train()
        for _ in range(EPOCHS):
            for mb in self._chunks(self.mb_rng.permutation(n), MINIBATCH):
                self.opt_c.zero_grad(set_to_none=True)
                tot = 0.0
                for idx in self._chunks(mb, MICRO):
                    ob = collate([rows[i].obs for i in idx])
                    loss = 0.5 * ((self._value(ob) - y[_t(idx, dev)]) ** 2).sum() / len(mb)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("launderer: non-finite critic loss")
                    loss.backward()
                    tot += float(loss.detach())
                g = float(torch.nn.utils.clip_grad_norm_(self.critic.parameters(), GRAD_CLIP))
                if not math.isfinite(g):
                    raise FloatingPointError("launderer: non-finite critic gradient")
                cgnorm.append(g)
                closs.append(tot)
                self.opt_c.step()
        r = np.concatenate(ratios) if ratios else np.ones(1)
        masks = np.stack([x.masks for x in rows])
        support = [n, int(masks[:, P.SLICES["gate"]][:, P.LAUNCH].sum()), int((masks[:, P.SLICES["dur"]].sum(1) > 1).sum())]
        st |= {"adv_scale": s_a, "return_mean": float(G.mean()), "value_bias": float((v_old - G).mean()),
               "steps_accepted": accepted, "steps_rejected": rejected, "passes_complete": passes,
               "kl": kls, "kl_final": list(kls[accepted - 1]) if accepted else [0.0, 0.0, 0.0], "kl_support": support,
               "ratio_q": np.quantile(r, [0.01, 0.5, 0.99]).tolist(), "clip_frac": clipped / max(len(r), 1),
               "entropy": float(np.concatenate(ents).mean()) if ents else None,
               "grad_actor": float(np.mean(gnorm)) if gnorm else None, "grad_actor_max": max(gnorm, default=None),
               "critic_loss_first": closs[0] if closs else None, "critic_loss_last": closs[-1] if closs else None,
               "grad_critic": float(np.mean(cgnorm)) if cgnorm else None,
               "decisions": decision_summary(rows, G)}
        self.stats = st
        return st

    def pop_stats(self) -> dict:
        st, self.stats = self.stats, {}
        return st or {"n": 0}

    # --- persistence
    def state(self) -> dict:
        return {"actor": self.actor.state_dict(), "critic": self.critic.state_dict(), "opt_a": self.opt_a.state_dict(),
                "opt_c": self.opt_c.state_dict(), "mb_rng": self.mb_rng.bit_generator.state, "n_updates": self.n_updates}

    def load_state(self, ck: dict, networks_only: bool = False) -> None:
        self.actor.load_state_dict(ck["actor"])
        self.critic.load_state_dict(ck["critic"])
        if not networks_only:
            self.opt_a.load_state_dict(ck["opt_a"])
            self.opt_c.load_state_dict(ck["opt_c"])
            self.mb_rng.bit_generator.state = ck["mb_rng"]
            self.n_updates = ck["n_updates"]


# ---------------------------------------------------------------------------- what the decisions were
BANDS = (1.5, 3.0)                          # log1p(source balance): under 3.5 BTC, 3.5 to 19 BTC, over 19 BTC
TIME_BINS = (250, 400, 450, 490)            # sub-steps of the world: 0-249, 250-399, 400-449, 450-489, 490-499


def band_of(row: Row) -> str:
    x = float(row.obs[4][1])
    return "low" if x < BANDS[0] else "mid" if x < BANDS[1] else "high"


def decision_summary(rows: list[Row], G: np.ndarray | None = None) -> dict:
    """Counts of what the launderers decided: waits by duration, launches by mode, recipient-free factors, no-ops,
    and per balance band x world-time bin ``[decisions where a launch was feasible, launches]``; the critic's bias
    ``V - G`` per band if returns are given."""
    out = {"decisions": len(rows), "scheduled": sum(not r.gate_free for r in rows),
           "waits": [0] * len(P.DURATIONS), "launches": [0] * P.N_MODES, "launch_noops": 0,
           "mules_reuse": 0, "change_reuse": 0, "hazard": {}}
    hz = defaultdict(lambda: [0, 0])
    bias = defaultdict(list)
    for j, r in enumerate(rows):
        a = r.action
        if a[0] == P.WAIT:
            out["waits"][int(a[1])] += 1
        else:
            out["launches"][int(a[2])] += 1
            out["launch_noops"] += r.noop
            out["mules_reuse"] += int(a[4])
            out["change_reuse"] += int(a[5])
        key = f"{band_of(r)}/{int(np.searchsorted(TIME_BINS, r.tw, side='right'))}"
        if r.masks[P.SLICES["gate"]][P.LAUNCH] and r.gate_free:
            hz[key][0] += 1
            hz[key][1] += int(a[0] == P.LAUNCH)
        if G is not None:
            bias[band_of(r)].append(r.v - G[j])
    out["hazard"] = dict(hz)
    if G is not None:
        out["bias_by_band"] = {k: [len(v), float(np.mean(v))] for k, v in bias.items()}
    return out


def calibration(rows: list[Row], G: np.ndarray) -> dict:
    """The critic's ``V`` before each decision against its plain return ``G`` (one world)."""
    if not rows:
        return {"n": 0}
    v = np.array([r.v for r in rows])
    m = np.array([r.m for r in rows])
    e = v - G
    w = (m - 1.0) * WEALTH_REF                                  # the BTC the launderer could spend
    var = float(G.var())
    strata = defaultdict(list)
    for j, r in enumerate(rows):
        strata[band_of(r)].append(e[j])
        strata[f"t{int(np.searchsorted(TIME_BINS, r.tw, side='right'))}"].append(e[j])
    return {"n": len(rows), "bias": float(e.mean()), "g_abs": float(np.abs(G).mean()), "g_mean": float(G.mean()),
            "bias_btc": float((e * w).sum() / w.sum()) if w.sum() > 0 else None, "rmse": float(np.sqrt((e ** 2).mean())),
            "rmse_norm": float(np.sqrt(((e / m) ** 2).mean())),
            "explained_variance": 1.0 - float(e.var()) / var if var > 0 else None,
            "strata": {k: {"n": len(x), "bias": float(np.mean(x))} for k, x in sorted(strata.items())}}
