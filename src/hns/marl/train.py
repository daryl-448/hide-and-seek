"""MARL pretraining run: launderers, innocents and the inspector co-evolve in SISTS-lite (SPEC §10, M5).

One run is ``episodes / 10`` worlds (epochs). Per world: draw its parameters
from the multiverse (§9.5), build the env with the three learned policies,
and play 10 episodes of 50 sub-steps. Every sub-step is one env step (the
policies act with epsilon-greedy soft-Q), then one gradient step per role
(launderer TD, innocent TD, detector-T ``TD + L_DA + lambda L_adv``) once
the role's buffer has warmed up. When the world ends, every open transition
becomes terminal, the world is exported (pruned and recomputed, §9.2), and
detector-W takes as many ``L_DA + lambda L_adv`` steps on the export as
detector-T took during the world.

No real label is read anywhere in this loop. The Full arm reads unlabelled
real fitting-split views for ``L_aux``; NDA reads no real batch at all.

Outputs under ``runs/marl/<arm>/<seed>/``: ``config.yaml``,
``metrics.jsonl`` (one ``episode`` record per episode and one ``epoch``
record per world, with the §10 health checks), ``ckpt/last.pt`` (every
``ckpt_worlds`` worlds: models, optimizers, buffers, the source pool and all
RNG states; a run resumes from it) and ``ckpt/final.pt`` plus
``result.json`` when the run completes.

Health checks per epoch (§10): per-role mean episode reward; detection rate
(share of illicit cash-out txs that were flagged); the illicit/licit ratio
among executed (committed) txs per episode; launderer success fraction (BTC
cashed out at a service undetected / BTC of proceeds received in the
world); each strategy's share of the started tranches. A run is
**degenerate** if the detection rate or the success fraction stays above
0.95 or below 0.05 over the last 20 epochs.

Usage: ``uv run python -m hns.marl.train [configs/marl.yaml] --arm full --seed 0 [--episodes N] [--out DIR]``
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import yaml

from hns.data.norm import NORM_DIR, Normalizer
from hns.eval.rf_anchor import git_sha
from hns.marl.learner import DetectorLearner, MarlConfig, RealViews, RoleLearner, SimW
from hns.marl.oppo import CURRICULUM_WORLDS, OppoLearner, OptionPolicy
from hns.marl.obs import Featurizer, detector_obs, innocent_obs, launderer_obs, pack_masks, world_readout
from hns.models.qnet import sample_actions
from hns.sim.entities import AMOUNTS, GO, STRATEGIES
from hns.sim.env import SimEnv
from hns.sim.export import export_world
from hns.sim.labels import ILLICIT, UNKNOWN
from hns.sim.live import LiveGraph
from hns.sim.multiverse import draw_world
from hns.sim.world import EPISODES, SUBSTEPS, WorldParams

RUNS = Path("runs/marl")
DEGENERATE_WINDOW, DEGENERATE_HI, DEGENERATE_LO = 20, 0.95, 0.05
_OBS = {"launderer": launderer_obs, "innocent": innocent_obs}


def read_records(path: Path) -> list[tuple[str, dict]]:
    """``(line, record)`` for every record of a ``metrics.jsonl``.

    A kill during an append can leave an incomplete last line; that line is dropped. A line that does not parse
    anywhere else is an error.
    """
    lines = path.read_text().splitlines()
    out = []
    for i, x in enumerate(lines):
        try:
            out.append((x, json.loads(x)))
        except json.JSONDecodeError:
            if i != len(lines) - 1:
                raise
    return out


def btc_fractions(env) -> dict[str, float]:
    """Where the world's launderer proceeds went, as fractions of the proceeds (§10 health).

    ``success_frac``: cashed out undetected. ``frozen_frac``: everything frozen in launderer txs, a flagged
    setup's change to the source included (the M5 runs of 2026-11-10 left that change out of ``frozen_frac``).
    ``frozen_change_frac``: that change alone. ``fee_frac``: fees paid. ``held_frac``: still spendable at
    launderer addresses at world end. ``btc_utility``: ``success_frac - frozen_frac``. ``accounting_gap``:
    ``1 - (cashed + frozen + fees + held) / proceeds``, zero up to rounding.
    """
    a = env.acct
    p = max(a["proceeds"], 1e-12)
    held = env.launderer_held()
    return {"success_frac": a["cashed"] / p, "frozen_frac": a["frozen"] / p, "frozen_change_frac": a["frozen_change"] / p,
            "fee_frac": a["fees"] / p, "held_frac": held / p, "btc_utility": (a["cashed"] - a["frozen"]) / p,
            "accounting_gap": 1.0 - (a["cashed"] + a["frozen"] + a["fees"] + held) / p}


def degenerate(epochs: list[dict], window: int = DEGENERATE_WINDOW) -> list[str]:
    """Which health series are stuck above 0.95 or below 0.05 over the last ``window`` epochs (§10)."""
    out = []
    if len(epochs) < window:
        return out
    for key in ("detection_rate", "success_frac"):
        x = [e[key] for e in epochs[-window:]]
        if all(v > DEGENERATE_HI for v in x):
            out.append(f"{key} > {DEGENERATE_HI}")
        if all(v < DEGENERATE_LO for v in x):
            out.append(f"{key} < {DEGENERATE_LO}")
    return out


class _AgentPolicy:
    """A role's shared Q-network as the env's policy; every state it sees becomes a buffer entry."""

    def __init__(self, trainer: "Trainer", role: str, salt: int):
        self.tr, self.role, self.salt = trainer, role, salt

    def act(self, env, states: dict) -> dict:
        if not states:
            return {}
        tr, L = self.tr, self.tr.roles[self.role]
        ids = list(states)
        ob = _OBS[self.role](env, tr.fz, ids, tr.gvec, tr.view_seed(self.salt), tr.cfg.k2)
        masks, active = pack_masks(self.role, states, ids)
        a = L.act(ob, masks, tr.eps_for(self.role), tr.rng)
        if self.role in tr.learn:
            for j, (i, o) in enumerate(zip(ids, ob.split())):
                oid = L.buf.add(o, masks[j], active[j], tr.step_no)
                prev = L.tail.get(i)
                if prev is not None:
                    L.buf.link(prev, oid)        # the agent's previous transition now has its next state
                L.tail[i] = oid
        names = list(L.branches)
        return {i: {b: int(a[j, bi]) for bi, b in enumerate(names) if b in states[i].branches}
                for j, i in enumerate(ids)}


class _DetectorPolicy:
    """detector-T as the acting inspector: global branches from the global readout, flags by ``Q_flag - Q_skip``.

    Exploration of ``select_tx`` (ours): the env flags the top ``b`` share of
    scores, so each candidate's score is its margin's rank in ``(0, 1)``, or,
    with probability epsilon, a uniform draw.
    """

    def __init__(self, trainer: "Trainer"):
        self.tr, self.depth, self.cand_ids = trainer, 1, []

    def act_global(self, env, state) -> dict:
        tr, D = self.tr, self.tr.det
        tr.gvec = world_readout(env, tr.last)
        a = sample_actions(D.global_q(tr.gvec), D.gq.sizes, D.g_tau, tr.eps_for("detector"), tr.rng)[0]
        if "detector" in tr.learn:
            D.gbuf.add(tr.gvec, a)
        self.depth = int(a[0]) + 1
        return {"analysis_depth": int(a[0]), "update_strategy": int(a[1])}

    def score(self, env, cands: list) -> np.ndarray:
        tr, D, n = self.tr, self.tr.det, len(cands)
        self.cand_ids = []
        if n == 0:
            return np.zeros(0)
        g, cfg = env.graph, tr.cfg
        tids = np.arange(g.n_tx - n, g.n_tx)      # the tentative txs, as if committed
        ob = detector_obs(g, tr.fz, tids, env.ts, tr.view_seed(3), cfg.k1, cfg.k2 if self.depth == 2 else 0,
                          cfg.obs_cap)
        margin = D.flag_margin(ob)
        rank = (np.argsort(np.argsort(margin, kind="stable"), kind="stable") + tr.rng.random(n)) / n
        score = np.where(tr.rng.random(n) < tr.eps_for("detector"), tr.rng.random(n), rank)
        if "detector" in tr.learn:
            ones, act = np.ones(2, bool), np.ones(1, bool)
            self.cand_ids = [D.flag_buf.add(o, ones, act, tr.step_no) for o in ob.split()]
        return score


ROLES = ("launderer", "innocent", "detector")


class Trainer:
    """The MARL loop. ``learn``: the roles that store transitions and update (all three in M5); a role outside it
    acts with its current networks at the fixed epsilon ``fixed_eps[role]`` and is never updated (frozen-opponent
    diagnostics, ``hns.marl.capability``). A frozen detector also skips detector-W and the world export."""

    def __init__(self, cfg: MarlConfig, device, centre: WorldParams, stats_t: dict, norm_w: Normalizer,
                 real_t: RealViews | None = None, real_w: RealViews | None = None, out: Path | None = None,
                 log=print, learn=ROLES, fixed_eps: dict | None = None):
        self.cfg, self.device, self.centre, self.norm_w, self.out, self.log = cfg, device, centre, norm_w, out, log
        self.learn, self.fixed_eps = frozenset(learn), dict(fixed_eps or {})
        if not self.learn <= set(ROLES) or any(r not in self.fixed_eps for r in ROLES if r not in self.learn):
            raise ValueError("every frozen role needs a fixed epsilon")
        torch.manual_seed(cfg.seed)
        self.fz = Featurizer(stats_t)
        self.oppo = cfg.launderer == "oppo"          # the launderer learns with OPPO-MC (hns.marl.oppo)
        self.roles = {"launderer": OppoLearner(cfg, device) if self.oppo else RoleLearner("launderer", cfg, device),
                      "innocent": RoleLearner("innocent", cfg, device)}
        self.det = DetectorLearner(cfg, device, self.fz, len(norm_w.stats["wallet_W"]), real_t, real_w)
        self.rng = np.random.default_rng([cfg.seed, 1])     # behaviour: every role's exploration
        self.replay_rng = np.random.default_rng([cfg.seed, 2])   # replay and batch sampling (its own stream, so a
        #                                                          replay variant leaves the exploration draws alone)
        self.after_update = None    # optional callable(trainer), run after each sub-step's updates (diagnostics)
        lpol = (OptionPolicy(self, self.roles["launderer"], np.random.default_rng([cfg.seed, 11]),
                             np.random.default_rng([cfg.seed, 12])) if self.oppo else _AgentPolicy(self, "launderer", 1))
        self.policies = (lpol, _AgentPolicy(self, "innocent", 2), _DetectorPolicy(self))
        self.world, self.step_no, self.episode = 0, 0, 0
        self.world_step, self.launches = 0, np.zeros((len(STRATEGIES), len(AMOUNTS), 2), np.int64)   # this world's
        self.gvec = np.zeros(8, np.float32)
        self.last: dict = {}
        self.epochs: list[dict] = []
        self.sec = defaultdict(float)

    # --- schedules and seeds
    @property
    def eps(self) -> float:
        c = self.cfg
        return max(c.eps_end, c.eps_start - (c.eps_start - c.eps_end) * self.episode / c.eps_episodes)

    def eps_for(self, role: str) -> float:
        """A role's exploration rate: fixed if the role is given one, else the schedule."""
        return self.fixed_eps.get(role, self.eps)

    def view_seed(self, salt: int) -> int:
        return (self.cfg.seed * 1_000_003 + self.step_no) * 16 + salt

    # --- one sub-step
    def after_step(self, env, res, ep: dict) -> None:
        """Turn the step's outcome into transitions, labelled source txs and health counts."""
        observe = getattr(env.launderer_policy, "observe", None)      # an option policy chains its own decisions
        for tr in res.transitions:
            if tr.role == "launderer" and observe is not None:
                observe(tr.agent, self.step_no, tr.reward, tr.noop)
            if tr.role in self.learn and not (self.oppo and tr.role == "launderer"):
                L = self.roles[tr.role]
                L.buf.set(L.tail[tr.agent], [tr.action.get(b, 0) for b in L.branches],
                          [b in tr.realized for b in L.branches], tr.reward)
            ep[f"r_{tr.role}"] += tr.reward
            if tr.role == "launderer":
                ep["r_launderer_tranche"] += tr.reward_tranche      # M5's first reward, a diagnostic (§9.3)
                # what the launderers did (repair cycle 2's logs): decisions by state and outcome, launches by
                # strategy x amount x change, and the world's reward discounted per sub-step from the world's start
                idle = len(tr.branches) > 1
                kind = "noop" if tr.noop else ("go" if tr.action["initiate"] == GO else "wait")
                ep[f"l_{'idle' if idle else 'flight'}_{kind}"] += 1
                ep["l_idle_no_launch"] += idle and not tr.masks["initiate"][GO]
                ep["r_launderer_disc"] += self.cfg.gamma ** (self.step_no - self.world_step) * tr.reward
                if idle and kind == "go":
                    self.launches[tr.action["strategy"], tr.action["amount"], tr.action["change"]] += 1
        inn = self.roles["innocent"]
        for i, r in res.continuation_rewards.items():
            if "innocent" in self.learn and i in inn.tail:
                # the forward route's hop 2 pays the decision that chose it, discounted to the decision's sub-step
                j = inn.tail[i]
                inn.buf.add_reward(j, r * self.cfg.gamma ** (self.step_no - inn.buf.t_of(j)))
            ep["r_innocent"] += r
        D, dpol, led = self.det, self.policies[2], env.ledger
        depth, fb, d_learn = dpol.depth, self.det.flag_buf, "detector" in self.learn
        lab_t, lab_y = [], []
        n_flag = n_frozen = n_commit = 0
        for j, c in enumerate(res.candidates):
            f, ill = c["flagged"], c["illicit"]
            if d_learn:
                r = ((1.0 if ill else -0.1) - 0.01 * depth) if f else (-1.0 if c["cashout"] else 0.0)
                oid = dpol.cand_ids[j]
                fb.set(oid, (int(f),), (True,), r)
                fb.end(oid)                          # a flag's outcome is known at once: terminal
            n_flag += f
            n_frozen += f and ill
            ep["cash"] += c["cashout"]
            ep["cash_flag"] += c["cashout"] and f
            tid = c["tid"]
            if tid >= 0:
                n_commit += 1
                ep["illicit" if ill else "licit"] += 1
                cls = led.tx_label[tid]
                if cls != UNKNOWN:
                    lab_t.append(tid)
                    lab_y.append(cls == ILLICIT)
        if d_learn:
            D.pool.add(lab_t, lab_y)
            D.gbuf.set_reward(res.detector_reward)
        ep["r_detector"] += res.detector_reward
        ep["n_cand"] += len(res.candidates)
        ep["n_flag"] += n_flag
        ep["n_frozen"] += n_frozen
        ep["n_noop"] += res.n_noop
        ep["depth2"] += depth == 2
        ep["budget"] += env.budget
        ep["n_commit"] += n_commit
        # this sub-step's flagging, and the episode's running commit total (not this sub-step's count, which
        # n_cand - n_flag + n_frozen already gives)
        self.last = {"n_cand": len(res.candidates), "n_flag": n_flag, "n_frozen": n_frozen,
                     "episode_committed": ep["n_commit"], "depth": depth}

    def update(self, losses: dict) -> None:
        """One gradient step per learning role (after its warm-up)."""
        for name, L in self.roles.items():
            if name not in self.learn or (self.oppo and name == "launderer"):
                continue
            t0 = time.perf_counter()
            v = L.update(self.replay_rng)
            self.sec[f"update_{name}"] += time.perf_counter() - t0
            if v is not None:
                losses[f"td_{name}"].append(v)
        if "detector" not in self.learn:
            return
        t0 = time.perf_counter()
        d = self.det.update(self.replay_rng, self.view_seed(4))
        self.sec["update_detector_t"] += time.perf_counter() - t0
        for k, v in (d or {}).items():
            losses[f"t_{k}"].append(v)

    # --- one world (epoch)
    def run_world(self) -> dict:
        cfg, w = self.cfg, self.world
        ss = np.random.SeedSequence([cfg.seed, w])
        params = draw_world(self.centre, np.random.default_rng(ss.spawn(1)[0]))
        env_seed = int(ss.generate_state(1)[0])
        env = SimEnv(params, env_seed, *self.policies)
        env.graph = LiveGraph()
        self.det.pool.begin(env.graph, env_seed, params.mask_rate)
        self.last = {}
        self.world_step, self.launches = self.step_no, np.zeros((len(STRATEGIES), len(AMOUNTS), 2), np.int64)
        if self.oppo:                                    # the first worlds launch on a schedule (curriculum, §10)
            self.policies[0].begin_world(scheduled=w < CURRICULUM_WORLDS)
        losses, tot = defaultdict(list), defaultdict(float)
        u0, ratio, t_world = self.det.n_updates, [], time.perf_counter()
        for e in range(EPISODES):
            self.episode = w * EPISODES + e
            env.start_episode(e)
            if self.last:
                self.last["episode_committed"] = 0       # a new episode has committed nothing yet
            ep = defaultdict(float)
            for _ in range(SUBSTEPS):
                t0 = time.perf_counter()
                res = env.step()
                self.after_step(env, res, ep)
                self.sec["env"] += time.perf_counter() - t0
                self.update(losses)
                if self.after_update is not None:
                    self.after_update(self)
                self.step_no += 1
            env.end_episode()
            ratio.append(ep["illicit"] / max(ep["licit"], 1.0))
            self._write({"kind": "episode", "episode": self.episode, "world": w, "eps": self.eps,
                         "reward_launderer": ep["r_launderer"] / params.n_launderers,
                         "reward_launderer_tranche": ep["r_launderer_tranche"] / params.n_launderers,
                         "reward_innocent": ep["r_innocent"] / params.n_innocents, "reward_detector": ep["r_detector"],
                         "illicit_licit_ratio": ratio[-1],
                         "detection_rate": ep["cash_flag"] / max(ep["cash"], 1.0), "n_tx": int(ep["n_commit"])})
            for k, v in ep.items():
                tot[k] += v
        # world end: terminal transitions, then detector-W on the pruned export (§10)
        for name, L in self.roles.items():
            if self.oppo and name == "launderer":         # every decision chain closes, then the world's update
                ro = self.policies[0].finish(self.step_no)
                if name in self.learn:
                    t0 = time.perf_counter()
                    L.learn(ro)
                    self.sec["update_launderer"] += time.perf_counter() - t0
            elif name in self.learn:
                L.end_world()
        sim = None
        if "detector" in self.learn:
            self.det.gbuf.end()
            t0 = time.perf_counter()
            n_w = self.det.n_updates - u0
            sim = SimW(export_world(env), self.norm_w, cfg, env_seed, params.mask_rate)
            if sim.n:
                for j in range(n_w):
                    for k, v in self.det.update_w(sim, self.replay_rng, self.view_seed(5) + 16 * j).items():
                        losses[f"w_{k}"].append(v)
            self.sec["detector_w"] += time.perf_counter() - t0
        mean = {k: float(np.mean(v)) for k, v in losses.items()}
        bad = [k for k, v in mean.items() if not math.isfinite(v)]
        if bad:
            raise FloatingPointError(f"world {w}: non-finite losses {bad}")
        a, n_sub = env.acct, EPISODES * SUBSTEPS
        started = int(a["strategy"].sum())
        rec = {
            "kind": "epoch", "world": w, "episode": (w + 1) * EPISODES, "eps": self.eps,
            "reward": {"launderer": tot["r_launderer"] / params.n_launderers / EPISODES,
                       "innocent": tot["r_innocent"] / params.n_innocents / EPISODES,
                       "detector": tot["r_detector"] / EPISODES,
                       "launderer_tranche": tot["r_launderer_tranche"] / params.n_launderers / EPISODES,
                       # per launderer, the world's reward discounted per sub-step from the world's start
                       "launderer_discounted": tot["r_launderer_disc"] / params.n_launderers},
            "detection_rate": tot["cash_flag"] / max(tot["cash"], 1.0),
            "illicit_licit_ratio": float(np.mean(ratio)),
            **btc_fractions(env),
            "strategy_share": {s: (int(c) / started if started else 0.0) for s, c in zip(STRATEGIES, a["strategy"])},
            "tranches": started, "flag_precision": tot["n_frozen"] / max(tot["n_flag"], 1.0),
            "flag_share": tot["n_flag"] / max(tot["n_cand"], 1.0), "budget_mean": tot["budget"] / n_sub,
            "depth2_share": tot["depth2"] / n_sub, "noop_share": tot["n_noop"] / max(tot["n_cand"] + tot["n_noop"], 1.0),
            "n_tx": int(tot["n_commit"]), "n_illicit": int(tot["illicit"]), "n_cashout": int(tot["cash"]),
            "world_params": {"n_launderers": params.n_launderers, "budget": params.budget,
                             "mask_rate": params.mask_rate, "offset": params.offset, "n_innocents": params.n_innocents},
            "export": {"n_tx": sim.n_tx, "n_labelled_wallets": sim.n} if sim is not None else None,
            "losses": mean,
            "launderer": {"decisions": {k[2:]: int(v) for k, v in tot.items() if k.startswith("l_")},
                          "launches": self.launches.tolist()},
            "learner": {name: L.pop_stats() for name, L in self.roles.items() if name in self.learn},
            "replay": {name: L.buf.support() for name, L in self.roles.items() if name in self.learn and hasattr(L, "buf")},
            "updates": {"launderer": self.roles["launderer"].n_updates, "innocent": self.roles["innocent"].n_updates,
                        "detector_t": self.det.n_updates, "detector_w": self.det.n_w_updates},
            "buffer": {name: L.buf.ready for name, L in self.roles.items() if hasattr(L, "buf")}
                      | {"flag": self.det.flag_buf.ready, "pool": self.det.pool.n},
            "sec": {"world": time.perf_counter() - t_world, **{k: round(v, 2) for k, v in self.sec.items()}},
        }
        self.sec.clear()
        self.epochs.append(rec)
        rec["degenerate"] = degenerate(self.epochs)
        self._write(rec)
        self.world += 1
        return rec

    # --- persistence
    def _write(self, rec: dict) -> None:
        if self.out is not None:
            with open(self.out / "metrics.jsonl", "a") as f:
                f.write(json.dumps(rec) + "\n")

    def _models(self) -> dict:
        D, L = self.det, self.roles["launderer"]
        lm = ({"launderer_actor": L.actor, "launderer_critic": L.critic} if self.oppo
              else {"launderer": L.net, "launderer_targ": L.targ})
        return lm | {"innocent": self.roles["innocent"].net, "innocent_targ": self.roles["innocent"].targ,
                "det_t": D.det_t, "det_w": D.det_w, "gq": D.gq, "gq_targ": D.gq_targ}

    def load_models(self, path: Path, roles) -> None:
        """Load the networks of ``roles`` (and their target networks) from a checkpoint's ``models``: another run's
        trained opponents, for a frozen-opponent diagnostic. Optimizers and buffers are left alone."""
        keys = {"launderer": ("launderer_actor", "launderer_critic") if self.oppo else ("launderer", "launderer_targ"),
                "innocent": ("innocent", "innocent_targ"),
                "detector": ("det_t", "det_w", "gq", "gq_targ")}
        ck = torch.load(path, map_location=self.device, weights_only=False)
        mods = self._models()
        for r in roles:
            for k in keys[r]:
                mods[k].load_state_dict(ck["models"][k])

    def _opts(self) -> dict:
        L = self.roles["launderer"]
        lo = {"launderer_actor": L.opt_a, "launderer_critic": L.opt_c} if self.oppo else {"launderer": L.opt}
        return lo | {"innocent": self.roles["innocent"].opt, "det_t": self.det.opt_t, "det_w": self.det.opt_w}

    def save(self, path: Path, full: bool = True) -> None:
        """``full``: everything a resume needs. Otherwise the models only (the transfer checkpoints)."""
        ck = {"config": asdict(self.cfg), "world": self.world, "git_sha": git_sha(),
              "models": {k: m.state_dict() for k, m in self._models().items()}}
        if full:
            D = self.det
            ck |= {"opts": {k: o.state_dict() for k, o in self._opts().items()},
                   "buffers": {name: L.buf for name, L in self.roles.items() if hasattr(L, "buf")}
                              | {"flag": D.flag_buf, "global": D.gbuf, "pool": D.pool},
                   "counters": {"step_no": self.step_no, "n_updates": D.n_updates, "n_w_updates": D.n_w_updates,
                                "launderer": self.roles["launderer"].n_updates,
                                "innocent": self.roles["innocent"].n_updates},
                   "epochs": self.epochs, "rng": self.rng.bit_generator.state,
                   "rng_replay": self.replay_rng.bit_generator.state, "rng_cpu": torch.get_rng_state(),
                   "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
            if self.oppo:                                # saved at a world boundary: no decision is pending
                pol = self.policies[0]
                ck["oppo"] = {"rng": pol.rng.bit_generator.state, "sched": pol.sched_rng.bit_generator.state,
                             "mb": self.roles["launderer"].mb_rng.bit_generator.state}
        tmp = path.with_suffix(".tmp")
        torch.save(ck, tmp, pickle_protocol=5)
        os.replace(tmp, path)       # never leave a half-written checkpoint

    def load(self, path: Path) -> None:
        ck = torch.load(path, map_location="cpu", weights_only=False)   # RNG states must stay on CPU
        if ck["config"] != asdict(self.cfg):
            raise ValueError(f"{path} was written by a different config")
        for k, m in self._models().items():
            m.load_state_dict(ck["models"][k])
        for k, o in self._opts().items():
            o.load_state_dict(ck["opts"][k])
        b, c, D = ck["buffers"], ck["counters"], self.det
        for name, L in self.roles.items():
            if hasattr(L, "buf"):
                L.buf = b[name]
        if self.oppo:
            pol = self.policies[0]
            pol.rng.bit_generator.state, pol.sched_rng.bit_generator.state = ck["oppo"]["rng"], ck["oppo"]["sched"]
            self.roles["launderer"].mb_rng.bit_generator.state = ck["oppo"]["mb"]
        D.flag_buf, D.gbuf, D.pool = b["flag"], b["global"], b["pool"]
        self.world, self.step_no, self.epochs = ck["world"], c["step_no"], ck["epochs"]
        D.n_updates, D.n_w_updates = c["n_updates"], c["n_w_updates"]
        self.roles["launderer"].n_updates, self.roles["innocent"].n_updates = c["launderer"], c["innocent"]
        self.rng.bit_generator.state = ck["rng"]
        self.replay_rng.bit_generator.state = ck["rng_replay"]
        torch.set_rng_state(ck["rng_cpu"])
        if ck["rng_cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(ck["rng_cuda"])

    # --- the run
    def run(self) -> dict:
        cfg, out = self.cfg, self.out
        n_worlds = cfg.episodes // EPISODES
        if cfg.episodes % EPISODES:
            raise ValueError(f"episodes must be a multiple of {EPISODES} (one world)")
        done = out / "result.json"
        if done.exists():
            res = json.loads(done.read_text())
            if res["config"] == asdict(cfg):
                self.log(f"{out}: done, reusing")
                return res
        last = out / "ckpt" / "last.pt"
        if last.exists():
            self.load(last)
            # drop records of worlds played after the checkpoint (a killed run), and a record the kill cut short:
            # those worlds are replayed
            keep = [x for x, r in read_records(out / "metrics.jsonl") if r["world"] < self.world]
            (out / "metrics.jsonl").write_text("".join(x + "\n" for x in keep))
            self.log(f"{out}: resuming at world {self.world}")
        else:
            (out / "ckpt").mkdir(parents=True, exist_ok=True)
            (out / "config.yaml").write_text(yaml.safe_dump(
                {**asdict(cfg), "git_sha": git_sha(), "device": str(self.device), "centre": self.centre.to_dict()},
                sort_keys=False))
            (out / "metrics.jsonl").write_text("")
        while self.world < n_worlds:
            r = self.run_world()
            ls = r["losses"]
            self.log(f"{cfg.arm} seed {cfg.seed} epoch {r['world'] + 1}/{n_worlds} eps {r['eps']:.2f} | "
                     f"R L {r['reward']['launderer']:+.3f} I {r['reward']['innocent']:+.3f} D {r['reward']['detector']:+.1f} | "
                     f"det {r['detection_rate']:.2f} succ {r['success_frac']:.2f} ill/lic {r['illicit_licit_ratio']:.3f} "
                     f"b {r['budget_mean']:.3f} | L_y T {ls.get('t_y_src', float('nan')):.3f} W {ls.get('w_y_src', float('nan')):.3f} | "
                     f"{r['sec']['world']:.0f}s" + (f" | DEGENERATE: {r['degenerate']}" if r["degenerate"] else ""))
            if self.world % cfg.ckpt_worlds == 0 or self.world == n_worlds:
                self.save(last)
        self.save(out / "ckpt" / "final.pt", full=False)
        res = {"config": asdict(cfg), "episodes": self.world * EPISODES, "degenerate": self.epochs[-1]["degenerate"],
               "final": {k: self.epochs[-1][k] for k in ("reward", "detection_rate", "success_frac",
                                                         "illicit_licit_ratio", "strategy_share")}}
        done.write_text(json.dumps(res, indent=1))
        return res


def build(cfg: MarlConfig, device, centre: WorldParams, out: Path | None, log=print) -> Trainer:
    """A trainer on the real frozen moments; the Full arm also loads the real fitting-split views."""
    from hns.models.hrg import TaskData

    stats_t = Normalizer.load(NORM_DIR / "T.json").stats
    norm_w = Normalizer.load(NORM_DIR / f"W_seed{cfg.seed}.json")
    real_t = real_w = None
    if cfg.da:
        real_t = RealViews.task_t(TaskData("T", cfg.seed, k1=cfg.k1, k2=cfg.k2, cap=cfg.cap))
        real_w = RealViews.task_w(TaskData("W", cfg.seed, k1=cfg.k1, k2=cfg.k2, cap=cfg.cap))
    return Trainer(cfg, device, centre, stats_t, norm_w, real_t, real_w, out, log)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("config", nargs="?", default="configs/marl.yaml")
    ap.add_argument("--arm", required=True, choices=["full", "nda"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--episodes", type=int, help="override (e.g. 20 for the benchmark)")
    ap.add_argument("--out", type=Path, default=RUNS)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--amp", choices=["none", "bf16"])
    a = ap.parse_args(argv)
    y = yaml.safe_load(Path(a.config).read_text())
    kw = dict(y["train"], arm=a.arm, seed=a.seed)
    if a.episodes is not None:
        kw["episodes"] = a.episodes
    if a.amp is not None:
        kw["amp"] = a.amp
    cfg = MarlConfig(**kw)
    centre = WorldParams.from_dict(yaml.safe_load(Path(y["centre"]).read_text()))
    out = a.out / a.arm / str(a.seed)
    out.mkdir(parents=True, exist_ok=True)
    res = build(cfg, torch.device(a.device), centre, out, lambda m: print(m, flush=True)).run()
    print(f"M5 {a.arm} seed {a.seed}: {res['episodes']} episodes, degenerate: {res['degenerate'] or 'no'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
