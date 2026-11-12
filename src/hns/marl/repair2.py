"""Repair cycle 2 of the launderer learner (SPEC §10, §16).

Declared on 2026-11-11, before any of its runs. Sources: the second Sol consult (runs/sol_consult/CONSULT2.md,
section C) and reports/m5_capability.md. Everything below is fixed in advance: cells, worlds, gates, selection.
Amended the same day, before any run had finished: stage 0 was added, and the SPEC's stop rule was removed at
the user's direction, so a failed cycle is followed by a new declared one.

**Why.** Under the per-proceeds reward the launderer learner found a profitable policy and lost it. In the saved
dropout-0 networks almost every unit of the head's trunk is inactive (2 to 6 of 128 fire), so most idle states
get identical Q-values; the shared value stream forces every branch's mean Q to equal ``V``; and the soft root
target adds an entropy bonus of the size of the whole return.

**Fixed conditions.** A fresh launderer learns against the frozen final inspector and innocents of M5's Full
seed 0 and NDA seed 1 (they act at epsilon 0.05, as in ``hns.marl.capability``). 30 worlds; launderer epsilon
1.0 -> 0.05 over 150 episodes; batch 128, lr 1e-3, Polyak 0.005, gamma 0.99 per sub-step, behaviour tau 0.01,
buffer 100k, warm-up 1,000, role-Q dropout 0, the root target, the per-proceeds reward, the simulator unchanged.
Seed 0 for the screen. Every cell starts from the same common weights (a branch's own value stream starts as a
copy of the shared one). Exploration, replay sampling, the archive's reservoir and the diagnostics each draw from
their own random stream.

**Cells** (``CELLS``): 12, each against both opponents, 24 screening runs.

- The factorial, 2 x 2 x 2, all with a LeakyReLU(0.01) trunk:
  S, the baseline: one ``V`` shared by all branches, or independent, one per branch (H1);
  T, the root target: soft ``tau logsumexp``, or hard ``max`` with tau for behaviour only (H4);
  R, replay: the 100k FIFO, or an 80k FIFO plus a 20k lifetime reservoir, 25 % of a batch from it (H2).
  Names: ``S|I`` + ``S|H`` + ``F|R``, e.g. ``IHF`` = independent, hard, FIFO.
- ``A``: the legacy learner (shared, soft, FIFO, ReLU trunk). Against ``SSF`` it isolates the trunk activation.
- ``W``: ``IHF`` with the branch-weighted loss (``hns.models.qnet.td_loss``, ``branch``) (H5).
- ``H``: ``IHF`` with the Huber-tailed loss (H6, the tails only).
- ``LN``: ``IHF`` with a LayerNorm on the readout, before the trunk (the encoder's own activations saturate too).

Every declared run trains its 30 worlds, in the order ``RUN_ORDER``, unless it aborts: a non-finite loss or
gradient, a non-finite Q at a feasible option of any state it acts in (training or evaluation) or of the panel,
or ``|Q| > 10,000`` over the fixed panel at two consecutive checks 100 updates apart. An aborted run is a failed
cell. Nothing is restarted with other settings.

**Stage 0, before the screen.** Cell ``A`` alone, against both opponents: the legacy learner with this cycle's
snapshots and diagnostics. No fix is tested, and the two runs are the screen's ``A`` runs. Question: is the
trunk already inactive when the policy is lost? ``timeline`` plays every snapshot from world 15 on two further
held-out worlds (``EVAL_SALT`` with j = 400 and 401, used for nothing else) and reads, per run:

- loss: with ``U`` the snapshots' natural BTC utility (sampled policy) and ``U*`` its maximum (the best
  snapshot: the earliest with it), a snapshot is low if ``U <= U* - 0.25``. If the world-30 snapshot is not low
  there is no loss. Otherwise the loss is the first world of the unbroken run of low snapshots that ends at
  world 30: a dip that recovered does not count;
- the trunk is inactive at a world if at least half of the panel's idle states have no trunk unit firing there;
- the reading: ``no loss``; ``trunk first`` if the trunk is inactive at the loss world (collapse: the first world
  of the unbroken inactive stretch that contains it); ``policy first`` if the trunk is active at every world from
  the best snapshot's through the loss world; ``inconclusive`` otherwise (inactive somewhere in between, active
  at the loss); ``incomplete`` if the run aborted, so that a snapshot from world 15 to 30 or a world's
  diagnostics is missing.

``trunk first`` in both runs: the screen runs as declared. ``policy first`` in either: the trunk is not what
loses the policy there, and the cells are revised before the screen. Any other combination is reported before
anything else runs. The panel's share of idle states whose best strategy is ``direct`` is printed next to it,
as a second view of the policy on fixed states. (The reading was tightened after a review and before any result
was read: the first wording took the last collapse and the first low snapshot, so a trunk that recovered after
the loss, or a dip that recovered before it, could have decided the order.)

**Logs.** Per world, in ``metrics.jsonl`` (``hns.marl.train``): rewards (also discounted), where the proceeds
went, launderer decisions by state and outcome, launches by strategy x amount x change, the learner's sampled
options, per-branch squared errors, rows and loss coefficients, gradient norms (total, encoder, head) and clip
share, the buffer's and the archive's contents and ages. In ``diag.jsonl``: on a fixed panel of 512 real states
per opponent (``panel``; half idle, half in flight), the online and the target network's Q-values (every option's
mean and maximum over the idle states, the root gap, strategy margins in tau units, the share of idle states with
the modal Q row, the correlation with the balance), the trunk's units that fire, the share of states where none
does or the output is all zero, the encoder's saturation; and on a replay batch, each branch's absolute TD
residuals (quantiles), the gap between the root's taken value and each launch branch's, and the root and
strategy losses' gradients into the shared layers (norms, cosine). Snapshots of the launderer's networks,
optimizer and random streams: before training, every 2 worlds, and every world from 15; the whole buffer at
worlds 15 and 30.

**Evaluation.** Worlds never used before: ``EVAL_SALT`` with j = 100..107 for selection and 200..207 for
confirmation; every policy plays the same worlds, so differences are paired by world. At epsilon 0:

- natural: the network decides everything; once sampling ``softmax(Q / tau)`` (the gated policy) and once taking
  each branch's argmax (reported);
- matched: each launderer launches once per episode at its first idle decision on or after a sub-step uniform on
  [0, 43], and always proceeds in flight; the network chooses the launch's options;
- references: ``wait``; ``uniform`` and ``direct50`` on the matched schedule; ``repeat50``, ``direct`` at 50 % up
  to six times per episode from the same first sub-step. Launching more than once is not a defect: ``repeat50``
  beat ``direct50`` in the consult's probe.

**Gates, per cell and opponent, on the selection worlds.**

- P1, sound: the run did not abort; the BTC accounting gap is at most 1e-6 in every training world; at world 30
  at most 1 % of the panel's states have an all-zero trunk output (a LeakyReLU trunk cannot have one; the share
  of states with no positive pre-activation and the modal Q row's share are logged); and the calibration bias
  ``mean(Q_initiate(s, a) - G)`` over the decisions is within ``max(0.5, 0.25 mean |G|)``. ``G`` matches the
  cell's target: for a soft cell, the sampled policy's return with the root entropy of every later decision
  added, ``G_t = r_t + gamma^gap (G_t+1 + tau H(pi_root(s_t+1)))``; for a hard cell, the argmax policy's plain
  return. The first capability test compared a soft Q with a plain return; this replaces that check. The largest
  Q against the most a launderer could cash is reported as a warning, not gated.
- P2, natural floor: the sampled policy's mean reward is at least half of ``direct50``'s and its paired difference
  to ``wait`` has a 95 % t-interval above 0; and both hold for BTC utility as well.
- P3, matched skill: on the matched schedule, BTC utility minus ``uniform``'s is at least 0.02 on average, with a
  paired 95 % t-interval above 0.

A cell passes the screen if it passes all three against both opponents.

**Selection.** Among the passing cells, the highest worst-opponent natural BTC utility (sampled policy) wins.
Cells within 0.02 of it count as tied, and the first of them in ``SIMPLEST_FIRST`` is taken (``IHF``, then fewer
added mechanisms). If no cell passes, the cycle has failed. A training-curve or snapshot peak never counts.
The reference snapshot for the retention check is the selected cell's world 16 or world 24 snapshot, whichever
has the higher worst-opponent natural BTC utility on worlds 100 and 101 (a tie goes to 24).

**Confirmation.** The selected cell is trained again with launderer seeds 1 and 2 against both opponents, and
all three seeds are evaluated on the confirmation worlds. Per opponent: P1 for every seed; P2 and P3 on the
per-world means over the three seeds (8 world clusters, not 24 rows); the matched skill difference at least 0 in
2 of the 3 seeds; retention, the final network's natural BTC utility minus the reference snapshot's, with a
paired 95 % t-interval above -0.05; and utilization (added 2026-11-12 with the user's acceptance, after stage 1
and before any confirmation; the fourth Sol consult's proposal): the natural policy's held share of the proceeds
minus ``direct50``'s, per world averaged over the three seeds, at most 0.05 on average over the 8 worlds (its
paired 95 % t-interval is reported). The cycle passes if all hold against both opponents. A policy that holds
much more than ``direct50`` can pass the floors while leaving most of its proceeds unspent, which says little
about laundering.

**After a pass**, one short live pilot (20 worlds per ecosystem, all three roles learning, from the selected
seed-0 launderer and the M5 final inspector and innocents; world seeds 900 and 901; at most 4 h of local GPU
time), judged on P2 and P3 against its own final opponents. Only then the six-run re-run (SPEC §15). If the screen,
the confirmation or the pilot fails, the cycle has failed, and a new cycle is declared before anything else runs.

**Not gated, reported:** the argmax policy; behaviour temperatures 0.005 and 0.03 on worlds 100 and 101 for the
selected cell and ``A`` (``sweep``); the initial and the world 16 / 24 snapshots on worlds 100 and 101.

**Deviations from the consult's section C:** the ``LN`` cell is added (user-approved). The activation, Q and
gradient diagnostics are logged every world, on the fixed panel and a replay batch, rather than at five chosen
worlds. Calibration is computed at evaluation, not in every training world, for the acting network only, and by
idle / in flight, three balance bands (under 3.5, 3.5 to 19, over 19 BTC) and the world's first and second half,
not by decile; the target network's values are logged on the panel. Sensitivity to the monetary inputs is
logged as the correlation of the root value with the balance. No-op reasons and the distribution of the gaps
between decisions are not logged.

Usage::

    python -m hns.marl.repair2 panel --opp full0
    python -m hns.marl.repair2 refs --opp full0 --worlds selection
    python -m hns.marl.repair2 train --cell IHF --opp full0 [--seed 0]
    python -m hns.marl.repair2 timeline --cell A --opp full0
    python -m hns.marl.repair2 select
    python -m hns.marl.repair2 confirm-eval --opp full0 --seed 1
    python -m hns.marl.repair2 confirm
    python -m hns.marl.repair2 report
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import shutil
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import yaml

import hns.marl.capability as cap
import hns.marl.train as mt
from hns.marl.obs import collate, launderer_obs, pack_masks
from hns.sim.entities import AMOUNTS, GO, STRATEGIES, WAIT
from hns.sim.env import SimEnv
from hns.sim.live import LiveGraph
from hns.sim.multiverse import draw_world
from hns.sim.world import EPISODES, SUBSTEPS

RUNS = Path("runs/repair2")
REPORT = Path("reports/m5_repair2.md")
NOTES = "<!-- findings: hand-written below; `report` keeps them -->"
OPPS = {"full0": Path("runs/marl/full/0"), "nda1": Path("runs/marl/nda/1")}


def _cell(baseline: str, backup: str, replay: str, **kw) -> dict:
    return {"baseline": baseline, "backup": backup, "replay": replay, "trunk_act": "leaky", "td": "mse",
            "head_norm": False} | kw


CELLS = {
    "A": _cell("shared", "soft", "fifo", trunk_act="relu"),
    "SSF": _cell("shared", "soft", "fifo"), "SSR": _cell("shared", "soft", "archive"),
    "SHF": _cell("shared", "hard", "fifo"), "SHR": _cell("shared", "hard", "archive"),
    "ISF": _cell("independent", "soft", "fifo"), "ISR": _cell("independent", "soft", "archive"),
    "IHF": _cell("independent", "hard", "fifo"), "IHR": _cell("independent", "hard", "archive"),
    "W": _cell("independent", "hard", "fifo", td="branch"),
    "H": _cell("independent", "hard", "fifo", td="huber"),
    "LN": _cell("independent", "hard", "fifo", head_norm=True),
}
# one-factor variants of the legacy learner for the staged tests (hns.marl.stage1); not cells of the screen
STAGE_CELLS = {"P": CELLS["A"] | {"scale": "wealth"}, "I": CELLS["A"] | {"baseline": "independent"},
               "PL": CELLS["A"] | {"scale": "wealth", "trunk_act": "leaky"},
               "PLR1": CELLS["A"] | {"scale": "wealth", "trunk_act": "leaky", "root_weight": "m"},
               # the joint critic is a bundle, not one factor: its own head, a leaky trunk, a hard double target
               "J": CELLS["A"] | {"critic": "joint", "backup": "hard", "trunk_act": "leaky"}}
ALL_CELLS = CELLS | STAGE_CELLS
RUN_ORDER = ("A", "SSF", "IHF", "IHR", "SHF", "ISF", "SSR", "SHR", "ISR", "W", "H", "LN")
SIMPLEST_FIRST = ("IHF", "A", "SSF", "ISF", "SHF", "SSR", "ISR", "SHR", "IHR", "W", "H", "LN")
WORLDS, EPS_EPISODES = 30, 150
PANELS = {"selection": tuple(range(100, 108)), "confirmation": tuple(range(200, 208))}
PANEL_WORLDS, PANEL_STATES = (300, 301), 256           # the diagnostic panel: 256 idle + 256 in-flight states
SNAP_REF = (16, 24)
TIMELINE_WORLDS, TIMELINE_FROM = (400, 401), 15         # stage 0: every snapshot from world 15, on two worlds
COLLAPSE, LOSS = 0.5, 0.25                              # share of idle panel states; BTC utility below the best
REFS = ("wait", "uniform", "direct50", "repeat50")
Q_ABORT, CHECK_EVERY = 1e4, 100
DEAD_MAX, ACCT_MAX, SKILL_MIN, TIE, RETAIN = 0.01, 1e-6, 0.02, 0.02, 0.05
HELD_SLACK = 0.05        # confirmation (2026-11-12): held share at most direct50's plus this, in proceeds fractions
SWEEP_TAUS = (0.005, 0.03)


class Abort(RuntimeError):
    """A run's numerical safety abort: the cell has failed."""


def make(cell: str, opp: str, seed: int, device, out: Path | None = None) -> mt.Trainer:
    return cap.make(OPPS[opp], device, out, seed=seed, q_dropout=0.0, target="root", design_roles="launderer",
                    **ALL_CELLS[cell])


def run_dir(cell: str, opp: str, seed: int, smoke: bool = False) -> Path:
    return (RUNS / "smoke" if smoke else RUNS) / f"{cell}_{opp}_s{seed}"


# ---------------------------------------------------------------------------- policies for evaluation
class Net:
    """The trained launderer at epsilon 0. ``mode``: ``boltzmann`` samples ``softmax(Q / tau)`` per branch;
    ``argmax`` takes each branch's best feasible option; ``matched`` samples the options like ``boltzmann`` but
    launches on ``sched``'s once-per-episode schedule and always proceeds in flight. Records, per decision, the
    root Q-values, the root policy's entropy, whether the launderer was idle, its log balance, and the strategy
    and amount it values most (whether or not it launches). ``eps``: the sampled modes' exploration rate (0 in
    every gated evaluation; the training rate for a reported look at the behaviour policy)."""

    def __init__(self, tr: mt.Trainer, mode: str = "boltzmann", tau: float | None = None,
                 sched: np.random.Generator | None = None, eps: float = 0.0):
        self.tr, self.mode, self.tau, self.sched, self.eps = tr, mode, tr.cfg.tau if tau is None else tau, sched, eps
        self.rec, self.ep, self.when, self.launched = {}, None, {}, set()

    def act(self, env, states: dict) -> dict:
        if not states:
            return {}
        tr, L = self.tr, self.tr.roles["launderer"]
        ids = list(states)
        ob = launderer_obs(env, tr.fz, ids, tr.gvec, tr.view_seed(1), tr.cfg.k2)
        masks, _ = pack_masks("launderer", states, ids)
        q = L.q_values(ob, masks)
        greedy = L.best(ob, masks, q)
        a = greedy.copy() if self.mode == "argmax" else L.act(ob, masks, self.eps, tr.rng, tau=self.tau)
        if self.mode == "matched" and env.episode != self.ep:
            self.ep, self.launched = env.episode, set()
            self.when = {x.id: int(self.sched.integers(0, cap.LAUNCH_LAST + 1)) for x in env.launderers}
        bs, ba = list(L.branches).index("strategy"), list(L.branches).index("amount")
        for j, i in enumerate(ids):
            st = states[i]
            idle = len(st.branches) > 1
            if self.mode == "matched":
                due = idle and i not in self.launched and env.sub >= self.when[i] and st.masks["initiate"][GO]
                a[j, 0] = GO if due or not idle else WAIT
                if due:
                    self.launched.add(i)
            z = q[j, :2][np.isfinite(q[j, :2])].astype(np.float64) / self.tau
            p = np.exp(z - z.max())
            p /= p.sum()
            ent = 0.0 if self.mode == "argmax" else float(-(p * np.log(np.maximum(p, 1e-300))).sum())
            self.rec[(tr.step_no, i)] = (float(q[j, 0]), float(q[j, 1]), ent, idle, float(ob.x_c[j, 1]),
                                         int(greedy[j, bs]), int(greedy[j, ba]))
        names = list(L.branches)
        return {i: {b: int(a[j, bi]) for bi, b in enumerate(names) if b in states[i].branches}
                for j, i in enumerate(ids)}


def reference(name: str, j: int) -> cap.Scheduled:
    """A scripted policy for world ``j``; every one of them, and ``matched``, launches on the same schedule."""
    rng, sched = np.random.default_rng([cap.EVAL_SALT, j, 2]), np.random.default_rng([cap.EVAL_SALT, j, 3])
    direct, half = STRATEGIES.index("direct"), AMOUNTS.index(0.5)
    if name == "wait":
        return cap.Scheduled(rng, never=True, sched_rng=sched)
    if name == "uniform":
        return cap.Scheduled(rng, sched_rng=sched)
    if name == "direct50":
        return cap.Scheduled(rng, direct, half, sched_rng=sched)
    if name == "repeat50":
        return cap.Scheduled(rng, direct, half, max_launches=6, sched_rng=sched)
    raise ValueError(name)


# ---------------------------------------------------------------------------- one evaluation world
BANDS = (1.5, 3.0)        # log1p(source balance): under 3.5 BTC, 3.5 to 19 BTC, over 19 BTC


def calibrate(dec: list[tuple], gamma: float, tau: float, mid: float = math.inf) -> dict:
    """``Q_initiate`` of the taken root option against the return from each decision, per sub-step discount to
    world end. ``dec``: ``(step, agent, a0, reward, (Q_wait, Q_go, root entropy, idle, log balance) or None)``.

    ``hard``: ``G_t = r_t + gamma^gap G_t+1``. ``soft``: the soft root target's own return, which also counts the
    root entropy of every later decision, ``G_t = r_t + gamma^gap (G_t+1 + tau H_t+1)``. Also by stratum: idle or
    in flight, the balance band (``BANDS``), and decisions before or from step ``mid`` (the world's halves).
    """
    by = defaultdict(list)
    for s, i, a0, r, rec in dec:
        by[i].append((s, a0, r, rec))
    rows = []                                         # (Q, G hard, G soft, idle, log balance, step)
    for seq in by.values():
        seq.sort(key=lambda x: x[0])
        g = gs = 0.0
        t_next, h_next = None, 0.0
        for s, a0, r, rec in reversed(seq):
            d = gamma ** (t_next - s) if t_next is not None else 0.0
            g, gs = r + d * g, r + d * (gs + tau * h_next)
            t_next, h_next = s, (rec[2] if rec is not None else 0.0)
            if rec is not None:
                rows.append((rec[a0], g, gs, rec[3], rec[4], s))
    if not rows:
        return {"n": 0}
    x = np.array(rows, np.float64)
    out = {"n": len(x), "q_max": float(x[:, 0].max()), "q_mean": float(x[:, 0].mean())}
    btc = np.expm1(x[:, 4])                                    # the source balance, as a weight
    for k, c in (("hard", 1), ("soft", 2)):
        out[f"bias_{k}"] = float((x[:, 0] - x[:, c]).mean())
        out[f"g_abs_{k}"] = float(np.abs(x[:, c]).mean())
        out[f"bias_{k}_btc"] = float(((x[:, 0] - x[:, c]) * btc).sum() / btc.sum()) if btc.sum() > 0 else 0.0
    for name, m in (("idle", x[:, 3] > 0), ("flight", x[:, 3] == 0), ("bal_low", x[:, 4] < BANDS[0]),
                    ("bal_mid", (x[:, 4] >= BANDS[0]) & (x[:, 4] < BANDS[1])), ("bal_high", x[:, 4] >= BANDS[1]),
                    ("early", x[:, 5] < mid), ("late", x[:, 5] >= mid)):
        out[name] = {"n": int(m.sum())} | ({f"bias_{k}": float((x[m, 0] - x[m, c]).mean()) for k, c in
                                            (("hard", 1), ("soft", 2))} if m.any() else {})
    return out


def gate_table(dec: list[tuple]) -> dict:
    """Per balance band, over the idle decisions where a launch was feasible: how many, how many launched, and at
    how many the network's best strategy was ``direct`` and its best amount 50 % (also where it waited: a bad
    preference that the gate hides still decides what a launch there would do)."""
    d, half = STRATEGIES.index("direct"), AMOUNTS.index(0.5)
    out = {k: [0, 0, 0, 0] for k in ("low", "mid", "high")}
    for _, _, a0, _, rec in dec:
        if rec is not None and rec[3] and math.isfinite(rec[1]):
            row = out["low" if rec[4] < BANDS[0] else "mid" if rec[4] < BANDS[1] else "high"]
            for i, hit in enumerate((True, a0 == GO, rec[5] == d, rec[6] == half)):
                row[i] += int(hit)
    return out


TIME_BINS = (250, 400, 450, 490)      # sub-steps of the world: 0-249, 250-399, 400-449, 450-489, 490-499


def hazard_table(dec: list[tuple], start: int, noops: frozenset = frozenset()) -> dict:
    """Per source-balance band (``BANDS``) and world-time bin (``TIME_BINS``; ``start``: the world's first step):
    ``[idle decisions where a launch was feasible, of them tried to launch, of those no-ops, in-flight decisions,
    of them tried to proceed, of those no-ops]``. ``noops``: the ``(step, agent)`` of no-op decisions, which the
    simulator records as waits although only ``go`` proposes a tx; tried minus no-ops is what happened."""
    out = {f"{k}/{b}": [0] * 6 for k in ("low", "mid", "high") for b in range(len(TIME_BINS) + 1)}
    for s, i, a0, _, rec in dec:
        if rec is None or (rec[3] and not math.isfinite(rec[1])):    # idle: only where a launch was feasible
            continue
        band = "low" if rec[4] < BANDS[0] else "mid" if rec[4] < BANDS[1] else "high"
        row, o = out[f"{band}/{int(np.searchsorted(TIME_BINS, s - start, side='right'))}"], 0 if rec[3] else 3
        noop = (s, i) in noops
        row[o], row[o + 1], row[o + 2] = row[o] + 1, row[o + 1] + int(a0 == GO or noop), row[o + 2] + int(noop)
    return out


def play(tr: mt.Trainer, lpol, j: int) -> dict:
    """Held-out world ``j`` with launderer policy ``lpol`` against ``tr``'s frozen opponents; nothing learns."""
    learn, tr.learn = tr.learn, frozenset()
    ss = np.random.SeedSequence([cap.EVAL_SALT, j])
    params = draw_world(tr.centre, np.random.default_rng(ss.spawn(1)[0]))
    env = SimEnv(params, int(ss.generate_state(1)[0]), lpol, tr.policies[1], tr.policies[2])
    env.graph = LiveGraph()
    tr.rng = np.random.default_rng([cap.EVAL_SALT, j, 1])      # the opponents' exploration and the policy's sampling
    tr.step_no, tr.last = cap.EVAL_STEP + j * EPISODES * SUBSTEPS, {}
    tr.world_step, tr.launches = tr.step_no, np.zeros((len(STRATEGIES), len(AMOUNTS), 2), np.int64)
    tot, dec, launched, noops = defaultdict(float), [], set(), set()
    for e in range(EPISODES):
        env.start_episode(e)
        if tr.last:
            tr.last["episode_committed"] = 0
        ep = defaultdict(float)
        for _ in range(SUBSTEPS):
            res = env.step()
            for t in res.transitions:
                if t.role == "launderer":
                    rec = lpol.rec.get((tr.step_no, t.agent)) if isinstance(lpol, Net) else None
                    dec.append((tr.step_no, t.agent, t.action["initiate"], t.reward, rec))
                    if len(t.branches) > 1 and not t.noop and t.action["initiate"] == GO:
                        launched.add((e, t.agent))                 # a launch that happened
                    if t.noop:
                        noops.add((tr.step_no, t.agent))
            tr.after_step(env, res, ep)
            tr.step_no += 1
        env.end_episode()
        for k, v in ep.items():
            tot[k] += v
    tr.learn = learn
    n_le = params.n_launderers * EPISODES
    out = {"world": j, "n_launderers": params.n_launderers, "budget0": params.budget,
           "reward": tot["r_launderer"] / n_le, "reward_discounted": tot["r_launderer_disc"] / params.n_launderers,
           "tranches": float(env.acct["strategy"].sum()) / n_le, "ep_launch_share": len(launched) / n_le,
           "detection_rate": tot["cash_flag"] / max(tot["cash"], 1.0), **mt.btc_fractions(env),
           "decisions": {k[2:]: int(v) for k, v in tot.items() if k.startswith("l_")},
           "launches": tr.launches.tolist(),
           "q_cap": float(env.proceeds.sum(0).max() / env.d_w)}
    if isinstance(lpol, Net):
        out["cal"] = calibrate(dec, tr.cfg.gamma, tr.cfg.tau, tr.world_step + EPISODES * SUBSTEPS / 2)
        out["gate"] = gate_table(dec)
        out["hazard"] = hazard_table(dec, tr.world_step, frozenset(noops))
        lpol.rec.clear()
    return out


def evaluate(tr: mt.Trainer, policy: str, worlds, tau: float | None = None, eps: float = 0.0) -> list[dict]:
    """``policy``: a ``Net`` mode or a reference name."""
    out = []
    for j in worlds:
        if policy in REFS:
            lpol = reference(policy, j)
        else:
            lpol = Net(tr, policy, tau, np.random.default_rng([cap.EVAL_SALT, j, 3]), eps)
        t0 = time.perf_counter()
        r = play(tr, lpol, j)
        out.append(r)
        print(f"  eval {policy} world {j}: nL {r['n_launderers']} reward {r['reward']:+.3f} utility "
              f"{r['btc_utility']:+.3f} tranches {r['tranches']:.2f} {time.perf_counter() - t0:.0f}s", flush=True)
    return out


# ---------------------------------------------------------------------------- snapshots
def snapshot(tr: mt.Trainer, path: Path) -> None:
    L = tr.roles["launderer"]
    torch.save({"world": tr.world, "launderer": L.net.state_dict(), "launderer_targ": L.targ.state_dict(),
                "opt": L.opt.state_dict(), "rng": tr.rng.bit_generator.state,
                "rng_replay": tr.replay_rng.bit_generator.state, "rng_cpu": torch.get_rng_state(),
                "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}, path)


def load_snapshot(tr: mt.Trainer, path: Path) -> None:
    ck = torch.load(path, map_location=tr.device, weights_only=False)
    L = tr.roles["launderer"]
    L.net.load_state_dict(ck["launderer"])
    L.targ.load_state_dict(ck["launderer_targ"])


# ---------------------------------------------------------------------------- the fixed diagnostic panel
def panel_path(opp: str) -> Path:
    return RUNS / f"panel_{opp}.pkl"


class _Collect:
    """Wraps a scripted launderer policy and keeps the observations of the states it sees."""

    def __init__(self, tr: mt.Trainer, inner):
        self.tr, self.inner, self.rows = tr, inner, []

    def act(self, env, states: dict) -> dict:
        if states:
            tr, ids = self.tr, list(states)
            ob = launderer_obs(env, tr.fz, ids, tr.gvec, tr.view_seed(1), tr.cfg.k2)
            masks, _ = pack_masks("launderer", states, ids)
            self.rows += [(o, masks[j]) for j, o in enumerate(ob.split())]
        return self.inner.act(env, states)


def cmd_panel(a) -> None:
    """The opponent's fixed panel: real launderer states from two worlds played with uniform options, up to three
    launches per episode, so every strategy's in-flight states occur."""
    tr = cap.make(OPPS[a.opp], torch.device(a.device))
    rng, rows = np.random.default_rng([cap.EVAL_SALT, 5]), []
    for j in PANEL_WORLDS:
        inner = cap.Scheduled(np.random.default_rng([cap.EVAL_SALT, j, 2]), max_launches=3)
        pol = _Collect(tr, inner)
        play(tr, pol, j)
        rows += pol.rows
    idle = np.array([o[4][0] == 0 for o, _ in rows])
    pick = np.concatenate([rng.permutation(np.flatnonzero(m))[:PANEL_STATES] for m in (idle, ~idle)])
    out = {"obs": [rows[i][0] for i in pick], "masks": np.stack([rows[i][1] for i in pick]), "idle": idle[pick],
           "bal": np.array([rows[i][0][4][1] for i in pick], np.float32)}
    RUNS.mkdir(parents=True, exist_ok=True)
    pickle.dump(out, open(panel_path(a.opp), "wb"))
    print(f"{panel_path(a.opp)}: {int(out['idle'].sum())} idle and {int((~out['idle']).sum())} in-flight states")


class Diagnostics:
    """Panel and gradient diagnostics of a training run, and its numerical safety check."""

    def __init__(self, tr: mt.Trainer, panel: dict):
        self.tr, self.L = tr, tr.roles["launderer"]
        self.ob, self.masks, self.idle, self.bal = collate(panel["obs"]), panel["masks"], panel["idle"], panel["bal"]
        self.rng = np.random.default_rng([tr.cfg.seed, 4])
        self.checked, self.over = 0, 0

    def check(self, tr: mt.Trainer) -> None:
        """Every ``CHECK_EVERY`` updates: the panel's feasible Q must be finite, and not above ``Q_ABORT`` twice
        in a row."""
        n = self.L.n_updates
        if n == self.checked or n % CHECK_EVERY:
            return
        self.checked = n
        try:
            top = self.L.q_abs_max(self.ob, self.masks)
        except FloatingPointError:
            raise Abort(f"non-finite Q at update {n}") from None
        self.over = self.over + 1 if top > Q_ABORT else 0
        if self.over >= 2:
            raise Abort(f"|Q| above {Q_ABORT:g} at two consecutive checks, update {n}")

    @torch.no_grad()
    def panel_stats(self) -> dict:
        L, got = self.L, {}
        hooks = [L.net.heads.trunk[0].register_forward_hook(lambda m, i, o: got.update(x=i[0], pre=o)),
                 L.net.heads.trunk.register_forward_hook(lambda m, i, o: got.update(z=o)),
                 L.net.enc.convs[-1].register_forward_hook(lambda m, i, o: got.update(conv=o))]
        q = L.q_values(self.ob, self.masks)
        for h in hooks:
            h.remove()
        qt = L._q(L.targ, self.ob, self.masks).cpu().numpy()
        pre, idle, tau = got["pre"].float().cpu().numpy(), self.idle, self.tr.cfg.tau
        on = pre > 0
        names = list(L.branches)
        o = np.concatenate([[0], np.cumsum(L.sizes)])
        si, d = names.index("strategy"), STRATEGIES.index("direct")
        root, strat = q[idle, :2], q[idle, o[si]:o[si + 1]]
        top = np.sort(strat, axis=1)
        m = L.scale_of(self.ob)                                # under the wealth scale Q = m q: the modal row is q's,
        m = np.ones((len(q), 1), np.float32) if m is None else m.cpu().numpy()    # so m cannot hide a blind network
        rows = np.round(np.concatenate([np.where(np.isfinite(root), root, 0.0), strat], axis=1) / m[idle], 4)
        modal = np.unique(rows, axis=0, return_counts=True)[1].max() / len(rows)
        best = np.where(np.isfinite(root), root, -np.inf).max(1)

        def table(x):                 # per branch, each option over the idle states where it is feasible (else None)
            ok = np.isfinite(x[idle])
            mean = np.where(ok, x[idle], 0.0).sum(0) / np.maximum(ok.sum(0), 1)
            top = np.where(ok, x[idle], -np.inf).max(0)
            cell = lambda v, i: float(v[i]) if ok[:, i].any() else None       # noqa: E731
            return {b: {"mean": [cell(mean, i) for i in range(o[k], o[k + 1])],
                        "max": [cell(top, i) for i in range(o[k], o[k + 1])]} for k, b in enumerate(names)}

        return {"q_table": table(q), "q_table_target": table(qt),
                "q_abs_max": L.q_abs_max(self.ob, self.masks),
                "q_net_abs_max": float(np.abs((q / m)[self.masks]).max()), "scale_max": float(m.max()),
                "q_wait_idle": float(root[:, WAIT].mean()),
                "go_minus_wait_tau": float(np.median((root[:, GO] - root[:, WAIT])[np.isfinite(root[:, GO])]) / tau),
                "direct_margin_tau": float(np.median(strat[:, d] - np.delete(strat, d, 1).max(1)) / tau),
                "top2_margin_tau": float(np.median(top[:, -1] - top[:, -2]) / tau),
                "direct_argmax_share": float((strat.argmax(1) == d).mean()),
                "modal_row_share": float(modal), "q_root_std_idle": float(best.std()),
                "q_balance_corr": float(np.corrcoef(best, self.bal[idle])[0, 1]) if best.std() > 0 < self.bal[idle].std() else 0.0,
                "target_gap": float(np.abs(np.where(self.masks[:, :2], q[:, :2] - qt[:, :2], 0.0)).mean()),
                "zero_trunk_share": float((got["z"] == 0).all(1).float().mean()),
                "units_active": int(on.any(0).sum()), "no_unit_share": float((~on.any(1)).mean()),
                "no_unit_share_idle": float((~on.any(1))[idle].mean()),
                "preact_max": np.quantile(pre.max(1), [0.05, 0.5, 0.95]).tolist(),
                "trunk_input_rms": float(got["x"].float().pow(2).mean().sqrt()),
                "encoder_saturated": float((got["conv"].float() < -5).float().mean())}

    def grad_stats(self) -> dict | None:
        """The root and strategy losses' gradients into the encoder and trunk, on one replay batch."""
        L = self.L
        if L.buf.ready < L.cfg.warmup:
            return None
        b = L.buf.sample(256, self.rng)
        y = L.batch_target(b)
        L.net.train()
        ob = collate(b["obs"])
        realized = torch.from_numpy(b["realized"]).to(y.device)
        qa = L.taken_q(L.net, ob, b["masks"], torch.from_numpy(b["action"]).to(y.device), realized)
        e = (y.unsqueeze(1) - qa) ** 2 * realized              # squared residuals in reward units
        m = L.scale_of(ob)
        e_loss = e if m is None else e / m ** 2                # what the loss weighs: divided by m(s)^2 under the scale
        if L.design["root_weight"] == "m":                     # and the root's by m(s) only
            e_loss = torch.cat([e[:, :1] / m, e_loss[:, 1:]], 1)
        names = list(L.branches)
        si = names.index("strategy")
        n_launch = int(realized[:, si].sum())
        res, gap = e.detach().sqrt().cpu().numpy(), (qa[:, :1] - qa).detach().cpu().numpy()
        rz = b["realized"]
        # |y - Q| of each branch over the rows that realize it (quantiles 50 / 90 / 99 %), and the root's taken value
        # minus the branch's over those rows (mean, mean absolute)
        out = {"launch_rows": n_launch,
               "residual_abs": {n: np.quantile(res[rz[:, i], i], [0.5, 0.9, 0.99]).tolist()
                                for i, n in enumerate(names) if rz[:, i].any()},
               "root_minus_branch": {n: [float(gap[rz[:, i], i].mean()), float(np.abs(gap[rz[:, i], i]).mean())]
                                     for i, n in enumerate(names) if i and rz[:, i].any()}}
        if not n_launch:
            return out
        shared = [p for p in (*L.net.enc.parameters(), *L.net.heads.trunk.parameters())]

        def grad(loss):
            g = torch.autograd.grad(loss, shared, retain_graph=True, allow_unused=True)
            return torch.cat([(torch.zeros_like(p) if x is None else x).reshape(-1) for x, p in zip(g, shared)])

        g_root, g_str = grad(e_loss[:, 0].mean()), grad(e_loss[:, si].sum() / n_launch)
        cos = torch.dot(g_root, g_str) / (g_root.norm() * g_str.norm()).clamp_min(1e-30)
        return out | {"root_norm": float(g_root.norm()), "strategy_norm": float(g_str.norm()), "cosine": float(cos)}


# ---------------------------------------------------------------------------- commands: train, evaluate
def _eval_final(tr: mt.Trainer, worlds) -> dict:
    return {m: evaluate(tr, m, worlds) for m in ("boltzmann", "argmax", "matched")}


def cmd_train(a) -> None:
    smoke = a.worlds != WORLDS or a.eval_worlds is not None
    out = run_dir(a.cell, a.opp, a.seed, smoke)
    if (out / "result.json").exists():
        print(f"{out}: done")
        return
    if out.exists():
        if a.cell in STAGE_CELLS and not smoke:                # a staged test's run is one attempt (hns.marl.stage1)
            raise SystemExit(f"{out}: an incomplete run of a stage cell is not restarted")
        shutil.rmtree(out)                                     # a killed run of the screen starts again from nothing
    (out / "snap").mkdir(parents=True)
    tr = make(a.cell, a.opp, a.seed, torch.device(a.device), out)
    L = tr.roles["launderer"]
    (out / "config.yaml").write_text(yaml.safe_dump(
        {**asdict(tr.cfg), "cell": a.cell, "opponent": str(OPPS[a.opp]), "opp_eps": cap.OPP_EPS, "learn": ["launderer"],
         "git_sha": mt.git_sha()}, sort_keys=False))
    (out / "metrics.jsonl").write_text("")
    diag = Diagnostics(tr, pickle.load(open(panel_path(a.opp), "rb")))
    tr.after_update = diag.check
    snapshot(tr, out / "snap" / "w00.pt")
    status = "ok"
    with open(out / "diag.jsonl", "w") as f:
        f.write(json.dumps({"world": 0, "panel": diag.panel_stats(), "grad": None}) + "\n")
        try:
            for w in range(1, a.worlds + 1):
                r = tr.run_world()
                d = {"world": w, "panel": diag.panel_stats(), "grad": diag.grad_stats()}
                f.write(json.dumps(d) + "\n")
                f.flush()
                if w % 2 == 0 or w >= 15:
                    snapshot(tr, out / "snap" / f"w{w:02d}.pt")
                if w in (15, 30):
                    torch.save(L.buf, out / "snap" / f"replay_w{w:02d}.pt", pickle_protocol=5)
                p = d["panel"]
                print(f"{out.name} world {w}/{a.worlds} eps {tr.eps:.2f} | R_L {r['reward']['launderer']:+.3f} "
                      f"utility {r['btc_utility']:+.3f} det {r['detection_rate']:.2f} tranches {r['tranches']} "
                      f"direct {r['strategy_share']['direct']:.2f} td {r['losses'].get('td_launderer', float('nan')):.3f}"
                      f" | units {p['units_active']} blind {p['zero_trunk_share']:.2f} margin {p['direct_margin_tau']:+.1f}"
                      f" tau | {r['sec']['world']:.0f}s", flush=True)
        except (Abort, FloatingPointError) as e:
            status = f"aborted: {e}"
            print(f"{out.name}: {status}", flush=True)
    res = {"cell": a.cell, "opp": a.opp, "seed": a.seed, "worlds": a.worlds, "status": status, "git_sha": mt.git_sha(),
           "accounting_gap_max": max((abs(e["accounting_gap"]) for e in tr.epochs), default=0.0),
           "zero_trunk_share": None, "modal_row_share": None}
    if status == "ok":
        res |= {k: v for k, v in diag.panel_stats().items() if k in ("zero_trunk_share", "modal_row_share")}
    if status == "ok" and not a.no_eval:                      # a staged test evaluates with its own command
        worlds = PANELS["selection"][:a.eval_worlds]
        try:
            ev = _eval_final(tr, worlds)
            for w in (0, *SNAP_REF):                           # diagnostic snapshots, on the first two worlds
                if (out / "snap" / f"w{w:02d}.pt").exists():
                    load_snapshot(tr, out / "snap" / f"w{w:02d}.pt")
                    ev[f"snap{w}"] = evaluate(tr, "boltzmann", worlds[:2])
            res["eval"] = ev
        except FloatingPointError as e:                        # a non-finite Q in an evaluation world: a failed cell
            res["status"] = f"aborted: {e}"
    (out / "result.json").write_text(json.dumps(res, indent=1))


def ref_path(opp: str, worlds: str, policy: str, smoke: bool = False) -> Path:
    return (RUNS / "smoke" if smoke else RUNS) / f"ref_{opp}_{worlds}_{policy}.json"


def cmd_refs(a) -> None:
    tr = cap.make(OPPS[a.opp], torch.device(a.device))
    for p in REFS:
        f = ref_path(a.opp, a.worlds, p, a.eval_worlds is not None)
        if not f.exists():
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(json.dumps({"opp": a.opp, "policy": p,
                                     "eval": evaluate(tr, p, PANELS[a.worlds][:a.eval_worlds])}, indent=1))


def cmd_confirm_eval(a) -> None:
    """The selected cell's seed ``a.seed`` on the confirmation worlds: the final network in the three modes, and the
    reference snapshot's natural policy."""
    sel = json.loads((RUNS / "selection.json").read_text())
    out = run_dir(sel["cell"], a.opp, a.seed)
    f = out / ("confirm.json" if a.eval_worlds is None else "confirm_smoke.json")
    if f.exists():
        return
    res = {"cell": sel["cell"], "opp": a.opp, "seed": a.seed,
           "status": json.loads((out / "result.json").read_text())["status"]}
    if res["status"] == "ok":                                  # an aborted run has no final network to evaluate
        tr = make(sel["cell"], a.opp, a.seed, torch.device(a.device))
        worlds = PANELS["confirmation"][:a.eval_worlds]
        try:
            load_snapshot(tr, out / "snap" / f"w{WORLDS:02d}.pt")
            ev = _eval_final(tr, worlds)
            load_snapshot(tr, out / "snap" / f"w{sel['reference_world']:02d}.pt")
            ev["reference"] = evaluate(tr, "boltzmann", worlds)
            res["eval"] = ev
        except FloatingPointError as e:
            res["status"] = f"aborted: {e}"
    f.write_text(json.dumps(res, indent=1))


def cmd_sweep(a) -> None:
    """Reported only: the final network's natural policy at other behaviour temperatures, worlds 100 and 101."""
    out = run_dir(a.cell, a.opp, a.seed)
    if json.loads((out / "result.json").read_text())["status"] != "ok":
        return
    tr = make(a.cell, a.opp, a.seed, torch.device(a.device))
    load_snapshot(tr, out / "snap" / f"w{WORLDS:02d}.pt")
    ev = {str(t): evaluate(tr, "boltzmann", PANELS["selection"][:2], tau=t) for t in SWEEP_TAUS}
    (out / "sweep.json").write_text(json.dumps(ev, indent=1))


# ---------------------------------------------------------------------------- stage 0: the timeline
def read_timeline(blind: dict[int, float], utility: dict[int, float]) -> dict:
    """Stage 0's reading of one run (the module docstring declares it). ``blind``: per training world, the share
    of the panel's idle states with no trunk unit firing. ``utility``: per snapshot world, its natural BTC utility
    on the timeline worlds."""
    out = {"collapse": None, "best": None, "loss": None}
    if set(utility) != set(range(TIMELINE_FROM, WORLDS + 1)) or not set(range(1, WORLDS + 1)) <= set(blind):
        return out | {"reading": "incomplete"}
    best = min(utility, key=lambda w: (-utility[w], w))
    low = {w for w in utility if utility[w] <= utility[best] - LOSS}
    out |= {"best": best, "best_utility": utility[best], "final_utility": utility[WORLDS], "blind_final": blind[WORLDS]}
    if WORLDS not in low:
        return out | {"reading": "no loss"}
    loss = WORLDS
    while loss - 1 in low:                                     # the run of low snapshots that ends at the last world
        loss -= 1
    dead = {w for w in blind if blind[w] >= COLLAPSE}
    if loss in dead:
        collapse = loss
        while collapse - 1 in dead:
            collapse -= 1
        return out | {"loss": loss, "collapse": collapse, "reading": "trunk first"}
    return out | {"loss": loss, "reading": "inconclusive" if dead & set(range(best, loss)) else "policy first"}


def cmd_timeline(a) -> None:
    out = run_dir(a.cell, a.opp, a.seed)
    diag = [json.loads(x) for x in (out / "diag.jsonl").read_text().splitlines()]
    epochs = {r["world"] + 1: r for _, r in mt.read_records(out / "metrics.jsonl") if r["kind"] == "epoch"}
    f, util, tr = out / "timeline.json", {}, None
    if f.exists():                                             # snapshots already played are not played again
        util = {r["world"]: r["utility"] for r in json.loads(f.read_text())["rows"] if r["utility"] is not None}
    for w in range(TIMELINE_FROM, WORLDS + 1):
        if w not in util and (out / "snap" / f"w{w:02d}.pt").exists():     # an aborted run stops short of world 30
            tr = tr or make(a.cell, a.opp, a.seed, torch.device(a.device))
            load_snapshot(tr, out / "snap" / f"w{w:02d}.pt")
            try:
                util[w] = float(np.mean([r["btc_utility"] for r in evaluate(tr, "boltzmann", TIMELINE_WORLDS)]))
            except FloatingPointError:                         # a non-finite Q: no utility, an incomplete timeline
                pass
    res = read_timeline({d["world"]: d["panel"]["no_unit_share_idle"] for d in diag if d["world"] >= 1}, util)
    res["rows"] = [{"world": d["world"], "units_active": d["panel"]["units_active"],
                    "blind_idle": d["panel"]["no_unit_share_idle"], "direct_argmax": d["panel"]["direct_argmax_share"],
                    "direct_margin_tau": d["panel"]["direct_margin_tau"], "modal_row": d["panel"]["modal_row_share"],
                    "train_utility": epochs[d["world"]]["btc_utility"],
                    "train_direct": epochs[d["world"]]["strategy_share"]["direct"], "utility": util.get(d["world"])}
                   for d in diag if d["world"] >= 1]
    f.write_text(json.dumps(res, indent=1))
    print(f"{out.name}: world | units firing | blind idle states | direct is best (panel) | margin (tau) | "
          f"training utility, direct share | snapshot utility on worlds {TIMELINE_WORLDS}")
    for r in res["rows"]:
        u = "" if r["utility"] is None else f"{r['utility']:+.3f}"
        print(f"  {r['world']:2d} | {r['units_active']:3d} | {r['blind_idle']:.2f} | {r['direct_argmax']:.2f} | "
              f"{r['direct_margin_tau']:+7.1f} | {r['train_utility']:+.3f}, {r['train_direct']:.2f} | {u}")
    print(f"best snapshot: world {res['best']}; loss: world {res['loss']}; collapse: world {res['collapse']}; "
          f"reading: {res['reading']}")


# ---------------------------------------------------------------------------- gates
def _col(rows: list[dict], key: str) -> np.ndarray:
    return np.array([r[key] for r in rows], np.float64)


def calibration_gate(ev: dict, backup: str) -> dict:
    """P1's calibration: the return that matches the cell's target, over the policy that target assumes."""
    rows, k = (ev["boltzmann"], "soft") if backup == "soft" else (ev["argmax"], "hard")
    n = sum(r["cal"]["n"] for r in rows)
    bias = sum(r["cal"][f"bias_{k}"] * r["cal"]["n"] for r in rows if r["cal"]["n"]) / max(n, 1)
    g_abs = sum(r["cal"][f"g_abs_{k}"] * r["cal"]["n"] for r in rows if r["cal"]["n"]) / max(n, 1)
    return {"bias": bias, "g_abs": g_abs, "kind": k, "ok": bool(n > 0 and abs(bias) <= max(0.5, 0.25 * g_abs))}


def floors(natural: dict, matched_utility: np.ndarray, refs: dict) -> dict:
    """P2 and P3 from per-world values (each array one value per world, in the same world order).

    ``natural``: ``{"reward": ..., "btc_utility": ...}`` of the sampled natural policy. ``refs``: the reference
    policies' world records.
    """
    out, p2 = {}, True
    for k in ("reward", "btc_utility"):
        x, wait, d50 = natural[k], _col(refs["wait"], k), _col(refs["direct50"], k)
        m, lo, hi = cap._t95(x - wait)
        ok = bool(x.mean() >= 0.5 * d50.mean() and lo > 0)
        out[k] = {"mean": float(x.mean()), "direct50": float(d50.mean()), "repeat50": float(_col(refs["repeat50"], k).mean()),
                  "wait": float(wait.mean()), "vs_wait": [m, lo, hi], "ok": ok}
        p2 &= ok
    m, lo, hi = cap._t95(matched_utility - _col(refs["uniform"], "btc_utility"))
    out["matched"] = {"mean": float(matched_utility.mean()), "uniform": float(_col(refs["uniform"], "btc_utility").mean()),
                      "direct50": float(_col(refs["direct50"], "btc_utility").mean()), "vs_uniform": [m, lo, hi]}
    return out | {"p2": p2, "p3": bool(m >= SKILL_MIN and lo > 0)}


def sound(res: dict, ev: dict | None) -> dict:
    """P1 of one run: no abort, closed accounting, a live trunk, and calibrated values on ``ev``."""
    if res["status"] != "ok" or ev is None:
        return {"p1": False, "status": res["status"]}
    cal = calibration_gate(ev, ALL_CELLS[res["cell"]]["backup"])
    q_cap = max((r["cal"]["q_max"] / max(r["q_cap"], 1e-12) for r in ev["boltzmann"] if r["cal"]["n"]), default=0.0)
    ok = res["accounting_gap_max"] <= ACCT_MAX and res["zero_trunk_share"] <= DEAD_MAX and cal["ok"]
    return {"p1": bool(ok), "status": "ok", "calibration": cal, "accounting_gap_max": res["accounting_gap_max"],
            "zero_trunk_share": res["zero_trunk_share"], "q_over_cap_warning": q_cap}


def screen_row(res: dict, refs: dict) -> dict:
    """One screening run's gates on the selection worlds."""
    row = {"cell": res["cell"], "opp": res["opp"], **sound(res, res.get("eval"))}
    if row["status"] != "ok":
        return row | {"p2": False, "p3": False, "pass": False}
    ev = res["eval"]
    nat = {k: _col(ev["boltzmann"], k) for k in ("reward", "btc_utility")}
    row |= floors(nat, _col(ev["matched"], "btc_utility"), refs)
    row |= {"argmax_utility": float(_col(ev["argmax"], "btc_utility").mean()),
            "snaps": {k: float(_col(v, "btc_utility").mean()) for k, v in ev.items() if k.startswith("snap")}}
    return row | {"pass": bool(row["p1"] and row["p2"] and row["p3"])}


def choose(rows: dict[tuple[str, str], dict]) -> dict:
    """The selection rule. ``rows``: ``(cell, opp) -> screen_row``; a cell missing a run cannot pass."""
    passing = [c for c in CELLS if all((c, o) in rows and rows[(c, o)]["pass"] for o in OPPS)]
    if not passing:
        return {"cell": None, "passing": []}
    score = {c: min(rows[(c, o)]["btc_utility"]["mean"] for o in OPPS) for c in passing}
    best = max(score.values())
    cell = min((c for c in passing if score[c] >= best - TIE), key=SIMPLEST_FIRST.index)
    ref = {w: min(rows[(cell, o)]["snaps"][f"snap{w}"] for o in OPPS) for w in SNAP_REF}
    return {"cell": cell, "passing": passing, "score": score,
            "reference_world": max(SNAP_REF, key=lambda w: (ref[w], w)), "reference_score": ref}


def _refs(opp: str, worlds: str) -> dict:
    return {p: json.loads(ref_path(opp, worlds, p).read_text())["eval"] for p in REFS}


def screen_rows() -> dict[tuple[str, str], dict]:
    rows = {}
    for cell in CELLS:
        for opp in OPPS:
            f = run_dir(cell, opp, 0) / "result.json"
            if f.exists():
                rows[(cell, opp)] = screen_row(json.loads(f.read_text()), _refs(opp, "selection"))
    return rows


def cmd_select(a) -> None:
    rows = screen_rows()
    missing = [f"{c}/{o}" for c in CELLS for o in OPPS if (c, o) not in rows]
    if missing:
        raise SystemExit(f"the screen is incomplete: {missing}")
    sel = choose(rows)
    (RUNS / "selection.json").write_text(json.dumps(sel, indent=1))
    print(json.dumps(sel, indent=1))


def confirmation(sel: dict) -> dict:
    """The confirmation gates of the selected cell, per opponent, from the three seeds' confirmation evaluations."""
    out = {"cell": sel["cell"], "opps": {}}
    for opp in OPPS:
        refs = _refs(opp, "confirmation")
        runs, aborted = [], []
        for seed in (0, 1, 2):
            d = run_dir(sel["cell"], opp, seed)
            con = json.loads((d / "confirm.json").read_text())
            if con["status"] != "ok":
                aborted.append(f"seed {seed}: {con['status']}")
            runs.append((json.loads((d / "result.json").read_text()), con.get("eval")))
        if aborted:                                            # an aborted seed fails the opponent, and the cycle
            out["opps"][opp] = {"aborted": aborted, "pass": False}
            continue
        p1 = [sound(res, ev) for res, ev in runs]
        nat = {k: np.mean([_col(ev["boltzmann"], k) for _, ev in runs], axis=0) for k in ("reward", "btc_utility")}
        matched = np.stack([_col(ev["matched"], "btc_utility") for _, ev in runs])
        fl = floors(nat, matched.mean(0), refs)
        seeds_ok = int(((matched - _col(refs["uniform"], "btc_utility")).mean(1) >= 0).sum())
        keep = cap._t95(np.mean([_col(ev["boltzmann"], "btc_utility") - _col(ev["reference"], "btc_utility")
                                 for _, ev in runs], axis=0))
        held = cap._t95(np.mean([_col(ev["boltzmann"], "held_frac") for _, ev in runs], axis=0)
                        - _col(refs["direct50"], "held_frac"))      # per world, over the seeds: 8 world clusters
        ok = (all(x["p1"] for x in p1) and fl["p2"] and fl["p3"] and seeds_ok >= 2 and keep[1] > -RETAIN
              and held[0] <= HELD_SLACK)
        out["opps"][opp] = {"p1": p1, **fl, "seeds_with_skill": seeds_ok, "retention": list(keep),
                            "held_over_direct50": list(held), "pass": bool(ok)}
    return out | {"pass": all(v["pass"] for v in out["opps"].values())}


def cmd_confirm(a) -> None:
    res = confirmation(json.loads((RUNS / "selection.json").read_text()))
    (RUNS / "confirmation.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({o: {k: v[k] for k in ("aborted", "p2", "p3", "seeds_with_skill", "retention", "held_over_direct50",
                                            "pass") if k in v}
                      for o, v in res["opps"].items()} | {"pass": res["pass"]}, indent=1))


# ---------------------------------------------------------------------------- report
def _yn(x: bool) -> str:
    return "yes" if x else "no"


def cmd_report(a) -> None:
    rows = screen_rows()
    lines = ["# M5 repair cycle 2: the launderer learner", "",
             "Design, gates and selection rule: `hns.marl.repair2` (declared before the runs). Each row: a fresh "
             f"launderer trained for {WORLDS} worlds against a frozen M5 inspector and innocents, then evaluated at "
             "epsilon 0 on selection worlds 100 to 107. Utility: (cashed - all frozen) / proceeds, of the sampled "
             "natural policy; argmax in brackets. Matched: BTC utility with one scheduled launch per episode, minus "
             "uniform options'. Blind: share of the panel's states with an all-zero trunk output at world 30.", "",
             "| Cell | Opponent | Baseline | Target | Replay | Other | Utility (argmax) | direct50 / repeat50 | "
             "Reward - wait (95 % CI) | Matched - uniform (95 % CI) | Q bias (mean abs G) | Blind | P1 | P2 | P3 | Pass |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for cell in RUN_ORDER:
        c = CELLS[cell]
        other = ", ".join(x for x in (c["trunk_act"], c["td"] if c["td"] != "mse" else "",
                                      "LayerNorm" if c["head_norm"] else "") if x)
        for opp in OPPS:
            r = rows.get((cell, opp))
            head = f"| {cell} | {opp} | {c['baseline']} | {c['backup']} | {c['replay']} | {other} | "
            if r is None:
                lines.append(head + "not run | | | | | | | | | |")
            elif r["status"] != "ok":
                lines.append(head + f"{r['status']} | | | | | | no | no | no | no |")
            else:
                u, w, m, cal = r["btc_utility"], r["reward"]["vs_wait"], r["matched"]["vs_uniform"], r["calibration"]
                lines.append(head + f"{u['mean']:+.3f} ({r['argmax_utility']:+.3f}) | {u['direct50']:+.3f} / "
                             f"{u['repeat50']:+.3f} | {w[0]:+.3f} ({w[1]:+.3f}, {w[2]:+.3f}) | {m[0]:+.3f} ({m[1]:+.3f}, "
                             f"{m[2]:+.3f}) | {cal['bias']:+.2f} ({cal['g_abs']:.2f}) | {r['zero_trunk_share']:.2f} | "
                             f"{_yn(r['p1'])} | {_yn(r['p2'])} | {_yn(r['p3'])} | {'**yes**' if r['pass'] else 'no'} |")
    lines.append("")
    if (RUNS / "selection.json").exists():
        sel = json.loads((RUNS / "selection.json").read_text())
        lines += ["## Selection", "", f"Passing cells: {', '.join(sel['passing']) or 'none'}. Selected: "
                  f"**{sel['cell'] or 'none (the cycle has failed)'}**"
                  + (f"; reference snapshot: world {sel['reference_world']}." if sel["cell"] else "."), ""]
    if (RUNS / "confirmation.json").exists():
        con = json.loads((RUNS / "confirmation.json").read_text())
        lines += ["## Confirmation (seeds 0 to 2, worlds 200 to 207)", "",
                  "| Opponent | Utility | direct50 / repeat50 | Reward - wait (95 % CI) | Matched - uniform (95 % CI) | "
                  "Seeds with skill | Final - reference (95 % CI) | P1 (each seed) | P2 | P3 | Pass |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        for opp, v in con["opps"].items():
            if "aborted" in v:
                lines.append(f"| {opp} | {'; '.join(v['aborted'])} | | | | | | | | | no |")
                continue
            u, w, m, k = v["btc_utility"], v["reward"]["vs_wait"], v["matched"]["vs_uniform"], v["retention"]
            lines.append(f"| {opp} | {u['mean']:+.3f} | {u['direct50']:+.3f} / {u['repeat50']:+.3f} | {w[0]:+.3f} "
                         f"({w[1]:+.3f}, {w[2]:+.3f}) | {m[0]:+.3f} ({m[1]:+.3f}, {m[2]:+.3f}) | {v['seeds_with_skill']} / 3 "
                         f"| {k[0]:+.3f} ({k[1]:+.3f}, {k[2]:+.3f}) | {' '.join(_yn(x['p1']) for x in v['p1'])} | "
                         f"{_yn(v['p2'])} | {_yn(v['p3'])} | {'**yes**' if v['pass'] else 'no'} |")
        lines += ["", f"Cycle verdict: **{'pass' if con['pass'] else 'fail'}**.", ""]
    old = REPORT.read_text() if REPORT.exists() else ""
    if NOTES in old:
        lines.append(old[old.index(NOTES):].rstrip("\n"))
    REPORT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    cmds = {"panel": cmd_panel, "refs": cmd_refs, "train": cmd_train, "confirm-eval": cmd_confirm_eval,
            "sweep": cmd_sweep, "timeline": cmd_timeline, "select": cmd_select, "confirm": cmd_confirm,
            "report": cmd_report}
    for name in cmds:
        p = sub.add_parser(name)
        if name in ("panel", "refs", "train", "confirm-eval", "sweep", "timeline"):
            p.add_argument("--opp", choices=list(OPPS), required=True)
            p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
        if name in ("train", "sweep", "timeline"):
            p.add_argument("--cell", choices=list(ALL_CELLS), required=True)
        if name in ("train", "confirm-eval", "sweep", "timeline"):
            p.add_argument("--seed", type=int, default=0)
        if name in ("train", "refs", "confirm-eval"):
            p.add_argument("--eval-worlds", type=int, help="fewer evaluation worlds: a smoke run, never a result")
        if name == "train":
            p.add_argument("--worlds", type=int, default=WORLDS, help="training worlds; anything else is a smoke run")
            p.add_argument("--no-eval", action="store_true", help="train, snapshot and log only (hns.marl.stage1 reads it)")
        if name == "refs":
            p.add_argument("--worlds", choices=list(PANELS), default="selection")
    a = ap.parse_args(argv)
    cmds[a.cmd](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
