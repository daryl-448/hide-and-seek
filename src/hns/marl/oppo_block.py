"""OPPO-MC capability cycle: the development blocks (SPEC §10). Declared on 2026-11-12, before any of it ran.

Source: the fifth Sol consult (runs/sol_consult/CONSULT5.md, sections C and D), adopted by the user on 2026-11-12
together with the SPEC revision. It supersedes stage 1's remaining blocks (``hns.marl.stage1``), whose verdicts
stand. The learner is ``hns.marl.oppo``.

**Why.** No Q-learner variant of repair cycle 2 and stage 1 kept a profitable launderer policy against both frozen
opponents: the policies formed and were lost, or rich launderers held their funds. The consult traced the common
failure to valuing a launch through a gate over the learner's own changing launch options, through long
bootstrapped chains of waits and in-flight pauses. OPPO-MC changes the estimator (on-policy PPO with complete-world
returns, a separate critic that never enters a return), the credit boundary (options) and the decision structure
(automatic flight, wait durations, a joint mode). It is a bundle, not a one-factor test: a pass shows that this
package can learn the known task, not which of its parts was needed.

**Fixed conditions.** A fresh OPPO-MC launderer, seed 0, learns against the frozen final inspector and innocents of
M5's Full seed 0 (``full0``) or NDA seed 1 (``nda1``), which act at epsilon 0.05 and never update
(``hns.marl.capability.make``). 30 training worlds, the multiverse draws of seed 0 (stage 1's). Worlds 1-5 are the
options-first curriculum, worlds 6-30 the natural actor (SPEC §10). The reward, the simulator, the observations and
the opponents are unchanged. Snapshots of the actor, critic, optimizers and random streams: before training, every
two worlds, and every world from 15; the complete decision rollout of worlds 15 and 30.

**Actor trust control (amended 2026-11-12 after the timing smoke, before either development block; the sixth Sol
consult, runs/sol_consult/CONSULT6.md).** In every training world, including the curriculum, the actor takes at
most four shuffled passes without replacement, logical minibatch 128, Adam lr 3e-4 and gradient-norm clip 1.0.
After each tentative actor Adam step, compute exact mean old-to-new joint KL over every collected decision,
conditional launch KL over every launch-feasible collected decision including sampled waits, and conditional
duration KL over every collected decision with more than one feasible duration. All three compare with the fixed
actor that collected this world, never with the preceding minibatch's actor. If any mean exceeds 0.02 nats, undo
only that step, restoring its immediately preceding actor parameters and complete Adam state, including moment
tensors and step counters; retain all earlier accepted steps and stop actor updates for this world. Do not retry the
rejected step or add passes. Non-finite quantities still abort the block. The critic completes its four scheduled
passes. No other learner, curriculum, block-budget or reading condition changes. (Why: a 7-world GPU timing smoke,
not a block, showed the per-pass guard undoing the first pass of every natural world, so the actor never changed
once timing was released; the choice used training-update mechanics only, no evaluation world.)

**Full-0's first attempt (2026-11-12 21:06) was voided** by the user: another job on the machine (a Rust build) starved
it after world 9 and the cap cut it before any reading; it is kept as ``OPPO_full0_s0_stalled1``. The block was run
again, unchanged.

**A block is one attempt**: one opponent, training and reading together, under a 45-minute cap on the local GPU
(``scripts/m5_repair2.sh oppo OPP``). ``run`` records that it started and refuses to start again; a block cut by the
cap, or aborted, is incomplete, never promising. ``read`` exists only to finish the reading of a block whose training
completed and whose reading did not run (a harness fault); it never re-reads a finished block. Full-0 runs first,
then NDA-1 as its own block, even if Full-0's reading fails, unless the failure is an implementation fault that
also invalidates the NDA run (then it is reported and a new declaration follows).

**Reading.** On development worlds 100-103 (``EVAL_SALT``; used by stage 1 already, so this is a development
reading, not a confirmation), against the same frozen opponents:

- natural: the world-20 and world-30 snapshots, the actor sampling as trained (no extra exploration);
- matched: the world-30 snapshot on the scheduled launch of the references (once per episode, the first idle
  sub-step on or after a uniform sub-step in [0, 43] where a launch is feasible, the same schedule stream as the
  references), always proceeding in flight, the actor choosing every launch factor;
- argmax: the world-30 snapshot taking every factor's most likely option (reported, never selected).

References: the scripted policies of ``hns.marl.repair2`` (``wait``, ``uniform``, ``direct50``, ``repeat50``), read
from the reference files on the selection worlds that repair2 computed on 2026-11-11 (local GPU), restricted to
100-103; legacy cell ``A``'s final network (its natural records on the same worlds). These are the files stage 1
read; they are not recomputed (the scripted policies and the opponents are unchanged; the consult proposed
recomputing them). Under them the floors below are U >= 0.401891 and R >= 0.322627 against Full-0, and
U >= 0.292472 and R >= 0.269037 against NDA-1; the formulas govern.

The block is **promising** if all five hold:

1. sound: all 30 worlds completed without an abort (a non-finite loss, gradient, probability, value or KL; an
   infeasible sampled action; a decision chain whose return differs from its rewards by more than 1e-6; a decision
   added after its world closed), the BTC accounting gap is at most 1e-6 in every world, and before every update the
   stored log-probabilities are reproduced within ``RATIO_TOL`` (the ratios start at one);
2. natural economics: the world-30 snapshot's mean BTC utility is positive and at least
   ``max(0.5 direct50's, A's - 0.02)``, and its mean reward at least half of ``direct50``'s;
3. retention: world-30 minus world-20 natural BTC utility is at least -0.05 on average over the four worlds;
4. the critic: with ``b_w = mean (V - G)`` and ``a_w = mean |G|`` over world ``w``'s decisions of the world-30
   natural rollout (``V`` before the decision, ``G`` its plain return), ``|mean_w b_w| <= max(0.5, 0.25 mean_w a_w)``;
5. utilization and option skill: the world-30 natural held share minus ``direct50``'s is at most 0.05 on average,
   and the matched BTC utility minus ``uniform``'s at least 0.02 on average.

Reported, not gated: every paired difference with its 95 % t-interval over the four worlds (to wait, ``direct50``,
``repeat50``, PL's world-30 network, retention, held, skill); where the proceeds went; launches by mode, wait
durations, launch rates by balance band and world time, launch no-ops; the critic's bias pooled, BTC-weighted, by
band and time, its RMSE and explained variance; per training world, complete and partial actor passes, actor Adam
steps accepted and rejected, the three KLs after every attempted step and at the retained final actor, and the
conditional-support counts, entropy, clip fraction, ratio quantiles and gradient norms; the argmax policy.

**What follows** (the consult's D3, fixed now): both blocks promising -> confirmation with seeds 1 and 2 on unused
worlds (SPEC §10), not the six runs. Natural economics or held funds fail while matched skill passes -> inspect the
learned durations, the gate's support and proposed-versus-executed launches; the next declaration changes only the
timing. Matched skill fails with adequate launch data -> audit the likelihoods, masks and credit on a small
enumerated problem first. World 20 good and world 30 lost -> inspect the KLs and undone passes. Economics good and
only the critic fails -> the critic, not the game, is revised. Every negative or incomplete block is reported before
a new declaration.

**Deviations from the consult.** References are reused, not recomputed (above). Encoder saturation is not
reported; route outcomes per exposed BTC are not logged. The reading's world-20 and world-30 evaluations and the
matched and argmax ones all count inside the block's cap. The live game's per-world acting copies and NMAI's
static option launderers are not implemented yet: neither is used by these blocks.

**Confirmation (declared 2026-11-12, after both development blocks read promising and before any confirmation run;
SPEC §10, step 2).** Both blocks were promising (``OPPO_full0_s0/oppo.json``, ``OPPO_nda1_s0/oppo.json``).

- Training: the unchanged design above, the amended trust control included, with launderer seeds 1 and 2 against
  each frozen opponent: four new runs (``train --opp O --seed S``), each one attempt with its seed's multiverse
  draws. Seed 0 is the development block's run. Seeds 1 and 2 run on a JarvisLabs A30 (seed 0 ran on the local GTX
  1650 SUPER) from the same code; every process has a 2-hour timeout, and a run cut by it is incomplete.
- Evaluation (``confirm-eval --opp O --seed S``, all six runs) on the eight confirmation worlds 200-207
  (``EVAL_SALT``; no cycle has used them): natural, the world-20 and world-30 snapshots; matched, the world-30
  snapshot and the untrained initial actor (the world-0 snapshot; its probability tables are exactly uniform, so
  every seed's initial actor plays the same game in a world, which is reported as a check); argmax, the world-30
  snapshot (reported). References: repair2's scripted ``wait``, ``uniform``, ``direct50`` and ``repeat50`` on
  worlds 200-207, computed for this confirmation (``python -m hns.marl.repair2 refs --opp O --worlds confirmation``).
  Every run is evaluated on the references' graph-view stream (``eval_views``; seed 0's development records are
  unchanged by it), so a world is the same game for every seed and for the references; only the launderer differs.

Per opponent, with ``u_sj``, ``r_sj``, ``h_sj`` seed ``s``'s world-30 natural BTC utility, reward and held share in
world ``j``, ``X_j`` the mean over the three seeds, and every interval a two-sided 95 % t-interval over the eight
worlds, the opponent is **confirmed** if all six hold:

1. sound: all three runs completed 30 worlds as in the block's criterion 1, and every evaluation world completed
   (no abort; each decision chain's return reproduced);
2. calibrated values, for each seed separately (SPEC acceptance item 2): ``|mean_j b_sj| <= max(0.5, 0.25 mean_j
   a_sj)`` over its world-30 natural rollouts;
3. natural economics: ``mean_j U_j > 0`` and ``mean_j u_sj > 0`` for at least 2 of the 3 seeds; ``mean U`` at least
   half of ``direct50``'s mean utility and ``mean R`` at least half of its mean reward; ``U_j - wait_j`` and
   ``R_j - wait_j`` with lower bounds above 0;
4. utilization: ``mean_j (H_j - direct50's held_j) <= 0.05``;
5. matched skill: with ``M_j`` the seeds' mean matched utility and ``I_j`` the initial actor's, ``M - uniform`` and
   ``M - I`` each have a mean of at least 0.02 and a lower bound above 0, and ``mean_j (matched_sj - uniform_j) >= 0``
   for at least 2 of the 3 seeds;
6. retention: ``mean_s (u_sj - world-20 u_sj)`` has a lower bound above -0.05 (each seed against its own world-20
   snapshot).

Not gated: legacy A's floor (it was for the development comparison only); the differences to ``direct50`` and
``repeat50``; every seed's numbers; the argmax policy; where the proceeds went; the critic's strata; launches by mode.

**What follows**: both opponents confirmed -> the live pilot (SPEC §10, step 3), declared before it runs. Otherwise
the failing criteria are reported with their numbers, nothing is re-run, and a new declaration follows. A run or
evaluation cut or aborted (or one that left no record) leaves its opponent incomplete, not passed; the other
opponent is still read, and the script exits non-zero.

**Confirmation reading (2026-11-13, A30, ``runs/repair2/oppo_confirmation.json``): not confirmed.** Every step
completed (0 failed).

- Full-0, **confirmed** (6 of 6): U +.486 (seeds .507, .517, .435; direct50 .659), R +.460; vs wait +.486 [+.437,
  +.535]; critic bias -.06, .00, +.14 (tolerance .5); held - direct50 -.029; matched +.620, vs uniform +1.48
  [+1.43, +1.53], vs initial +1.44 [+1.37, +1.51]; retention +.247 [+.221, +.274].
- NDA-1, **not confirmed**: criterion 4 fails, held - direct50 +.246 [+.227, +.266] (bar .05). The other five hold:
  U +.361 (seeds .520, .085, .479; direct50 .569), R +.317; vs wait +.361 [+.294, +.429]; critic bias -.46, -.27,
  +.04; matched +.424, vs uniform +1.34 [+1.27, +1.42], vs initial +1.32 [+1.22, +1.43]; retention +.313. The failure
  is seed 1: held .89 at world 30 (seeds 0 and 2: .04, .08). In training, every seed held .6-.9 around worlds 15-21
  as the actor's entropy fell. Seeds 0 and 2 recovered by about world 24 (entropy back to about 3); seed 1's entropy
  kept falling (1.2 to .7) and it held .8-.97 through world 30.
- Initial-actor check: identical across seeds in every world but one (Full-0 world 202, seed 2: -.837 against
  -.822). Re-evaluated alone in fresh processes, all three seeds give -.82204. The code path is seed-independent;
  the difference came from the run (six processes on one GPU, or state carried between evaluations in a process).

As declared, nothing is re-run; a new declaration follows.

**Next attempt: OPPO-MC-60 (declared 2026-11-13 and approved by the user, before any of it ran; the seventh Sol
consult, runs/sol_consult/CONSULT7.md and r11/DECLARATION_PROPOSAL.md).** The attempt above (``--attempt 30``) stays
as read: Full-0 confirmed, NDA-1 not confirmed on utilization. Neither verdict is amended, and no old run is pooled
with a new one. A CPU diagnostic after that reading continued NDA-1 seed 1 from world 30 on training worlds, unchanged
(not an exact resume: the snapshots lacked the opponents' exploration stream), and its gate reopened (held .90 at
world 31, .015 at world 48). That diagnostic is not an acceptance run, and its actor is never an initialization or a
candidate. The user chose this budget-only attempt instead of the timing-only branch of "What follows" above.

- The only change is the training budget: 60 complete worlds per run instead of 30 (``--attempt 60``). Worlds 1-5
  are the options-first curriculum, worlds 6-60 natural. Networks, policy, wait durations, gamma, rewards, wealth
  scale, advantage scale, entropy, optimizers, minibatches, the three 0.02-nat per-Adam-step KL checks and their
  rollback, the simulator, the observations and the frozen opponents are unchanged. No launch floor, longer
  curriculum, optimizer reset, guard relaxation, extra pass, temperature change or peak selection.
- Development: fresh seed 3 against Full-0, then against NDA-1 as its own block, 60 worlds each; NDA-1 runs even if
  Full-0 fails, unless the failure is a shared harness or implementation fault. Each block is one attempt, training
  and reading together, under a 60-minute cap, on the JarvisLabs A30 (the user's choice), one block at a time. Read
  on development worlds 100-103 from the world-40 and world-60 snapshots (natural), world-60 matched and world-60
  argmax (reported), with the same references and the same five criteria, 40 in place of 20 and 60 in place of 30.
  Only world 60 can pass. The development references are the published selection-world files (computed on the local
  GPU on 2026-11-11), not recomputed.
- Confirmation, only if both development blocks are promising: fresh seeds 4 and 5 against both opponents, 60 worlds
  each, one attempt each (two-hour timeout per process). Seeds 3, 4 and 5 are evaluated on unused worlds 208-215
  (``r2.PANELS["confirmation60"]``) with the references' view stream: natural at worlds 40 and 60, matched at world
  60 and the initial actor, argmax at world 60 (reported). The scripted references are computed on 208-215 before
  any learner evaluation runs. Each run's evaluation is its own fresh process, one at a time. The six criteria of
  the confirmation above apply unchanged to seeds 3, 4 and 5, with 60 in place of 30 and 40 in place of 20; in
  particular utilization stays the mean held excess over ``direct50`` of at most 0.05, over all three seeds. Full-0 is
  re-confirmed: its training budget and endpoint changed too. Worlds 200-207 are not used again.
- Records: ``OPPO60_{opp}_s{seed}``, markers ``oppo60_{opp}_s{seed}.json``, references ``ref_{opp}_confirmation60_*``,
  the reading ``oppo60_confirmation.json``; nothing of attempt 30 is read or overwritten except the development
  references and legacy A's and PL's records that the development reading uses. Snapshots keep the cadence (every
  two worlds, every world from 15) through 60 and add the opponents' exploration and replay streams; complete
  rollouts are saved at worlds 15, 30, 40 and 60.

**What follows** (fixed now): both development blocks promising -> the confirmation. Held funds or natural economics
fail while matched skill passes -> the 60-world budget test failed; diagnose support at the endpoint, and the next
declaration is the timing-coverage package (a smooth launch floor of 0.10 and waits of 1 or 5 sub-steps), not an
extension to 100 worlds or an earlier snapshot. Matched skill fails -> audit likelihoods, masks, credit and the
amount and change choices on training states. Only the critic fails -> the critic is revised, not its tolerance.
Retention fails -> audit the kept KL steps, optimizer geometry and support; world 40 is never eligible instead. An
incomplete or numerically failed step is never a pass; only the user may void an attempt for an established harness
fault. Both opponents confirmed -> the live pilot (SPEC §10, step 3: 60 fresh worlds per arm, a world-30 fork, worlds
216-223), declared in full before it runs. Otherwise the failing criteria are reported and a new declaration follows.

**Development reading of OPPO-MC-60** (2026-11-13, A30, run r_05cc8ecc; each block one attempt, read once;
``oppo60_{opp}_s3.json`` and ``logs/oppo60_dev_{opp}.log``):

- NDA-1, seed 3: promising, all five criteria. U60 .664 (bar .292), R .635 (bar .269); U60 - U40 +.005
  [-.119, +.128]; critic bias +.11 (tolerance .5); held excess over ``direct50`` -.052 [-.099, -.005]; matched minus
  uniform +1.52 [+1.36, +1.67].
- Full-0, seed 3: not promising, c3 only. Mean U60 - U40 -.0522 against the bar of -.05 (worlds 100-103: -.013, -.151,
  +.019, -.062; world 101's held share rose from .026 to .215 and its utility fell from .601 to .450). The rest pass:
  U60 .549 (bar .402), R .529 (bar .323), critic bias +.01, held excess -.007 [-.140, +.126], matched minus uniform
  +1.51 [+1.36, +1.65].

**Deviation (the user's, 2026-11-13).** As declared, only both blocks promising permit the confirmation. The user
approved running it anyway ("run it"), knowing Full-0's block missed c3 by .002. Both development readings stand as
read above and are not amended. The confirmation runs exactly as declared: seeds 4 and 5 fresh, seeds 3, 4 and 5
read on worlds 208-215, the six criteria and every tolerance unchanged, and Full-0's c6 (the paired U60 - U40 lower
bound above -.05) applies in full. A confirmation pass therefore stands on the confirmation alone. It does not undo
the development miss, and the report states both. The script runs this with ``scripts/m5_oppo60.sh confirm``.

Usage::

    scripts/m5_repair2.sh oppo full0       (attempt 30: oppo_block run, train then read; the 45-minute cap)
    python -m hns.marl.oppo_block read --opp full0     (only after a harness fault in the reading)
    scripts/m5_oppo_confirm.sh             (attempt 30, on the A30: the confirmation)
    scripts/m5_oppo60.sh [smoke]           (attempt 60, on the A30: both development blocks, then the confirmation)
    scripts/m5_oppo60.sh confirm           (attempt 60: the confirmation only; the deviation above)
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import torch
import yaml

import hns.marl.capability as cap
import hns.marl.repair2 as r2
import hns.marl.train as mt
from hns.marl.capability import _t95
from hns.marl.oppo import OptionPolicy, calibration, decision_summary
from hns.sim.world import EPISODES

DEV_WORLDS = (100, 101, 102, 103)
RATIO_TOL = 1e-3
ACCT_MAX, A_SLACK, RETAIN, HELD_SLACK, SKILL_MIN = 1e-6, 0.02, 0.05, 0.05, 0.02
ABORTS = (FloatingPointError, RuntimeError, ValueError)     # numerical, chain and legality faults (and out of memory)


@dataclass(frozen=True)
class Attempt:
    """One declared attempt of the cycle (module docstring). Its records carry its own name, so no attempt reads or
    overwrites another's."""
    name: str                   # run directories ``{name}_{opp}_s{seed}``; markers and the reading in lower case
    worlds: int                 # training worlds per run
    snaps: tuple[int, int]      # the retention reference and the eligible endpoint
    dev_seed: int               # the development blocks' launderer seed (also a confirmation seed)
    seeds: tuple[int, ...]      # the confirmation's seeds
    panel: str                  # ``r2.PANELS`` key of the confirmation worlds
    rollouts: tuple[int, ...]   # training worlds whose complete decision rollout is saved

    @property
    def conf_worlds(self) -> tuple[int, ...]:
        return tuple(r2.PANELS[self.panel])


ATTEMPTS = {"30": Attempt("OPPO", 30, (20, 30), 0, (0, 1, 2), "confirmation", (15, 30)),          # closed 2026-11-13
            "60": Attempt("OPPO60", 60, (40, 60), 3, (3, 4, 5), "confirmation60", (15, 30, 40, 60))}


def run_dir(att: Attempt, opp: str, seed: int, smoke: bool = False) -> Path:
    return (r2.RUNS / "smoke" if smoke else r2.RUNS) / f"{att.name}_{opp}_s{seed}"


def mark_path(att: Attempt, opp: str, seed: int, smoke: bool = False) -> Path:
    return (r2.RUNS / "smoke" if smoke else r2.RUNS) / f"{att.name.lower()}_{opp}_s{seed}.json"


def make(att: Attempt, opp: str, seed: int, device, out: Path | None = None) -> mt.Trainer:
    return cap.make(r2.OPPS[opp], device, out, seed=seed, launderer="oppo", episodes=att.worlds * EPISODES)


# ---------------------------------------------------------------------------- snapshots
def snapshot(tr: mt.Trainer, path: Path) -> None:
    pol = tr.policies[0]
    torch.save({"world": tr.world, "step_no": tr.step_no, **tr.roles["launderer"].state(),
                "rng": pol.rng.bit_generator.state, "sched": pol.sched_rng.bit_generator.state,
                "opp_rng": tr.rng.bit_generator.state, "replay_rng": tr.replay_rng.bit_generator.state,
                "rng_cpu": torch.get_rng_state(),
                "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}, path)


def load_snapshot(tr: mt.Trainer, path: Path) -> None:
    tr.roles["launderer"].load_state(torch.load(path, map_location=tr.device, weights_only=False), networks_only=True)


# ---------------------------------------------------------------------------- evaluation
def evaluate(tr: mt.Trainer, mode: str, worlds) -> list[dict]:
    """The loaded actor on held-out ``worlds`` against ``tr``'s frozen opponents. ``mode``: ``natural``,
    ``matched`` (the references' launch schedule) or ``argmax``. Each record adds the critic's calibration and
    what the launderers decided. Every seed plays on the references' graph-view stream (``eval_views``)."""
    out = []
    with eval_views(tr):
        for j in worlds:
            pol = OptionPolicy(tr, tr.roles["launderer"], np.random.default_rng([cap.EVAL_SALT, j, 4]),
                               np.random.default_rng([cap.EVAL_SALT, j, 3]), greedy=mode == "argmax",
                               scheduled=mode == "matched")
            t0 = time.perf_counter()
            r = r2.play(tr, pol, j)
            ro = pol.finish(tr.step_no)
            G = ro.returns()
            ro.check(G)
            r |= {"cal": calibration(ro.rows, G), "options": decision_summary(ro.rows, G) | {"waits": pol.waits.tolist()}}
            out.append(r)
            print(f"  eval {mode} world {j}: nL {r['n_launderers']} reward {r['reward']:+.3f} "
                  f"utility {r['btc_utility']:+.3f} held {r['held_frac']:.3f} tranches {r['tranches']:.2f} "
                  f"V-G {r['cal'].get('bias', float('nan')):+.2f} {time.perf_counter() - t0:.0f}s", flush=True)
    return out


@contextmanager
def eval_views(tr: mt.Trainer):
    """Evaluation samples every role's graph views from the references' stream (``cap.make``'s seed), so a world is
    the same game for every launderer seed; only the configuration's seed is swapped, and only for ``view_seed`` (no
    other evaluation path reads it). Training keeps its seed's stream. Seed 0's records are unchanged."""
    cfg, tr.cfg = tr.cfg, replace(tr.cfg, seed=cap.SEED)
    try:
        yield
    finally:
        tr.cfg = cfg


# ---------------------------------------------------------------------------- the reading
def _dev(rows: list[dict]) -> list[dict]:
    by = {r["world"]: r for r in rows}
    return [by[j] for j in DEV_WORLDS if j in by]


def read_block(res: dict, snaps: dict | None, matched: list[dict] | None, refs: dict, a_final: list[dict],
               n_worlds: int) -> dict:
    """The five criteria (module docstring). ``snaps``: world -> natural records on ``DEV_WORLDS`` (``None`` if
    the run or its evaluation aborted); ``refs``: the reference policies' records on the same worlds; ``n_worlds``:
    the attempt's training worlds."""
    if res["status"] != "ok" or snaps is None or matched is None:
        return {"status": res["status"], **{f"c{i}": False for i in range(1, 6)}, "promising": False}
    col = r2._col
    first, last = min(snaps), max(snaps)
    u = {w: col(v, "btc_utility") for w, v in snaps.items()}
    rew = col(snaps[last], "reward")
    d50 = refs["direct50"]
    ud, rd, ua = col(d50, "btc_utility"), col(d50, "reward"), col(a_final, "btc_utility")
    u_bar = max(0.5 * ud.mean(), ua.mean() - A_SLACK)
    c1 = (res["worlds_done"] == n_worlds and res["accounting_gap_max"] <= ACCT_MAX
          and res["ratio_dev_max"] is not None and res["ratio_dev_max"] <= RATIO_TOL)
    c2 = u[last].mean() > 0 and u[last].mean() >= u_bar and rew.mean() >= 0.5 * rd.mean()
    keep = u[last] - u[first]
    cal = [r["cal"] for r in snaps[last]]
    bias = float(np.mean([c["bias"] for c in cal])) if all(c["n"] for c in cal) else float("nan")
    g_abs = float(np.mean([c["g_abs"] for c in cal])) if all(c["n"] for c in cal) else float("nan")
    c4 = bool(np.isfinite(bias) and abs(bias) <= max(0.5, 0.25 * g_abs))
    held = col(snaps[last], "held_frac") - col(d50, "held_frac")
    skill = col(matched, "btc_utility") - col(refs["uniform"], "btc_utility")
    c5 = held.mean() <= HELD_SLACK and skill.mean() >= SKILL_MIN
    out = {"status": "ok", "c1": bool(c1), "c2": bool(c2), "c3": bool(keep.mean() >= -RETAIN), "c4": c4, "c5": bool(c5),
           "utility": {str(w): float(x.mean()) for w, x in u.items()}, "reward": float(rew.mean()),
           "utility_bar": float(u_bar), "reward_bar": float(0.5 * rd.mean()),
           "direct50": {"utility": float(ud.mean()), "reward": float(rd.mean()),
                        "held": float(col(d50, "held_frac").mean())},
           "a_final_utility": float(ua.mean()),
           "retention": {"differences": keep.tolist(), "interval": list(_t95(keep))},
           "critic": {"bias": bias, "g_abs": g_abs, "tolerance": max(0.5, 0.25 * g_abs) if np.isfinite(g_abs) else None},
           "held_over_direct50": {"differences": held.tolist(), "interval": list(_t95(held))},
           "matched": {"utility": float(col(matched, "btc_utility").mean()),
                       "uniform": float(col(refs["uniform"], "btc_utility").mean()),
                       "skill": {"differences": skill.tolist(), "interval": list(_t95(skill))}},
           "worlds_done": res["worlds_done"], "accounting_gap_max": res["accounting_gap_max"],
           "ratio_dev_max": res["ratio_dev_max"]}
    return out | {"promising": all(out[f"c{i}"] for i in range(1, 6))}


def reported(snaps: dict, refs: dict, opp: str, argmax: list[dict] | None) -> dict:
    """Not gated: paired comparisons, where the proceeds went, the critic's strata and the launch behaviour."""
    col, last = r2._col, snaps[max(snaps)]
    u, rw = col(last, "btc_utility"), col(last, "reward")
    out = {"vs": {}}
    for name in ("wait", "direct50", "repeat50"):
        out["vs"][name] = {"utility": list(_t95(u - col(refs[name], "btc_utility"))),
                           "reward": list(_t95(rw - col(refs[name], "reward")))}
    f = r2.run_dir("PL", opp, 0) / "stage1.json"
    if f.exists() and json.loads(f.read_text())["snaps"] is not None:
        pl = _dev(json.loads(f.read_text())["snaps"]["30"])                # PL's world-30 network
        pl = [r for r in pl if r["world"] in {x["world"] for x in last}]
        out["vs"]["PL_w30"] = {"utility": list(_t95(u - col(pl, "btc_utility"))),
                               "held": float(col(pl, "held_frac").mean())}
    out["proceeds"] = {k: float(col(last, k).mean()) for k in ("success_frac", "frozen_frac", "fee_frac", "held_frac",
                                                                 "frozen_change_frac", "tranches", "detection_rate")}
    cal = [r["cal"] for r in last]
    out["critic"] = {k: float(np.mean([c[k] for c in cal if c.get(k) is not None]))
                     for k in ("bias_btc", "rmse", "rmse_norm", "explained_variance", "g_mean")}
    strata: dict = {}
    for c in cal:
        for k, v in c["strata"].items():
            s = strata.setdefault(k, [0, 0.0])
            s[0] += v["n"]
            s[1] += v["bias"] * v["n"]
    out["critic"]["strata"] = {k: {"n": n, "bias": b / n} for k, (n, b) in sorted(strata.items())}
    opts = [r["options"] for r in last]
    hz: dict = {}
    for o in opts:
        for k, v in o["hazard"].items():
            h = hz.setdefault(k, [0, 0])
            h[0], h[1] = h[0] + v[0], h[1] + v[1]
    out["options"] = {"decisions": int(sum(o["decisions"] for o in opts)),
                      "waits": np.sum([o["waits"] for o in opts], axis=0).tolist(),
                      "launches_by_mode": np.sum([o["launches"] for o in opts], axis=0).tolist(),
                      "launch_noops": int(sum(o["launch_noops"] for o in opts)), "hazard": hz}
    if argmax is not None:
        out["argmax_utility"] = float(col(argmax, "btc_utility").mean())
    return out


def training_summary(out: Path) -> dict:
    """Per training world, from ``metrics.jsonl``: the actor steps kept and undone, complete passes, the KLs and the
    learner's other statistics, and the world's economics."""
    rows = []
    for _, e in mt.read_records(out / "metrics.jsonl"):
        if e["kind"] != "epoch":
            continue
        st = e["learner"].get("launderer", {})
        rows.append({"world": e["world"] + 1, "utility": e["btc_utility"], "held": e["held_frac"],
                     **{k: st.get(k) for k in ("n", "steps_accepted", "steps_rejected", "passes_complete", "kl",
                                               "kl_final", "kl_support", "entropy", "clip_frac", "ratio_q",
                                               "adv_scale", "value_bias", "grad_actor", "grad_critic",
                                               "ratio_dev_max")}})
    return {"worlds": rows}


# ---------------------------------------------------------------------------- commands
def train(a) -> dict:
    att = ATTEMPTS[a.attempt]
    smoke = a.worlds != att.worlds
    out = run_dir(att, a.opp, a.seed, smoke)
    if out.exists():
        if not smoke:
            raise SystemExit(f"{out}: a block's run is not restarted")
        shutil.rmtree(out)
    (out / "snap").mkdir(parents=True)
    tr = make(att, a.opp, a.seed, torch.device(a.device), out)
    (out / "config.yaml").write_text(yaml.safe_dump(
        {**asdict(tr.cfg), "opponent": str(r2.OPPS[a.opp]), "opp_eps": cap.OPP_EPS,
         "learn": ["launderer"], "attempt": a.attempt, "git_sha": mt.git_sha()}, sort_keys=False))
    (out / "metrics.jsonl").write_text("")
    snapshot(tr, out / "snap" / "w00.pt")
    status, t0 = "ok", time.perf_counter()
    try:
        for w in range(1, a.worlds + 1):
            r = tr.run_world()
            if w % 2 == 0 or w >= 15 or w == a.worlds:
                snapshot(tr, out / "snap" / f"w{w:02d}.pt")
            if w in att.rollouts:
                torch.save(tr.policies[0].rollout, out / "snap" / f"rollout_w{w:02d}.pt", pickle_protocol=5)
            st = r["learner"]["launderer"]
            kl = max(st.get("kl_final") or [float("nan")])
            print(f"{out.name} world {w}/{a.worlds} | R_L {r['reward']['launderer']:+.3f} utility {r['btc_utility']:+.3f} "
                  f"held {r['held_frac']:.2f} det {r['detection_rate']:.2f} tranches {r['tranches']} "
                  f"direct {r['strategy_share']['direct']:.2f} | decisions {st.get('n')} steps "
                  f"{st.get('steps_accepted')}+{st.get('steps_rejected')} passes {st.get('passes_complete')} kl {kl:.4f} "
                  f"H {st.get('entropy') or float('nan'):.2f} "
                  f"| {r['sec']['world']:.0f}s, {(time.perf_counter() - t0) / 60:.1f} min", flush=True)
    except ABORTS as e:
        status = f"aborted: {type(e).__name__}: {e}"
        print(f"{out.name}: {status}", flush=True)
    st = [e["learner"].get("launderer", {}) for e in tr.epochs]
    res = {"attempt": a.attempt, "opp": a.opp, "seed": a.seed, "worlds": a.worlds, "worlds_done": tr.world, "status": status,
           "git_sha": mt.git_sha(), "minutes": (time.perf_counter() - t0) / 60,
           "accounting_gap_max": max((abs(e["accounting_gap"]) for e in tr.epochs), default=0.0),
           "ratio_dev_max": max((s["ratio_dev_max"] for s in st if s.get("n")), default=None),
           "value_dev_max": max((s["value_dev_max"] for s in st if s.get("n")), default=None),
           "return_check_max": max((float(s["return_check"]) for s in st if "return_check" in s), default=None)}
    (out / "result.json").write_text(json.dumps(res, indent=1))
    return res


def read(a) -> dict:
    att = ATTEMPTS[a.attempt]
    smoke = a.worlds != att.worlds or a.eval_worlds is not None
    out = run_dir(att, a.opp, a.seed, smoke)
    res = json.loads((out / "result.json").read_text())
    worlds = DEV_WORLDS[:a.eval_worlds]
    snap_worlds = att.snaps if not smoke else (0, a.worlds)      # a smoke run reads its first and last snapshot
    snaps = matched = argmax = None
    if res["status"] == "ok":
        tr = make(att, a.opp, a.seed, torch.device(a.device))
        try:
            snaps = {}
            for w in snap_worlds:
                load_snapshot(tr, out / "snap" / f"w{w:02d}.pt")
                snaps[w] = evaluate(tr, "natural", worlds)
            matched = evaluate(tr, "matched", worlds)                  # the endpoint's snapshot is loaded
        except ABORTS as e:
            res, snaps = res | {"status": f"aborted in the reading: {type(e).__name__}: {e}"}, None
        if snaps is not None:                                          # reported only
            try:
                argmax = evaluate(tr, "argmax", worlds)
            except ABORTS:
                argmax = None
    refs = {p: [r for r in r2._refs(a.opp, "selection")[p] if r["world"] in worlds] for p in r2.REFS}
    a_final = [r for r in json.loads((r2.run_dir("A", a.opp, 0) / "result.json").read_text())["eval"]["boltzmann"]
               if r["world"] in worlds]
    reading = read_block(res, snaps, matched, refs, a_final, att.worlds)
    rep = reported(snaps, refs, a.opp, argmax) if snaps is not None else None
    (out / "oppo.json").write_text(json.dumps(
        {"attempt": a.attempt, "opp": a.opp, "seed": a.seed, "reading": reading, "reported": rep, "training": training_summary(out),
         "snaps": None if snaps is None else {str(w): v for w, v in snaps.items()}, "matched": matched,
         "argmax": argmax}, indent=1))
    print(json.dumps(reading, indent=1))
    if rep is not None:
        print(json.dumps(rep, indent=1))
    return reading


def _start(a, smoke: bool) -> tuple[Path, dict]:
    """One attempt: the marker written first makes a second start refuse."""
    mark = mark_path(ATTEMPTS[a.attempt], a.opp, a.seed, smoke)
    if mark.exists() and not smoke:
        raise SystemExit(f"{mark}: this run was already attempted ({json.loads(mark.read_text())['state']}); "
                         "it is not run again")
    mark.parent.mkdir(parents=True, exist_ok=True)
    info = {"attempt": a.attempt, "opp": a.opp, "seed": a.seed, "started": time.strftime("%Y-%m-%d %H:%M:%S"), "state": "started"}
    mark.write_text(json.dumps(info, indent=1))
    return mark, info


def cmd_run(a) -> None:
    """One block, one attempt: train, then read."""
    att = ATTEMPTS[a.attempt]
    smoke = a.worlds != att.worlds or a.eval_worlds is not None
    if not smoke and a.seed != att.dev_seed:
        raise SystemExit(f"a development block of attempt {a.attempt} uses seed {att.dev_seed}")
    mark, info = _start(a, smoke)
    train(a)
    reading = read(a)
    mark.write_text(json.dumps(info | {"state": "done", "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
                                       "promising": reading["promising"]}, indent=1))


def cmd_train(a) -> None:
    """A confirmation seed's run, one attempt: training only (``confirm-eval`` reads it)."""
    att = ATTEMPTS[a.attempt]
    smoke = a.worlds != att.worlds
    if not smoke and (a.seed == att.dev_seed or a.seed not in att.seeds):
        raise SystemExit(f"attempt {a.attempt}'s confirmation trains seeds {[s for s in att.seeds if s != att.dev_seed]}; "
                         f"seed {att.dev_seed} is the development blocks' run")
    mark, info = _start(a, smoke)
    res = train(a)
    mark.write_text(json.dumps(info | {"state": f"trained: {res['status']}",
                                       "finished": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=1))


# ---------------------------------------------------------------------------- the confirmation
def cmd_confirm_eval(a) -> None:
    """One run on the confirmation worlds: natural at both snapshots (attempt 30: worlds 20 and 30), matched at the
    endpoint and the initial actor, argmax at the endpoint (module docstring). A smoke run reads its first and last
    snapshot in their place."""
    att = ATTEMPTS[a.attempt]
    smoke = a.worlds != att.worlds or a.eval_worlds is not None
    if not smoke and a.seed not in att.seeds:
        raise SystemExit(f"attempt {a.attempt}'s confirmation reads seeds {list(att.seeds)}")
    out = run_dir(att, a.opp, a.seed, smoke)
    f = out / "confirm.json"
    if f.exists():
        raise SystemExit(f"{f}: this run was already evaluated")
    res = json.loads((out / "result.json").read_text())
    worlds = att.conf_worlds[:a.eval_worlds]
    w20, w30 = att.snaps if a.worlds == att.worlds else (0, a.worlds)
    k20, k30 = (f"natural_{w}" for w in att.snaps)
    rec = {"attempt": a.attempt, "opp": a.opp, "seed": a.seed, "worlds": list(worlds), "status": res["status"], "git_sha": mt.git_sha()}
    if res["status"] == "ok":
        tr = make(att, a.opp, a.seed, torch.device(a.device))
        try:
            for key, w, mode in ((k20, w20, "natural"), (k30, w30, "natural"),
                                 ("matched", w30, "matched"), ("initial", 0, "matched")):
                load_snapshot(tr, out / "snap" / f"w{w:02d}.pt")
                rec[key] = evaluate(tr, mode, worlds)
        except ABORTS as e:
            rec["status"] = f"aborted in the evaluation: {type(e).__name__}: {e}"
        if rec["status"] == "ok":                                          # reported only
            try:
                load_snapshot(tr, out / "snap" / f"w{w30:02d}.pt")
                rec["argmax"] = evaluate(tr, "argmax", worlds)
            except ABORTS:
                rec["argmax"] = None
    f.write_text(json.dumps(rec, indent=1))


def _load(f: Path) -> dict:
    """A run's record; a run that left none (cut by its timeout, or crashed) is incomplete."""
    return json.loads(f.read_text()) if f.exists() else {"status": f"incomplete: no {f}"}


def _by_world(rows: list[dict], worlds) -> list[dict]:
    by = {r["world"]: r for r in rows}
    return [by[j] for j in worlds]


def read_confirmation(results: dict, evals: dict, refs: dict, worlds, att: Attempt) -> dict:
    """The six criteria for one opponent (module docstring). ``results`` / ``evals``: seed -> ``result.json`` /
    ``confirm.json`` (``_load``: a missing file is an incomplete record); ``refs``: the scripted policies' records on
    ``worlds``, or ``None`` if any is missing. Anything not ok leaves the opponent incomplete, never read."""
    seeds = sorted(evals)
    bad = [f"seed {s} {k}: {x['status']}" for s in seeds for k, x in (("training", results[s]), ("evaluation", evals[s]))
           if x["status"] != "ok"] + (["references: incomplete"] if refs is None else [])
    if bad:
        return {"status": "; ".join(bad), **{f"c{i}": False for i in range(1, 7)}, "confirmed": False}
    col = lambda rows, k: r2._col(_by_world(rows, worlds), k)                     # noqa: E731
    m = lambda key, k: np.stack([col(evals[s][key], k) for s in seeds])          # noqa: E731  (seed, world)
    k20, k30 = (f"natural_{w}" for w in att.snaps)
    U, R, H, U20 = m(k30, "btc_utility"), m(k30, "reward"), m(k30, "held_frac"), m(k20, "btc_utility")
    M, M0 = m("matched", "btc_utility"), m("initial", "btc_utility")
    ref = {p: {k: col(refs[p], k) for k in ("btc_utility", "reward", "held_frac")} for p in refs}
    d50, wait, unif = ref["direct50"], ref["wait"], ref["uniform"]
    c1 = all(results[s]["worlds_done"] == att.worlds and results[s]["accounting_gap_max"] <= ACCT_MAX
             and results[s]["ratio_dev_max"] is not None and results[s]["ratio_dev_max"] <= RATIO_TOL for s in seeds)
    critic = {}
    for s in seeds:
        cal = [r["cal"] for r in _by_world(evals[s][k30], worlds)]
        b, g = float(np.mean([c["bias"] for c in cal])), float(np.mean([c["g_abs"] for c in cal]))
        critic[s] = {"bias": b, "g_abs": g, "tolerance": max(0.5, 0.25 * g), "ok": abs(b) <= max(0.5, 0.25 * g)}
    c2 = all(c["ok"] for c in critic.values())
    u, r = U.mean(0), R.mean(0)
    vs_wait_u, vs_wait_r = _t95(u - wait["btc_utility"]), _t95(r - wait["reward"])
    c3 = bool(u.mean() > 0 and (U.mean(1) > 0).sum() >= 2 and u.mean() >= 0.5 * d50["btc_utility"].mean()
              and r.mean() >= 0.5 * d50["reward"].mean() and vs_wait_u[1] > 0 and vs_wait_r[1] > 0)
    held = H.mean(0) - d50["held_frac"]
    c4 = bool(held.mean() <= HELD_SLACK)
    sk_u, sk_i = _t95(M.mean(0) - unif["btc_utility"]), _t95(M.mean(0) - M0.mean(0))
    seed_skill = (M - unif["btc_utility"]).mean(1)
    c5 = bool(sk_u[0] >= SKILL_MIN and sk_u[1] > 0 and sk_i[0] >= SKILL_MIN and sk_i[1] > 0 and (seed_skill >= 0).sum() >= 2)
    keep = _t95((U - U20).mean(0))
    c6 = bool(keep[1] > -RETAIN)
    out = {"status": "ok", "c1": bool(c1), "c2": c2, "c3": c3, "c4": c4, "c5": c5, "c6": c6,
           "utility": float(u.mean()), "reward": float(r.mean()), "utility_by_seed": dict(zip(seeds, U.mean(1).tolist())),
           "direct50": {"utility": float(d50["btc_utility"].mean()), "reward": float(d50["reward"].mean()),
                        "held": float(d50["held_frac"].mean())},
           "vs_wait": {"utility": list(vs_wait_u), "reward": list(vs_wait_r)},
           "critic_by_seed": critic, "held_over_direct50": list(_t95(held)),
           "matched": {"utility": float(M.mean()), "uniform": float(unif["btc_utility"].mean()),
                       "initial": float(M0.mean()), "vs_uniform": list(sk_u), "vs_initial": list(sk_i),
                       "vs_uniform_by_seed": dict(zip(seeds, seed_skill.tolist())),
                       "initial_seed_spread": float(np.abs(M0 - M0[0]).max())},
           "retention": list(keep)}
    return out | {"confirmed": all(out[f"c{i}"] for i in range(1, 7))}


def confirmation_reported(evals: dict, refs: dict, worlds, att: Attempt) -> dict:
    """Not gated: differences to direct50 and repeat50, the argmax policy, proceeds, critic strata, launch modes."""
    seeds, k30 = sorted(evals), f"natural_{att.snaps[1]}"
    col = lambda rows, k: r2._col(_by_world(rows, worlds), k)                     # noqa: E731
    u = np.mean([col(evals[s][k30], "btc_utility") for s in seeds], 0)
    rw = np.mean([col(evals[s][k30], "reward") for s in seeds], 0)
    out = {"vs": {p: {"utility": list(_t95(u - col(refs[p], "btc_utility"))),
                      "reward": list(_t95(rw - col(refs[p], "reward")))} for p in ("direct50", "repeat50")}}
    out["argmax_by_seed"] = {s: (float(col(evals[s]["argmax"], "btc_utility").mean()) if evals[s].get("argmax") else None)
                             for s in seeds}
    out["proceeds"] = {k: float(np.mean([col(evals[s][k30], k) for s in seeds]))
                       for k in ("success_frac", "frozen_frac", "fee_frac", "held_frac", "tranches", "detection_rate")}
    strata: dict = {}
    for s in seeds:
        for r in evals[s][k30]:
            for k, v in r["cal"].get("strata", {}).items():
                x = strata.setdefault(k, [0, 0.0])
                x[0], x[1] = x[0] + v["n"], x[1] + v["bias"] * v["n"]
    out["critic_strata"] = {k: {"n": n, "bias": b / n} for k, (n, b) in sorted(strata.items())}
    out["launches_by_mode"] = np.sum([r["options"]["launches"] for s in seeds for r in evals[s][k30]], 0).tolist()
    return out


def cmd_confirm(a) -> None:
    """The confirmation's reading for both opponents (module docstring)."""
    att = ATTEMPTS[a.attempt]
    smoke = a.worlds != att.worlds or a.eval_worlds is not None
    f = (r2.RUNS / "smoke" if smoke else r2.RUNS) / f"{att.name.lower()}_confirmation.json"
    if f.exists() and not smoke:
        raise SystemExit(f"{f}: this confirmation was already read; it is not read again")
    worlds = att.conf_worlds[:a.eval_worlds]
    out = {"attempt": a.attempt, "worlds": list(worlds), "opponents": {}}
    for opp in r2.OPPS:
        results = {s: _load(run_dir(att, opp, s, smoke) / "result.json") for s in att.seeds}
        evals = {s: _load(run_dir(att, opp, s, smoke) / "confirm.json") for s in att.seeds}
        files = {p: r2.ref_path(opp, att.panel, p, smoke) for p in r2.REFS}
        refs = ({p: [r for r in json.loads(f.read_text())["eval"] if r["world"] in worlds] for p, f in files.items()}
                if all(f.exists() for f in files.values()) else None)
        verdict = read_confirmation(results, evals, refs, worlds, att)
        rep = confirmation_reported(evals, refs, worlds, att) if verdict["status"] == "ok" else None
        out["opponents"][opp] = {"reading": verdict, "reported": rep}
    out["confirmed"] = all(o["reading"]["confirmed"] for o in out["opponents"].values())
    f.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


def cmd_read(a) -> None:
    att = ATTEMPTS[a.attempt]
    smoke = a.worlds != att.worlds or a.eval_worlds is not None
    out = run_dir(att, a.opp, a.seed, smoke)
    if not (out / "result.json").exists() or (out / "oppo.json").exists():
        raise SystemExit(f"{out}: read finishes a block whose training completed and whose reading did not run")
    read(a)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    cmds = {"run": cmd_run, "read": cmd_read, "train": cmd_train, "confirm-eval": cmd_confirm_eval,
            "confirm": cmd_confirm}
    for name in cmds:
        p = sub.add_parser(name)
        p.add_argument("--attempt", choices=list(ATTEMPTS), default="30", help="the declared attempt (docstring)")
        if name != "confirm":
            p.add_argument("--opp", choices=list(r2.OPPS), required=True)
            p.add_argument("--seed", type=int, help="default: the attempt's development seed")
            p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
        p.add_argument("--worlds", type=int, help="default: the attempt's; anything else is a smoke run")
        p.add_argument("--eval-worlds", type=int, help="a smoke run's evaluation worlds")
    a = ap.parse_args(argv)
    att = ATTEMPTS[a.attempt]
    a.worlds = att.worlds if a.worlds is None else a.worlds
    if a.cmd != "confirm" and a.seed is None:
        a.seed = att.dev_seed
    cmds[a.cmd](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
