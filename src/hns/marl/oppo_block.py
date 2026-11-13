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

**Confirmation reading of OPPO-MC-60 (2026-11-13, A30, run r_64ac9a83, ``runs/repair2/oppo60_confirmation.json``):
not confirmed.** Every step completed (0 failed); both readings are complete.

- Full-0, **not confirmed**: criterion 6 fails. Retention, the seeds' mean U60 - U40, is -.036 [-.089, +.016] (the
  lower bound must be above -.05). Per seed: -.011, +.008, -.106 (seed 5: .664 to .559 with held .032, so the drop
  is not held funds). Per world it runs from -.136 to +.055. The other five hold: U +.585 (seeds .583, .614, .559;
  direct50 .701), R +.564; vs wait +.585 [+.535, +.635]; critic bias +.03, +.02, -.01 (tolerance .5); held -
  direct50 -.065 [-.080, -.050]; matched +.650, vs uniform +1.55 [+1.47, +1.63], vs initial +1.50 [+1.38, +1.61].
- NDA-1, **confirmed** (6 of 6): U +.619 (seeds .642, .640, .575; direct50 .682), R +.596; vs wait +.619 [+.567,
  +.672]; critic bias +.25, +.01, -.05; held - direct50 -.027 [-.050, -.005]; matched +.622, vs uniform +1.55 [+1.40,
  +1.69], vs initial +1.50 [+1.31, +1.69]; retention +.022 [-.042, +.087]. In training, seeds 3 and 5 held about .5
  over worlds 21-30, as seed 1 of attempt 30 did. Both reopened by worlds 31-40 (held .05 and .03).
- Reported, not gated: final utility minus ``direct50``'s is -.116 [-.190, -.041] for Full-0 and -.063 [-.198,
  +.072] for NDA-1. Argmax by seed: Full-0 .644, .646, .659; NDA-1 .638, .633, .485. Natural launches are 96 % (Full-0)
  and 99 % (NDA-1) direct at 0.5. Initial-actor spread across seeds: 0 in both.
- In training, Full-0's utility levels off from about world 31 (ten-world means .57-.66 for every seed). The world-40
  and world-60 policies are both on that plateau, and their difference on worlds 208-215 has a world-to-world
  standard deviation of .063.

The six criteria, as declared, are not met for both opponents in one attempt. Full-0 passed attempt 30 and failed
attempt 60; NDA-1 failed attempt 30 and passed attempt 60. Runs of different attempts are not combined, and neither
opponent's learner is chosen separately, so the live pilot is not permitted. Nothing is re-run; a new declaration
follows (the user's call).

**Third attempt: OPPO-MC-60R (declared 2026-11-13 and approved by the user, before any of it ran).** Attempts 30 and
60 stay as read: no verdict is amended, no old run is read again, and no run of one attempt is pooled with another's.

*What a review of attempt 60's records found* (2026-11-13, CPU only, nothing re-run):

- Criterion 6 as declared (a lower 95 % bound above -.05 over eight worlds) passes a learner that has lost nothing
  49 % of the time at Full-0's noise (the seeds' mean difference has a world-to-world standard deviation of .063) and
  36 % at NDA-1's (.077): ``retention_power``. No declaration had checked that.
- By seed, world 60 minus world 40 on worlds 208-215 is -.011, +.008, -.106 (Full-0) and -.066, -.076, +.209 (NDA-1).
  NDA-1 passed criterion 6 only because seed 5 was still recovering at world 40. Four of the five runs that had
  converged by world 40 are lower at world 60; the ten-world training means are flat.
- The actor's choices barely depend on the state (a torchlens inspection of all twelve runs,
  ``runs/inspect_m5/oppo_lens.py``): one launch probability, one mode mix and one wait mix serve every decision.
  Between worlds 40 and 60 that launch probability rose in all six runs (.63-.83 to .78-.94), with shorter waits and
  more launches per episode. The small loss after world 40 is therefore probably a drift of that setting and not only
  noise.

*What changes, and what does not.* The learner, its 60-world budget, the curriculum, the three KL checks, the
simulator, the observations and the frozen opponents are attempt 60's, unchanged. The attempt changes the seeds, the
evaluation worlds and criterion 6:

- Runs: fresh launderer seeds 6, 7 and 8 against both opponents, 60 worlds each (``train --attempt 60r``), one
  attempt each with its seed's multiverse draws, on a JarvisLabs A30, the six runs and the two reference jobs at
  once under CUDA MPS (attempt 60's confirmation ran four and two), a two-hour timeout per process. There is no
  development block: the learner is attempt 60's, whose development readings stand as read (NDA-1 promising, Full-0
  not), and ``run`` refuses this attempt.
- Evaluation: seeds 6, 7 and 8 on the sixteen unused worlds 224-239 (``r2.PANELS["confirmation60r"]``; 216-223 and
  240-247 are kept for the live pilot), on the references' view stream: natural at worlds 40 and 60, matched at world
  60 and the initial actor, argmax at world 60 (reported). The scripted references are computed on 224-239 before any
  learner evaluation. Each run's evaluation is its own fresh process, one at a time.
- Reading: the six criteria of the confirmation above, with 60 in place of 30 and 40 in place of 20, every interval a
  two-sided 95 % t-interval over the sixteen worlds. Criteria 1 to 5 and their tolerances are unchanged (utilization:
  the mean held excess over ``direct50`` at most .05, over all three seeds). **Criterion 6 is amended**: the lower
  bound of ``mean_s (u_sj - world-40 u_sj)`` must be above **-.15** (``Attempt.retain``).

*The amendment is made after a failure.* Attempt 60's Full-0 failed the old criterion 6, and this margin was chosen
knowing that, so a pass here is weaker evidence than a pass of a test fixed in advance; the report says so. The
user first approved -.10 over sixteen worlds. Before any run, resampling attempt 60's records by world and by seed
(all six runs, per opponent) showed that -.10 would pass both opponents 58 % of the time: the five runs that had
converged by world 40 lose .050 on average after it (all six, with NDA-1 seed 5's +.209, average -.007). The user
then chose -.15 over sixteen worlds (the alternatives shown were -.10 over sixteen, and -.125
over thirty-two). The margin is set by what the test has to catch: the Q-learners' losses were .25 to .43.

*Power, stated before the runs* (SPEC §16; ``python -m hns.marl.oppo_block power --attempt 60r --source 60``):

- Criterion 6 alone, at a world-to-world standard deviation of .063 / .077: it passes a true difference of 0 or
  -.05 in 100 % of runs, -.07 in 99.7 % / 97 %, -.10 in 84 % / 68 %, -.15 in 2.5 % (its size), -.25 in none.
- All six criteria, resampling attempt 60's own records by world and by seed (20,000 draws): Full-0 96.9 %, NDA-1
  97.4 %, about 94 % for both; criteria 1 to 5 pass in every draw. This knows only what seeds 3-5 did. A seed still
  holding its funds at world 60 would fail criterion 4, and has probability zero here; one of the six NDA-1 seeds of
  both attempts still held .89 at world 30, and none of the three did at world 60.

*Reported, not gated:* everything the confirmation above reports, and: criterion 6's difference for each seed, and
in discounted reward (the quantity the launderer is trained on) with its interval; each run's kept actor steps; and
**state use** (``state_use``) at the world-40 and world-60 snapshots, over the run's saved decisions of training
worlds 40 and 60 at which a launch is feasible: the launch probability's 5 %, 50 % and 95 % quantiles and its mean
by balance band, and the mean change of the launch probability and of the mode distribution (total variation) when
every use of the controller features, or the whole graph readout, is replaced by its mean. On attempt 60's runs it
reproduces the inspection's numbers.

*Records:* ``OPPO60R_{opp}_s{seed}``, markers ``oppo60r_{opp}_s{seed}.json``, references
``ref_{opp}_confirmation60r_*``, the reading ``oppo60r_confirmation.json``; the script is ``scripts/m5_oppo60r.sh``.
Nothing of attempts 30 or 60 is read or overwritten.

**What follows** (fixed now): both opponents confirmed -> the live pilot (SPEC §10, step 3: 60 fresh worlds per arm,
a world-30 fork, worlds 216-223 and 240-247), declared in full before it runs. Held funds or natural economics fail
while matched skill passes -> the timing-coverage package declared for attempt 60 (a smooth launch floor of 0.10 and
waits of 1 or 5 sub-steps), not more worlds. Retention fails -> it was not shown that the learner keeps its utility
within the margin after world 40 (a real loss, or too little precision); the launch probability's drift is read
first (state use at 40 and 60), and the next declaration changes the learner, not the margin and not the endpoint. Matched skill fails -> audit likelihoods, masks and
credit. Only the critic fails -> the critic is revised, not its tolerance. An incomplete or numerically failed step
is never a pass; only the user may void an attempt for an established harness fault.

**Reading of OPPO-MC-60R (2026-11-13, A30, run r_fe75184b, ``runs/repair2/oppo60r_confirmation.json``): confirmed
against both opponents.** Every step completed (0 failed), from commit 88c24be; all six runs trained 60 worlds.

- Full-0, **confirmed** (6 of 6): U +.579 (seeds .510, .658, .571; direct50 .677), R +.556; vs wait +.579 [+.553,
  +.606]; critic bias +.19, +.16, +.00 (tolerance .5); held - direct50 -.071 [-.081, -.061]; matched +.640, vs
  uniform +1.50 [+1.47, +1.52], vs initial +1.49 [+1.44, +1.54]; retention -.058 [-.081, -.034] (margin -.15).
- NDA-1, **confirmed** (6 of 6): U +.504 (seeds .328, .582, .603; direct50 .663), R +.485; vs wait +.504 [+.473,
  +.535]; critic bias +.20, -.01, -.06; held - direct50 -.046 [-.061, -.031]; matched +.627, vs uniform +1.57
  [+1.53, +1.61], vs initial +1.56 [+1.51, +1.60]; retention -.091 [-.133, -.049] (margin -.15).
- What the amended margin decided. Under the -.10 margin first approved, NDA-1 fails criterion 6 (its lower bound is
  -.133) and Full-0 passes; under attempt 60's -.05 both fail. Both retention intervals exclude zero: in this attempt
  the learner does lose utility between worlds 40 and 60. The pass is of the test as amended after attempt 60's
  failure, not of the test first declared.
- Reported, not gated. Retention by seed: -.122, +.015, -.066 (Full-0) and -.260, +.011, -.025 (NDA-1); in discounted
  reward +.007 [-.088, +.102] and -.057 [-.173, +.058]. Seed 6 loses most against both opponents: its world-60 actor
  puts 5.0 % and 5.8 % of its launches on non-direct modes (1.2 % and 0.3 % at world 40), and its frozen share rises
  (.17 to .24, .19 to .31); its argmax policy scores .612 and .498. Final utility minus ``direct50``'s is -.098
  [-.161, -.034] and -.159 [-.218, -.099]; minus ``repeat50``'s, -.150 [-.185, -.115] and -.109 [-.162, -.057].
  Argmax by seed: .612, .634, .428 and .498, .621, .626. Natural launches are 90 % (Full-0) and 95 % (NDA-1) direct
  at 0.5, 98 % direct at any amount. Kept actor steps per run: 693 to 1,319.
- State use, as in attempts 30 and 60: the launch probability's 5 %-95 % range is at most 5.4 points wide in every
  run and snapshot (at most 2.9 in eleven of the twelve); replacing the controller features moves it by at most 0.2
  points, the graph readout by at most 1.4. From world 40 to world 60 its median rose in all six runs (.72 to .76,
  .77 to .79, .65 to .87; .69 to .81, .81 to .90, .73 to .97).
- Reproducibility. The three seeds' initial actors differ in one world per opponent (233 by .007, 237 by .009),
  although each evaluation was its own process: GPU evaluation here is not bitwise reproducible. In training, two of
  the six runs departed from the smoke run's first two worlds (same seeds and code, eight processes on the GPU).
- The power line gave 97 % per opponent for runs like attempt 60's. These runs lost more after world 40 than attempt
  60's did (-.058 and -.091 against -.036 and +.022).

As declared, the next step is the live pilot (SPEC §10, step 3), declared in full before it runs.

Usage::

    scripts/m5_repair2.sh oppo full0       (attempt 30: oppo_block run, train then read; the 45-minute cap)
    python -m hns.marl.oppo_block read --opp full0     (only after a harness fault in the reading)
    scripts/m5_oppo_confirm.sh             (attempt 30, on the A30: the confirmation)
    scripts/m5_oppo60.sh [smoke]           (attempt 60, on the A30: both development blocks, then the confirmation)
    scripts/m5_oppo60.sh confirm           (attempt 60: the confirmation only; the deviation above)
    scripts/m5_oppo60r.sh [smoke]          (attempt 60R, on the A30: six runs, their evaluations, the reading)
    python -m hns.marl.oppo_block power --attempt 60r --source 60      (a declaration's power lines; CPU)
"""

from __future__ import annotations

import argparse
import json
import math
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
from hns.marl.obs import collate
from hns.marl.oppo import BANDS, OptionPolicy, calibration, decision_summary
from hns.models import policy as P
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
    dev_seed: int | None        # the development blocks' launderer seed (also a confirmation seed); None: no block
    seeds: tuple[int, ...]      # the confirmation's seeds
    panel: str                  # ``r2.PANELS`` key of the confirmation worlds
    rollouts: tuple[int, ...]   # training worlds whose complete decision rollout is saved
    retain: float = RETAIN      # the confirmation's retention margin (criterion 6)

    @property
    def conf_worlds(self) -> tuple[int, ...]:
        return tuple(r2.PANELS[self.panel])


ATTEMPTS = {"30": Attempt("OPPO", 30, (20, 30), 0, (0, 1, 2), "confirmation", (15, 30)),          # closed 2026-11-13
            "60": Attempt("OPPO60", 60, (40, 60), 3, (3, 4, 5), "confirmation60", (15, 30, 40, 60)),  # closed 2026-11-13
            "60r": Attempt("OPPO60R", 60, (40, 60), None, (6, 7, 8), "confirmation60r", (15, 30, 40, 60), retain=0.15)}


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
            if w in att.rollouts or (smoke and w == a.worlds):
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
           "actor_steps": int(sum(s.get("steps_accepted") or 0 for s in st)),
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
    if att.dev_seed is None:
        raise SystemExit(f"attempt {a.attempt} has no development block (seed: its confirmation only)")
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
        raise SystemExit(f"attempt {a.attempt}'s confirmation trains seeds {[s for s in att.seeds if s != att.dev_seed]}"
                         + ("" if att.dev_seed is None else f"; seed {att.dev_seed} is the development blocks' run"))
    mark, info = _start(a, smoke)
    res = train(a)
    mark.write_text(json.dumps(info | {"state": f"trained: {res['status']}",
                                       "finished": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=1))


# ---------------------------------------------------------------------------- state use (reported)
@torch.no_grad()
def state_use(L, rows: list, chunk: int = 256) -> dict:
    """How far the actor's choices depend on the state: reported, never gated (2026-11-13; the torchlens inspection
    of attempts 30 and 60, ``runs/inspect_m5/oppo_lens.py``, found that they barely do). Over the saved decisions
    ``rows`` at which a launch is feasible: the launch probability's 5 %, 50 % and 95 % quantiles and its mean by
    balance band, and how far the launch probability (mean absolute change) and the mode distribution (mean total
    variation) move at a decision when one source of state information is replaced by its mean over ``rows``:
    ``controller``, every use of the controller features (their skip into the heads and the controller node); or
    ``graph``, the whole graph readout after its LayerNorm. ``L``: the launderer's learner, its actor loaded."""
    L.actor.eval()
    batches = [(collate([r.obs for r in rows[i:i + chunk]]), np.stack([r.masks for r in rows[i:i + chunk]]))
               for i in range(0, len(rows), chunk)]
    x_c = np.concatenate([ob.x_c for ob, _ in batches])
    feas = np.concatenate([m[:, P.SLICES["gate"]][:, P.LAUNCH] for _, m in batches])
    x_mean = x_c.mean(0).astype(np.float32)

    def tables(hook=None, ctrl_mean: bool = False):
        h = L.actor.trunk.norm.register_forward_hook(hook) if hook is not None else None
        try:
            out = []
            for ob, masks in batches:
                if ctrl_mean:
                    ob = replace(ob, x_c=np.broadcast_to(x_mean, ob.x_c.shape).copy())
                t = L._forward(ob, masks)
                out.append((t["gate"].exp()[:, P.LAUNCH].cpu().numpy(), t["mode"].exp().cpu().numpy()))
        finally:
            if h is not None:
                h.remove()
        return np.concatenate([x for x, _ in out])[feas], np.concatenate([x for _, x in out])[feas]

    seen = []
    p, mode = tables(lambda m, i, o: seen.append(o.sum(0)))            # also collects the readout's mean
    if not feas.any():
        return {"n": 0}
    r_mean = torch.stack(seen).sum(0) / len(rows)
    edits = {"controller": tables(ctrl_mean=True), "graph": tables(lambda m, i, o: r_mean.expand_as(o).clone())}
    bal = x_c[feas, 1]
    bands = {"low": bal < BANDS[0], "mid": (bal >= BANDS[0]) & (bal < BANDS[1]), "high": bal >= BANDS[1]}
    return {"n": int(feas.sum()), "p_launch": [float(q) for q in np.quantile(p, [0.05, 0.5, 0.95])],
            "p_launch_band": {k: (float(p[m].mean()) if m.any() else None) for k, m in bands.items()},
            "band_n": {k: int(m.sum()) for k, m in bands.items()},
            "dp_launch": {k: float(np.abs(p - q).mean()) for k, (q, _) in edits.items()},
            "mode_tv": {k: float(0.5 * np.abs(mode - z).sum(1).mean()) for k, (_, z) in edits.items()}}


def state_use_of_run(tr: mt.Trainer, out: Path, att: Attempt, snaps: tuple[int, int]) -> dict | None:
    """``state_use`` of both snapshots of a run on its own saved training decisions (the attempt's last two saved
    rollouts; a smoke run's last world). ``None`` if the run saved no rollout."""
    files = [f for f in dict.fromkeys(out / "snap" / f"rollout_w{w:02d}.pt" for w in (*att.rollouts[-2:], snaps[1]))
             if f.exists()]
    rows = [r for f in files for r in torch.load(f, map_location="cpu", weights_only=False).rows]
    if not rows:
        return None
    res = {"rollouts": [f.name for f in files], "decisions": len(rows)}
    for w in snaps:
        load_snapshot(tr, out / "snap" / f"w{w:02d}.pt")
        res[str(w)] = state_use(tr.roles["launderer"], rows)
    return res


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
            try:
                rec["state_use"] = state_use_of_run(tr, out, att, (w20, w30))
            except ABORTS:
                rec["state_use"] = None
            rec["actor_steps"] = res.get("actor_steps")
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
    c6 = bool(keep[1] > -att.retain)
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
           "retention": list(keep), "retention_margin": att.retain}
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
    k20 = f"natural_{att.snaps[0]}"                  # retention again, per seed and in the learner's own objective
    d = lambda k: np.stack([col(evals[s][k30], k) - col(evals[s][k20], k) for s in seeds])   # noqa: E731
    out["retention_by_seed"] = dict(zip(seeds, d("btc_utility").mean(1).tolist()))
    out["retention_discounted"] = list(_t95(d("reward_discounted").mean(0)))
    out["retention_discounted_by_seed"] = dict(zip(seeds, d("reward_discounted").mean(1).tolist()))
    out["state_use"] = {s: evals[s].get("state_use") for s in seeds}
    out["actor_steps"] = {s: evals[s].get("actor_steps") for s in seeds}
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


# ---------------------------------------------------------------------------- power (SPEC §16: stated before a run)
def retention_power(true: float, sd: float, n: int, margin: float) -> float:
    """The probability that a paired lower 95 % bound over ``n`` worlds lies above ``-margin`` when the true mean
    difference is ``true`` and its world-to-world standard deviation ``sd`` (normal differences; noncentral t)."""
    from scipy import stats
    return float(stats.nct.sf(stats.t.ppf(0.975, n - 1), n - 1, (true + margin) * math.sqrt(n) / sd))


def bootstrap_power(src: Attempt, att: Attempt, draws: int = 2000, seed: int = 0, smoke: bool = False) -> dict:
    """How often runs like ``src``'s recorded ones would pass ``att``'s confirmation, per opponent: each draw
    resamples ``src``'s confirmation worlds (with replacement, to ``att``'s number) and its seeds (with replacement)
    and reads the six criteria under ``att``'s rules. It knows only what ``src``'s seeds did: a behaviour none of
    them showed (a seed still holding at the endpoint) has probability zero here."""
    if (src.worlds, src.snaps) != (att.worlds, att.snaps):
        raise SystemExit("power: the recorded attempt must have the declared attempt's budget and snapshots")
    rng, n_w, W = np.random.default_rng([seed, 99]), len(att.conf_worlds), src.conf_worlds
    keys = [f"natural_{w}" for w in src.snaps] + ["matched", "initial"]
    out = {}
    for opp in r2.OPPS:
        results = {s: _load(run_dir(src, opp, s, smoke) / "result.json") for s in src.seeds}
        evals = {s: _load(run_dir(src, opp, s, smoke) / "confirm.json") for s in src.seeds}
        files = {p: r2.ref_path(opp, src.panel, p, smoke) for p in r2.REFS}
        if any(x["status"] != "ok" for x in (*results.values(), *evals.values())) or not all(f.exists() for f in files.values()):
            out[opp] = None
            continue
        ev = {s: {k: _by_world(evals[s][k], W) for k in keys} for s in src.seeds}
        refs = {p: _by_world(json.loads(f.read_text())["eval"], W) for p, f in files.items()}
        n = np.zeros(7)
        for _ in range(draws):
            ws, ss = rng.integers(len(W), size=n_w), rng.choice(src.seeds, len(att.seeds))
            pick = lambda rows: [rows[i] | {"world": j} for j, i in enumerate(ws)]      # noqa: E731
            r = read_confirmation({k: results[s] for k, s in enumerate(ss)},
                                  {k: {"status": "ok"} | {key: pick(ev[s][key]) for key in keys} for k, s in enumerate(ss)},
                                  {p_: pick(x) for p_, x in refs.items()}, range(n_w), att)
            n += [r[f"c{i}"] for i in range(1, 7)] + [r["confirmed"]]
        out[opp] = {**{f"c{i + 1}": float(x) for i, x in enumerate(n[:6] / draws)}, "confirmed": float(n[6] / draws)}
    return out


def cmd_power(a) -> None:
    """A declaration's power lines: retention alone at the recorded noise, and the bootstrap of a recorded attempt."""
    att, src = ATTEMPTS[a.attempt], ATTEMPTS[a.source]
    n = len(att.conf_worlds)
    print(f"retention, lower 95 % bound above -{att.retain:g} over {n} worlds (attempt {a.attempt}):")
    for sd in a.sd:
        print(f"  world-to-world sd {sd:g}: " + ", ".join(
            f"true {t:+.3f} -> {retention_power(t, sd, n, att.retain):.3f}" for t in (0.0, -0.036, -0.07, -att.retain)))
    print(json.dumps({"source": a.source, "draws": a.draws, "pass": bootstrap_power(src, att, a.draws)}, indent=1))


def cmd_read(a) -> None:
    att = ATTEMPTS[a.attempt]
    if att.dev_seed is None:
        raise SystemExit(f"attempt {a.attempt} has no development block (seed: its confirmation only)")
    smoke = a.worlds != att.worlds or a.eval_worlds is not None
    out = run_dir(att, a.opp, a.seed, smoke)
    if not (out / "result.json").exists() or (out / "oppo.json").exists():
        raise SystemExit(f"{out}: read finishes a block whose training completed and whose reading did not run")
    read(a)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    cmds = {"run": cmd_run, "read": cmd_read, "train": cmd_train, "confirm-eval": cmd_confirm_eval,
            "confirm": cmd_confirm, "power": cmd_power}
    for name in cmds:
        p = sub.add_parser(name)
        p.add_argument("--attempt", choices=list(ATTEMPTS), default="30", help="the declared attempt (docstring)")
        if name == "power":
            p.add_argument("--source", choices=list(ATTEMPTS), required=True, help="the recorded attempt to resample")
            p.add_argument("--sd", type=float, nargs="+", default=[0.063, 0.077], help="retention's world-to-world sd")
            p.add_argument("--draws", type=int, default=2000)
            continue
        if name != "confirm":
            p.add_argument("--opp", choices=list(r2.OPPS), required=True)
            p.add_argument("--seed", type=int, help="default: the attempt's development seed")
            p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
        p.add_argument("--worlds", type=int, help="default: the attempt's; anything else is a smoke run")
        p.add_argument("--eval-worlds", type=int, help="a smoke run's evaluation worlds")
    a = ap.parse_args(argv)
    if a.cmd == "power":
        cmd_power(a)
        return 0
    att = ATTEMPTS[a.attempt]
    a.worlds = att.worlds if a.worlds is None else a.worlds
    if a.cmd != "confirm" and a.seed is None:
        if att.dev_seed is None:
            raise SystemExit(f"attempt {a.attempt} has no development seed: give --seed (one of {list(att.seeds)})")
        a.seed = att.dev_seed
    cmds[a.cmd](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
