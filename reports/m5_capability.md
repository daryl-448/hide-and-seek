# M5 follow-up: frozen-opponent learner capability

Design and pass criteria: `hns.marl.capability` (declared before the runs). Each row: a fresh launderer trained for 30 worlds against a frozen M5 inspector and innocents (epsilon 0.05), then evaluated at epsilon 0 on 8 held-out worlds shared by every row. Reward: per launderer and episode. Utility: (cashed - all frozen) / proceeds. Max Q: over the evaluation decisions; ÷ cap, the largest ratio to P1's bound over the worlds (at most 1 to pass).

## 1. M5's per-tranche reward (commit 98369af): `(cashed - frozen tranche) / tranche` per tranche.

| Opponent | Target | Q dropout | Reward | wait / uniform / direct50 | Learned - wait (95 % CI) | Utility | direct50 utility | Max Q (÷ cap) | Q bias (mean abs G) | Dropout gap | P1 | P2 | Pass |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| full0 | mean | 0 | +0.910 | -0.050 / -0.821 / +0.833 | +0.960 (+0.660, +1.260) | +0.248 | +0.719 | 12.7 (0.13) | +5.27 (1.30) | +0.00 | no | yes | no |
| full0 | mean | 0.5 | +0.163 | -0.050 / -0.821 / +0.833 | +0.213 (+0.131, +0.295) | -0.081 | +0.719 | 855.1 (8.49) | +66.97 (0.52) | -2.75 | no | no | no |
| full0 | root | 0 | +4.839 | -0.050 / -0.821 / +0.833 | +4.889 (+4.472, +5.306) | +0.317 | +0.719 | 9.3 (0.09) | -5.26 (10.34) | +0.00 | no | yes | no |
| full0 | root | 0.5 | -0.253 | -0.050 / -0.821 / +0.833 | -0.203 (-0.230, -0.176) | -0.410 | +0.719 | 1208.0 (12.00) | +596.54 (0.37) | -14.12 | no | no | no |
| nda1 | mean | 0 | -0.043 | -0.050 / -0.911 / +0.808 | +0.007 (-0.001, +0.014) | -0.002 | +0.697 | 12.0 (0.12) | +5.87 (0.10) | +0.00 | no | no | no |
| nda1 | mean | 0.5 | -0.319 | -0.050 / -0.911 / +0.808 | -0.269 (-0.326, -0.211) | -0.339 | +0.697 | 3901.1 (38.74) | +148.35 (0.42) | -5.06 | no | no | no |
| nda1 | root | 0 | +1.573 | -0.050 / -0.911 / +0.808 | +1.623 (+1.417, +1.828) | +0.018 | +0.697 | 27.7 (0.28) | +0.49 (3.56) | +0.00 | yes | yes | **yes** |
| nda1 | root | 0.5 | +0.005 | -0.050 / -0.911 / +0.808 | +0.055 (+0.042, +0.068) | +0.037 | +0.697 | 123.8 (1.23) | +82.70 (0.08) | -1.17 | no | no | no |

Learned launch mix (share of launches, mean over evaluation worlds):

| Opponent | Target | Q dropout | Tranches / L / episode | Per-tranche reward | Top strategies | Amounts 10 / 25 / 50 / 100 % | Detection rate |
|---|---|---|---|---|---|---|---|
| full0 | mean | 0 | 1.21 | +0.910 | `direct` 0.96, `gather_scatter` 0.02, `cycle` 0.01 | 0.43 / 0.02 / 0.55 / 0.00 | 0.07 |
| full0 | mean | 0.5 | 0.60 | +0.163 | `direct` 0.74, `random` 0.19, `scatter_gather` 0.03 | 0.53 / 0.00 / 0.38 / 0.09 | 0.16 |
| full0 | root | 0 | 7.02 | +4.839 | `direct` 0.95, `bipartite` 0.02, `random` 0.02 | 0.18 / 0.11 / 0.70 / 0.01 | 0.14 |
| full0 | root | 0.5 | 0.30 | -0.253 | `gather_scatter` 0.88, `stack` 0.12, `direct` 0.00 | 0.00 / 0.00 / 0.00 / 1.00 | 0.88 |
| nda1 | mean | 0 | 0.01 | -0.043 | `direct` 0.50, `peel` 0.11, `cycle` 0.01 | 0.21 / 0.08 / 0.34 / 0.00 | 0.10 |
| nda1 | mean | 0.5 | 0.47 | -0.319 | `peel` 0.51, `scatter_gather` 0.22, `random` 0.12 | 0.14 / 0.05 / 0.66 / 0.15 | 0.30 |
| nda1 | root | 0 | 4.29 | +1.573 | `direct` 0.76, `gather_scatter` 0.16, `scatter_gather` 0.03 | 0.05 / 0.25 / 0.69 / 0.01 | 0.20 |
| nda1 | root | 0.5 | 0.11 | +0.005 | `direct` 0.93, `peel` 0.07, `bipartite` 0.00 | 0.00 / 0.00 / 0.19 / 0.81 | 0.22 |

## 2. The per-proceeds reward (§9.3, revised): `(cashed - all frozen) / D_w`. The per-tranche reward is logged as a diagnostic.

| Opponent | Target | Q dropout | Reward | wait / uniform / direct50 | Learned - wait (95 % CI) | Utility | direct50 utility | Max Q (÷ cap) | Q bias (mean abs G) | Dropout gap | P1 | P2 | Pass |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| full0 | root | 0 | -0.058 | -0.050 / -0.904 / +0.628 | -0.008 (-0.064, +0.047) | -0.013 | +0.676 | 6.8 (0.28) | +0.43 (0.27) | +0.00 | yes | no | no |
| full0 | root | 0.5 | +0.423 | -0.050 / -0.904 / +0.628 | +0.473 (+0.066, +0.880) | +0.423 | +0.676 | 7.7 (0.35) | -0.97 (1.66) | +0.10 | no | yes | no |
| nda1 | root | 0 | +0.245 | -0.050 / -0.967 / +0.622 | +0.295 (+0.264, +0.325) | +0.269 | +0.669 | 24.4 (1.25) | +0.24 (0.47) | +0.00 | no | no | no |
| nda1 | root | 0.5 | +0.108 | -0.050 / -0.967 / +0.622 | +0.158 (-0.140, +0.456) | +0.108 | +0.669 | 4.2 (0.28) | -1.93 (1.51) | +0.68 | no | no | no |

Learned launch mix (share of launches, mean over evaluation worlds):

| Opponent | Target | Q dropout | Tranches / L / episode | Per-tranche reward | Top strategies | Amounts 10 / 25 / 50 / 100 % | Detection rate |
|---|---|---|---|---|---|---|---|
| full0 | root | 0 | 1.07 | +0.053 | `direct` 0.55, `cycle` 0.42, `bipartite` 0.02 | 0.57 / 0.03 / 0.40 / 0.00 | 0.10 |
| full0 | root | 0.5 | 1.00 | -0.050 | `direct` 1.00, `fan_out` 0.00, `scatter_gather` 0.00 | 0.00 / 0.00 / 0.00 / 1.00 | 0.53 |
| nda1 | root | 0 | 5.41 | +3.888 | `direct` 0.96, `bipartite` 0.03, `random` 0.01 | 0.00 / 0.03 / 0.96 / 0.02 | 0.15 |
| nda1 | root | 0.5 | 1.00 | -0.076 | `direct` 1.00, `fan_out` 0.00, `scatter_gather` 0.00 | 0.00 / 0.00 / 0.00 / 1.00 | 0.54 |

<!-- findings: hand-written below; `report` keeps them -->
## Findings (2026-11-10)

**Verdict.** No variant passes under either reward. By the rule declared before the runs, the launderer learner
needs further repair before any live pilot. The test does not decide the Q-network dropout either, because neither
setting passes.

**Dropout 0.5 is unstable under both rewards.**
- Per tranche: max `Q` 124 to 3,901 in 4 of 4 runs.
- Per proceeds: the launderer TD loss climbs through training, from 0.2 to 2,715 against NDA-1 and from 0.2 to 0.48
  against Full-0.
- Per proceeds, the evaluated policy is `direct` at 100 % every time, flagged on about 53 % of cash-outs.
- Dropout 0 never diverged.

**Root target, dropout 0, per-proceeds reward: it finds a good policy, then loses it.**
- Against Full-0, training worlds 18 to 24 of 30 (epsilon 0.05) were 87 to 96 % `direct`, with detection 0.08 to 0.13
  and BTC utility +0.37 to +0.49. For scale, `direct50` evaluates at +0.68, but on other worlds and at epsilon 0.
- From world 25, `stack`, `scatter_gather`, `cycle` and `bipartite` come back, and get caught. By world 30 the
  policy was 35 % `direct` with utility -0.66. The final network evaluates at utility -0.01, about `wait`.
- Against NDA-1 the peak was worlds 19 to 22 (`direct` 81 to 93 %, utility up to +0.29), followed by the same decline.
- The TD loss stays near 0.05 throughout, so this is drift, not divergence.
- Likely mechanism (a hypothesis, not tested):
  - Once the policy concentrates on `direct`, the replay buffer loses the other strategies' transitions. It holds
    100k entries, about 8 worlds.
  - From then on, only the shared trunk and the dueling mean-subtraction move those strategies' advantages.
    Nothing anchors them.
  - The near-greedy policy (tau 0.01, epsilon 0.05) then picks whichever strategy rises above `direct`, until that
    strategy is caught again.

**It over-launches.**
- At its peak the learned policy launched 4.5 to 6.9 tranches per launderer and episode against Full-0, and 1.0 to
  4.1 against NDA-1. `direct50` launches once.
- Each launch puts the source's change at risk: frozen change took 6 to 24 % of the proceeds in those worlds.
- The per-proceeds reward charges that loss, but 30 worlds did not teach the learner to avoid it.

**Calibration.**
- At dropout 0, `Q_initiate` over-estimates by +0.24 to +0.43 on average, against a mean |G| of 0.27 to 0.47.
- Against NDA-1 the largest `Q` (24.4) exceeds the most any launderer could cash in that world (1.25 times the cap).
