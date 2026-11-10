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
