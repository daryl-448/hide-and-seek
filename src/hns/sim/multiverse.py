"""Multiverse: per-world parameter draws around the calibrated centre (SPEC §9.5).

Uncalibrated parameters are drawn from their §9.5 ranges: launderer count
``n_L`` in {10, 25, 50}, initial detector budget in {2, 5, 10} %, feature
mask rate in [0, 0.3], window offset in [0, 39]. Every calibrated generator
parameter (§9.7) gets +-10 % jitter around the centre, clipped to its
calibration bounds; for the four that §9.5 also lists with a range (service
count, demand rate, ``p_fresh``, feerate), that range is the clip (ours:
this keeps the calibrated centre the default world). Lognormal locations
(``*_mu``) are jittered on their median, ``exp(mu) x (1 +- 10 %)``. The label rates are
the centre's, never jittered (§9.6).
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from hns.sim.world import CALIBRATED, EPISODES, WorldParams

N_LAUNDERERS = (10, 25, 50)
BUDGETS = (0.02, 0.05, 0.10)
MASK_RATE = (0.0, 0.3)
MAX_OFFSET = 49 - EPISODES   # 39
JITTER = 0.10


def draw_world(centre: WorldParams, rng: np.random.Generator) -> WorldParams:
    upd = {}
    for name, (lo, hi, is_int) in CALIBRATED.items():
        v, u = getattr(centre, name), rng.uniform(-JITTER, JITTER)
        # lognormal locations (*_mu) are jittered on the median exp(mu), i.e. mu + log(1 + u)
        x = v + np.log1p(u) if name.endswith("_mu") else v * (1 + u)
        x = float(np.clip(x, lo, hi))
        upd[name] = int(round(x)) if is_int else x
    upd |= {
        "n_launderers": int(rng.choice(N_LAUNDERERS)),
        "budget": float(rng.choice(BUDGETS)),
        "mask_rate": float(rng.uniform(*MASK_RATE)),
        "offset": int(rng.integers(0, MAX_OFFSET + 1)),
    }
    return replace(centre, **upd)
