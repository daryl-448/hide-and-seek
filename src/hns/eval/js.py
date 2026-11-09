"""Jensen-Shannon divergence over fixed bins (SPEC §7, §5 D5).

Bins are ``bins`` equal-width bins over ``[lo, hi]`` on the transformed
(pre-standardization) scale; the outermost bins collect everything beyond
the range. Log base 2, so JS is in [0, 1].
"""

from __future__ import annotations

import numpy as np


def histogram(v: np.ndarray, lo: float, hi: float, bins: int) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    if np.isnan(v).any():
        raise ValueError("NaN in JS input")
    idx = np.floor((v - lo) / (hi - lo) * bins).astype(np.int64)
    return np.bincount(np.clip(idx, 0, bins - 1), minlength=bins).astype(np.float64)


def js(p: np.ndarray, q: np.ndarray) -> float:
    """JS divergence (base 2) of two count vectors."""
    p = np.asarray(p, np.float64)
    q = np.asarray(q, np.float64)
    if p.sum() <= 0 or q.sum() <= 0:
        return 1.0  # one side empty: maximal divergence
    p, q = p / p.sum(), q / q.sum()
    m = 0.5 * (p + q)

    def kl(a):
        nz = a > 0
        return float(np.sum(a[nz] * np.log2(a[nz] / m[nz])))

    return float(np.clip(0.5 * kl(p) + 0.5 * kl(q), 0.0, 1.0))


def range_rule(v: np.ndarray) -> tuple[float, float]:
    """The D5 range: [0.5th, 99.5th] percentile widened by 10 % (5 % per side)."""
    lo, hi = np.percentile(np.asarray(v, np.float64), [0.5, 99.5])
    pad = 0.05 * (hi - lo) if hi > lo else 0.5
    return float(lo - pad), float(hi + pad)
