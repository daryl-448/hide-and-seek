"""Seeded uniform neighbour sampling over CSR rows (SPEC §5 D2, E20).

Training never needs ``NeighborLoader`` or a compiled PyG extension (E12).
CSR rows built by ``hns.data.graph`` hold distinct neighbours per edge type,
so sampling entries uniformly without replacement is uniform over distinct
neighbours, which keeps hubs (max wallet degree 22,996) bounded.
"""

from __future__ import annotations

import numba as nb
import numpy as np

_M64 = np.uint64(0xFFFFFFFFFFFFFFFF)


@nb.njit(cache=True)
def _splitmix64(x):
    x = (x + np.uint64(0x9E3779B97F4A7C15)) & _M64
    z = x
    z = ((z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)) & _M64
    z = ((z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)) & _M64
    return x, z ^ (z >> np.uint64(31))


@nb.njit(cache=True)
def _sample(indptr, indices, rows, ends, k, seed, keys):
    n = rows.shape[0]
    counts = np.empty(n, dtype=np.int64)
    for i in range(n):
        lo = indptr[rows[i]]
        hi = indptr[rows[i] + 1] if ends[i] < 0 else ends[i]
        counts[i] = min(k, hi - lo)
    ptr = np.zeros(n + 1, dtype=np.int64)
    for i in range(n):
        ptr[i + 1] = ptr[i] + counts[i]
    out = np.empty(ptr[n], dtype=indices.dtype)
    for i in range(n):
        lo = indptr[rows[i]]
        hi = indptr[rows[i] + 1] if ends[i] < 0 else ends[i]
        deg = hi - lo
        m = counts[i]
        state = (np.uint64(seed) * np.uint64(0x2545F4914F6CDD1D)) ^ np.uint64(keys[i])
        state, _ = _splitmix64(state)
        if m == deg:
            for j in range(m):
                out[ptr[i] + j] = indices[lo + j]
            continue
        # Floyd's algorithm: m distinct offsets from range(deg), O(m) memory.
        chosen = np.empty(m, dtype=np.int64)
        c = 0
        for j in range(deg - m, deg):
            state, r = _splitmix64(state)
            t = np.int64(r % np.uint64(j + 1))
            dup = False
            for q in range(c):
                if chosen[q] == t:
                    dup = True
                    break
            chosen[c] = j if dup else t
            c += 1
        for j in range(m):
            out[ptr[i] + j] = indices[lo + chosen[j]]
    return ptr, out


def sample_neighbors(indptr: np.ndarray, indices: np.ndarray, rows: np.ndarray, k: int, seed: int,
                     keys: np.ndarray | None = None, ends: np.ndarray | None = None):
    """Sample up to ``k`` distinct neighbours of each row, uniformly, without replacement.

    ``keys`` (default: the row ids) seed each draw together with ``seed``, so a
    target node's sample does not depend on the batch it is in. ``ends``
    optionally cuts each row at an absolute CSR offset (for causal prefixes;
    -1 means the full row). Returns ``(ptr, neighbours)`` in CSR form.
    """
    rows = np.asarray(rows, dtype=np.int64)
    keys = rows if keys is None else np.asarray(keys, dtype=np.int64)
    ends = np.full(len(rows), -1, dtype=np.int64) if ends is None else np.asarray(ends, dtype=np.int64)
    return _sample(indptr, indices, rows, ends, int(k), int(seed) & 0x7FFFFFFFFFFFFFFF, keys)
