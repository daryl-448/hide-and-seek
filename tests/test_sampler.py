import numpy as np

from hns.models.sampler import sample_neighbors


def _csr(rows):
    indptr = np.cumsum([0] + [len(r) for r in rows]).astype(np.int64)
    return indptr, np.concatenate([np.asarray(r, dtype=np.int32) for r in rows])


def test_distinct_subset_and_caps():
    indptr, idx = _csr([list(range(100, 130)), [7, 8], []])
    ptr, out = sample_neighbors(indptr, idx, np.array([0, 1, 2]), k=10, seed=0)
    assert ptr.tolist() == [0, 10, 12, 12]
    s0 = out[0:10]
    assert len(set(s0.tolist())) == 10 and set(s0.tolist()) <= set(range(100, 130))
    assert sorted(out[10:12].tolist()) == [7, 8]


def test_seeded_per_key_not_per_batch():
    indptr, idx = _csr([list(range(50)), list(range(50, 100))])
    _, a = sample_neighbors(indptr, idx, np.array([0, 1]), k=5, seed=42)
    _, b = sample_neighbors(indptr, idx, np.array([1]), k=5, seed=42)
    assert a[5:].tolist() == b.tolist()
    _, c = sample_neighbors(indptr, idx, np.array([0, 1]), k=5, seed=43)
    assert a.tolist() != c.tolist()


def test_uniform_over_hub():
    indptr, idx = _csr([list(range(20))])
    counts = np.zeros(20)
    for key in range(4000):
        _, o = sample_neighbors(indptr, idx, np.array([0]), k=5, seed=1, keys=np.array([key]))
        counts[o] += 1
    expected = 4000 * 5 / 20
    assert np.abs(counts - expected).max() < 0.15 * expected


def test_causal_prefix_end():
    indptr, idx = _csr([[1, 2, 3, 4, 5, 6]])
    _, o = sample_neighbors(indptr, idx, np.array([0]), k=10, seed=0, ends=np.array([3]))
    assert sorted(o.tolist()) == [1, 2, 3]
