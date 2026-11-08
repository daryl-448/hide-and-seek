import math

import numpy as np
import polars as pl

from hns.data.split import nested_fractions, official_t, official_w, t_split, w_split


def _txs():
    ts = np.repeat(np.arange(1, 50), 20)
    rng = np.random.default_rng(1)
    cls = rng.choice([1, 2, 3], size=len(ts), p=[0.1, 0.4, 0.5])
    size = np.where(np.arange(len(ts)) % 97 == 0, np.nan, 1.0)
    return pl.DataFrame({"tid": np.arange(len(ts), dtype=np.int32), "ts": ts.astype(np.int16),
                         "class": cls.astype(np.int8), "size": size}).with_columns(pl.col("size").fill_nan(None))


def test_official_t_rule():
    txs = _txs()
    t = official_t(txs)
    assert t["y"].null_count() == 0
    assert not t["tid"].is_in(txs.filter(pl.col("size").is_null())["tid"].implode()).any()
    assert (t.filter(pl.col("official") == "test")["ts"] >= 35).all()


def test_t_split_temporal_val_and_nested_fractions():
    d = t_split(_txs(), seed=0)
    j = d.join(_txs().select("tid", "ts"), on="tid")
    assert set(j.filter(pl.col("split") == "fit")["ts"].unique()) <= set(range(1, 30))
    assert set(j.filter(pl.col("split") == "val")["ts"].unique()) <= set(range(30, 35))
    assert not (d["in_f0.25"] & (d["split"] != "fit")).any()
    assert not (d["in_f0.05"] & ~d["in_f0.25"]).any()  # nested


def test_official_w_is_file_order_tail():
    n = 101
    w = pl.DataFrame({"wid": np.arange(n, dtype=np.int32),
                      "class": np.array([3 if i % 5 == 0 else (1 if i % 7 == 0 else 2) for i in range(n)], dtype=np.int8)})
    o = official_w(w)
    lab = w.filter(pl.col("class") != 3)["wid"].to_list()
    n_test = math.ceil(0.3 * len(lab))
    assert o.filter(pl.col("official") == "test")["wid"].to_list() == lab[-n_test:]
    s0, s1 = w_split(w, 0), w_split(w, 1)
    assert (s0["split"] == "unl").sum() == (w["class"] == 3).sum()
    assert s0.filter(pl.col("split") == "test")["wid"].to_list() == lab[-n_test:]
    assert s0.filter(pl.col("split") == "val")["wid"].to_list() != s1.filter(pl.col("split") == "val")["wid"].to_list()


def test_nested_fractions_stratified():
    y = np.array([1] * 40 + [0] * 360)
    strata = np.repeat([1, 2], 200)
    fr = nested_fractions(y, strata, seed=3)
    assert not (fr[0.05] & ~fr[0.25]).any()
    for f in (0.05, 0.25):
        for s in (1, 2):
            for c in (0, 1):
                m = (strata == s) & (y == c)
                assert fr[f][m].sum() == round(f * m.sum())
    again = nested_fractions(y, strata, seed=3)
    assert all((again[f] == fr[f]).all() for f in fr)
