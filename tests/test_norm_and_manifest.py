import numpy as np
import polars as pl
import pytest

from hns.data.load import AUG_FEATURES
from hns.data.norm import Normalizer, feature_stats
from hns.feats import manifest


def test_manifest_covers_shared_channel():
    assert manifest.features("tx") == AUG_FEATURES
    assert manifest.features("wallet_T") == ["c_in", "c_out", "c_tot", "c_steps"]
    w = manifest.features("wallet_W")
    assert len(w) == 55 + 3 and w[-3:] == ["self_loop_frac", "aa_in_degree", "aa_out_degree"]
    m = manifest.load()["wallet_W"]
    assert m["fees_as_share_total"]["transform"] == "log1p"
    assert all(m[f"fees_as_share_{s}"]["transform"] == "identity" for s in ("min", "max", "mean", "median"))


def test_transforms():
    assert np.allclose(manifest.apply("log1p", np.array([0.0, np.e - 1])), [0, 1])
    bt = manifest.apply("block_ts", np.array([0.0, 391200.0, 391200.0 + 1975 * 49]))
    assert bt.tolist() == [-1.0, 0.0, 49.0]
    with pytest.raises(ValueError):
        manifest.apply("log1p", np.array([-1.0]))
    with pytest.raises(ValueError):
        manifest.apply("block_ts", np.array([1000.0]))
    with pytest.raises(ValueError):
        manifest.apply("identity", np.array([np.nan]))


def test_stats_and_standardize():
    x = np.random.default_rng(0).lognormal(size=10_000)
    s = feature_stats(x, "log1p", "x", js=True)
    lo, hi = np.percentile(np.log1p(x), [0.5, 99.5])
    assert np.isclose(s["js_hi"] - s["js_lo"], 1.1 * (hi - lo)) and s["js_bins"] == 50
    n = Normalizer({"tx": {"in_BTC_total": s}})
    z = n.standardize("tx", pl.DataFrame({"in_BTC_total": x}))
    assert z.shape == (10_000, 1) and abs(z.mean()) < 1e-5 and abs(z.std() - 1) < 1e-4
    assert feature_stats(np.zeros(5), "identity", "c", js=False)["sd"] == 1.0
