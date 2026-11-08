"""D5: frozen, label-blind normalization per task (SPEC §5 D5, E16).

Each feature is first transformed per the manifest (anonymized tx features:
identity), then standardized with a mean and sd fitted once on that task's
real fitting split. The same moments serve both domains and every arm.

- T (``data/norm/T.json``): txs in ts 1-29 (labelled and unknown, nulls
  dropped), plus D3 causal wallet rows with ts <= 29.
- W (``data/norm/W_seed{s}.json``): official-train wallets minus that seed's
  val wallets, plus all unknown wallets; and, for the W view's tx nodes, its
  own moments for the 17 shared tx features (no adapter in W, §11), fitted on
  the non-null txs linked to those wallets. T's moments are never reused. The
  W val split is drawn per seed (§6), so W has one moment file per seed.

Per shared feature the file also stores the JS histogram range (§7): the
fitting split's [0.5th, 99.5th] percentile on the transformed (pre-
standardization) scale, widened by 10 % (5 % per side), with 50 equal-width
bins whose outermost bins collect everything beyond it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl

from hns.data.graph import GRAPH_DIR
from hns.data.load import ANON_FEATURES, PARQUET_DIR
from hns.data.split import SEEDS, SPLIT_DIR, T_VAL_FROM
from hns.feats import manifest

NORM_DIR = Path("data/norm")
JS_BINS = 50
T_FIT_MAX_TS = T_VAL_FROM - 1  # 29


def feature_stats(x: np.ndarray, transform: str, name: str, js: bool) -> dict:
    v = manifest.apply(transform, x, name)
    mean, sd = float(v.mean()), float(v.std())
    out = {"transform": transform, "mean": mean, "sd": sd if sd > 0 else 1.0}
    if js:
        lo, hi = np.percentile(v, [0.5, 99.5])
        pad = 0.05 * (hi - lo) if hi > lo else 0.5
        out |= {"js_lo": float(lo - pad), "js_hi": float(hi + pad), "js_bins": JS_BINS}
    return out


def group_stats(df: pl.DataFrame, group: str | None, names: list[str]) -> dict:
    spec = manifest.load()[group] if group else {}
    return {n: feature_stats(df[n].to_numpy(), spec[n]["transform"] if group else "identity", n, js=bool(group))
            for n in names}


def fit_t(parquet_dir: Path, graph_dir: Path) -> dict:
    tx_names = manifest.features("tx")
    txs = pl.read_parquet(parquet_dir / "txs.parquet", columns=["ts", *ANON_FEATURES, *tx_names])
    txs = txs.filter((pl.col("ts") <= T_FIT_MAX_TS) & pl.col("size").is_not_null())
    cw = pl.read_parquet(graph_dir / "causal_wallet.parquet").filter(pl.col("ts") <= T_FIT_MAX_TS)
    return {
        "task": "T",
        "fitted_on": {"tx_rows": txs.height, "wallet_T_rows": cw.height, "rule": "ts <= 29, label-blind"},
        "tx": group_stats(txs, "tx", tx_names),
        "tx_anon": group_stats(txs, None, ANON_FEATURES),
        "wallet_T": group_stats(cw, "wallet_T", manifest.features("wallet_T")),
    }


def fit_w(parquet_dir: Path, graph_dir: Path, split_dir: Path, seed: int) -> dict:
    names = manifest.features("wallet_W")
    w = pl.read_parquet(parquet_dir / "wallets.parquet").join(
        pl.read_parquet(graph_dir / "wallet_derived.parquet"), on="wid")
    sp = pl.read_parquet(split_dir / f"W_seed{seed}.parquet", columns=["wid", "split"])
    w = w.join(sp, on="wid").filter(pl.col("split").is_in(["fit", "unl"]))
    fit_wid = w["wid"].implode()
    linked = pl.concat([pl.read_parquet(parquet_dir / f, columns=["wid", "tid"]) for f in ("addr_tx.parquet", "tx_addr.parquet")])
    tid = linked.filter(pl.col("wid").is_in(fit_wid))["tid"].unique().implode()
    tx_names = manifest.features("tx")
    txs = pl.read_parquet(parquet_dir / "txs.parquet", columns=["tid", *tx_names])
    txs = txs.filter(pl.col("tid").is_in(tid) & pl.col("size").is_not_null())
    return {
        "task": "W", "seed": seed,
        "fitted_on": {"wallet_W_rows": w.height, "tx_rows": txs.height,
                      "rule": "official train minus val, plus unknown; txs linked to them; label-blind"},
        "wallet_W": group_stats(w, "wallet_W", names),
        "tx": group_stats(txs, "tx", tx_names),
    }


class Normalizer:
    """Apply frozen moments: ``standardize(group, df)`` -> float32 ``(n, d)`` in manifest order."""

    def __init__(self, stats: dict):
        self.stats = stats

    @classmethod
    def load(cls, path: Path) -> "Normalizer":
        return cls(json.loads(Path(path).read_text()))

    def standardize(self, group: str, df: pl.DataFrame) -> np.ndarray:
        cols = []
        for name, s in self.stats[group].items():
            v = manifest.apply(s["transform"], df[name].to_numpy(), name)
            cols.append((v - s["mean"]) / s["sd"])
        return np.stack(cols, axis=1).astype(np.float32)


def run(parquet_dir: Path = PARQUET_DIR, graph_dir: Path = GRAPH_DIR, split_dir: Path = SPLIT_DIR,
        out_dir: Path = NORM_DIR, seeds=SEEDS) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "T.json").write_text(json.dumps(fit_t(parquet_dir, graph_dir), indent=1))
    for s in seeds:
        (out_dir / f"W_seed{s}.json").write_text(json.dumps(fit_w(parquet_dir, graph_dir, split_dir, s), indent=1))
    print(f"D5: wrote {out_dir}")


if __name__ == "__main__":
    run()
