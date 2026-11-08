"""D4: official splits, val splits and nested label fractions (SPEC §5 D4, §2 E3).

Writes ``data/splits/{T,W}_seed{s}.parquet``. Columns:

- T: ``tid, y, split, in_f0.05, in_f0.25``. ``split`` is ``fit`` (ts 1-29),
  ``val`` (ts 30-34) or ``test`` (ts >= 35). Labelled, non-null txs only.
- W: ``wid, y, split, in_f0.05, in_f0.25``. ``split`` is ``fit``, ``val``,
  ``test`` or ``unl`` (unknown wallets: in the fitting split, never targets).

``y`` is 1 for illicit, 0 for licit, null for unknown. ``in_f*`` mark the
nested, stratified label fractions of the labelled fitting set (``f = 1.0`` is
all of ``split == fit``). The T split is the same for every seed; only its
fractions change. For W the seed also draws the val split.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.model_selection import train_test_split

from hns.data.load import PARQUET_DIR

SPLIT_DIR = Path("data/splits")
SEEDS = (0, 1, 2)
FRACTIONS = (0.05, 0.25)
T_TEST_FROM, T_VAL_FROM = 35, 30
W_TEST_SIZE, W_VAL_SIZE = 0.30, 0.10


def _y(cls: pl.Expr) -> pl.Expr:
    return pl.when(cls == 1).then(1).when(cls == 2).then(0).otherwise(None).cast(pl.Int8).alias("y")


def official_t(txs: pl.DataFrame) -> pl.DataFrame:
    """Notebook rule: drop rows with nulls, drop unknown, train ts < 35, test ts >= 35."""
    df = txs.filter((pl.col("class") != 3) & pl.col("size").is_not_null())
    return df.select("tid", "ts", _y(pl.col("class")),
                     pl.when(pl.col("ts") >= T_TEST_FROM).then(pl.lit("test")).otherwise(pl.lit("train")).alias("official"))


def official_w(wallets: pl.DataFrame) -> pl.DataFrame:
    """Notebook rule: de-duplicated wallets in file (= wid) order, drop unknown,
    ``train_test_split(test_size=0.30, shuffle=False)``: the last ceil(0.3 n) rows are test."""
    df = wallets.filter(pl.col("class") != 3).sort("wid")
    n_test = math.ceil(W_TEST_SIZE * df.height)
    official = np.array(["train"] * (df.height - n_test) + ["test"] * n_test)
    return df.select("wid", _y(pl.col("class"))).with_columns(official=pl.Series(official))


def nested_fractions(y: np.ndarray, strata: np.ndarray, seed: int, fractions=FRACTIONS) -> dict[float, np.ndarray]:
    """Stratified (by ``strata`` and ``y``), seeded, nested subsets: each smaller
    fraction is a prefix of the same per-stratum permutation, so it sits inside
    every larger one. ``round(f * n)`` per stratum."""
    rng = np.random.default_rng(seed)
    out = {f: np.zeros(len(y), dtype=bool) for f in fractions}
    key = np.stack([strata, y], axis=1)
    for k in np.unique(key, axis=0):
        idx = np.flatnonzero((key == k).all(axis=1))
        perm = rng.permutation(idx)
        for f in fractions:
            out[f][perm[: int(round(f * len(idx)))]] = True
    return out


def t_split(txs: pl.DataFrame, seed: int) -> pl.DataFrame:
    df = official_t(txs).with_columns(
        split=pl.when(pl.col("official") == "test").then(pl.lit("test"))
        .when(pl.col("ts") >= T_VAL_FROM).then(pl.lit("val")).otherwise(pl.lit("fit"))
    )
    fit = df.filter(pl.col("split") == "fit")
    fr = nested_fractions(fit["y"].to_numpy(), fit["ts"].to_numpy(), seed)
    marks = fit.select("tid", *[pl.Series(f"in_f{f}", m) for f, m in fr.items()])
    return (df.join(marks, on="tid", how="left")
            .with_columns([pl.col(f"in_f{f}").fill_null(False) for f in FRACTIONS])
            .select("tid", "y", "split", *[f"in_f{f}" for f in FRACTIONS]).sort("tid"))


def w_split(wallets: pl.DataFrame, seed: int) -> pl.DataFrame:
    off = official_w(wallets)
    train = off.filter(pl.col("official") == "train")
    _, val_wid = train_test_split(train["wid"].to_numpy(), test_size=W_VAL_SIZE,
                                  stratify=train["y"].to_numpy(), random_state=seed)
    df = off.with_columns(
        split=pl.when(pl.col("official") == "test").then(pl.lit("test"))
        .when(pl.col("wid").is_in(pl.Series(val_wid).implode())).then(pl.lit("val")).otherwise(pl.lit("fit"))
    ).drop("official")
    fit = df.filter(pl.col("split") == "fit")
    fr = nested_fractions(fit["y"].to_numpy(), np.zeros(fit.height, dtype=np.int64), seed)
    marks = fit.select("wid", *[pl.Series(f"in_f{f}", m) for f, m in fr.items()])
    unl = wallets.filter(pl.col("class") == 3).select(
        "wid", pl.lit(None, pl.Int8).alias("y"), pl.lit("unl").alias("split"))
    return (pl.concat([df, unl]).join(marks, on="wid", how="left")
            .with_columns([pl.col(f"in_f{f}").fill_null(False) for f in FRACTIONS])
            .select("wid", "y", "split", *[f"in_f{f}" for f in FRACTIONS]).sort("wid"))


def run(parquet_dir: Path = PARQUET_DIR, out_dir: Path = SPLIT_DIR, seeds=SEEDS) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    txs = pl.read_parquet(parquet_dir / "txs.parquet", columns=["tid", "ts", "class", "size"])
    wallets = pl.read_parquet(parquet_dir / "wallets.parquet", columns=["wid", "class"])
    for s in seeds:
        t_split(txs, s).write_parquet(out_dir / f"T_seed{s}.parquet")
        w_split(wallets, s).write_parquet(out_dir / f"W_seed{s}.parquet")
    print(f"D4: wrote {out_dir} for seeds {list(seeds)}")


if __name__ == "__main__":
    run()
