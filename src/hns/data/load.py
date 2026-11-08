"""D0: load the nine Elliptic++ CSVs into typed parquet tables (SPEC §5 D0).

Outputs, in ``data/parquet/``:

- ``txs.parquet``: ``tid`` (dense, ``txId`` sorted order), ``txId``, ``ts``,
  ``class``, then the 182 tx features as float32.
- ``wallets.parquet``: one row per address (E4). ``wid`` (dense, sorted
  address order, so ``wid`` order equals the combined file's order),
  ``address``, ``class``, the 55 wallet features. No ``Time step``.
- ``tx_edges.parquet`` (``src``, ``dst`` tids), ``addr_tx.parquet``
  (``wid``, ``tid``: address is an input), ``tx_addr.parquet`` (``tid``,
  ``wid``: address receives an output), ``addr_addr.parquet`` (``src``,
  ``dst`` wids, one row per raw row, multiplicity kept).
- ``raw_stats.json``: facts about the raw files that de-duplication erases,
  checked later by the D1 audit.
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from hns.data import ids

RAW_DIR = Path("dataset")
PARQUET_DIR = Path("data/parquet")

AUG_FEATURES = [
    "in_txs_degree", "out_txs_degree", "total_BTC", "fees", "size",
    "num_input_addresses", "num_output_addresses",
    "in_BTC_min", "in_BTC_max", "in_BTC_mean", "in_BTC_median", "in_BTC_total",
    "out_BTC_min", "out_BTC_max", "out_BTC_mean", "out_BTC_median", "out_BTC_total",
]
ANON_FEATURES = [f"Local_feature_{i}" for i in range(1, 94)] + [
    f"Aggregate_feature_{i}" for i in range(1, 73)
]
TX_FEATURES = ANON_FEATURES + AUG_FEATURES  # 182, the notebook's feature set


def _header(path: Path) -> list[str]:
    with open(path) as f:
        return f.readline().rstrip("\n").split(",")


def wallet_feature_names(raw_dir: Path = RAW_DIR) -> list[str]:
    return [c for c in _header(raw_dir / "wallets_features.csv") if c not in ("address", "Time step")]


def load_txs(raw_dir: Path) -> pl.DataFrame:
    cols = _header(raw_dir / "txs_features.csv")
    feats = [c for c in cols if c not in ("txId", "Time step")]
    if feats != TX_FEATURES:
        raise ValueError("unexpected txs_features.csv columns")
    schema = {"txId": pl.Int64, "Time step": pl.Int16} | {c: pl.Float32 for c in feats}
    f = pl.read_csv(raw_dir / "txs_features.csv", schema=schema)
    c = pl.read_csv(raw_dir / "txs_classes.csv", schema={"txId": pl.Int64, "class": pl.Int8})
    if f.height != c.height or f["txId"].n_unique() != f.height:
        raise ValueError("txs_features / txs_classes row mismatch or duplicate txId")
    df = f.join(c, on="txId", how="inner", validate="1:1").rename({"Time step": "ts"})
    if df.height != f.height:
        raise ValueError("txs_classes does not cover txs_features")
    df = df.sort("txId").with_row_index("tid").with_columns(pl.col("tid").cast(pl.Int32))
    return df.select(["tid", "txId", "ts", "class", *TX_FEATURES])


def load_wallets(raw_dir: Path) -> tuple[pl.DataFrame, dict]:
    feats = wallet_feature_names(raw_dir)
    schema = {"address": pl.String, "Time step": pl.Int16} | {c: pl.Float32 for c in feats}
    raw = pl.read_csv(raw_dir / "wallets_features.csv", schema=schema)
    pair_counts = raw.group_by(["address", "Time step"]).len()
    stats = {
        "wallets_features_rows": raw.height,
        "wallet_distinct_addr_ts_pairs": pair_counts.height,
        "wallet_duplicated_pairs": int((pair_counts["len"] > 1).sum()),
        "wallet_max_pair_multiplicity": int(pair_counts["len"].max()),
    }
    dedup = raw.drop("Time step").unique()
    stats["wallet_rows_after_dedup"] = dedup.height
    stats["wallet_distinct_addresses"] = raw["address"].n_unique()
    # Every feature constant per address <=> one distinct feature row per address.
    stats["wallet_features_constant_per_address"] = dedup.height == stats["wallet_distinct_addresses"]

    cls = pl.read_csv(raw_dir / "wallets_classes.csv", schema={"address": pl.String, "class": pl.Int8})
    stats["wallets_classes_rows"] = cls.height
    df = dedup.join(cls, on="address", how="inner", validate="1:1")
    if df.height != dedup.height or df.height != cls.height:
        raise ValueError("wallets_classes does not match the de-duplicated wallet features")

    # The W split runs in the combined file's order; check it equals sorted address order.
    comb = pl.read_csv(raw_dir / "wallets_features_classes_combined.csv", columns=["address", "class"],
                       schema_overrides={"address": pl.String, "class": pl.Int8})
    stats["combined_rows"] = comb.height
    stats["combined_sorted_by_address"] = bool(comb["address"].is_sorted())
    first = comb.unique(subset="address", keep="first", maintain_order=True)
    stats["combined_dedup_order_equals_sorted"] = bool(
        first["address"].equals(first["address"].sort())
    )
    stats["combined_classes_match"] = bool(
        first.join(cls, on="address", how="inner").filter(pl.col("class") != pl.col("class_right")).height == 0
        and first.height == cls.height
    )

    wid = ids.address_ids(df["address"])
    df = df.join(wid, on="address").sort("wid")
    return df.select(["wid", "address", "class", *feats]), stats


def load_edges(raw_dir: Path, txs: pl.DataFrame, wallets: pl.DataFrame) -> dict[str, pl.DataFrame]:
    tmap = txs.select("txId", "tid")
    wmap = wallets.select("address", "wid")

    def tx_ids(df: pl.DataFrame, col: str, new: str) -> pl.DataFrame:
        out = df.join(tmap.rename({"txId": col, "tid": new}), on=col, how="left")
        if out[new].null_count():
            raise ValueError(f"{col}: txId not in txs_features")
        return out.drop(col)

    def w_ids(df: pl.DataFrame, col: str, new: str) -> pl.DataFrame:
        out = df.join(wmap.rename({"address": col, "wid": new}), on=col, how="left")
        if out[new].null_count():
            raise ValueError(f"{col}: address not in wallet tables")
        return out.drop(col)

    e = pl.read_csv(raw_dir / "txs_edgelist.csv", schema={"txId1": pl.Int64, "txId2": pl.Int64})
    e = tx_ids(tx_ids(e.with_row_index("_r"), "txId1", "src"), "txId2", "dst").sort("_r").drop("_r")

    a = pl.read_csv(raw_dir / "AddrTx_edgelist.csv", schema={"input_address": pl.String, "txId": pl.Int64})
    a = tx_ids(w_ids(a.with_row_index("_r"), "input_address", "wid"), "txId", "tid").sort("_r").drop("_r")

    t = pl.read_csv(raw_dir / "TxAddr_edgelist.csv", schema={"txId": pl.Int64, "output_address": pl.String})
    t = w_ids(tx_ids(t.with_row_index("_r"), "txId", "tid"), "output_address", "wid").sort("_r").drop("_r")

    aa = pl.read_csv(raw_dir / "AddrAddr_edgelist.csv",
                     schema={"input_address": pl.String, "output_address": pl.String})
    aa = w_ids(w_ids(aa.with_row_index("_r"), "input_address", "src"), "output_address", "dst")
    aa = aa.sort("_r").drop("_r")
    return {
        "tx_edges": e.select("src", "dst"),
        "addr_tx": a.select("wid", "tid"),
        "tx_addr": t.select("tid", "wid"),
        "addr_addr": aa.select("src", "dst"),
    }


def run(raw_dir: Path = RAW_DIR, out_dir: Path = PARQUET_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    txs = load_txs(raw_dir)
    wallets, stats = load_wallets(raw_dir)
    edges = load_edges(raw_dir, txs, wallets)
    txs.write_parquet(out_dir / "txs.parquet")
    wallets.write_parquet(out_dir / "wallets.parquet")
    for name, df in edges.items():
        df.write_parquet(out_dir / f"{name}.parquet")
    (out_dir / "raw_stats.json").write_text(json.dumps(stats, indent=2))
    print(f"D0: wrote {out_dir} (txs {txs.height}, wallets {wallets.height})")


if __name__ == "__main__":
    run()
