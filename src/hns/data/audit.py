"""D1: audit the loaded data against SPEC §2 and fail loudly on any mismatch.

Writes ``data/audit.json``: one record per check, with expected and actual
values. Exits non-zero if any check fails, so nothing downstream runs.

Match rates are exact equality, except ``fees``, which is compared within a
float32 tolerance (the CSV stores rounded floats, so ``in - out`` differs from
``fees`` in the last digits). Rates over tx features use the 202,804 txs whose
augmented features are present; rates over wallet features use all wallets.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from hns.data.causal import causal_table
from hns.data.graph import typed_links
from hns.data.load import AUG_FEATURES, PARQUET_DIR, TX_FEATURES
from hns.data.split import official_t, official_w

AUDIT_PATH = Path("data/audit.json")
MATCH_SLACK = 0.5  # percentage points (§5 D1)
FEES_RTOL, FEES_ATOL = 1e-6, 1e-8

# §2 "Feature-definition audit" match rates, in percent.
MATCH_SPEC = {
    "in_txs_degree": 99.7, "out_txs_degree": 99.7,
    "num_input_addresses": 94.6, "num_output_addresses": 99.8,
    "total_BTC": 100.0, "fees": 100.0,
    "num_txs_as_sender": 99.97, "num_txs_as receiver": 98.96,
    "total_txs": 100.0, "num_timesteps_appeared_in": 100.0,
}


class Audit:
    def __init__(self):
        self.checks: list[dict] = []

    def eq(self, name: str, actual, expected):
        self.checks.append({"check": name, "expected": expected, "actual": actual, "ok": actual == expected})

    def ge(self, name: str, actual: float, minimum: float):
        self.checks.append({"check": name, "expected": f">= {minimum}", "actual": actual, "ok": actual >= minimum})

    @property
    def failures(self) -> list[dict]:
        return [c for c in self.checks if not c["ok"]]


def _pct(mask) -> float:
    return float(np.mean(np.asarray(mask))) * 100.0


def _class_counts(s: pl.Series) -> dict:
    return {str(k): int(v) for k, v in s.value_counts().sort(s.name).rows()}


def audit(parquet_dir: Path = PARQUET_DIR) -> Audit:
    a = Audit()
    raw = json.loads((parquet_dir / "raw_stats.json").read_text())
    txs = pl.read_parquet(parquet_dir / "txs.parquet")
    w = pl.read_parquet(parquet_dir / "wallets.parquet")
    e = pl.read_parquet(parquet_dir / "tx_edges.parquet")
    at = pl.read_parquet(parquet_dir / "addr_tx.parquet")
    ta = pl.read_parquet(parquet_dir / "tx_addr.parquet")
    aa = pl.read_parquet(parquet_dir / "addr_addr.parquet")

    # --- file rows and classes
    a.eq("txs_features rows", txs.height, 203_769)
    a.eq("txs classes {1,2,3}", _class_counts(txs["class"]), {"1": 4_545, "2": 42_019, "3": 157_205})
    a.eq("txs_edgelist rows", e.height, 234_355)
    a.eq("wallets_features rows", raw["wallets_features_rows"], 1_268_260)
    a.eq("wallets_classes rows", raw["wallets_classes_rows"], 822_942)
    a.eq("wallet classes {1,2,3}", _class_counts(w["class"]), {"1": 14_266, "2": 251_088, "3": 557_588})
    a.eq("combined rows", raw["combined_rows"], 1_268_260)
    a.eq("combined sorted by address", raw["combined_sorted_by_address"], True)
    a.eq("combined de-dup order == sorted (wid) order", raw["combined_dedup_order_equals_sorted"], True)
    a.eq("combined classes == wallets_classes", raw["combined_classes_match"], True)
    a.eq("AddrAddr rows", aa.height, 2_868_964)
    a.eq("AddrTx rows", at.height, 477_117)
    a.eq("TxAddr rows", ta.height, 837_124)
    a.eq("AddrTx all distinct", at.unique().height, at.height)
    a.eq("TxAddr all distinct", ta.unique().height, ta.height)

    # --- tx graph structure
    ts = txs.select("tid", "ts")
    ets = e.join(ts.rename({"tid": "src", "ts": "ts1"}), on="src").join(ts.rename({"tid": "dst", "ts": "ts2"}), on="dst")
    a.eq("tx edges crossing a time step", int((ets["ts1"] != ets["ts2"]).sum()), 0)
    a.eq("tx edge self-loops", int((e["src"] == e["dst"]).sum()), 0)
    n = txs.height
    g = coo_matrix((np.ones(e.height), (e["src"].to_numpy(), e["dst"].to_numpy())), shape=(n, n))
    n_comp, comp = connected_components(g, directed=True, connection="weak")
    comp_per_ts = pl.DataFrame({"ts": txs["ts"].to_numpy(), "c": comp}).group_by("ts").agg(pl.col("c").n_unique())
    a.eq("time steps are exactly 1..49", sorted(txs["ts"].unique().to_list()), list(range(1, 50)))
    a.eq("weak components == time steps (one per step)", n_comp, 49)
    a.eq("every time step is one weak component", int(comp_per_ts["c"].max()), 1)
    per_ts = txs.group_by("ts").len()["len"]
    a.eq("txs per step [min, median, max]", [int(per_ts.min()), int(per_ts.median()), int(per_ts.max())],
         [1_089, 4_291, 7_880])

    # --- AddrAddr
    pairs = aa.unique()
    a.eq("AddrAddr distinct pairs", pairs.height, 2_784_344)
    a.eq("AddrAddr self-loop rows", int((aa["src"] == aa["dst"]).sum()), 45_981)
    deg = pl.concat([pairs.select(pl.col("src").alias("w")), pairs.select(pl.col("dst").alias("w"))]).group_by("w").len()["len"]
    a.eq("AddrAddr total degree over distinct pairs [max, median]", [int(deg.max()), float(deg.median())], [22_996, 2.0])
    j = at.join(ta, on="tid").select(pl.col("wid").alias("src"), pl.col("wid_right").alias("dst"))
    a.eq("AddrAddr == AddrTx join TxAddr (multiset)", j.sort(["src", "dst"]).equals(aa.sort(["src", "dst"])), True)
    linked_w = pl.concat([at["wid"], ta["wid"]]).unique()
    a.eq("wallets without address-tx links", w.height - linked_w.len(), 0)

    # --- null augmented txs (E7)
    allnull = txs.filter(pl.all_horizontal([pl.col(c).is_null() for c in AUG_FEATURES]))
    anynull = txs.filter(pl.any_horizontal([pl.col(c).is_null() for c in TX_FEATURES]))
    linked_t = pl.concat([at["tid"], ta["tid"]]).unique()
    a.eq("txs with all 17 augmented null", allnull.height, 965)
    a.eq("txs with any null == all-augmented-null", anynull.height, allnull.height)
    a.eq("null txs by class", _class_counts(allnull["class"]), {"2": 519, "3": 446})
    a.eq("null txs with address links", int(allnull["tid"].is_in(linked_t.implode()).sum()), 0)

    # --- wallet file (E4, E5)
    a.eq("wallet distinct (address, ts) pairs", raw["wallet_distinct_addr_ts_pairs"], 920_691)
    a.eq("wallet duplicated (address, ts) pairs", raw["wallet_duplicated_pairs"], 246_545)
    a.eq("wallet max (address, ts) multiplicity", raw["wallet_max_pair_multiplicity"], 568)
    a.eq("wallet features constant per address", raw["wallet_features_constant_per_address"], True)
    a.eq("wallet rows after de-dup", raw["wallet_rows_after_dedup"], 822_942)

    # --- labels (E6)
    links = pl.concat([at.select("wid", "tid"), ta.select("wid", "tid")]).unique()
    lc = (links.join(txs.select("tid", "class"), on="tid").group_by("wid")
          .agg((pl.col("class") == 1).any().alias("ill"), (pl.col("class") == 2).any().alias("lic")))
    x = w.select("wid", "class").join(lc, on="wid", how="left").fill_null(False)
    a.eq("illicit wallets linked to an illicit tx", int(x.filter(pl.col("class") == 1)["ill"].sum()), 14_266)
    a.eq("non-illicit wallets linked to an illicit tx", int(x.filter(pl.col("class") != 1)["ill"].sum()), 0)
    a.eq("licit wallets linked to a licit tx (%)", _pct(x.filter(pl.col("class") == 2)["lic"]), 100.0)
    a.eq("unknown wallets linked to a licit tx (%, 1 dp)", round(_pct(x.filter(pl.col("class") == 3)["lic"]), 1), 14.3)

    # --- feature-definition audit (§2 table)
    d = txs.filter(pl.col("size").is_not_null())
    rec = {
        "in_txs_degree": e.group_by("dst").len().rename({"dst": "tid"}),
        "out_txs_degree": e.group_by("src").len().rename({"src": "tid"}),
        "num_input_addresses": at.group_by("tid").len(),
        "num_output_addresses": ta.group_by("tid").len(),
    }
    rates = {}
    for f, cnt in rec.items():
        r = d.select("tid", f).join(cnt, on="tid", how="left").fill_null(0)
        rates[f] = _pct(r[f] == r["len"])
        if f.startswith("num_"):
            a.eq(f"{f} never below link count", int((r[f] < r["len"]).sum()), 0)
    rates["total_BTC"] = _pct(d["total_BTC"] == d["out_BTC_total"])
    fee_err = (d["fees"].cast(pl.Float64) - (d["in_BTC_total"].cast(pl.Float64) - d["out_BTC_total"].cast(pl.Float64))).abs()
    rates["fees"] = _pct(fee_err <= FEES_RTOL * d["in_BTC_total"].cast(pl.Float64).abs() + FEES_ATOL)

    ws = w.select("wid", "num_txs_as_sender", "num_txs_as receiver", "total_txs", "num_timesteps_appeared_in")
    lw = links.join(ts, on="tid").group_by("wid").agg(pl.len().alias("tot"), pl.col("ts").n_unique().alias("nts"))
    ws = (ws.join(at.group_by("wid").len().rename({"len": "ns"}), on="wid", how="left")
          .join(ta.group_by("wid").len().rename({"len": "nr"}), on="wid", how="left")
          .join(lw, on="wid", how="left").fill_null(0))
    for f, c in [("num_txs_as_sender", "ns"), ("num_txs_as receiver", "nr"), ("total_txs", "tot"),
                 ("num_timesteps_appeared_in", "nts")]:
        rates[f] = _pct(ws[f] == ws[c])
    # D3 causal counts at each wallet's last step must equal the lifetime recomputation.
    final = causal_table(typed_links(ts, at, ta)).group_by("wid").agg(pl.all().sort_by("ts").last())
    cf = ws.join(final, on="wid", how="left")
    a.eq("D3 final state == lifetime (c_in, c_out, c_tot, c_steps) (%)",
         [_pct(cf["c_in"] == cf["ns"]), _pct(cf["c_out"] == cf["nr"]), _pct(cf["c_tot"] == cf["tot"]),
          _pct(cf["c_steps"] == cf["nts"])], [100.0, 100.0, 100.0, 100.0])
    for f, spec in MATCH_SPEC.items():
        a.ge(f"match rate % {f}", round(rates[f], 3), round(spec - MATCH_SLACK, 3))

    # --- official splits (E3)
    t = official_t(txs)
    for part, (ill, lic) in {"train": (3_462, 26_237), "test": (1_083, 15_263)}.items():
        p = t.filter(pl.col("official") == part)
        a.eq(f"T {part} (illicit, licit)", [int((p["y"] == 1).sum()), int((p["y"] == 0).sum())], [ill, lic])
    ow = official_w(w).join(w.select("wid", "address"), on="wid")
    first_ts = links.join(ts, on="tid").group_by("wid").agg(pl.col("ts").min().alias("fts"))
    for part, (rows, ill, lic, p2pkh, p2sh) in {"train": (185_747, 11_377, 174_370, 185_747, 0),
                                                "test": (79_607, 2_889, 76_718, 43_011, 36_596)}.items():
        p = ow.filter(pl.col("official") == part)
        a.eq(f"W {part} (rows, illicit, licit)", [p.height, int((p["y"] == 1).sum()), int((p["y"] == 0).sum())],
             [rows, ill, lic])
        a.eq(f"W {part} address types (1..., 3...)",
             [int(p["address"].str.starts_with("1").sum()), int(p["address"].str.starts_with("3").sum())], [p2pkh, p2sh])
        f = p.join(first_ts, on="wid")["fts"]
        a.eq(f"W {part} first-appearance ts p10/p50/p90",
             [int(f.quantile(q, "nearest")) for q in (0.1, 0.5, 0.9)], [2, 22, 45])
    return a


def run(parquet_dir: Path = PARQUET_DIR, out: Path = AUDIT_PATH) -> None:
    a = audit(parquet_dir)
    out.write_text(json.dumps({"n_checks": len(a.checks), "n_failed": len(a.failures), "checks": a.checks}, indent=1))
    if a.failures:
        for c in a.failures:
            print(f"AUDIT FAIL: {c['check']}: expected {c['expected']}, got {c['actual']}", file=sys.stderr)
        raise SystemExit(f"D1 audit failed ({len(a.failures)} of {len(a.checks)} checks); see {out}")
    print(f"D1: all {len(a.checks)} audit checks passed; wrote {out}")


if __name__ == "__main__":
    run()
