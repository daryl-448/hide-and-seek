import json

import numpy as np
import polars as pl

from hns.data.graph import LINK_IN, LINK_OUT


def test_load_dedups_wallets_and_maps_ids(built):
    pq = built["parquet"]
    w = pl.read_parquet(pq / "wallets.parquet")
    assert w["address"].to_list() == ["1A", "1B", "1D", "3C"]  # sorted, wid order
    assert w["wid"].to_list() == [0, 1, 2, 3]
    assert "Time step" not in w.columns
    assert dict(zip(w["address"], w["class"])) == {"1A": 1, "1B": 1, "1D": 3, "3C": 2}
    stats = json.loads((pq / "raw_stats.json").read_text())
    assert stats["wallets_features_rows"] == 7
    assert stats["wallet_distinct_addr_ts_pairs"] == 6
    assert stats["wallet_duplicated_pairs"] == 1 and stats["wallet_max_pair_multiplicity"] == 2
    assert stats["wallet_features_constant_per_address"] and stats["combined_dedup_order_equals_sorted"]

    tx = pl.read_parquet(pq / "txs.parquet")
    assert tx["txId"].to_list() == [10, 20, 30, 40, 50] and tx["tid"].to_list() == [0, 1, 2, 3, 4]
    assert tx.filter(pl.col("txId") == 50)["size"].is_null().all()
    e = pl.read_parquet(pq / "tx_edges.parquet")
    assert e.rows() == [(0, 1), (2, 3), (3, 4)]
    a = pl.read_parquet(pq / "addr_tx.parquet")
    assert a.rows() == [(0, 0), (1, 2), (0, 2)]  # (wid, tid)


def test_tx_flow_drops_null_tx_edges(built):
    g = np.load(built["graph"] / "tx_flow.npz")
    assert int(g["n_dropped_null_edges"]) == 1
    out = {i: g["out_indices"][g["out_indptr"][i]:g["out_indptr"][i + 1]].tolist() for i in range(5)}
    assert out == {0: [1], 1: [], 2: [3], 3: [], 4: []}
    inn = {i: g["in_indices"][g["in_indptr"][i]:g["in_indptr"][i + 1]].tolist() for i in range(5)}
    assert inn == {0: [], 1: [0], 2: [], 3: [2], 4: []}


def test_addr_tx_csr_typed_and_ts_sorted(built):
    g = np.load(built["graph"] / "addr_tx.npz")
    lo, hi = g["w2t_indptr"][0], g["w2t_indptr"][1]  # wallet 1A
    got = list(zip(g["w2t_tid"][lo:hi].tolist(), g["w2t_type"][lo:hi].tolist(), g["w2t_ts"][lo:hi].tolist()))
    # 1A: input of tx 10 (tid 0, ts 1), output of tx 10, input of tx 30 (tid 2, ts 2); sorted by (ts, tid)
    assert sorted(got[:2]) == [(0, LINK_IN, 1), (0, LINK_OUT, 1)] and got[2] == (2, LINK_IN, 2)
    assert np.all(np.diff(g["w2t_ts"][lo:hi]) >= 0)


def test_addr_addr_collapse_and_derived(built):
    g = np.load(built["graph"] / "addr_addr.npz")
    d = pl.read_parquet(built["graph"] / "wallet_derived.parquet")
    # rows: 1A->1B, 1A->1A (self), 1B->1D, 1A->1D ; wids 1A=0, 1B=1, 1D=2, 3C=3
    out0 = g["out_indices"][g["out_indptr"][0]:g["out_indptr"][1]].tolist()
    assert out0 == [1, 2]  # self-loop removed
    assert np.allclose(g["out_logmult"][g["out_indptr"][0]:g["out_indptr"][1]], np.log1p(1))
    assert g["out_first_ts"][g["out_indptr"][0]:g["out_indptr"][1]].tolist() == [1, 2]
    row = dict(zip(d["wid"], d["self_loop_frac"]))
    assert np.isclose(row[0], 1 / 3) and row[1] == 0 and row[3] == 0  # 1A: 1 self row of 3 as input
    assert d["aa_in_degree"].to_list() == [0, 1, 2, 0] and d["aa_out_degree"].to_list() == [2, 1, 0, 0]


def test_causal_wallet_state(built):
    from hns.data.causal import CausalWalletState

    s = CausalWalletState.load(built["graph"] / "causal_wallet.parquet")
    # 1A (wid 0): ts1 in+out of tx10 -> c_in 1, c_out 1, c_tot 1 (same tx), c_steps 1; ts2 adds input of tx30.
    q = s.query(np.array([0, 0, 0, 2, 2, 3]), np.array([0, 1, 49, 1, 2, 1]))
    assert q.tolist() == [[0, 0, 0, 0], [1, 1, 1, 1], [2, 1, 2, 2], [0, 0, 0, 0], [0, 2, 2, 1], [0, 1, 1, 1]]


def test_audit_runs_on_fixture(built):
    """On the fixture the §2 counts fail, but the structural and consistency checks hold."""
    from hns.data.audit import audit

    a = audit(built["parquet"])
    names = {c["check"]: c for c in a.checks}
    assert not names["txs_features rows"]["ok"]
    assert names["tx edge self-loops"]["ok"] and names["AddrAddr == AddrTx join TxAddr (multiset)"]["ok"]
    assert names["wallet features constant per address"]["ok"]
    assert names["non-illicit wallets linked to an illicit tx"]["ok"]
    assert names["D3 final state == lifetime (c_in, c_out, c_tot, c_steps) (%)"]["ok"]
    assert names["match rate % total_txs"]["ok"] and names["match rate % num_timesteps_appeared_in"]["ok"]
