"""Tiny synthetic Elliptic++-shaped fixtures. Never real data (SPEC §4)."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from hns.data.load import AUG_FEATURES, TX_FEATURES

WALLET_FEATS = ["num_txs_as_sender", "num_txs_as receiver", "total_txs", "num_timesteps_appeared_in"]

# txId -> (time step, class); tx 50 has null augmented features (E7).
TXS = {10: (1, 1), 20: (1, 2), 30: (2, 3), 40: (2, 2), 50: (2, 2)}
TX_EDGES = [(10, 20), (30, 40), (40, 50)]
ADDR_TX = [("1A", 10), ("1B", 30), ("1A", 30)]  # inputs
TX_ADDR = [(10, "1B"), (10, "1A"), (20, "3C"), (30, "1D"), (40, "1D")]  # outputs
WALLET_CLASS = {"1A": 1, "1B": 1, "3C": 2, "1D": 3}
# (address, ts) rows of wallets_features.csv, with a duplicated pair; features constant per address.
WALLET_ROWS = [("1A", 1), ("1A", 2), ("1A", 2), ("1B", 1), ("1B", 2), ("3C", 1), ("1D", 2)]


def _write(path: Path, header: list[str], rows) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


@pytest.fixture
def raw_dir(tmp_path: Path) -> Path:
    raw = tmp_path / "dataset"
    raw.mkdir()
    rng = np.random.default_rng(0)
    rows = []
    for txid, (ts, _) in TXS.items():
        vals = list(rng.random(len(TX_FEATURES)).round(4))
        if txid == 50:
            for f in AUG_FEATURES:
                vals[TX_FEATURES.index(f)] = ""
        rows.append([txid, ts, *vals])
    _write(raw / "txs_features.csv", ["txId", "Time step", *TX_FEATURES], rows)
    _write(raw / "txs_classes.csv", ["txId", "class"], [[t, c] for t, (_, c) in TXS.items()])
    _write(raw / "txs_edgelist.csv", ["txId1", "txId2"], TX_EDGES)
    _write(raw / "AddrTx_edgelist.csv", ["input_address", "txId"], ADDR_TX)
    _write(raw / "TxAddr_edgelist.csv", ["txId", "output_address"], TX_ADDR)
    aa = [(a, b) for a, t in ADDR_TX for t2, b in TX_ADDR if t == t2]
    _write(raw / "AddrAddr_edgelist.csv", ["input_address", "output_address"], aa)
    # consistent with the links above: (as sender, as receiver, total txs, distinct time steps)
    feats = {"1A": [2.0, 1.0, 2.0, 2.0], "1B": [1.0, 1.0, 2.0, 2.0], "1D": [0.0, 2.0, 2.0, 1.0], "3C": [0.0, 1.0, 1.0, 1.0]}
    _write(raw / "wallets_features.csv", ["address", "Time step", *WALLET_FEATS],
           [[a, ts, *feats[a]] for a, ts in WALLET_ROWS])
    # classes in a different order than the sorted features file
    _write(raw / "wallets_classes.csv", ["address", "class"], [[a, c] for a, c in WALLET_CLASS.items()])
    _write(raw / "wallets_features_classes_combined.csv", ["address", "Time step", "class", *WALLET_FEATS],
           [[a, ts, WALLET_CLASS[a], *feats[a]] for a, ts in sorted(WALLET_ROWS)])
    return raw


@pytest.fixture
def built(raw_dir: Path, tmp_path: Path) -> dict[str, Path]:
    from hns.data import causal, graph, load

    pq, gr = tmp_path / "parquet", tmp_path / "graph"
    load.run(raw_dir, pq)
    graph.run(pq, gr)
    causal.run(pq, gr)
    return {"parquet": pq, "graph": gr}
