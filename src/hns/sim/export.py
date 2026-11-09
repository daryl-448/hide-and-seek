"""Per-world export: prune every time step, then recompute (SPEC §9.2, §9.4).

1. **Prune.** Keep only the largest weakly connected component of each time
   step's tx graph (ties: the component holding the lowest tid), with its
   address links. That mirrors Elliptic++'s one-component-per-step graphs.
2. **Recompute over the retained txs only:** the 17 tx features, wallet
   labels (E6 rule), the 55 wallet features, the D3 causal counts, the
   AddrAddr pairs, ``self_loop_frac`` and AddrAddr degrees. So no exported
   value describes a dropped tx.

The tables use the real pipeline's schemas and ids (``tid``, ``wid`` dense,
in ledger order), and the graphs are built by the same ``hns.data`` code as
the real ones (D2, D3).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from hns.data import graph
from hns.data.causal import causal_table
from hns.data.load import AUG_FEATURES
from hns.feats import tx_feats, wallet_feats
from hns.sim.labels import wallet_labels


@dataclass
class WorldExport:
    txs: pl.DataFrame        # tid, ltid, ts, block, class, truth, role, frozen, 17 features
    tx_edges: pl.DataFrame   # src, dst
    inputs: pl.DataFrame     # tid, wid, amount: one row per spent output
    outputs: pl.DataFrame    # tid, wid, amount: one row per output
    addr_tx: pl.DataFrame    # wid, tid: distinct (address is an input)
    tx_addr: pl.DataFrame    # tid, wid: distinct (address receives an output)
    wallets: pl.DataFrame    # wid, addr, class, licit_rule, 55 named + 3 derived features
    causal: pl.DataFrame     # D3 table: wid, ts, c_in, c_out, c_tot, c_steps
    graphs: dict             # tx_flow, addr_tx, addr_addr in the D2 npz layout
    steps: pl.DataFrame      # ts, n_live, n_retained


def largest_components(n: int, edges: np.ndarray, ts: np.ndarray) -> np.ndarray:
    """Boolean mask of txs in the largest weak component of their time step."""
    if len(edges):
        g = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n, n))
    else:
        g = coo_matrix((n, n))
    _, comp = connected_components(g, directed=True, connection="weak")
    keep = np.zeros(n, bool)
    for t in np.unique(ts):
        idx = np.flatnonzero(ts == t)
        c = comp[idx]
        labels, counts = np.unique(c, return_counts=True)
        best = labels[counts == counts.max()]
        winner = c[np.isin(c, best)][0]   # the component holding the lowest tid among the largest
        keep[idx[c == winner]] = True
    return keep


def export_world(env) -> WorldExport:
    led = env.ledger
    n = led.n_tx
    ts = np.asarray(led.tx_ts, dtype=np.int64)
    edges = np.array(list(led.edges), dtype=np.int64).reshape(-1, 2)
    keep = largest_components(n, edges, ts)
    new = np.full(n, -1, np.int64)
    new[keep] = np.arange(int(keep.sum()))

    def rows(r):
        a = np.array(r, dtype=np.float64).reshape(-1, 3)
        k = keep[a[:, 0].astype(np.int64)]
        return a[k, 0].astype(np.int64), a[k, 1].astype(np.int64), a[k, 2]

    it, ia, iv = rows(led.in_rows)
    ot, oa, ov = rows(led.out_rows)
    addrs = np.unique(np.concatenate([ia, oa]))
    inputs = pl.DataFrame({"tid": new[it], "wid": np.searchsorted(addrs, ia), "amount": iv})
    outputs = pl.DataFrame({"tid": new[ot], "wid": np.searchsorted(addrs, oa), "amount": ov})
    ek = keep[edges[:, 0]] & keep[edges[:, 1]] if len(edges) else np.zeros(0, bool)
    tx_edges = pl.DataFrame({"src": new[edges[ek, 0]], "dst": new[edges[ek, 1]]}, schema={"src": pl.Int64, "dst": pl.Int64})

    kept = np.flatnonzero(keep)
    tids = pl.Series("tid", np.arange(len(kept), dtype=np.int64))
    base = pl.DataFrame({
        "tid": tids, "ltid": kept, "ts": ts[kept].astype(np.int16),
        "block": np.asarray(led.tx_block, np.int64)[kept],
        "class": np.asarray(led.tx_label, np.int8)[kept],
        "truth": np.asarray(led.tx_truth, bool)[kept],
        "role": np.asarray(led.tx_role, np.int8)[kept],
        "frozen": np.asarray(led.tx_frozen, bool)[kept],
    })
    feats = tx_feats.tx_features(tids, inputs, outputs, tx_edges)
    txs = base.join(feats, on="tid").sort("tid")

    agg = lambda df: df.group_by("wid", "tid").agg(pl.col("amount").sum())  # noqa: E731
    lin, lout = agg(inputs), agg(outputs)
    addr_tx = lin.select("wid", "tid").sort("wid", "tid")
    tx_addr = lout.select("tid", "wid").sort("tid", "wid")
    n_w = len(addrs)

    wf = wallet_feats.wallet_features(lin, lout, txs)
    links = pl.concat([addr_tx, tx_addr.select("wid", "tid")]).unique()
    wl = wallet_labels(links, txs.select("tid", "class"), env.p.p_wunk, env.label_rng)
    aa_graph, derived = graph.addr_addr(txs, addr_tx, tx_addr, None, n_w)
    wallets = (pl.DataFrame({"wid": np.arange(n_w), "addr": addrs}).join(wl, on="wid")
               .join(wf, on="wid").join(derived.with_columns(pl.col("wid").cast(pl.Int64)), on="wid").sort("wid"))
    tl = graph.typed_links(txs, addr_tx, tx_addr)
    graphs = {"tx_flow": graph.tx_flow(txs, tx_edges), "addr_tx": graph.addr_tx_csr(tl, txs.height, n_w),
              "addr_addr": aa_graph}
    steps = (pl.DataFrame({"ts": ts, "kept": keep}).group_by("ts")
             .agg(pl.len().alias("n_live"), pl.col("kept").sum().alias("n_retained")).sort("ts"))
    return WorldExport(txs.select("tid", "ltid", "ts", "block", "class", "truth", "role", "frozen", *AUG_FEATURES),
                       tx_edges, inputs, outputs, addr_tx, tx_addr, wallets, causal_table(tl), graphs, steps)
