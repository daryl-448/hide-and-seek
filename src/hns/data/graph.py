"""D2: CSR graphs and edge-list-derived wallet features (SPEC §5 D2, E20).

Outputs, in ``data/graph/``:

- ``tx_flow.npz``: tx->tx money flow, CSR both directions (``out_*``, ``in_*``).
  Edges touching the 965 null-feature txs are dropped, because T drops those
  txs from the graph (E7) and they must not be reachable from any view.
- ``addr_tx.npz``: typed address-tx links (``LINK_IN``: address is an input of
  the tx, ``LINK_OUT``: address receives an output), with the tx's time step.
  CSR wallet->tx (``w2t_*``) and tx->wallet (``t2w_*``). Each row is sorted by
  ``(ts, other id)``, so a causal prefix (ts <= t) is one ``searchsorted``.
- ``addr_addr.npz``: AddrAddr collapsed to distinct directed non-self pairs,
  with ``logmult = log1p(multiplicity)`` and ``first_ts`` (first time step of a
  tx producing the pair). CSR both directions.
- ``wallet_derived.parquet``: ``self_loop_frac``, ``aa_in_degree``,
  ``aa_out_degree`` per wid.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from hns.data.load import PARQUET_DIR

GRAPH_DIR = Path("data/graph")
LINK_IN, LINK_OUT = 0, 1


def csr(src: np.ndarray, dst: np.ndarray, n: int, *attrs: np.ndarray, sort_keys: tuple = ()):
    """CSR of ``src -> dst`` over ``n`` rows. Rows are sorted by ``sort_keys`` then ``dst``.

    Returns ``(indptr int64[n+1], indices int32, *attrs)`` in CSR order.
    """
    order = np.lexsort((dst, *reversed(sort_keys), src)) if sort_keys else np.lexsort((dst, src))
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(np.bincount(src, minlength=n), out=indptr[1:])
    return (indptr, dst[order].astype(np.int32), *(a[order] for a in attrs))


def tx_flow(txs: pl.DataFrame, edges: pl.DataFrame) -> dict[str, np.ndarray]:
    bad = txs.filter(pl.col("size").is_null())["tid"]
    e = edges.filter(~pl.col("src").is_in(bad.implode()) & ~pl.col("dst").is_in(bad.implode()))
    s, d, n = e["src"].to_numpy(), e["dst"].to_numpy(), txs.height
    oi, oj = csr(s, d, n)
    ii, ij = csr(d, s, n)
    return {"out_indptr": oi, "out_indices": oj, "in_indptr": ii, "in_indices": ij,
            "n_dropped_null_edges": np.int64(edges.height - e.height)}


def typed_links(txs: pl.DataFrame, addr_tx: pl.DataFrame, tx_addr: pl.DataFrame) -> pl.DataFrame:
    """One row per link: ``wid, tid, type, ts``."""
    ts = txs.select("tid", "ts")
    return pl.concat([
        addr_tx.select("wid", "tid").with_columns(pl.lit(LINK_IN, pl.Int8).alias("type")),
        tx_addr.select("wid", "tid").with_columns(pl.lit(LINK_OUT, pl.Int8).alias("type")),
    ]).join(ts, on="tid", how="left")


def addr_tx_csr(links: pl.DataFrame, n_tx: int, n_w: int) -> dict[str, np.ndarray]:
    w, t = links["wid"].to_numpy(), links["tid"].to_numpy()
    ty, ts = links["type"].to_numpy(), links["ts"].to_numpy()
    wi, wj, wty, wts = csr(w, t, n_w, ty, ts, sort_keys=(ts,))
    ti, tj, tty, tts = csr(t, w, n_tx, ty, ts, sort_keys=(ts,))
    return {"w2t_indptr": wi, "w2t_tid": wj, "w2t_type": wty, "w2t_ts": wts,
            "t2w_indptr": ti, "t2w_wid": tj, "t2w_type": tty, "t2w_ts": tts}


def addr_addr(txs: pl.DataFrame, addr_tx: pl.DataFrame, tx_addr: pl.DataFrame,
              aa_rows: pl.DataFrame, n_w: int) -> tuple[dict[str, np.ndarray], pl.DataFrame]:
    # AddrAddr == AddrTx join TxAddr on txId (audited in D1); the join gives each row's time step.
    rows = (addr_tx.join(tx_addr, on="tid").join(txs.select("tid", "ts"), on="tid")
            .select(pl.col("wid").alias("src"), pl.col("wid_right").alias("dst"), "ts"))
    if rows.height != aa_rows.height:
        raise ValueError("AddrAddr is not AddrTx join TxAddr")
    self_rows = rows.filter(pl.col("src") == pl.col("dst")).group_by("src").len()
    as_input = rows.group_by("src").len()
    pairs = (rows.filter(pl.col("src") != pl.col("dst")).group_by(["src", "dst"])
             .agg(pl.len().alias("mult"), pl.col("ts").min().alias("first_ts")))
    s, d = pairs["src"].to_numpy(), pairs["dst"].to_numpy()
    lm = np.log1p(pairs["mult"].to_numpy()).astype(np.float32)
    ft = pairs["first_ts"].to_numpy().astype(np.int16)
    oi, oj, olm, oft = csr(s, d, n_w, lm, ft)
    ii, ij, ilm, ift = csr(d, s, n_w, lm, ft)
    g = {"out_indptr": oi, "out_indices": oj, "out_logmult": olm, "out_first_ts": oft,
         "in_indptr": ii, "in_indices": ij, "in_logmult": ilm, "in_first_ts": ift}

    n_in = np.bincount(as_input["src"].to_numpy(), as_input["len"].to_numpy(), minlength=n_w)
    n_self = np.bincount(self_rows["src"].to_numpy(), self_rows["len"].to_numpy(), minlength=n_w)
    frac = np.divide(n_self, n_in, out=np.zeros(n_w), where=n_in > 0)
    derived = pl.DataFrame({
        "wid": np.arange(n_w, dtype=np.int32),
        "self_loop_frac": frac.astype(np.float32),
        "aa_in_degree": np.diff(ii).astype(np.float32),
        "aa_out_degree": np.diff(oi).astype(np.float32),
    })
    return g, derived


def run(parquet_dir: Path = PARQUET_DIR, out_dir: Path = GRAPH_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    txs = pl.read_parquet(parquet_dir / "txs.parquet", columns=["tid", "ts", "size"])
    n_w = pl.scan_parquet(parquet_dir / "wallets.parquet").select(pl.len()).collect().item()
    edges = pl.read_parquet(parquet_dir / "tx_edges.parquet")
    a = pl.read_parquet(parquet_dir / "addr_tx.parquet")
    t = pl.read_parquet(parquet_dir / "tx_addr.parquet")
    aa = pl.read_parquet(parquet_dir / "addr_addr.parquet")

    np.savez(out_dir / "tx_flow.npz", **tx_flow(txs, edges))
    np.savez(out_dir / "addr_tx.npz", **addr_tx_csr(typed_links(txs, a, t), txs.height, n_w))
    g, derived = addr_addr(txs, a, t, aa, n_w)
    np.savez(out_dir / "addr_addr.npz", **g)
    derived.write_parquet(out_dir / "wallet_derived.parquet")
    print(f"D2: wrote {out_dir}")


if __name__ == "__main__":
    run()
