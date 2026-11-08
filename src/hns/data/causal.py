"""D3: causal wallet state for task T (SPEC §5 D3, E5).

For every wallet and every time step at which it has a link, the cumulative
counts over linked txs with time step <= t:

- ``c_in``: distinct txs where the wallet is an input,
- ``c_out``: distinct txs where it receives an output,
- ``c_tot``: distinct linked txs (a tx can be both, so ``c_tot <= c_in + c_out``),
- ``c_steps``: distinct time steps.

Stored sorted by ``(wid, ts)``. ``query(wid, t)`` returns the state of the last
row with ``ts <= t``, or zeros if the wallet has no link yet. These are the
only wallet features T may use.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from hns.data.graph import GRAPH_DIR, LINK_IN, LINK_OUT, typed_links
from hns.data.load import PARQUET_DIR

COUNTS = ["c_in", "c_out", "c_tot", "c_steps"]
_TS_STRIDE = 64  # > max time step (49)


def causal_table(links: pl.DataFrame) -> pl.DataFrame:
    per_step = (
        links.group_by(["wid", "ts"])
        .agg(
            (pl.col("type") == LINK_IN).sum().alias("n_in"),
            (pl.col("type") == LINK_OUT).sum().alias("n_out"),
            pl.col("tid").n_unique().alias("n_tot"),
        )
        .sort(["wid", "ts"])
    )
    return per_step.select(
        "wid", "ts",
        pl.col("n_in").cum_sum().over("wid").cast(pl.Int32).alias("c_in"),
        pl.col("n_out").cum_sum().over("wid").cast(pl.Int32).alias("c_out"),
        pl.col("n_tot").cum_sum().over("wid").cast(pl.Int32).alias("c_tot"),
        pl.int_range(1, pl.len() + 1).over("wid").cast(pl.Int32).alias("c_steps"),
    )


class CausalWalletState:
    def __init__(self, table: pl.DataFrame):
        self.key = table["wid"].to_numpy().astype(np.int64) * _TS_STRIDE + table["ts"].to_numpy()
        self.wid = table["wid"].to_numpy()
        self.counts = table.select(COUNTS).to_numpy().astype(np.float32)
        if np.any(np.diff(self.key) <= 0):
            raise ValueError("causal table must be strictly sorted by (wid, ts)")

    @classmethod
    def load(cls, path: Path = GRAPH_DIR / "causal_wallet.parquet") -> "CausalWalletState":
        return cls(pl.read_parquet(path))

    def query(self, wid: np.ndarray, t: np.ndarray) -> np.ndarray:
        """``(n, 4)`` float32 counts of ``wid`` as of time step ``t`` (inclusive)."""
        wid = np.asarray(wid, dtype=np.int64)
        q = wid * _TS_STRIDE + np.asarray(t, dtype=np.int64)
        i = np.searchsorted(self.key, q, side="right") - 1
        ok = (i >= 0) & (self.wid[np.maximum(i, 0)] == wid)
        out = np.zeros((len(q), len(COUNTS)), dtype=np.float32)
        out[ok] = self.counts[i[ok]]
        return out


def run(parquet_dir: Path = PARQUET_DIR, out_dir: Path = GRAPH_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    txs = pl.read_parquet(parquet_dir / "txs.parquet", columns=["tid", "ts"])
    links = typed_links(txs, pl.read_parquet(parquet_dir / "addr_tx.parquet"),
                        pl.read_parquet(parquet_dir / "tx_addr.parquet"))
    causal_table(links).write_parquet(out_dir / "causal_wallet.parquet")
    print(f"D3: wrote {out_dir / 'causal_wallet.parquet'}")


if __name__ == "__main__":
    run()
