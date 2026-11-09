"""The 17 named tx features, computed once for real data and the simulator (SPEC §7, E9).

Inputs are row tables:

- ``inputs``: ``tid, wid[, amount]``, one row per spent output (simulator) or
  per AddrTx link (real data, which has no amounts);
- ``outputs``: ``tid, wid[, amount]``, one row per created output (simulator)
  or per TxAddr link (real);
- ``edges``: ``src, dst``, the tx->tx flow edges, already restricted to one
  time step (§9.2).

``num_input_addresses`` and ``num_output_addresses`` count rows, i.e. inputs
and outputs, so they are never below the distinct-address link count: the
audited real behaviour (§2). The D1 audit calls :func:`flow_degrees` and
:func:`address_counts` on the real edge lists.
"""

from __future__ import annotations

import polars as pl

from hns.data.load import AUG_FEATURES

SIZE_BASE, SIZE_IN, SIZE_OUT = 10, 148, 34


def tx_size(n_in: int, n_out: int) -> int:
    """Bytes: ``10 + 148 n_in + 34 n_out`` (§9.2)."""
    return SIZE_BASE + SIZE_IN * n_in + SIZE_OUT * n_out


def flow_degrees(edges: pl.DataFrame) -> pl.DataFrame:
    """``tid, in_txs_degree, out_txs_degree`` for txs with at least one edge."""
    i = edges.group_by("dst").len().rename({"dst": "tid", "len": "in_txs_degree"})
    o = edges.group_by("src").len().rename({"src": "tid", "len": "out_txs_degree"})
    return i.join(o, on="tid", how="full", coalesce=True).fill_null(0)


def address_counts(inputs: pl.DataFrame, outputs: pl.DataFrame) -> pl.DataFrame:
    """``tid, num_input_addresses, num_output_addresses`` (row counts)."""
    i = inputs.group_by("tid").len().rename({"len": "num_input_addresses"})
    o = outputs.group_by("tid").len().rename({"len": "num_output_addresses"})
    return i.join(o, on="tid", how="full", coalesce=True).fill_null(0)


def _amount_stats(rows: pl.DataFrame, side: str) -> pl.DataFrame:
    a = pl.col("amount")
    return rows.group_by("tid").agg(
        a.min().alias(f"{side}_BTC_min"), a.max().alias(f"{side}_BTC_max"), a.mean().alias(f"{side}_BTC_mean"),
        a.median().alias(f"{side}_BTC_median"), a.sum().alias(f"{side}_BTC_total"))


def tx_features(tids: pl.Series, inputs: pl.DataFrame, outputs: pl.DataFrame, edges: pl.DataFrame) -> pl.DataFrame:
    """``tid`` plus the 17 features in manifest order, for every tid in ``tids``.

    Every tx has at least one input and one output. ``size`` is the simulator's
    byte formula; ``fees = in_BTC_total - out_BTC_total`` and
    ``total_BTC = out_BTC_total`` (both audited, §2).
    """
    df = (pl.DataFrame({"tid": tids})
          .join(flow_degrees(edges), on="tid", how="left")
          .join(address_counts(inputs, outputs), on="tid", how="left")
          .join(_amount_stats(inputs, "in"), on="tid", how="left")
          .join(_amount_stats(outputs, "out"), on="tid", how="left")
          .fill_null(0))
    df = df.with_columns(
        size=SIZE_BASE + SIZE_IN * pl.col("num_input_addresses") + SIZE_OUT * pl.col("num_output_addresses"),
        total_BTC=pl.col("out_BTC_total"),
        fees=pl.col("in_BTC_total") - pl.col("out_BTC_total"),
    )
    return df.select("tid", *[pl.col(c).cast(pl.Float64) for c in AUG_FEATURES])
