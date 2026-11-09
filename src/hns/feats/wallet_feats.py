"""The 55 named lifetime wallet features (W view), computed once (SPEC §7, E9).

Inputs:

- ``lin``: ``wid, tid[, amount]``, one row per distinct (address, tx) where the
  address is an input; ``amount`` is the BTC it spent in that tx;
- ``lout``: ``tid, wid[, amount]``, the same for outputs (BTC received);
- ``txs``: ``tid, ts[, block, fees, in_BTC_total]``.

The real data has no per-link amounts or blocks, so only the count features
(:func:`link_counts`) can be recomputed there; the D1 audit calls it. The rest
follow the manifest's *assumed* definitions. Empty histories are 0 (§7): a
statistic over no values, a block of "never", or a gap list of a single tx.

Per-tx values used by the statistics, for a wallet ``w`` and a linked tx:

- ``btc_transacted``: BTC sent plus received by ``w`` in that tx;
- ``fees``: the full tx fee, for senders and receivers alike (§7);
- ``fees_as_share``: tx fee / ``in_BTC_total`` (ours: the fee's share of the
  BTC the tx moves);
- ``transacted_w_address``: distinct counterparties of ``w`` in the tx (its
  output addresses if ``w`` is an input, its input addresses if ``w`` is an
  output; never ``w`` itself);
- ``blocks_btwn_*``: gaps between consecutive blocks of the linked (input,
  output) txs, sorted by block.
"""

from __future__ import annotations

import polars as pl

from hns.feats import manifest

STATS = ("total", "min", "max", "mean", "median")
DERIVED = ("self_loop_frac", "aa_in_degree", "aa_out_degree")  # from hns.data.graph.addr_addr


def named_features() -> list[str]:
    """The 55 named wallet features, in manifest order."""
    return [f for f in manifest.features("wallet_W") if f not in DERIVED]


def link_counts(lin: pl.DataFrame, lout: pl.DataFrame, txs: pl.DataFrame) -> pl.DataFrame:
    """``wid, num_txs_as_sender, num_txs_as receiver, total_txs, num_timesteps_appeared_in``."""
    links = pl.concat([lin.select("wid", "tid"), lout.select("wid", "tid")]).unique()
    s = lin.group_by("wid").agg(pl.col("tid").n_unique().alias("num_txs_as_sender"))
    r = lout.group_by("wid").agg(pl.col("tid").n_unique().alias("num_txs_as receiver"))
    t = (links.join(txs.select("tid", "ts"), on="tid").group_by("wid")
         .agg(pl.len().alias("total_txs"), pl.col("ts").n_unique().alias("num_timesteps_appeared_in")))
    return (t.join(s, on="wid", how="left").join(r, on="wid", how="left").fill_null(0)
            .select("wid", "num_txs_as_sender", "num_txs_as receiver", "total_txs", "num_timesteps_appeared_in"))


def _stats(df: pl.DataFrame, col: str, prefix: str) -> pl.DataFrame:
    c = pl.col(col)
    return df.group_by("wid").agg(c.sum().alias(f"{prefix}_total"), c.min().alias(f"{prefix}_min"),
                                  c.max().alias(f"{prefix}_max"), c.mean().alias(f"{prefix}_mean"),
                                  c.median().alias(f"{prefix}_median"))


def _gaps(links: pl.DataFrame, prefix: str) -> pl.DataFrame:
    g = (links.group_by("wid").agg(pl.col("block").sort().diff().drop_nulls().alias("g"))
         .explode("g").drop_nulls("g"))
    return _stats(g, "g", prefix)


def wallet_features(lin: pl.DataFrame, lout: pl.DataFrame, txs: pl.DataFrame) -> pl.DataFrame:
    """``wid`` plus the 55 named features in manifest order, one row per linked wallet."""
    t = txs.select("tid", "ts", "block", "fees", share=pl.col("fees") / pl.col("in_BTC_total"))
    lin = lin.select("wid", "tid", sent="amount").join(t, on="tid")
    lout = lout.select("wid", "tid", recv="amount").join(t, on="tid")
    per_tx = (lin.select("wid", "tid", "sent").join(lout.select("wid", "tid", "recv"), on=["wid", "tid"],
                                                       how="full", coalesce=True)
              .with_columns(pl.col("sent", "recv").fill_null(0.0))
              .with_columns(btc=pl.col("sent") + pl.col("recv")).join(t, on="tid"))

    pairs = lin.select(src="wid", tid="tid").join(lout.select(dst="wid", tid="tid"), on="tid")
    cp = pl.concat([pairs.select(w="src", c="dst", tid="tid"), pairs.select(w="dst", c="src", tid="tid")])
    cp = cp.filter(pl.col("w") != pl.col("c")).unique().rename({"w": "wid"})
    n_cp = cp.group_by("wid", "tid").len().rename({"len": "n_cp"})
    per_tx = per_tx.join(n_cp, on=["wid", "tid"], how="left").with_columns(pl.col("n_cp").fill_null(0))
    multiple = (cp.group_by("wid", "c").agg(pl.col("tid").n_unique().alias("n")).filter(pl.col("n") > 1)
                .group_by("wid").len().rename({"len": "num_addr_transacted_multiple"}))

    life = per_tx.group_by("wid").agg(pl.col("block").min().alias("first_block_appeared_in"),
                                      pl.col("block").max().alias("last_block_appeared_in"))
    life = life.with_columns(lifetime_in_blocks=pl.col("last_block_appeared_in") - pl.col("first_block_appeared_in"))
    first_sent = lin.group_by("wid").agg(pl.col("block").min().alias("first_sent_block"))
    first_recv = lout.group_by("wid").agg(pl.col("block").min().alias("first_received_block"))

    out = (link_counts(lin, lout, t)
           .join(life, on="wid", how="left")
           .join(first_sent, on="wid", how="left").join(first_recv, on="wid", how="left")
           .join(_stats(per_tx, "btc", "btc_transacted"), on="wid", how="left")
           .join(_stats(lin, "sent", "btc_sent"), on="wid", how="left")
           .join(_stats(lout, "recv", "btc_received"), on="wid", how="left")
           .join(_stats(per_tx, "fees", "fees"), on="wid", how="left")
           .join(_stats(per_tx, "share", "fees_as_share"), on="wid", how="left")
           .join(_gaps(per_tx, "blocks_btwn_txs"), on="wid", how="left")
           .join(_gaps(lin, "blocks_btwn_input_txs"), on="wid", how="left")
           .join(_gaps(lout, "blocks_btwn_output_txs"), on="wid", how="left")
           .join(multiple, on="wid", how="left")
           .join(_stats(per_tx, "n_cp", "transacted_w_address"), on="wid", how="left")
           .fill_null(0))
    return out.select("wid", *[pl.col(c).cast(pl.Float64) for c in named_features()]).sort("wid")
