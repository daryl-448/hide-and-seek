"""Dense integer ids (SPEC §5 D0)."""

from __future__ import annotations

import polars as pl


def address_ids(addresses: pl.Series) -> pl.DataFrame:
    """Map address strings to dense int32 ``wid`` in sorted-unique (byte) order.

    The combined wallet file is sorted by address, so ``wid`` order equals its
    file order, which the official W split relies on (SPEC §2, E3).
    """
    uniq = addresses.unique().sort()
    return pl.DataFrame({"address": uniq, "wid": pl.int_range(0, uniq.len(), dtype=pl.Int32, eager=True)})
