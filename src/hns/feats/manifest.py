"""Load the shared-feature manifest and apply its transforms (SPEC §7, §5 D5)."""

from __future__ import annotations

from functools import cache
from pathlib import Path

import numpy as np
import yaml

MANIFEST = Path(__file__).with_name("manifest.yaml")
GROUPS = ("tx", "wallet_T", "wallet_W")


@cache
def load() -> dict:
    m = yaml.safe_load(MANIFEST.read_text())
    for g in GROUPS:
        for name, e in m[g].items():
            if e["transform"] not in ("log1p", "identity", "block_ts") or e["status"] not in ("audited", "assumed"):
                raise ValueError(f"bad manifest entry {g}/{name}")
    return m


def features(group: str) -> list[str]:
    return list(load()[group])


def apply(transform: str, x: np.ndarray, name: str = "?") -> np.ndarray:
    """Transform raw values to the pre-standardization scale. Fails on invalid input."""
    x = np.asarray(x, dtype=np.float64)
    if np.isnan(x).any():
        raise ValueError(f"{name}: NaN input")
    if transform == "identity":
        return x
    if transform == "log1p":
        if (x < 0).any():
            raise ValueError(f"{name}: negative value under log1p")
        return np.log1p(x)
    if transform == "block_ts":
        b = load()["block_ts"]
        out = (x - b["first_block"]) / b["blocks_per_step"]
        out[x == 0] = b["sentinel_to"]
        if (out[x != 0] < 0).any():
            raise ValueError(f"{name}: block height before the dataset's first block")
        return out
    raise ValueError(transform)
