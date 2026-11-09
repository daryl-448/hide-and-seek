"""World parameters, time and blocks (SPEC §9.2, §9.5, §9.7).

An episode is one time step of ``SUBSTEPS`` sub-steps; a world is
``EPISODES`` consecutive episodes placed at ``offset`` (time steps
``offset + 1 .. offset + EPISODES``). Blocks follow the real data's
1,975 blocks per time step from block 391,200 (E19).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields

FIRST_BLOCK, BLOCKS_PER_STEP = 391_200, 1_975
SUBSTEPS = 50
EPISODES = 10
DUST = 1e-5  # BTC; smallest payment output (§9.3 masks)

ROLE_INNOCENT, ROLE_SERVICE, ROLE_LAUNDERER = 0, 1, 2

# §9.7 calibration search bounds: name -> (lo, hi, is_int).
CALIBRATED = {
    "n_innocents": (3000, 6000, True),
    "addr_mean": (1.0, 5.0, False),
    "bal_mu": (-3.0, 1.0, False),
    "bal_sigma": (1.0, 3.0, False),
    "demand_a": (0.5, 5.0, False),
    "demand_b": (0.5, 5.0, False),
    "p_ii": (0.5, 0.9, False),
    "pa_exp": (0.5, 1.5, False),
    "demand_rate": (0.01, 0.03, False),
    "n_services": (20, 60, True),
    "batch_size": (2, 100, True),
    "consol_interval": (1, 10, True),
    "p_fresh": (0.5, 0.95, False),
    "fee_mu": (-15.0, -14.0, False),
    "fee_sigma": (0.4, 1.0, False),
}


@dataclass
class WorldParams:
    # --- calibrated generator parameters (§9.7); defaults are mid-range placeholders
    n_innocents: int = 4000
    addr_mean: float = 2.0          # addresses per innocent: geometric with this mean
    bal_mu: float = -1.0            # innocent address starting balance: lognormal BTC
    bal_sigma: float = 2.0
    demand_a: float = 1.0           # demand amount: Beta(a, b) fraction of the innocent's balance
    demand_b: float = 3.0
    p_ii: float = 0.7               # demand recipient is an innocent (else a service)
    pa_exp: float = 1.0             # preferential-attachment exponent on degree
    demand_rate: float = 0.02       # share of innocents that get a demand per sub-step
    n_services: int = 40
    batch_size: int = 20            # service payout batch (outputs)
    consol_interval: int = 5        # service deposit consolidation interval (sub-steps)
    p_fresh: float = 0.8            # change to a fresh address (static policies)
    fee_mu: float = -14.5           # feerate: lognormal BTC/byte
    fee_sigma: float = 0.7
    # --- not calibrated (§9.5 multiverse, or ours)
    n_launderers: int = 25
    proceeds_mu: float = 1.0        # launderer proceeds per episode: lognormal BTC (§9.7 bounds' centre)
    proceeds_sigma: float = 1.0
    budget: float = 0.05            # detector flag budget fraction b
    mask_rate: float = 0.1          # detector feature-mask rate (used by M5)
    offset: int = 0                 # world window offset in [0, 39]
    service_float: float = 1000.0   # ours: genesis BTC of each service hot wallet
    payout_max_wait: int = 10       # ours: a queued payout waits at most this many sub-steps for a full batch
    # --- observed-label rates (§9.4), solved by calibration
    p_ill: float = 1.0
    p_lic: float = 0.2
    p_wunk: float = 0.0

    def ts(self, episode: int) -> int:
        """Absolute time step (1-based) of an episode."""
        return self.offset + episode + 1

    def block(self, episode: int, substep: int) -> int:
        """Block height of a sub-step: about 39.5 blocks per sub-step."""
        return FIRST_BLOCK + (self.ts(episode) - 1) * BLOCKS_PER_STEP + (substep * BLOCKS_PER_STEP) // SUBSTEPS

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "WorldParams":
        names = {f.name for f in fields(cls)}
        unknown = set(d) - names
        if unknown:
            raise ValueError(f"unknown world parameters: {sorted(unknown)}")
        return cls(**d)
