"""Entities, their state and their action branches (SPEC §9.3, §9.7).

- **Services** (scripted): per-customer deposit addresses, a hot wallet funded
  by genesis, consolidation of funded deposits into the hot wallet every
  ``consol_interval`` sub-steps, and FIFO payouts in batches of up to
  ``batch_size`` outputs with change back to the hot wallet (which makes
  AddrAddr self-loops). A queued payout waits for a full batch at most
  ``payout_max_wait`` sub-steps (ours).
- **Innocents**: individuals with geometric(``addr_mean``) addresses, each
  funded by one genesis output. A scripted demand process gives a share
  ``demand_rate`` of idle innocents a demand per sub-step: an amount
  ``Beta(a, b) x balance`` and a fixed recipient (an innocent's address,
  chosen by preferential attachment on degree, with probability ``p_ii``;
  else the innocent's deposit address at a uniform service).
  Ours, so that money keeps circulating: every committed deposit an innocent
  makes *to a service as the demand's recipient* queues a withdrawal of the
  same amount from that service to a preferential-attachment innocent.
- **Launderers**: an illicit source address that receives fresh proceeds by
  genesis at the start of every episode (ours, so laundering continues over
  the world's 10 time steps), and a mule pool that grows on demand.

Action branches (Eqs. 16-17) are the ``*_BRANCHES`` tables below, with
options indexed in the order of the constants next to them.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

AMOUNTS = (0.10, 0.25, 0.50, 1.00)
STRATEGIES = ("direct", "fan_out", "scatter_gather", "gather_scatter", "stack", "bipartite", "cycle", "peel", "random")
K_RECIPIENTS = 16
FRESH, REUSE = 0, 1

LAUNDERER_BRANCHES = {"initiate": 2, "amount": len(AMOUNTS), "recipient": K_RECIPIENTS,
                      "strategy": len(STRATEGIES), "mules": 2, "change": 2}
INNOCENT_BRANCHES = {"initiate": 2, "split": 3, "route": 3, "change": 2}   # initiate: 0 now, 1 defer
DETECTOR_BRANCHES = {"analysis_depth": 2, "update_strategy": 3}
ROUTE_DIRECT, ROUTE_FORWARD, ROUTE_SERVICE = 0, 1, 2
INIT_NOW, INIT_DEFER = 0, 1          # innocents
WAIT, GO = 0, 1                      # launderers
BUDGET_MULT = (0.8, 1.0, 1.25)
BUDGET_RANGE = (0.01, 0.20)


@dataclass
class AgentState:
    """``B(s)`` and ``M_b(s)``: functions of the state only (§9.3)."""
    branches: tuple[str, ...]
    masks: dict[str, np.ndarray]


@dataclass
class Service:
    id: int
    hot: int
    phase: int
    deposits: list[int] = field(default_factory=list)
    queue: deque = field(default_factory=deque)   # (address, amount, global sub-step queued)


@dataclass
class Demand:
    rcpt: int                 # recipient address (fixed by demand)
    amount: float
    to_service: int           # service id if the recipient is a service deposit address, else -1
    stage: int = 0            # 0: awaiting the innocent's decision; 1: forward hop 2 pending
    second: int = -1          # forward: the innocent's own intermediate address
    split: int = 1
    change: int = FRESH


@dataclass
class Tranche:
    strategy: int
    plan: list[tuple]         # remaining hops
    holders: list[int]        # addresses holding tranche funds
    recipient: int            # cash-out deposit address
    mules: int                # FRESH or REUSE
    amount: float = 0.0       # BTC sent into the route by the setup hop
    first_mule: int = -1      # cycle: m1


@dataclass
class Launderer:
    id: int
    source: int
    pool: list[int] = field(default_factory=list)
    tranche: Tranche | None = None
    # stochastic parts of the state, drawn at the start of each sub-step
    k: int = 2
    feerate: float = 0.0
    slots: list[int] = field(default_factory=list)   # service ids behind the K recipient slots
