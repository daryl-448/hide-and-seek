"""Laundering route templates (SPEC §9.3 table), executed one hop per sub-step.

A plan is a list of hops. Hop counts (setup + body + terminal) are <= 6:

==================  =====================================================  ====
strategy            hops                                                     n
==================  =====================================================  ====
``direct``          setup: source -> recipient                               1
``fan_out``         setup scatter -> k mules; terminal                       2
``scatter_gather``  setup scatter -> k; gather -> 1 mule; terminal           3
``gather_scatter``  setup scatter -> k; gather -> 1; split -> k fresh;       4
                    terminal
``stack``           setup scatter -> k; each mule -> 1 further mule;         3
                    terminal
``bipartite``       setup scatter -> k; 1 tx with k inputs, k outputs;       3
                    terminal
``cycle``           setup -> m1; m1 -> m2; m2 -> m1 (reuses m1); terminal    4
``peel``            h in 2..5 peel hops (5-20 % to the recipient, the rest   <=6
                    to a fresh mule), the first from the source; terminal
``random``          setup -> 1 mule; h - 1 one-to-one hops (h in 2..5);      <=6
                    terminal
==================  =====================================================  ====

The setup hop spends all of the source's outputs. Its tranche is
``amount x (source balance - fee)``, with the fee taken first; the rest is
change to the source (``reuse``) or to a fresh source address (``fresh``),
and at 100 % there is no change output. Later hops spend all spendable
outputs of the current holders. The terminal hop pays the recipient from
every holder in its own tx (parallel cash-outs). Mules come from the
``mules`` branch (fresh, or ``reuse``: drawn uniformly from the pool, topped
up with fresh ones if the pool runs short), except where a template says
fresh (``split``, ``peel``).

Each spec's ``meta`` holds ``next`` (the addresses that hold tranche funds
after the hop, if the tx commits unfrozen) and ``cash`` (the recipient).
"""

from __future__ import annotations

import numpy as np

from hns.sim.entities import AMOUNTS, FRESH, REUSE, STRATEGIES, Launderer, Tranche
from hns.sim.utxo import TxSpec
from hns.sim.world import ROLE_LAUNDERER


def plan(strategy: int, k: int, rng: np.random.Generator) -> list[tuple]:
    s = STRATEGIES[strategy]
    if s == "direct":
        return [("setup", 0)]
    if s == "fan_out":
        return [("setup", k), ("terminal",)]
    if s == "scatter_gather":
        return [("setup", k), ("gather",), ("terminal",)]
    if s == "gather_scatter":
        return [("setup", k), ("gather",), ("split", k), ("terminal",)]
    if s == "stack":
        return [("setup", k), ("one_to_one",), ("terminal",)]
    if s == "bipartite":
        return [("setup", k), ("bipartite",), ("terminal",)]
    if s == "cycle":
        return [("setup", 1), ("one_to_one",), ("back",), ("terminal",)]
    if s == "peel":
        h = int(rng.integers(2, 6))
        return [("peel_setup",)] + [("peel",)] * (h - 1) + [("terminal",)]
    if s == "random":
        h = int(rng.integers(2, 6))
        return [("setup", 1)] + [("one_to_one",)] * (h - 1) + [("terminal",)]
    raise ValueError(s)


def _spec(L: Launderer, utxos, pay=(), rest=(), **meta) -> TxSpec:
    return TxSpec(utxos=list(utxos), pay=list(pay), rest=list(rest), feerate=L.feerate,
                  role=ROLE_LAUNDERER, entity=L.id, illicit=True, meta=meta)


def setup_spec(env, L: Launderer, T: Tranche, amount: int, change: int) -> TxSpec:
    """The first hop, from the source. Sets ``T.holders`` candidates via ``meta['next']``."""
    led = env.ledger
    op = T.plan[0]
    p = AMOUNTS[amount]
    if op[0] == "peel_setup":
        mules = [env.fresh_mule(L)]
        frac = float(env.rng.uniform(0.05, 0.20))
        payees, w = [T.recipient, mules[0]], [frac, 1.0 - frac]
    elif op[1] == 0:
        mules, payees, w = [], [T.recipient], [1.0]
    else:
        mules = env.draw_mules(L, op[1], T.mules, exclude=())
        payees, w = mules, [1.0 / len(mules)] * len(mules)
    utxos = led.spendable([L.source])
    in_total = sum(led.u_amt[u] for u in utxos)
    new_source = -1
    if p < 1.0:
        if change == REUSE:
            ch = L.source
        else:
            ch = new_source = env.launderer_addr(L)
        s = _spec(L, utxos, pay=[(a, 0.0) for a in payees], rest=[ch])
        tranche = p * (in_total - s.fee())
        s.pay = [(a, wi * tranche) for a, wi in zip(payees, w)]
    elif op[0] == "peel_setup":
        s = _spec(L, utxos, pay=[(T.recipient, 0.0)], rest=mules)
        tranche = in_total - s.fee()
        s.pay = [(T.recipient, w[0] * tranche)]
    else:
        s = _spec(L, utxos, rest=payees)
        tranche = in_total - s.fee()
    s.meta.update(next=mules, cash=T.recipient, tranche=tranche, new_source=new_source, has_change=p < 1.0)
    return s


def body_specs(env, L: Launderer, T: Tranche) -> list[TxSpec]:
    """Specs of the next non-setup hop over the holders that still have spendable outputs."""
    led = env.ledger
    op = T.plan[0][0]
    H = [h for h in T.holders if led.addr_utxos[h]]
    if op == "gather":
        m = env.draw_mules(L, 1, T.mules, exclude=H)
        return [_spec(L, led.spendable(H), rest=m, next=m, cash=T.recipient)]
    if op == "split":
        ms = [env.fresh_mule(L) for _ in range(T.plan[0][1])]
        return [_spec(L, led.spendable(H), rest=ms, next=ms, cash=T.recipient)]
    if op == "bipartite":
        ms = env.draw_mules(L, len(H), T.mules, exclude=H)
        return [_spec(L, led.spendable(H), rest=ms, next=ms, cash=T.recipient)]
    out = []
    used = list(H)
    for h in H:
        u = led.spendable([h])
        if op == "one_to_one":
            m = env.draw_mules(L, 1, T.mules, exclude=used)
            used += m
            out.append(_spec(L, u, rest=m, next=m, cash=T.recipient))
        elif op == "back":
            out.append(_spec(L, u, rest=[T.first_mule], next=[T.first_mule], cash=T.recipient))
        elif op == "peel":
            m = env.fresh_mule(L)
            s = _spec(L, u, pay=[(T.recipient, 0.0)], rest=[m], next=[m], cash=T.recipient)
            frac = float(env.rng.uniform(0.05, 0.20))
            s.pay = [(T.recipient, frac * (sum(led.u_amt[x] for x in u) - s.fee()))]
            out.append(s)
        elif op == "terminal":
            out.append(_spec(L, u, rest=[T.recipient], next=[], cash=T.recipient))
        else:
            raise ValueError(op)
    return out


__all__ = ["plan", "setup_spec", "body_specs", "FRESH", "REUSE"]
