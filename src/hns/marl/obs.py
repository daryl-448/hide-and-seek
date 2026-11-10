"""Agent observations on the live HRG (SPEC §9.3, §10).

Every observation is a sampled view plus node features:

- **tx nodes**: the 17 shared tx features, standardized with T's frozen
  moments (D5), as the detector sees them.
- **wallet nodes**: the D3 causal counts as of the view's time step,
  standardized the same way. In an agent view (not in the detector's) each
  of the agent's own addresses also carries ``W_PRIV`` private values:
  ``log1p(balance)``, ``log1p(#spendable outputs)`` and two role flags
  (launderer: is the source, holds tranche funds; innocent: is funded, 0).
- **controller node** (agent views only, never visible to the detector):
  the agent's private state, followed by the world's global readout.

The **global readout** (ours; §9.3 leaves it open) is :func:`world_readout`:
time within the episode and the world, the inspector's budget, what the
previous sub-step's flagging did (candidates, flag share, flag precision,
analysis depth), and how many txs the episode has committed so far. All of
it is observable in the world. The running total is the one value that is
not about the previous sub-step: that sub-step's own commit count is already
fixed by the other three (candidates - flagged + frozen).

Launderer controller state (``L_CTRL`` values): tranche in flight; ``log1p``
of the source balance, the tranche amount and the holders' funds; hops left
/ 6; the tranche's strategy (one-hot, 9); ``k / 5``; ``ln(feerate) + 14.5``;
``log1p(#mules) / 5``. Its ``recipient`` pointer branch scores ``K = 16``
candidate services, each described by ``D_CAND`` values: the standardized
causal counts of the service's hot wallet, whether the launderer already has
a deposit address there, and ``log1p`` of that address's received-tx count.

Innocent controller state (``I_CTRL`` values): ``log1p`` of its balance and
of the demand amount; amount / balance; recipient is a service; sub-steps
the demand has waited / 50; funded addresses / 5; addresses / 10.

An agent controls at most ``MAX_ADDR`` address nodes in its view (ours):
the launderer's source, tranche holders, then newest mules; the innocent's
addresses by balance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from hns.data.causal import COUNTS
from hns.data.load import AUG_FEATURES
from hns.feats import manifest
from hns.models.detector import KIND_TX, KIND_W, Graph
from hns.models.hrg import Views
from hns.sim.entities import (DETECTOR_BRANCHES, INNOCENT_BRANCHES, K_RECIPIENTS, LAUNDERER_BRANCHES, STRATEGIES)
from hns.sim.live import split_ids
from hns.sim.world import EPISODES, ROLE_LAUNDERER, SUBSTEPS

G_DIM = 8
L_CTRL = 8 + len(STRATEGIES) + G_DIM
I_CTRL = 7 + G_DIM
W_PRIV = 4
D_W_AGENT = len(COUNTS) + W_PRIV
D_CAND = len(COUNTS) + 2
MAX_ADDR = 10
BRANCHES = {"launderer": LAUNDERER_BRANCHES, "innocent": INNOCENT_BRANCHES, "detector": DETECTOR_BRANCHES}


class Featurizer:
    """Frozen D5 standardization of the T view's shared channel, from ``Normalizer.stats``-shaped moments."""

    def __init__(self, stats: dict):
        self.tx = self._group(stats["tx"], AUG_FEATURES)
        self.wallet = self._group(stats["wallet_T"], COUNTS)

    @staticmethod
    def _group(st: dict, names: list[str]):
        if list(st) != list(names):
            raise ValueError("moments do not match the manifest order")
        if any(st[n]["transform"] != "log1p" for n in names):
            raise ValueError("the T view's shared features are all log1p (manifest)")
        mean = np.array([st[n]["mean"] for n in names], np.float32)
        sd = np.array([st[n]["sd"] for n in names], np.float32)
        return lambda raw: ((np.log1p(np.asarray(raw, np.float32)) - mean) / sd).astype(np.float32)


def identity_stats() -> dict:
    """Moments that only apply the manifest transforms (mean 0, sd 1): for tests without real data."""
    m = manifest.load()
    return {g: {n: {"transform": e["transform"], "mean": 0.0, "sd": 1.0} for n, e in m[g].items()}
            for g in ("tx", "wallet_T", "wallet_W")}


# ---------------------------------------------------------------------------- batches of observations
@dataclass
class ObsBatch:
    v: Views                         # with ``kind``; the target (or controller) is each view's first node
    x_tx: np.ndarray
    x_w: np.ndarray
    x_c: np.ndarray | None = None    # (n_views, d_ctrl)
    cand: np.ndarray | None = None   # (n_views, K, D_CAND)

    def __len__(self) -> int:
        return len(self.v.target)

    def split(self) -> list[tuple]:
        """One compact observation per view, for the replay buffer."""
        v, B = self.v, len(self.v.target)
        edges = np.arange(B + 1)
        n_ptr = np.searchsorted(v.view, edges)
        e_ptr = np.searchsorted(v.view[v.dst], edges)
        t_ptr = np.concatenate([[0], np.cumsum(v.kind == KIND_TX)])[n_ptr]
        w_ptr = np.concatenate([[0], np.cumsum(v.kind == KIND_W)])[n_ptr]
        kind, hop = v.kind.astype(np.int8), v.hop.astype(np.int8)
        src, dst, typ = v.src, v.dst, v.etype.astype(np.int8)
        out = []
        for j in range(B):
            a, b, ea, eb = n_ptr[j], n_ptr[j + 1], e_ptr[j], e_ptr[j + 1]
            out.append((kind[a:b].copy(), hop[a:b].copy(), self.x_tx[t_ptr[j]:t_ptr[j + 1]].copy(),
                        self.x_w[w_ptr[j]:w_ptr[j + 1]].copy(), None if self.x_c is None else self.x_c[j].copy(),
                        (src[ea:eb] - a).astype(np.int16), (dst[ea:eb] - a).astype(np.int16), typ[ea:eb].copy(),
                        None if self.cand is None else self.cand[j].copy()))
        return out

    def tensors(self, device) -> tuple[Graph, dict[str, torch.Tensor]]:
        t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(device, non_blocking=True)  # noqa: E731
        f = {"x_tx": t(self.x_tx), "x_w": t(self.x_w)}
        if self.x_c is not None:
            f["x_c"] = t(self.x_c)
        return Graph(self.v, 0, device), f


def collate(obs: list[tuple]) -> ObsBatch:
    """Stack replay observations into one batch."""
    sizes = np.array([len(o[0]) for o in obs], np.int64)
    off = np.concatenate([[0], np.cumsum(sizes)])
    cat = lambda i, dt: np.concatenate([o[i] for o in obs]).astype(dt, copy=False)  # noqa: E731
    n_e = sum(len(o[5]) for o in obs)
    v = Views(node_g=np.zeros(off[-1], np.int64), hop=cat(1, np.int8),
              view=np.repeat(np.arange(len(obs), dtype=np.int32), sizes), target=off[:-1].copy(),
              src=np.concatenate([o[5] + off[j] for j, o in enumerate(obs)]).astype(np.int64, copy=False),
              dst=np.concatenate([o[6] + off[j] for j, o in enumerate(obs)]).astype(np.int64, copy=False),
              etype=cat(7, np.int8), logmult=np.zeros(n_e, np.float32), kind=cat(0, np.int8))
    return ObsBatch(v, cat(2, np.float32), cat(3, np.float32),
                    None if obs[0][4] is None else np.stack([o[4] for o in obs]),
                    None if obs[0][8] is None else np.stack([o[8] for o in obs]))


def merge(batches: list[ObsBatch]) -> ObsBatch:
    """Concatenate batches of the same node feature widths (no controller nodes) into one graph batch."""
    n_off = np.concatenate([[0], np.cumsum([len(b.v.hop) for b in batches])])
    v_off = np.concatenate([[0], np.cumsum([len(b) for b in batches])])
    cat = lambda f: np.concatenate([f(b, j) for j, b in enumerate(batches)])  # noqa: E731
    v = Views(node_g=cat(lambda b, j: b.v.node_g), hop=cat(lambda b, j: b.v.hop),
              view=cat(lambda b, j: b.v.view + v_off[j]).astype(np.int32), target=cat(lambda b, j: b.v.target + n_off[j]),
              src=cat(lambda b, j: b.v.src + n_off[j]), dst=cat(lambda b, j: b.v.dst + n_off[j]),
              etype=cat(lambda b, j: b.v.etype), logmult=cat(lambda b, j: b.v.logmult), kind=cat(lambda b, j: b.v.kind))
    return ObsBatch(v, cat(lambda b, j: b.x_tx), cat(lambda b, j: b.x_w))


def with_kind(v: Views, n_tx: int) -> Views:
    """A static-graph view (``hns.models.hrg``) with its node kinds made explicit, so it can be merged."""
    if v.kind is None:
        v.kind = (v.node_g >= n_tx).astype(np.int8)
    return v


# ---------------------------------------------------------------------------- what the roles see
def world_readout(env, last: dict) -> np.ndarray:
    """The world's global readout (ours).

    ``last``: the previous sub-step's ``n_cand``, ``n_flag``, ``n_frozen`` and ``depth``, and
    ``episode_committed``, the txs committed so far in the episode. ``Trainer`` refreshes ``last`` after every
    sub-step, so at an episode's first sub-step ``episode_committed`` still holds the previous episode's total.
    """
    n = max(last.get("n_cand", 0), 1)
    fl = last.get("n_flag", 0)
    return np.array([
        env.sub / SUBSTEPS, env.episode / EPISODES, env.budget / 0.20,
        math.log1p(last.get("n_cand", 0)) / math.log1p(500.0), fl / n, last.get("n_frozen", 0) / max(fl, 1),
        math.log1p(last.get("episode_committed", 0)) / 10.0, float(last.get("depth", 1) - 1),
    ], np.float32)


def detector_obs(g, fz: Featurizer, tids: np.ndarray, t_cut, seed: int, k1: int, k2: int, cap: int) -> ObsBatch:
    """T views of tx targets on live graph ``g``, with causal wallet counts as of each view's time step."""
    v = g.tx_views(tids, t_cut, seed, k1, k2, cap)
    tx, addr = split_ids(v)
    ts_w = np.broadcast_to(np.asarray(t_cut, np.int64), np.shape(tids))[v.view[v.kind == KIND_W]]
    return ObsBatch(v, fz.tx(g.tx_feat[tx]), fz.wallet(g.counts_asof(addr, ts_w)))


def _agent_batch(env, fz: Featurizer, ids: list[int], own: list[list[int]], priv: list[tuple], x_c: np.ndarray,
                 cand: np.ndarray | None, seed: int, k2: int) -> ObsBatch:
    g = env.graph
    g.reserve(len(env.ledger.addr_role))
    a_ptr = np.zeros(len(ids) + 1, np.int64)
    np.cumsum([len(o) for o in own], out=a_ptr[1:])
    a_addr = np.array([a for o in own for a in o], np.int64)
    v = g.agent_views(a_ptr, a_addr, np.asarray(ids, np.int64), seed, k2)
    tx, addr = split_ids(v)
    assert len(addr) == len(a_addr)
    x_w = np.concatenate([fz.wallet(g.w_cnt[addr]), np.asarray(priv, np.float32).reshape(-1, W_PRIV)], axis=1)
    return ObsBatch(v, fz.tx(g.tx_feat[tx]), x_w, x_c, cand)


def launderer_obs(env, fz: Featurizer, ids: list[int], gvec: np.ndarray, seed: int, k2: int = 5) -> ObsBatch:
    led, g = env.ledger, env.graph
    g.reserve(len(led.addr_role))
    bal, utx = led.balance, led.addr_utxos
    x_c = np.zeros((len(ids), L_CTRL), np.float32)
    cand = np.zeros((len(ids), K_RECIPIENTS, D_CAND), np.float32)
    own, priv = [], []
    for j, i in enumerate(ids):
        L = env.launderers[i]
        T = L.tranche
        holders = T.holders if T is not None else []
        o = list(dict.fromkeys([L.source, *holders, *reversed(L.pool[-MAX_ADDR:])]))[:MAX_ADDR]
        hs = set(holders)
        own.append(o)
        priv.extend((math.log1p(bal[a]), math.log1p(len(utx[a])), a == L.source, a in hs) for a in o)
        c = x_c[j]
        c[1] = math.log1p(bal[L.source])
        if T is not None:
            c[0] = 1.0
            c[2] = math.log1p(T.amount)
            c[3] = math.log1p(sum(bal[h] for h in holders))
            c[4] = len(T.plan) / 6.0
            c[8 + T.strategy] = 1.0
        c[5] = L.k / 5.0
        c[6] = math.log(L.feerate) + 14.5
        c[7] = math.log1p(len(L.pool)) / 5.0
        c[8 + len(STRATEGIES):] = gvec
        hot = np.array([env.services[s].hot for s in L.slots], np.int64)
        cand[j, :len(hot), :len(COUNTS)] = fz.wallet(g.w_cnt[hot])
        for q, s in enumerate(L.slots):
            dep = env.deposit.get((ROLE_LAUNDERER, i, s))
            if dep is not None:
                cand[j, q, len(COUNTS)] = 1.0
                cand[j, q, len(COUNTS) + 1] = math.log1p(g.w_cnt[dep, 1])
    return _agent_batch(env, fz, ids, own, priv, x_c, cand, seed, k2)


def innocent_obs(env, fz: Featurizer, ids: list[int], gvec: np.ndarray, seed: int, k2: int = 5) -> ObsBatch:
    led = env.ledger
    bal, utx = led.balance, led.addr_utxos
    x_c = np.zeros((len(ids), I_CTRL), np.float32)
    own, priv = [], []
    for j, i in enumerate(ids):
        d = env.pending[i]
        addrs = env.inn_addrs[i]
        o = sorted(addrs, key=lambda a: -bal[a])[:MAX_ADDR]
        own.append(o)
        priv.extend((math.log1p(bal[a]), math.log1p(len(utx[a])), bal[a] > 0, 0.0) for a in o)
        tot = sum(bal[a] for a in addrs)
        c = x_c[j]
        c[0] = math.log1p(tot)
        c[1] = math.log1p(d.amount)
        c[2] = min(d.amount / max(tot, 1e-12), 1.0)
        c[3] = d.to_service >= 0
        c[4] = min((env.g - d.born) / SUBSTEPS, 1.0)
        c[5] = min(sum(bal[a] > 0 for a in addrs) / 5.0, 1.0)
        c[6] = min(len(addrs) / 10.0, 1.0)
        c[7:] = gvec
    return _agent_batch(env, fz, ids, own, priv, x_c, None, seed, k2)


def pack_masks(role: str, states: dict, ids: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """``(masks, active)``: ``M_b(s)`` of every branch side by side (all-true for a branch outside ``B(s)``),
    and which branches are in ``B(s)``."""
    br = BRANCHES[role]
    masks = np.ones((len(ids), sum(br.values())), bool)
    active = np.zeros((len(ids), len(br)), bool)
    for j, i in enumerate(ids):
        st, o = states[i], 0
        for bi, (b, n) in enumerate(br.items()):
            if b in st.branches:
                masks[j, o:o + n] = st.masks[b]
                active[j, bi] = True
            o += n
    return masks, active
