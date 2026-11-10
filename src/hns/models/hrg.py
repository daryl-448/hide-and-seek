"""Hierarchical relationship graph views for T and W (SPEC §8.1, §5 leakage rules).

Nodes live in one global id space: txs are ``0 .. n_tx-1`` (level 1), wallets
are ``n_tx + wid`` (level 2). Each relation is a CSR over that space, and
holds distinct neighbours of one type, so uniform sampling over a row is
uniform over distinct neighbours (E20):

==============  =============  ======================================
relation        from -> to     meaning
==============  =============  ======================================
``FLOW_OUT``    tx -> tx       txs spending this tx's outputs
``FLOW_IN``     tx -> tx       txs whose outputs this tx spends
``TX_INW``      tx -> wallet   input addresses of the tx
``TX_OUTW``     tx -> wallet   output addresses of the tx
``W_INTX``      wallet -> tx   txs where the wallet is an input
``W_OUTTX``     wallet -> tx   txs where the wallet receives an output
``AA_OUT``      wallet -> w    AddrAddr successors (W view only)
``AA_IN``       wallet -> w    AddrAddr predecessors (W view only)
==============  =============  ======================================

A view is a target's sampled 2-hop neighbourhood: up to ``k1`` neighbours per
relation at hop 1 and ``k2`` at hop 2, at most ``cap`` nodes, seeded per
(target, parent, relation), so a target's view does not depend on its batch.
Every sampled pair gives two messages: ``child -> parent`` typed by the
relation as seen from the parent, and ``parent -> child`` typed by its inverse.
Exact duplicate messages within a view are dropped.

**T view (causal).** Rows whose children are txs are sorted by the child's
time step, and cut at the target's time step ``t``, so the view holds only txs
with ``ts <= t``. (Flow edges join txs of one time step; tx -> wallet rows
need no cut.) Wallet nodes get D3 causal counts as of ``t``. AddrAddr is not
in the T view (§8.1).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numba as nb
import numpy as np
import polars as pl

from hns.data.causal import COUNTS, CausalWalletState
from hns.data.graph import GRAPH_DIR, LINK_IN, LINK_OUT
from hns.data.load import ANON_FEATURES, PARQUET_DIR
from hns.data.norm import NORM_DIR, Normalizer
from hns.data.split import SPLIT_DIR
from hns.feats import manifest
from hns.models.sampler import _splitmix64

FLOW_OUT, FLOW_IN, TX_INW, TX_OUTW, W_INTX, W_OUTTX, AA_OUT, AA_IN = range(8)
N_REL = 8
SELF_LOOP = 8           # message type of the self-loop the encoder adds
N_EDGE_TYPES = 9
INVERSE = np.array([FLOW_IN, FLOW_OUT, W_INTX, W_OUTTX, TX_INW, TX_OUTW, AA_IN, AA_OUT], dtype=np.int64)
TX_RELS = (FLOW_OUT, FLOW_IN, TX_INW, TX_OUTW)
W_RELS = (W_INTX, W_OUTTX, AA_OUT, AA_IN)
HOP_TARGET, HOP_1, HOP_2 = 0, 1, 2


@dataclass
class Relations:
    """All relations stacked into one CSR: row ``g`` of relation ``r`` is
    ``indices[indptr[r, g]:indptr[r, g + 1]]``."""

    indptr: np.ndarray      # int64 (N_REL, n_nodes + 1), offsets into the stacked arrays
    indices: np.ndarray     # int64 global child ids
    ts: np.ndarray          # int16 child time step (tx children), else -1
    logmult: np.ndarray     # float32 log1p(multiplicity) (AddrAddr), else 0
    n_tx: int
    n_nodes: int


def build_relations(n_tx: int, n_w: int, flow: dict, links: dict, aa: dict | None, tx_ts: np.ndarray) -> Relations:
    """``flow``, ``links``, ``aa``: the D2 npz contents. ``aa=None`` leaves AddrAddr out (T view)."""
    n = n_tx + n_w
    srcs, dsts, lms = [[] for _ in range(N_REL)], [[] for _ in range(N_REL)], [[] for _ in range(N_REL)]

    def rows(indptr):
        return np.repeat(np.arange(len(indptr) - 1, dtype=np.int64), np.diff(indptr))

    fs, fd = rows(flow["out_indptr"]), flow["out_indices"].astype(np.int64)
    srcs[FLOW_OUT], dsts[FLOW_OUT] = fs, fd
    srcs[FLOW_IN], dsts[FLOW_IN] = fd, fs
    t, w, ty = rows(links["t2w_indptr"]), links["t2w_wid"].astype(np.int64) + n_tx, links["t2w_type"]
    for rel_tw, rel_wt, kind in ((TX_INW, W_INTX, LINK_IN), (TX_OUTW, W_OUTTX, LINK_OUT)):
        m = ty == kind
        srcs[rel_tw], dsts[rel_tw] = t[m], w[m]
        srcs[rel_wt], dsts[rel_wt] = w[m], t[m]
    if aa is not None:
        s, d = rows(aa["out_indptr"]) + n_tx, aa["out_indices"].astype(np.int64) + n_tx
        lm = aa["out_logmult"].astype(np.float32)
        srcs[AA_OUT], dsts[AA_OUT], lms[AA_OUT] = s, d, lm
        srcs[AA_IN], dsts[AA_IN], lms[AA_IN] = d, s, lm

    indptr = np.zeros((N_REL, n + 1), dtype=np.int64)
    ind, tss, lmult = [], [], []
    off = 0
    for r in range(N_REL):
        s = np.asarray(srcs[r], dtype=np.int64)
        d = np.asarray(dsts[r], dtype=np.int64)
        lm = np.asarray(lms[r], dtype=np.float32) if len(lms[r]) else np.zeros(len(s), np.float32)
        cts = np.where(d < n_tx, tx_ts[np.minimum(d, n_tx - 1)], -1).astype(np.int16)
        order = np.lexsort((d, cts, s))  # rows sorted by (child ts, child id): causal prefix cut
        indptr[r, 1:] = np.cumsum(np.bincount(s, minlength=n)) + off
        indptr[r, 0] = off
        ind.append(d[order]); tss.append(cts[order]); lmult.append(lm[order])
        off += len(s)
    return Relations(indptr, np.concatenate(ind), np.concatenate(tss), np.concatenate(lmult), n_tx, n)


@nb.njit(cache=True)
def _floyd(m, deg, state, chosen):
    """``m`` distinct offsets from ``range(deg)`` into ``chosen[:m]``."""
    c = 0
    for j in range(deg - m, deg):
        state, r = _splitmix64(state)
        t = np.int64(r % np.uint64(j + 1))
        dup = False
        for q in range(c):
            if chosen[q] == t:
                dup = True
                break
        chosen[c] = j if dup else t
        c += 1
    return state


@nb.njit(cache=True)
def _mix(seed, a, b, r):
    s = (np.uint64(seed) * np.uint64(0x2545F4914F6CDD1D)) ^ np.uint64(a)
    s, z = _splitmix64(s)
    s, z = _splitmix64(z ^ (np.uint64(b) * np.uint64(0x9E3779B97F4A7C15)))
    s, z = _splitmix64(z ^ np.uint64(r + 1))
    return z


@nb.njit(cache=True)
def _row(indptr, ts, r, g, t_cut):
    lo, hi = indptr[r, g], indptr[r, g + 1]
    if t_cut >= 0 and hi > lo and ts[lo] >= 0:  # tx children: keep the prefix with ts <= t_cut
        a, b = lo, hi
        while a < b:
            mid = (a + b) // 2
            if ts[mid] <= t_cut:
                a = mid + 1
            else:
                b = mid
        hi = a
    return lo, hi


@nb.njit(cache=True)
def _build(indptr, indices, ts, logmult, n_tx, rel_on, inverse, targets, t_cut, seed, k1, k2, cap, local):
    """Sampled views of ``targets``. ``local`` is an all ``-1`` scratch array over global ids; it is
    restored before returning. Returns node and message arrays over the whole batch."""
    B = targets.shape[0]
    max_e = 2 * (k1 * 4 + k1 * 4 * k2 * 4)
    node_g = np.empty(B * cap, np.int64)
    node_hop = np.empty(B * cap, np.int8)
    node_view = np.empty(B * cap, np.int32)
    tgt = np.empty(B, np.int64)
    e_src = np.empty(B * max_e, np.int64)
    e_dst = np.empty(B * max_e, np.int64)
    e_typ = np.empty(B * max_e, np.int8)
    e_lm = np.empty(B * max_e, np.float32)
    vkey = np.empty(max_e, np.int64)
    vlm = np.empty(max_e, np.float32)
    chosen = np.empty(max(k1, k2), np.int64)
    order = np.empty(cap, np.int64)
    n_nodes, n_edges = 0, 0
    for v in range(B):
        base = n_nodes
        g0 = targets[v]
        local[g0] = 0
        node_g[n_nodes] = g0; node_hop[n_nodes] = HOP_TARGET; node_view[n_nodes] = v
        n_nodes += 1
        tgt[v] = base
        ne = 0
        for hop in range(1, 3):
            k = k1 if hop == 1 else k2
            # parents: the target (hop 1) or the hop-1 nodes in a seeded random order (hop 2)
            if hop == 1:
                n_par = 1
                order[0] = 0
            else:
                n_par = 0
                for i in range(base, n_nodes):
                    if node_hop[i] == HOP_1:
                        order[n_par] = i - base
                        n_par += 1
                st = _mix(seed, g0, 0, 99)  # r = 99: distinct from relation draws
                for i in range(n_par - 1, 0, -1):
                    st, z = _splitmix64(st)
                    j = np.int64(z % np.uint64(i + 1))
                    tmp = order[i]; order[i] = order[j]; order[j] = tmp
            for pi in range(n_par):
                p_loc = order[pi]
                p = node_g[base + p_loc]
                is_tx = p < n_tx
                for r in range(8):
                    if not rel_on[r] or (r < 4) != is_tx:
                        continue
                    lo, hi = _row(indptr, ts, r, p, t_cut[v])
                    deg = hi - lo
                    m = min(k, deg)
                    if m == 0:
                        continue
                    if m == deg:
                        for j in range(m):
                            chosen[j] = j
                    else:
                        _floyd(m, deg, _mix(seed, g0, p, r), chosen)
                    for j in range(m):
                        c = indices[lo + chosen[j]]
                        c_loc = local[c]
                        if c_loc < 0:
                            if n_nodes - base >= cap:
                                continue
                            c_loc = n_nodes - base
                            local[c] = c_loc
                            node_g[n_nodes] = c; node_hop[n_nodes] = hop; node_view[n_nodes] = v
                            n_nodes += 1
                        # message child -> parent typed r, parent -> child typed inverse(r)
                        vkey[ne] = (c_loc * 1024 + p_loc) * 16 + r
                        vlm[ne] = logmult[lo + chosen[j]]
                        ne += 1
                        vkey[ne] = (p_loc * 1024 + c_loc) * 16 + inverse[r]
                        vlm[ne] = logmult[lo + chosen[j]]
                        ne += 1
        perm = np.argsort(vkey[:ne])
        last = -1
        for q in range(ne):
            kq = vkey[perm[q]]
            if kq == last:
                continue
            last = kq
            e_src[n_edges] = base + kq // 16 // 1024
            e_dst[n_edges] = base + (kq // 16) % 1024
            e_typ[n_edges] = kq % 16
            e_lm[n_edges] = vlm[perm[q]]
            n_edges += 1
        for i in range(base, n_nodes):
            local[node_g[i]] = -1
    return (node_g[:n_nodes], node_hop[:n_nodes], node_view[:n_nodes], tgt,
            e_src[:n_edges], e_dst[:n_edges], e_typ[:n_edges], e_lm[:n_edges])


@dataclass
class Views:
    node_g: np.ndarray      # global id per node
    hop: np.ndarray         # 0 target, 1, 2
    view: np.ndarray        # view (target) index per node
    target: np.ndarray      # node index of each view's target
    src: np.ndarray         # message source node index
    dst: np.ndarray         # message destination node index
    etype: np.ndarray       # relation as seen from dst
    logmult: np.ndarray
    kind: np.ndarray | None = None   # node kind (0 tx, 1 wallet, 2 controller); None: ``node_g >= n_tx`` is a wallet


class ViewSampler:
    def __init__(self, rel: Relations, active: tuple[int, ...], causal: bool, k1: int = 10, k2: int = 5, cap: int = 512):
        if cap > 1024:
            raise ValueError("cap must be <= 1024 (message key packing)")
        self.rel, self.causal, self.k1, self.k2, self.cap = rel, causal, k1, k2, cap
        self.rel_on = np.zeros(N_REL, dtype=np.bool_)
        self.rel_on[list(active)] = True
        self._local = np.full(rel.n_nodes, -1, dtype=np.int64)

    def sample(self, targets: np.ndarray, t: np.ndarray | None, seed: int) -> Views:
        targets = np.ascontiguousarray(targets, dtype=np.int64)
        t_cut = (np.asarray(t, dtype=np.int64) if self.causal else np.full(len(targets), -1, np.int64))
        r = self.rel
        out = _build(r.indptr, r.indices, r.ts, r.logmult, np.int64(r.n_tx), self.rel_on, INVERSE, targets,
                     np.ascontiguousarray(t_cut), np.int64(seed & 0x7FFFFFFFFFFFFFFF), self.k1, self.k2, self.cap,
                     self._local)
        return Views(*out)


class TaskData:
    """Everything a T0 run reads for one task and seed: views, standardized features, targets.

    - T: tx nodes carry the 17 shared features (T moments) and the 165 anonymized
      adapter features; wallet nodes carry D3 causal counts as of the view's
      target time step. Txs with null augmented features are unreachable (E7).
    - W: tx nodes carry the 17 shared features (W moments, no adapter); wallet
      nodes carry the 58 W-view features.
    """

    def __init__(self, task: str, seed: int, parquet_dir: Path = PARQUET_DIR, graph_dir: Path = GRAPH_DIR,
                 split_dir: Path = SPLIT_DIR, norm_dir: Path = NORM_DIR, k1: int = 10, k2: int = 5, cap: int = 512):
        self.task = task
        tx_names = manifest.features("tx")
        cols = ["tid", "ts", *tx_names] + (ANON_FEATURES if task == "T" else [])
        txs = pl.read_parquet(parquet_dir / "txs.parquet", columns=cols).sort("tid")
        self.tx_ts = txs["ts"].to_numpy().astype(np.int16)
        n_tx = txs.height
        n_w = pl.scan_parquet(parquet_dir / "wallets.parquet").select(pl.len()).collect().item()
        self.n_tx = n_tx
        norm = Normalizer.load(norm_dir / ("T.json" if task == "T" else f"W_seed{seed}.json"))
        self.norm = norm
        ok = txs["size"].is_not_null().to_numpy()
        self.x_tx = np.full((n_tx, len(tx_names)), np.nan, dtype=np.float32)
        self.x_tx[ok] = norm.standardize("tx", txs.filter(pl.col("size").is_not_null()))
        self.x_anon = None
        if task == "T":
            self.x_anon = np.full((n_tx, len(ANON_FEATURES)), np.nan, dtype=np.float32)
            self.x_anon[ok] = norm.standardize("tx_anon", txs.filter(pl.col("size").is_not_null()))
        flow = dict(np.load(graph_dir / "tx_flow.npz"))
        links = dict(np.load(graph_dir / "addr_tx.npz"))
        if task == "T":
            self.causal = CausalWalletState.load(graph_dir / "causal_wallet.parquet")
            st = norm.stats["wallet_T"]
            self._wt = [(manifest.load()["wallet_T"][c]["transform"], st[c]["mean"], st[c]["sd"]) for c in COUNTS]
            rel = build_relations(n_tx, n_w, flow, links, None, self.tx_ts)
            self.sampler = ViewSampler(rel, (FLOW_OUT, FLOW_IN, TX_INW, TX_OUTW, W_INTX, W_OUTTX), True, k1, k2, cap)
            sp = pl.read_parquet(split_dir / f"T_seed{seed}.parquet")
            self.split = sp.join(txs.select("tid", "ts"), on="tid").rename({"tid": "id"})
            self.d_w = len(COUNTS)
        else:
            w = pl.read_parquet(parquet_dir / "wallets.parquet").join(
                pl.read_parquet(graph_dir / "wallet_derived.parquet"), on="wid").sort("wid")
            self.x_w = norm.standardize("wallet_W", w)
            aa = dict(np.load(graph_dir / "addr_addr.npz"))
            rel = build_relations(n_tx, n_w, flow, links, aa, self.tx_ts)
            self.sampler = ViewSampler(rel, tuple(range(N_REL)), False, k1, k2, cap)
            sp = pl.read_parquet(split_dir / f"W_seed{seed}.parquet")
            self.split = sp.with_columns(pl.lit(-1, pl.Int16).alias("ts")).rename({"wid": "id"})
            self.d_w = self.x_w.shape[1]
        self.d_tx = self.x_tx.shape[1]
        self.d_anon = 0 if self.x_anon is None else self.x_anon.shape[1]

    def targets(self, split: str, fraction: float = 1.0) -> dict[str, np.ndarray]:
        """Labelled targets of ``split`` (``fit`` restricted to the nested label fraction)."""
        df = self.split.filter((pl.col("split") == split) & pl.col("y").is_not_null())
        if split == "fit" and fraction < 1.0:
            df = df.filter(pl.col(f"in_f{fraction}"))
        ids = df["id"].to_numpy().astype(np.int64)
        g = ids if self.task == "T" else ids + self.n_tx
        return {"g": g, "t": df["ts"].to_numpy().astype(np.int64), "y": df["y"].to_numpy().astype(np.float32)}

    def batch(self, g: np.ndarray, t: np.ndarray, seed: int) -> tuple[Views, dict[str, np.ndarray]]:
        """Views of targets ``g`` (at time steps ``t`` for T) and their node features."""
        v = self.sampler.sample(g, t, seed)
        is_tx = v.node_g < self.n_tx
        tid, wid = v.node_g[is_tx], v.node_g[~is_tx] - self.n_tx
        feats = {"x_tx": self.x_tx[tid]}
        if self.x_anon is not None:
            feats["x_anon"] = self.x_anon[tid]
        if self.task == "T":
            raw = self.causal.query(wid, t[v.view[~is_tx]])
            feats["x_w"] = np.stack([(manifest.apply(tr, raw[:, i]) - m) / s for i, (tr, m, s) in enumerate(self._wt)],
                                    axis=1).astype(np.float32)
        else:
            feats["x_w"] = self.x_w[wid]
        for k, x in feats.items():
            if not np.isfinite(x).all():
                raise ValueError(f"non-finite {k} in a view (null-feature tx reached?)")
        return v, feats
