"""Live HRG over a world's ledger: the graph the agents and the inspector see (SPEC §8.1, §9.3, §10).

The export (``hns.sim.export``) rebuilds a finished world with the real
pipeline's code. This module keeps the same graph and the same shared
features up to date sub-step by sub-step, so policies can observe ``s_t``
and the detector can score tentative txs "as if committed" (§9.3 step 4).

Graph. Six relations as in ``hns.models.hrg`` (no AddrAddr: the T view
leaves it out): ``FLOW_OUT``/``FLOW_IN`` (tx -> tx, same time step),
``TX_INW``/``TX_OUTW`` (tx -> distinct input / output addresses) and
``W_INTX``/``W_OUTTX`` (address -> txs). Each row is an append-only linked
list of distinct neighbours, newest first. A batch of txs can be added and
then popped again exactly (the tentative txs of a sub-step), which leaves no
trace.

Features, all raw (pre-transform):

- ``tx_feat``: the 17 named tx features (``AUG_FEATURES`` order), by the
  definitions of ``hns.feats.tx_feats``. ``out_txs_degree`` grows as later
  txs of the same time step spend a tx's outputs.
- ``w_cnt``: the D3 causal counts ``c_in, c_out, c_tot, c_steps`` of every
  address as of now; ``snap[ts]`` holds them as of the end of a finished
  time step.

Views. :meth:`LiveGraph.tx_views` is the T view (§8.1) of tx targets: the
sampled 2-hop neighbourhood restricted to txs with ``ts <= t_cut``.
:meth:`LiveGraph.agent_views` is the agent view (§9.3): one controller
node per agent linked to the addresses it controls, and up to ``k2`` txs
per address and link type. Sampling is uniform over distinct neighbours
(reservoir sampling over the row), seeded per (target, parent, relation).
In a view's id space a tx is its ledger tid and an address is
``WBASE + address``; the controller has id ``-1``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numba as nb
import numpy as np

from hns.feats.tx_feats import tx_size
from hns.models.hrg import _mix, Views
from hns.models.sampler import _splitmix64
from hns.sim.utxo import GENESIS

# relation ids as in hns.models.hrg (numba needs plain module constants)
FLOW_OUT, FLOW_IN, TX_INW, TX_OUTW, W_INTX, W_OUTTX = 0, 1, 2, 3, 4, 5
N_LIVE_REL = 6
CTRL_ADDR, ADDR_CTRL = 9, 10      # messages address -> its controller, controller -> address (agent view only)
N_AGENT_EDGE_TYPES = 11
KIND_TX, KIND_W, KIND_CTRL = 0, 1, 2
WBASE = 1 << 40
N_FEAT = 17
F_OUT_DEGREE = 1                  # index of out_txs_degree in AUG_FEATURES


@dataclass
class TxBatch:
    """Txs of one sub-step, in commit order: features and distinct neighbours in CSR form."""

    ts: int
    feat: np.ndarray        # float32 (n, 17) raw features, out_txs_degree = 0
    in_ptr: np.ndarray
    in_w: np.ndarray        # distinct input addresses
    out_ptr: np.ndarray
    out_w: np.ndarray       # distinct output addresses
    par_ptr: np.ndarray
    par: np.ndarray         # distinct parent tids in the same time step

    @property
    def n(self) -> int:
        return self.feat.shape[0]

    def take(self, keep: np.ndarray) -> "TxBatch":
        """The sub-batch of the rows where ``keep`` is true, order kept."""
        idx = np.flatnonzero(keep)

        def sel(ptr, val):
            parts = [val[ptr[j]:ptr[j + 1]] for j in idx]
            p = np.zeros(len(idx) + 1, np.int64)
            if parts:
                np.cumsum([len(x) for x in parts], out=p[1:])
            return p, (np.concatenate(parts) if parts else np.zeros(0, np.int64))

        return TxBatch(self.ts, self.feat[idx], *sel(self.in_ptr, self.in_w), *sel(self.out_ptr, self.out_w),
                       *sel(self.par_ptr, self.par))


def _stats(x: list[float]) -> tuple[float, float, float, float, float]:
    s = sorted(x)
    n, tot = len(s), sum(s)
    med = s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])
    return s[0], s[-1], tot / n, med, tot


def tx_batch(led, resolved: list, ts: int) -> TxBatch:
    """Features and links of resolved proposals, as if committed at time step ``ts`` on ledger ``led``."""
    n = len(resolved)
    feat = np.zeros((n, N_FEAT), np.float32)
    in_ptr, out_ptr, par_ptr = [0], [0], [0]
    in_w, out_w, par = [], [], []
    u_addr, u_amt, u_tid, tx_ts = led.u_addr, led.u_amt, led.u_tid, led.tx_ts
    for j, r in enumerate(resolved):
        ut = r.spec.utxos
        ia = [u_amt[u] for u in ut]
        oa = [a for _, a in r.outputs]
        ps = dict.fromkeys(p for u in ut if (p := u_tid[u]) != GENESIS and tx_ts[p] == ts)
        i_st, o_st = _stats(ia), _stats(oa)
        feat[j] = (len(ps), 0.0, o_st[4], i_st[4] - o_st[4], tx_size(len(ut), len(oa)), len(ut), len(oa), *i_st, *o_st)
        in_w.extend(dict.fromkeys(u_addr[u] for u in ut))
        out_w.extend(dict.fromkeys(a for a, _ in r.outputs))
        par.extend(ps)
        in_ptr.append(len(in_w)); out_ptr.append(len(out_w)); par_ptr.append(len(par))
    a = lambda x: np.asarray(x, dtype=np.int64)  # noqa: E731
    return TxBatch(int(ts), feat, a(in_ptr), a(in_w), a(out_ptr), a(out_w), a(par_ptr), a(par))


# ---------------------------------------------------------------------------- numba kernels
@nb.njit(cache=True)
def _push(head, r, owner, child, e_child, e_next, e_rel, e_owner, n_e):
    e_child[n_e] = child
    e_next[n_e] = head[r, owner]
    e_rel[n_e] = r
    e_owner[n_e] = owner
    head[r, owner] = n_e
    return n_e + 1


@nb.njit(cache=True)
def _touch(w, ts, sign, w_cnt, w_cur_ts, w_n_cur):
    """One more (or one fewer) distinct linked tx of address ``w`` at time step ``ts``."""
    if sign > 0:
        if w_cur_ts[w] != ts:
            w_cur_ts[w] = ts
            w_n_cur[w] = 0
        if w_n_cur[w] == 0:
            w_cnt[w, 3] += 1
        w_n_cur[w] += 1
        w_cnt[w, 2] += 1
    else:
        w_n_cur[w] -= 1
        w_cnt[w, 2] -= 1
        if w_n_cur[w] == 0:
            w_cnt[w, 3] -= 1


@nb.njit(cache=True)
def _is_input(w, j, in_ptr, in_w):
    for q in range(in_ptr[j], in_ptr[j + 1]):
        if in_w[q] == w:
            return True
    return False


@nb.njit(cache=True)
def _add(n_tx, n_e, ts, feat, in_ptr, in_w, out_ptr, out_w, par_ptr, par, tx_ts, tx_feat, head, w_cnt, w_cur_ts,
         w_n_cur, e_child, e_next, e_rel, e_owner):
    for j in range(feat.shape[0]):
        t = n_tx + j
        tx_ts[t] = ts
        tx_feat[t, :] = feat[j, :]
        for q in range(par_ptr[j], par_ptr[j + 1]):
            p = par[q]
            n_e = _push(head, FLOW_OUT, p, t, e_child, e_next, e_rel, e_owner, n_e)
            n_e = _push(head, FLOW_IN, t, p, e_child, e_next, e_rel, e_owner, n_e)
            tx_feat[p, F_OUT_DEGREE] += 1.0
        for q in range(in_ptr[j], in_ptr[j + 1]):
            w = in_w[q]
            n_e = _push(head, TX_INW, t, w, e_child, e_next, e_rel, e_owner, n_e)
            n_e = _push(head, W_INTX, w, t, e_child, e_next, e_rel, e_owner, n_e)
            w_cnt[w, 0] += 1
            _touch(w, ts, 1, w_cnt, w_cur_ts, w_n_cur)
        for q in range(out_ptr[j], out_ptr[j + 1]):
            w = out_w[q]
            n_e = _push(head, TX_OUTW, t, w, e_child, e_next, e_rel, e_owner, n_e)
            n_e = _push(head, W_OUTTX, w, t, e_child, e_next, e_rel, e_owner, n_e)
            w_cnt[w, 1] += 1
            if not _is_input(w, j, in_ptr, in_w):
                _touch(w, ts, 1, w_cnt, w_cur_ts, w_n_cur)
    return n_e


@nb.njit(cache=True)
def _pop(n_e0, n_e, ts, in_ptr, in_w, out_ptr, out_w, par_ptr, par, tx_feat, head, w_cnt, w_cur_ts, w_n_cur,
         e_next, e_rel, e_owner):
    for e in range(n_e - 1, n_e0 - 1, -1):   # every row is a stack: undo the pushes newest first
        head[e_rel[e], e_owner[e]] = e_next[e]
    for j in range(in_ptr.shape[0] - 1):
        for q in range(par_ptr[j], par_ptr[j + 1]):
            tx_feat[par[q], F_OUT_DEGREE] -= 1.0
        for q in range(in_ptr[j], in_ptr[j + 1]):
            w = in_w[q]
            w_cnt[w, 0] -= 1
            _touch(w, ts, -1, w_cnt, w_cur_ts, w_n_cur)
        for q in range(out_ptr[j], out_ptr[j + 1]):
            w = out_w[q]
            w_cnt[w, 1] -= 1
            if not _is_input(w, j, in_ptr, in_w):
                _touch(w, ts, -1, w_cnt, w_cur_ts, w_n_cur)


@nb.njit(cache=True)
def _reservoir(head, e_child, e_next, r, owner, tx_ts, t_cut, k, state, chosen):
    """Up to ``k`` uniform distinct children of row ``(r, owner)``; wallet -> tx rows keep ``ts <= t_cut`` only."""
    cut = r >= W_INTX
    c = 0
    e = head[r, owner]
    while e >= 0:
        ch = e_child[e]
        if not cut or tx_ts[ch] <= t_cut:
            if c < k:
                chosen[c] = ch
            else:
                state, z = _splitmix64(state)
                j = np.int64(z % np.uint64(c + 1))
                if j < k:
                    chosen[j] = ch
            c += 1
        e = e_next[e]
    return min(c, k)


@nb.njit(cache=True)
def _tx_views(head, e_child, e_next, tx_ts, targets, t_cut, seed, k1, k2, cap, ltx, lw):
    """T views of tx ``targets``; mirrors ``hns.models.hrg._build`` on the live rows."""
    B = targets.shape[0]
    max_e = 2 * (k1 * 4 + k1 * 4 * k2 * 4)
    node_id = np.empty(B * cap, np.int64)
    node_kind = np.empty(B * cap, np.int8)
    node_hop = np.empty(B * cap, np.int8)
    node_view = np.empty(B * cap, np.int32)
    tgt = np.empty(B, np.int64)
    e_src = np.empty(B * max_e, np.int64)
    e_dst = np.empty(B * max_e, np.int64)
    e_typ = np.empty(B * max_e, np.int8)
    vkey = np.empty(max_e, np.int64)
    chosen = np.empty(max(k1, k2, 1), np.int64)
    order = np.empty(cap, np.int64)
    inverse = np.array([FLOW_IN, FLOW_OUT, W_INTX, W_OUTTX, TX_INW, TX_OUTW], np.int64)
    n_nodes, n_edges = 0, 0
    for v in range(B):
        base = n_nodes
        t0 = targets[v]
        ltx[t0] = 0
        node_id[n_nodes] = t0; node_kind[n_nodes] = KIND_TX; node_hop[n_nodes] = 0; node_view[n_nodes] = v
        n_nodes += 1
        tgt[v] = base
        ne = 0
        for hop in range(1, 3):
            k = k1 if hop == 1 else k2
            if k == 0:
                break
            if hop == 1:
                n_par = 1
                order[0] = 0
            else:
                n_par = 0
                for i in range(base, n_nodes):
                    if node_hop[i] == 1:
                        order[n_par] = i - base
                        n_par += 1
                st = _mix(seed, t0, 0, 99)
                for i in range(n_par - 1, 0, -1):
                    st, z = _splitmix64(st)
                    j = np.int64(z % np.uint64(i + 1))
                    tmp = order[i]; order[i] = order[j]; order[j] = tmp
            for pi in range(n_par):
                p_loc = order[pi]
                p = node_id[base + p_loc]
                p_tx = node_kind[base + p_loc] == KIND_TX
                for r in range(N_LIVE_REL):
                    if (r < 4) != p_tx:
                        continue
                    m = _reservoir(head, e_child, e_next, r, p, tx_ts, t_cut[v], k,
                                   _mix(seed, t0, p + (0 if p_tx else WBASE), r), chosen)
                    c_tx = r < 2 or r >= 4
                    for j in range(m):
                        c = chosen[j]
                        c_loc = ltx[c] if c_tx else lw[c]
                        if c_loc < 0:
                            if n_nodes - base >= cap:
                                continue
                            c_loc = n_nodes - base
                            if c_tx:
                                ltx[c] = c_loc
                            else:
                                lw[c] = c_loc
                            node_id[n_nodes] = c
                            node_kind[n_nodes] = KIND_TX if c_tx else KIND_W
                            node_hop[n_nodes] = hop
                            node_view[n_nodes] = v
                            n_nodes += 1
                        vkey[ne] = (c_loc * 1024 + p_loc) * 16 + r
                        ne += 1
                        vkey[ne] = (p_loc * 1024 + c_loc) * 16 + inverse[r]
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
            n_edges += 1
        for i in range(base, n_nodes):
            if node_kind[i] == KIND_TX:
                ltx[node_id[i]] = -1
            else:
                lw[node_id[i]] = -1
    return (node_id[:n_nodes], node_kind[:n_nodes], node_hop[:n_nodes], node_view[:n_nodes], tgt,
            e_src[:n_edges], e_dst[:n_edges], e_typ[:n_edges])


@nb.njit(cache=True)
def _agent_views(head, e_child, e_next, tx_ts, a_ptr, a_addr, keys, seed, k2, ltx):
    """Agent views: controller (hop 0) -> its addresses (hop 1) -> up to ``k2`` txs per address and link type."""
    B = a_ptr.shape[0] - 1
    n_a = a_addr.shape[0]
    node_id = np.empty(B + n_a * (1 + 2 * k2), np.int64)
    node_kind = np.empty(node_id.shape[0], np.int8)
    node_hop = np.empty(node_id.shape[0], np.int8)
    node_view = np.empty(node_id.shape[0], np.int32)
    tgt = np.empty(B, np.int64)
    e_src = np.empty(2 * n_a * (1 + 2 * k2), np.int64)
    e_dst = np.empty(e_src.shape[0], np.int64)
    e_typ = np.empty(e_src.shape[0], np.int8)
    chosen = np.empty(max(k2, 1), np.int64)
    n, ne = 0, 0
    for v in range(B):
        base = n
        tgt[v] = base
        node_id[n] = -1; node_kind[n] = KIND_CTRL; node_hop[n] = 0; node_view[n] = v
        n += 1
        a0, a1 = a_ptr[v], a_ptr[v + 1]
        for q in range(a0, a1):
            node_id[n] = a_addr[q]; node_kind[n] = KIND_W; node_hop[n] = 1; node_view[n] = v
            e_src[ne] = n; e_dst[ne] = base; e_typ[ne] = CTRL_ADDR
            ne += 1
            e_src[ne] = base; e_dst[ne] = n; e_typ[ne] = ADDR_CTRL
            ne += 1
            n += 1
        if k2 > 0:
            for q in range(a0, a1):
                w = a_addr[q]
                p = base + 1 + (q - a0)
                for r in range(W_INTX, W_OUTTX + 1):
                    m = _reservoir(head, e_child, e_next, r, w, tx_ts, np.int64(1 << 30), k2,
                                   _mix(seed, keys[v], w, r), chosen)
                    for j in range(m):
                        c = chosen[j]
                        c_idx = ltx[c]
                        if c_idx < 0:
                            c_idx = n
                            ltx[c] = n
                            node_id[n] = c; node_kind[n] = KIND_TX; node_hop[n] = 2; node_view[n] = v
                            n += 1
                        e_src[ne] = c_idx; e_dst[ne] = p; e_typ[ne] = r                 # tx -> address
                        ne += 1
                        e_src[ne] = p; e_dst[ne] = c_idx; e_typ[ne] = r - 2             # address -> tx (inverse)
                        ne += 1
        for i in range(base, n):
            if node_kind[i] == KIND_TX:
                ltx[node_id[i]] = -1
    return (node_id[:n], node_kind[:n], node_hop[:n], node_view[:n], tgt, e_src[:ne], e_dst[:ne], e_typ[:ne])


@nb.njit(cache=True)
def _keep_mask(seeds, node_g, d, rate):
    """``keep[i, f]`` is false when ``hash(seeds[i], node_g[i], f) < rate`` (§8.4: the per-episode mask)."""
    out = np.ones((node_g.shape[0], d), np.bool_)
    if rate <= 0.0:
        return out
    for i in range(node_g.shape[0]):
        for f in range(d):
            z = _mix(seeds[i], node_g[i], f, 7)
            if (z >> np.uint64(11)) * (1.0 / 9007199254740992.0) < rate:
                out[i, f] = False
    return out


def keep_mask(seeds: np.ndarray, node_g: np.ndarray, d: int, rate: float) -> np.ndarray:
    """Per-node feature keep-mask of an episode: a pure function of (episode seed, node id, feature), so a node
    created later in the episode is covered by the same draw (SPEC §8.4)."""
    return _keep_mask(np.ascontiguousarray(seeds, np.int64), np.ascontiguousarray(node_g, np.int64), int(d),
                      float(rate))


# ---------------------------------------------------------------------------- the graph
class LiveGraph:
    def __init__(self, cap: int = 1 << 14, e_cap: int = 1 << 16):
        self.n_tx, self.n_e, self.cap, self.e_cap = 0, 0, cap, e_cap
        self.tx_ts = np.zeros(cap, np.int16)
        self.tx_feat = np.zeros((cap, N_FEAT), np.float32)
        self.head = np.full((N_LIVE_REL, cap), -1, np.int32)
        self.w_cnt = np.zeros((cap, 4), np.int32)
        self.w_cur_ts = np.full(cap, -1, np.int16)
        self.w_n_cur = np.zeros(cap, np.int32)
        self.e_child = np.zeros(e_cap, np.int32)
        self.e_next = np.zeros(e_cap, np.int32)
        self.e_rel = np.zeros(e_cap, np.int8)
        self.e_owner = np.zeros(e_cap, np.int32)
        self.snap: dict[int, np.ndarray] = {}     # finished time step -> counts (n_addr, 4) at its end
        self._ltx = np.full(cap, -1, np.int64)
        self._lw = np.full(cap, -1, np.int64)

    def _grow(self, n: int, n_e: int) -> None:
        if n > self.cap:
            new = max(n, 2 * self.cap)
            pad = new - self.cap
            self.tx_ts = np.concatenate([self.tx_ts, np.zeros(pad, np.int16)])
            self.tx_feat = np.concatenate([self.tx_feat, np.zeros((pad, N_FEAT), np.float32)])
            self.head = np.concatenate([self.head, np.full((N_LIVE_REL, pad), -1, np.int32)], axis=1)
            self.w_cnt = np.concatenate([self.w_cnt, np.zeros((pad, 4), np.int32)])
            self.w_cur_ts = np.concatenate([self.w_cur_ts, np.full(pad, -1, np.int16)])
            self.w_n_cur = np.concatenate([self.w_n_cur, np.zeros(pad, np.int32)])
            self._ltx = np.full(new, -1, np.int64)
            self._lw = np.full(new, -1, np.int64)
            self.cap = new
        if n_e > self.e_cap:
            new = max(n_e, 2 * self.e_cap)
            pad = new - self.e_cap
            self.e_child = np.concatenate([self.e_child, np.zeros(pad, np.int32)])
            self.e_next = np.concatenate([self.e_next, np.zeros(pad, np.int32)])
            self.e_rel = np.concatenate([self.e_rel, np.zeros(pad, np.int8)])
            self.e_owner = np.concatenate([self.e_owner, np.zeros(pad, np.int32)])
            self.e_cap = new

    def reserve(self, n_addr: int) -> None:
        """Make room for addresses ``0 .. n_addr - 1``."""
        self._grow(max(self.n_tx, n_addr), self.n_e)

    # --- updates
    def add(self, b: TxBatch) -> None:
        """Append the batch's txs; they get tids ``n_tx .. n_tx + b.n - 1``."""
        top = max(int(b.in_w.max(initial=-1)), int(b.out_w.max(initial=-1))) + 1
        self._grow(max(self.n_tx + b.n, top), self.n_e + 2 * (len(b.in_w) + len(b.out_w) + len(b.par)))
        self.n_e = _add(self.n_tx, self.n_e, b.ts, b.feat, b.in_ptr, b.in_w, b.out_ptr, b.out_w, b.par_ptr, b.par,
                        self.tx_ts, self.tx_feat, self.head, self.w_cnt, self.w_cur_ts, self.w_n_cur, self.e_child,
                        self.e_next, self.e_rel, self.e_owner)
        self.n_tx += b.n

    def pop(self, b: TxBatch) -> None:
        """Undo the most recent :meth:`add` (which must have been of ``b``) exactly."""
        n_e0 = self.n_e - 2 * (len(b.in_w) + len(b.out_w) + len(b.par))
        _pop(n_e0, self.n_e, b.ts, b.in_ptr, b.in_w, b.out_ptr, b.out_w, b.par_ptr, b.par, self.tx_feat, self.head,
             self.w_cnt, self.w_cur_ts, self.w_n_cur, self.e_next, self.e_rel, self.e_owner)
        self.n_e = n_e0
        self.n_tx -= b.n

    def end_step(self, ts: int, n_addr: int) -> None:
        """Freeze the causal counts of time step ``ts`` (call when its episode ends)."""
        self.reserve(n_addr)
        self.snap[int(ts)] = self.w_cnt[:n_addr].copy()

    # --- features
    def counts_asof(self, addr: np.ndarray, ts: np.ndarray) -> np.ndarray:
        """Causal counts ``(n, 4)`` of ``addr`` as of the end of time step ``ts`` (the live counts for a step
        that has not ended)."""
        addr = np.asarray(addr, np.int64)
        ts = np.broadcast_to(np.asarray(ts, np.int64), addr.shape)
        out = np.zeros((len(addr), 4), np.int32)
        for t in np.unique(ts):
            m = ts == t
            s = self.snap.get(int(t))
            if s is None:
                out[m] = self.w_cnt[addr[m]]
            else:
                a = addr[m]
                ok = a < len(s)
                rows = np.zeros((len(a), 4), np.int32)
                rows[ok] = s[a[ok]]
                out[m] = rows
        return out

    # --- views
    def tx_views(self, tids: np.ndarray, t_cut: np.ndarray, seed: int, k1: int = 10, k2: int = 5,
                 cap: int = 512) -> Views:
        if cap > 1024:
            raise ValueError("cap must be <= 1024 (message key packing)")
        tids = np.ascontiguousarray(tids, np.int64)
        t_cut = np.ascontiguousarray(np.broadcast_to(np.asarray(t_cut, np.int64), tids.shape))
        nid, kind, hop, view, tgt, src, dst, typ = _tx_views(
            self.head, self.e_child, self.e_next, self.tx_ts, tids, t_cut, np.int64(seed & 0x7FFFFFFFFFFFFFFF),
            k1, k2, cap, self._ltx, self._lw)
        return Views(np.where(kind == KIND_TX, nid, nid + WBASE), hop, view, tgt, src, dst, typ,
                     np.zeros(len(src), np.float32), kind)

    def agent_views(self, a_ptr: np.ndarray, a_addr: np.ndarray, keys: np.ndarray, seed: int, k2: int = 5) -> Views:
        nid, kind, hop, view, tgt, src, dst, typ = _agent_views(
            self.head, self.e_child, self.e_next, self.tx_ts, np.ascontiguousarray(a_ptr, np.int64),
            np.ascontiguousarray(a_addr, np.int64), np.ascontiguousarray(keys, np.int64),
            np.int64(seed & 0x7FFFFFFFFFFFFFFF), k2, self._ltx)
        return Views(np.where(kind == KIND_W, nid + WBASE, nid), hop, view, tgt, src, dst, typ,
                     np.zeros(len(src), np.float32), kind)


def split_ids(v: Views) -> tuple[np.ndarray, np.ndarray]:
    """``(tids, addresses)`` of a live view's tx and wallet nodes, each in node order."""
    return v.node_g[v.kind == KIND_TX], v.node_g[v.kind == KIND_W] - WBASE
