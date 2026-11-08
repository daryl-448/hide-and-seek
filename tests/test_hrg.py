"""HRG views (SPEC §8.1): relation typing, causality, caps, dedup and per-target seeding."""

import numpy as np
import polars as pl
import pytest

from hns.data.graph import addr_addr, addr_tx_csr, tx_flow
from hns.models.hrg import (AA_IN, AA_OUT, FLOW_IN, FLOW_OUT, INVERSE, N_REL, TX_INW, TX_OUTW, W_INTX, W_OUTTX,
                            ViewSampler, build_relations)

N_TX, N_W = 60, 40


@pytest.fixture(scope="module")
def graph():
    rng = np.random.default_rng(0)
    ts = rng.integers(1, 6, N_TX).astype(np.int16)
    txs = pl.DataFrame({"tid": np.arange(N_TX, dtype=np.int32), "ts": ts, "size": np.ones(N_TX, np.float32)})
    pairs = {(a, b) for a, b in rng.integers(0, N_TX, (200, 2)) if a != b and ts[a] == ts[b]}  # flow: same ts
    edges = pl.DataFrame({"src": [a for a, _ in pairs], "dst": [b for _, b in pairs]}, schema={"src": pl.Int32, "dst": pl.Int32})
    ins = {(w, t) for w, t in zip(rng.integers(0, N_W, 150), rng.integers(0, N_TX, 150))}
    outs = {(w, t) for w, t in zip(rng.integers(0, N_W, 150), rng.integers(0, N_TX, 150))}
    a = pl.DataFrame({"wid": [w for w, _ in ins], "tid": [t for _, t in ins]}, schema={"wid": pl.Int32, "tid": pl.Int32})
    t = pl.DataFrame({"tid": [t for _, t in outs], "wid": [w for w, _ in outs]}, schema={"tid": pl.Int32, "wid": pl.Int32})
    from hns.data.graph import typed_links
    links = addr_tx_csr(typed_links(txs, a, t), N_TX, N_W)
    aa_rows = a.join(t, on="tid").select(pl.col("wid").alias("src"), pl.col("wid_right").alias("dst"))
    aa, _ = addr_addr(txs, a, t, aa_rows, N_W)
    flow = tx_flow(txs, edges)
    sets = {r: set() for r in range(N_REL)}
    for s, d in pairs:
        sets[FLOW_OUT].add((s, d)); sets[FLOW_IN].add((d, s))
    for w, x in ins:
        sets[TX_INW].add((x, N_TX + w)); sets[W_INTX].add((N_TX + w, x))
    for w, x in outs:
        sets[TX_OUTW].add((x, N_TX + w)); sets[W_OUTTX].add((N_TX + w, x))
    for s, d in aa_rows.filter(pl.col("src") != pl.col("dst")).iter_rows():
        sets[AA_OUT].add((N_TX + s, N_TX + d)); sets[AA_IN].add((N_TX + d, N_TX + s))
    return {"ts": ts, "flow": flow, "links": links, "aa": aa, "sets": sets}


def _check_messages(v, sets):
    seen = set()
    for s, d, ty in zip(v.src, v.dst, v.etype):
        gs, gd = v.node_g[s], v.node_g[d]
        assert v.view[s] == v.view[d]
        assert (gd, gs) in sets[ty], (gd, gs, ty)          # src is dst's neighbour under relation ty
        assert (gs, gd) in sets[INVERSE[ty]]
        key = (s, d, ty)
        assert key not in seen
        seen.add(key)


def test_w_view_types_dedup_and_cap(graph):
    rel = build_relations(N_TX, N_W, graph["flow"], graph["links"], graph["aa"], graph["ts"])
    sm = ViewSampler(rel, tuple(range(N_REL)), causal=False, k1=10, k2=5, cap=25)
    targets = N_TX + np.arange(N_W)
    v = sm.sample(targets, None, seed=3)
    _check_messages(v, graph["sets"])
    assert np.bincount(v.view).max() <= 25
    assert (v.node_g[v.target] == targets).all() and (v.hop[v.target] == 0).all()
    for i in range(len(targets)):  # one node per global id within a view
        g = v.node_g[v.view == i]
        assert len(np.unique(g)) == len(g)
    assert (sm._local == -1).all()


def test_t_view_causal_and_no_addr_addr(graph):
    ts = graph["ts"]
    rel = build_relations(N_TX, N_W, graph["flow"], graph["links"], None, ts)
    sm = ViewSampler(rel, (FLOW_OUT, FLOW_IN, TX_INW, TX_OUTW, W_INTX, W_OUTTX), causal=True)
    targets = np.arange(N_TX)
    v = sm.sample(targets, ts[targets], seed=1)
    _check_messages(v, graph["sets"])
    is_tx = v.node_g < N_TX
    assert (ts[v.node_g[is_tx]] <= ts[targets][v.view[is_tx]]).all()
    assert not np.isin(v.etype, [AA_OUT, AA_IN]).any()
    # without the cut, some views do reach later txs, so the check above is not vacuous
    sm_nc = ViewSampler(rel, (FLOW_OUT, FLOW_IN, TX_INW, TX_OUTW, W_INTX, W_OUTTX), causal=False)
    u = sm_nc.sample(targets, None, seed=1)
    u_tx = u.node_g < N_TX
    assert (ts[u.node_g[u_tx]] > ts[targets][u.view[u_tx]]).any()


def test_view_seeded_per_target_not_per_batch(graph):
    rel = build_relations(N_TX, N_W, graph["flow"], graph["links"], graph["aa"], graph["ts"])
    sm = ViewSampler(rel, tuple(range(N_REL)), causal=False, k1=2, k2=2)
    a = sm.sample(np.array([N_TX + 5, N_TX + 7]), None, seed=9)
    b = sm.sample(np.array([N_TX + 7]), None, seed=9)
    assert sorted(a.node_g[a.view == 1].tolist()) == sorted(b.node_g.tolist())
