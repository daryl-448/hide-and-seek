"""M5: the live HRG equals the committed ledger, and its views follow the T-view rules (SPEC §8.1, §9.3)."""

from dataclasses import replace

import numpy as np
import polars as pl
import pytest

from hns.data.load import AUG_FEATURES
from hns.feats import tx_feats
from hns.models.hrg import FLOW_IN, FLOW_OUT, INVERSE, TX_INW, TX_OUTW, W_INTX, W_OUTTX
from hns.sim.env import SimEnv
from hns.sim.export import export_world
from hns.sim.live import (ADDR_CTRL, CTRL_ADDR, KIND_CTRL, KIND_TX, KIND_W, WBASE, LiveGraph, _reservoir, keep_mask,
                          split_ids)
from hns.sim.world import EPISODES, WorldParams

SMALL = WorldParams(n_innocents=300, n_services=5, n_launderers=5, batch_size=5, consol_interval=3,
                    demand_rate=0.03, p_ill=0.5, p_lic=0.3, p_wunk=0.2, budget=0.10, offset=7)


@pytest.fixture(scope="module")
def world():
    env = SimEnv(SMALL, seed=5)       # RandomDetector at b = 10 %: licit flags are reverted every sub-step
    env.graph = LiveGraph(cap=64, e_cap=64)   # tiny capacities: growth is exercised
    env.run_world()
    return env


def _rows(g, r, owner):
    out, e = [], g.head[r, owner]
    while e >= 0:
        out.append(int(g.e_child[e]))
        e = g.e_next[e]
    return out


def _ledger_sets(led):
    ins = {(t, a) for t, a, _ in led.in_rows}
    outs = {(t, a) for t, a, _ in led.out_rows}
    return ins, outs


def test_graph_equals_ledger_after_reverts(world):
    led, g = world.ledger, world.graph
    assert g.n_tx == led.n_tx and (g.tx_ts[:g.n_tx] == np.array(led.tx_ts)).all()
    ins, outs = _ledger_sets(led)
    got = {r: set() for r in range(6)}
    for t in range(g.n_tx):
        for r in (FLOW_OUT, FLOW_IN, TX_INW, TX_OUTW):
            row = _rows(g, r, t)
            assert len(row) == len(set(row))                  # distinct neighbours
            got[r] |= {(t, c) for c in row}
    for a in range(len(led.addr_role)):
        for r in (W_INTX, W_OUTTX):
            row = _rows(g, r, a)
            assert len(row) == len(set(row))
            assert row == sorted(row, reverse=True)           # newest first: a ts cut keeps a suffix
            got[r] |= {(a, c) for c in row}
    assert got[FLOW_OUT] == set(led.edges) and got[FLOW_IN] == {(b, a) for a, b in led.edges}
    assert got[TX_INW] == ins and got[W_INTX] == {(a, t) for t, a in ins}
    assert got[TX_OUTW] == outs and got[W_OUTTX] == {(a, t) for t, a in outs}
    assert g.n_e == sum(len(s) for s in got.values())         # no leftover edge of a popped tentative tx


def test_live_features_match_the_shared_definitions(world):
    led, g = world.ledger, world.graph
    f = lambda rows: pl.DataFrame({"tid": [r[0] for r in rows], "wid": [r[1] for r in rows],  # noqa: E731
                                   "amount": [r[2] for r in rows]})
    edges = pl.DataFrame({"src": [a for a, _ in led.edges], "dst": [b for _, b in led.edges]},
                         schema={"src": pl.Int64, "dst": pl.Int64})
    want = tx_feats.tx_features(pl.Series("tid", np.arange(led.n_tx)), f(led.in_rows), f(led.out_rows), edges)
    want = want.sort("tid").select(AUG_FEATURES).to_numpy()
    np.testing.assert_allclose(g.tx_feat[:g.n_tx], want, rtol=2e-5, atol=1e-9)
    # and, on the retained txs, the export's features (degrees are within one component)
    ex = export_world(world)
    np.testing.assert_allclose(g.tx_feat[ex.txs["ltid"].to_numpy()], ex.txs.select(AUG_FEATURES).to_numpy(),
                               rtol=2e-5, atol=1e-9)


def test_causal_counts_live_and_as_of_each_step(world):
    led, g = world.ledger, world.graph
    ins, outs = _ledger_sets(led)
    ts = np.array(led.tx_ts)
    n_addr = len(led.addr_role)
    steps = sorted(set(ts.tolist()))
    assert sorted(g.snap) == steps == [SMALL.offset + 1 + e for e in range(EPISODES)]
    for t_cut in (steps[0], steps[4], steps[-1]):
        want = np.zeros((n_addr, 4), np.int64)
        seen_steps = [set() for _ in range(n_addr)]
        both = [set() for _ in range(n_addr)]
        for (t, a) in ins:
            if ts[t] <= t_cut:
                want[a, 0] += 1; both[a].add(t); seen_steps[a].add(ts[t])
        for (t, a) in outs:
            if ts[t] <= t_cut:
                want[a, 1] += 1; both[a].add(t); seen_steps[a].add(ts[t])
        want[:, 2] = [len(b) for b in both]
        want[:, 3] = [len(s) for s in seen_steps]
        got = g.counts_asof(np.arange(n_addr), np.full(n_addr, t_cut))
        assert (got == want).all()
    assert (g.w_cnt[:n_addr] == g.snap[steps[-1]]).all()
    # an address created after a step has zero counts as of that step
    late = n_addr - 1
    assert (g.counts_asof(np.array([late]), np.array([steps[0]])) == 0).all() or len(g.snap[steps[0]]) > late


def _check_view(v, g, led, t_cut, sets):
    tids, addrs = split_ids(v)
    assert (np.array(led.tx_ts)[tids] <= t_cut[v.view[v.kind == KIND_TX]]).all()      # causal
    seen = set()
    for s, d, ty in zip(v.src, v.dst, v.etype):
        assert v.view[s] == v.view[d]
        ks = lambda i: int(v.node_g[i] - WBASE) if v.kind[i] == KIND_W else int(v.node_g[i])  # noqa: E731
        assert (ks(d), ks(s)) in sets[int(ty)]                 # src is dst's neighbour under relation ty
        assert (ks(s), ks(d)) in sets[int(INVERSE[ty])]
        assert (s, d, ty) not in seen
        seen.add((s, d, ty))


def test_tx_views_are_causal_typed_capped_and_seeded(world):
    led, g = world.ledger, world.graph
    ins, outs = _ledger_sets(led)
    sets = {FLOW_OUT: set(led.edges), FLOW_IN: {(b, a) for a, b in led.edges}, TX_INW: ins, TX_OUTW: outs,
            W_INTX: {(a, t) for t, a in ins}, W_OUTTX: {(a, t) for t, a in outs}}
    rng = np.random.default_rng(0)
    tids = rng.choice(led.n_tx, 200, replace=False)
    t_cut = np.array(led.tx_ts)[tids]
    v = g.tx_views(tids, t_cut, seed=3, k1=10, k2=5, cap=40)
    _check_view(v, g, led, t_cut, sets)
    assert np.bincount(v.view).max() <= 40 and (v.node_g[v.target] == tids).all()
    assert (v.hop[v.target] == 0).all() and v.hop.max() == 2
    # a target's view does not depend on its batch
    one = g.tx_views(tids[7:8], t_cut[7:8], seed=3, k1=10, k2=5, cap=40)
    assert (one.node_g == v.node_g[v.view == 7]).all()
    # analysis depth 1: no hop-2 node
    d1 = g.tx_views(tids, t_cut, seed=3, k1=10, k2=0, cap=40)
    assert d1.hop.max() == 1 and len(d1.hop) < len(v.hop)


def test_reservoir_is_uniform_over_the_causal_prefix(world):
    led, g = world.ledger, world.graph
    hub = max(range(len(led.addr_role)), key=lambda a: len(_rows(g, W_OUTTX, a)))
    ts = np.array(led.tx_ts)
    t_cut = int(np.median(ts[_rows(g, W_OUTTX, hub)]))
    row = [t for t in _rows(g, W_OUTTX, hub) if ts[t] <= t_cut]
    assert len(row) > 30 and len(row) < len(_rows(g, W_OUTTX, hub))
    counts, ch = np.zeros(led.n_tx), np.empty(10, np.int64)
    for s in range(2000):
        m = _reservoir(g.head, g.e_child, g.e_next, W_OUTTX, hub, g.tx_ts, t_cut, 10, np.uint64(s * 7919 + 1), ch)
        assert m == 10 and len(set(ch.tolist())) == 10
        counts[ch] += 1
    assert counts.sum() == counts[row].sum()                    # never a tx after the cut
    p = counts[row] / counts.sum()
    assert abs(p - 1 / len(row)).max() < 4 * np.sqrt(1 / len(row) / 20000)


def test_agent_views(world):
    led, g = world.ledger, world.graph
    own = [world.inn_addrs[i] for i in range(20)]
    a_ptr = np.cumsum([0] + [len(o) for o in own])
    v = g.agent_views(a_ptr, np.concatenate(own), np.arange(20), seed=1, k2=5)
    ins, outs = _ledger_sets(led)
    assert (v.kind[v.target] == KIND_CTRL).all() and (np.bincount(v.view, v.kind == KIND_CTRL) == 1).all()
    for j in range(20):
        nodes = np.flatnonzero(v.view == j)
        addrs = v.node_g[nodes][v.kind[nodes] == KIND_W] - WBASE
        assert addrs.tolist() == own[j]                         # every controlled address, in order, at hop 1
    for s, d, ty in zip(v.src, v.dst, v.etype):
        ks, kd = v.kind[s], v.kind[d]
        if ty == CTRL_ADDR:
            assert ks == KIND_W and kd == KIND_CTRL
        elif ty == ADDR_CTRL:
            assert ks == KIND_CTRL and kd == KIND_W
        elif ty == W_INTX:                                      # tx -> address: the address is an input of the tx
            assert (int(v.node_g[s]), int(v.node_g[d] - WBASE)) in ins
        elif ty == W_OUTTX:
            assert (int(v.node_g[s]), int(v.node_g[d] - WBASE)) in outs
        elif ty == TX_INW:
            assert (int(v.node_g[d]), int(v.node_g[s] - WBASE)) in ins
        else:
            assert ty == TX_OUTW and (int(v.node_g[d]), int(v.node_g[s] - WBASE)) in outs
    assert (v.hop[v.kind == KIND_TX] == 2).all() and (v.kind != KIND_TX).sum() == 20 + a_ptr[-1]


def test_tentative_txs_are_visible_to_the_detector_then_gone():
    seen = {}

    class Peek:
        def act_global(self, env, state):
            return {"analysis_depth": 0, "update_strategy": 1}

        def score(self, env, cands):
            g = env.graph
            seen["n"] = (g.n_tx, env.ledger.n_tx, len(cands))
            if cands:                                           # the tentative txs are the last len(cands) tids
                tids = np.arange(g.n_tx - len(cands), g.n_tx)
                v = g.tx_views(tids, np.full(len(tids), env.ts), seed=0)
                seen["feat"] = g.tx_feat[tids].copy()
                seen["sizes"] = np.bincount(v.view)
            return np.ones(len(cands))                          # flag in order

    env = SimEnv(replace(SMALL, budget=0.2), seed=2, detector=Peek())
    env.graph = LiveGraph()
    env.start_episode(0)
    for _ in range(6):
        before = env.ledger.n_tx
        res = env.step()
        n_g, n_led, n_c = seen["n"]
        assert n_led == before and n_g == before + n_c          # scored on s_t plus the tentative txs
        assert env.graph.n_tx == env.ledger.n_tx                # reverted ones are gone again
        if n_c:
            assert (seen["feat"][:, 4] > 0).all() and (seen["sizes"] >= 3).all()   # size; tx + in + out wallet
    assert any(c["outcome"] == "reverted" for c in res.candidates)


def test_keep_mask_is_a_function_of_seed_node_and_feature():
    node = np.arange(5000) + WBASE
    seeds = np.full(5000, 11)
    m = keep_mask(seeds, node, 17, 0.2)
    assert abs((~m).mean() - 0.2) < 0.01
    assert (keep_mask(seeds[100:110], node[100:110], 17, 0.2) == m[100:110]).all()     # batch-independent
    assert (keep_mask(seeds + 1, node, 17, 0.2) != m).any()                            # a new episode: a new draw
    assert keep_mask(seeds, node, 17, 0.0).all()
