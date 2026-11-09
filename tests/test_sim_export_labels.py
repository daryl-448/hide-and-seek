"""M4: label rules (E6), the shortcut guard (§9.4), and the export's prune-then-recompute (§9.2)."""

from dataclasses import replace

import numpy as np
import polars as pl
import pytest
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from hns.data.causal import COUNTS
from hns.eval import js as jsd
from hns.feats import wallet_feats
from hns.sim import labels
from hns.sim.env import SimEnv, run_static_world
from hns.sim.export import export_world, largest_components
from hns.sim.multiverse import draw_world
from hns.sim.world import CALIBRATED, ROLE_INNOCENT, ROLE_LAUNDERER, ROLE_SERVICE, WorldParams

SMALL = WorldParams(n_innocents=400, n_services=6, n_launderers=5, batch_size=5, consol_interval=3,
                    demand_rate=0.03, p_ill=0.6, p_lic=0.3, p_wunk=0.25)


@pytest.fixture(scope="module")
def world():
    env = run_static_world(SMALL, seed=11)
    return env, export_world(env)


def test_tx_label_rule():
    assert labels.tx_label(0.99, False, ROLE_SERVICE, 0.0, 0.0) == labels.LICIT        # services always licit
    assert labels.tx_label(0.1, False, ROLE_INNOCENT, 1.0, 0.2) == labels.LICIT
    assert labels.tx_label(0.3, False, ROLE_INNOCENT, 1.0, 0.2) == labels.UNKNOWN
    assert labels.tx_label(0.3, True, ROLE_LAUNDERER, 0.5, 1.0) == labels.ILLICIT
    assert labels.tx_label(0.7, True, ROLE_LAUNDERER, 0.5, 1.0) == labels.UNKNOWN


def test_wallet_label_rule_by_hand():
    links = pl.DataFrame({"wid": [0, 0, 1, 2, 3], "tid": [0, 1, 1, 2, 3]})
    cls = pl.DataFrame({"tid": [0, 1, 2, 3], "class": [1, 2, 3, 2]}, schema={"tid": pl.Int64, "class": pl.Int8})
    out = labels.wallet_labels(links, cls, 0.0, np.random.default_rng(0))
    assert dict(zip(out["wid"], out["class"])) == {0: 1, 1: 2, 2: 3, 3: 2}   # illicit wins; unknown-only stays unknown
    out = labels.wallet_labels(links, cls, 1.0, np.random.default_rng(0))
    assert dict(zip(out["wid"], out["class"])) == {0: 1, 1: 3, 2: 3, 3: 3}   # p_wunk only touches the licit rule


def test_export_wallet_labels_follow_e6(world):
    _, ex = world
    links = pl.concat([ex.addr_tx, ex.tx_addr.select("wid", "tid")]).unique()
    lc = (links.join(ex.txs.select("tid", "class"), on="tid").group_by("wid")
          .agg((pl.col("class") == 1).any().alias("ill"), (pl.col("class") == 2).any().alias("lic")))
    w = ex.wallets.select("wid", "class", "licit_rule").join(lc, on="wid")
    assert w.height == ex.wallets.height
    assert ((w["class"] == 1) == w["ill"]).all()                              # illicit <=> linked to an illicit tx
    assert (w.filter(pl.col("class") == 2)["lic"]).all()                      # licit => linked to a licit tx
    assert (w["licit_rule"] == (~w["ill"] & w["lic"])).all()
    rule = w.filter(pl.col("licit_rule"))
    frac_unk = float((rule["class"] == 3).mean())
    assert abs(frac_unk - SMALL.p_wunk) < 0.05                                # p_wunk share set to unknown
    t = ex.txs
    assert (t.filter(pl.col("role") == ROLE_SERVICE)["class"] == 2).all()
    assert (t.filter(pl.col("truth"))["class"].is_in([1, 3])).all()
    assert (t.filter(~pl.col("truth"))["class"].is_in([2, 3])).all()


def test_rate_solvers_hit_readme_targets():
    r = labels.solve_tx_rates(n=10_000, n_illicit=1_000, n_service=1_000, n_other=8_000)
    assert r["labelled_frac"] == pytest.approx(labels.TX_LABELLED) and r["illicit_share"] == pytest.approx(
        labels.TX_ILLICIT_SHARE)
    r = labels.solve_tx_rates(n=10_000, n_illicit=100, n_service=1_000, n_other=8_900)     # too few illicit txs
    assert r["p_ill"] == 1.0 and r["illicit_share"] < labels.TX_ILLICIT_SHARE
    w = labels.solve_p_wunk(n_wallets=10_000, n_illicit=200, n_licit_rule=4_000)
    assert w["labelled_frac"] == pytest.approx(labels.W_LABELLED) and 0 <= w["p_wunk"] <= 0.5
    w = labels.solve_p_wunk(n_wallets=10_000, n_illicit=200, n_licit_rule=9_000)          # clipped at 0.5
    assert w["p_wunk"] == 0.5 and w["shortfall"] < 0


def test_shortcut_guard(world):
    _, ex = world
    assert labels.shortcut_auc(ex.txs) <= labels.GUARD_MAX_AUC
    # positive control: a label that depends on a feature must fail the guard
    leaky = ex.txs.with_columns(
        pl.when(pl.col("size") > pl.col("size").median()).then(2).otherwise(3).cast(pl.Int8).alias("class"))
    assert labels.shortcut_auc(leaky) > 0.9


def test_largest_component_per_step_by_hand():
    #   ts 1: {0,1,2} chain and {3,4}; ts 2: {5,6} and {7,8} (tie: the one holding the lowest tid)
    edges = np.array([[0, 1], [1, 2], [3, 4], [5, 6], [7, 8]])
    ts = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2])
    assert np.flatnonzero(largest_components(9, edges, ts)).tolist() == [0, 1, 2, 5, 6]


def test_export_prunes_to_one_component_per_step(world):
    env, ex = world
    led = env.ledger
    n = ex.txs.height
    e = ex.tx_edges
    g = coo_matrix((np.ones(e.height), (e["src"].to_numpy(), e["dst"].to_numpy())), shape=(n, n))
    _, comp = connected_components(g, directed=True, connection="weak")
    per_ts = pl.DataFrame({"ts": ex.txs["ts"], "c": comp}).group_by("ts").agg(pl.col("c").n_unique())
    assert (per_ts["c"] == 1).all()
    assert (ex.steps["n_retained"] < ex.steps["n_live"]).all()
    # retained ltids are committed ledger txs; edges are exactly the ledger edges among them
    lt = ex.txs["ltid"].to_numpy()
    kept = {(int(a), int(b)) for a, b in led.edges if a in set(lt) and b in set(lt)}
    assert {(int(lt[a]), int(lt[b])) for a, b in e.iter_rows()} == kept


def test_export_recomputes_over_retained_txs_only(world):
    env, ex = world
    led = env.ledger
    lt = set(ex.txs["ltid"].to_list())
    # every link row describes a retained tx, and amounts match the ledger rows
    ins = [(t, a, v) for t, a, v in led.in_rows if t in lt]
    assert ex.inputs.height == len(ins) and ex.inputs["amount"].sum() == pytest.approx(sum(v for *_, v in ins))
    # some wallet appears in both retained and dropped txs: its features count retained txs only
    addr = ex.wallets["addr"].to_numpy()
    live_txs = {}
    for t, a, _ in led.in_rows + led.out_rows:
        live_txs.setdefault(a, set()).add(t)
    w = ex.wallets.with_columns(pl.Series("live", [len(live_txs[int(a)]) for a in addr]))
    assert (w["total_txs"] <= w["live"]).all() and (w["total_txs"] < w["live"]).any()
    kept = {int(a): len(live_txs[int(a)] & lt) for a in addr}
    assert w["total_txs"].to_list() == [kept[int(a)] for a in addr]
    # tx degrees = exported flow edges
    deg = ex.tx_edges.group_by("dst").len()
    d = ex.txs.select("tid", "in_txs_degree").join(deg.rename({"dst": "tid"}), on="tid", how="left").fill_null(0)
    assert (d["in_txs_degree"] == d["len"]).all()
    # D3 causal counts at each wallet's last step equal its lifetime counts in the export
    last = ex.causal.sort("wid", "ts").group_by("wid").last().sort("wid")
    lc = wallet_feats.link_counts(ex.addr_tx, ex.tx_addr, ex.txs).sort("wid")
    assert last["c_in"].to_list() == lc["num_txs_as_sender"].to_list()
    assert last["c_tot"].to_list() == lc["total_txs"].to_list()
    assert last["c_steps"].to_list() == lc["num_timesteps_appeared_in"].to_list()
    assert set(COUNTS) <= set(ex.causal.columns)
    # derived W features exist and the flow CSR has one row per retained tx
    assert {"self_loop_frac", "aa_in_degree", "aa_out_degree"} <= set(ex.wallets.columns)
    assert len(ex.graphs["tx_flow"]["out_indptr"]) == ex.txs.height + 1
    assert ex.txs.select(pl.all().is_null().any()).row(0) == (False,) * ex.txs.width


def test_self_loops_come_from_change_reuse(world):
    _, ex = world
    assert (ex.wallets["self_loop_frac"] > 0).any()


def test_multiverse_draws_stay_in_bounds():
    rng = np.random.default_rng(0)
    centre = WorldParams()
    for _ in range(50):
        p = draw_world(centre, rng)
        for k, (lo, hi, _) in CALIBRATED.items():
            assert lo <= getattr(p, k) <= hi
        assert p.n_launderers in (10, 25, 50) and p.budget in (0.02, 0.05, 0.10) and 0 <= p.offset <= 39
        assert (p.p_ill, p.p_lic, p.p_wunk) == (centre.p_ill, centre.p_lic, centre.p_wunk)
        assert abs(np.exp(p.fee_mu) / np.exp(centre.fee_mu) - 1) <= 0.1 + 1e-9
    # a world at the largest offset ends at the real data's last time step
    env = SimEnv(replace(SMALL, offset=39), 0)
    env.start_episode(9)
    assert env.ts == 49


def test_js():
    h = jsd.histogram(np.array([-5.0, 0.1, 0.5, 9.0]), 0.0, 1.0, 2)
    assert h.tolist() == [2, 2]                                  # outermost bins collect beyond the range
    assert jsd.js(np.array([1, 0]), np.array([0, 1])) == pytest.approx(1.0)
    assert jsd.js(np.array([3, 1]), np.array([3, 1])) == pytest.approx(0.0)
