"""M5: BDQ soft-Q pieces, replay chains, DGDA-lite heads, observations and the MARL loop (SPEC §8.3, §10)."""

import json
import math
from dataclasses import replace

import numpy as np
import pytest
import torch

import hns.marl.train as mt
from hns.data.norm import Normalizer
from hns.marl.buffer import PENDING, TERMINAL, Replay, VecReplay
from hns.marl.learner import MarlConfig, RealViews
from hns.marl.obs import (BRANCHES, D_CAND, I_CTRL, L_CTRL, Featurizer, ObsBatch, collate, detector_obs, identity_stats,
                          innocent_obs, launderer_obs, pack_masks, world_readout)
from hns.models.detector import Detector, Graph, kl_normal, target_rows
from hns.models.hrg import Views
from hns.models.qnet import AgentQ, BranchHeads, root_value, sample_actions, soft_value, td_loss
from hns.sim.entities import K_RECIPIENTS
from hns.sim.env import SimEnv
from hns.sim.live import KIND_CTRL, KIND_TX, KIND_W, LiveGraph
from hns.sim.world import WorldParams

TINY = WorldParams(n_innocents=80, n_services=3, n_launderers=4, batch_size=4, consol_interval=3, demand_rate=0.05,
                   p_ill=0.6, p_lic=0.5, p_wunk=0.1)
NEG = float("-inf")


# ---------------------------------------------------------------------------- soft-Q pieces
def test_branch_heads_are_dueling_over_feasible_options():
    torch.manual_seed(0)
    h = BranchHeads(6, {"a": 3, "ptr": 4}, pointer={"ptr": 5})
    x, cand = torch.randn(2, 6), torch.randn(2, 4, 5)
    masks = torch.tensor([[True, False, True, True, True, False, False], [True] * 7])
    q = h(x, masks, {"ptr": cand})
    assert q.shape == (2, 7) and (q[~masks] == NEG).all() and torch.isfinite(q[masks]).all()
    v = h.value(h.trunk(x))
    # mean over the feasible options of a branch is V(s): Q = V + A - mean_feasible(A)
    assert torch.allclose(q[0, [0, 2]].mean(), v[0, 0], atol=1e-6)
    assert torch.allclose(q[0, 3:5].mean(), v[0, 0], atol=1e-6)
    assert torch.allclose(q[1, 3:].mean(), v[1, 0], atol=1e-6)
    # the pointer branch depends on the candidates' features
    assert not torch.allclose(h(x, masks, {"ptr": cand + 1.0})[1, 3:], q[1, 3:])


def test_soft_value_by_hand_and_max_limit():
    sizes = [2, 3]
    q = torch.tensor([[1.0, 2.0, 0.5, NEG, 1.5], [0.0, 0.0, 3.0, 1.0, NEG]])
    active = torch.tensor([[True, True], [True, False]])
    tau = 0.5
    want0 = 0.5 * (tau * math.log(math.exp(1 / tau) + math.exp(2 / tau))
                   + tau * math.log(math.exp(0.5 / tau) + math.exp(1.5 / tau)))
    want1 = tau * math.log(2.0)                              # only the first branch is in B(s')
    got = soft_value(q, sizes, active, tau)
    assert torch.allclose(got, torch.tensor([want0, want1]), atol=1e-6)
    # tau -> 0 is Algorithm 1's max, averaged over the active branches
    assert torch.allclose(soft_value(q, sizes, active, 1e-4), torch.tensor([(2.0 + 1.5) / 2, 0.0]), atol=1e-3)


def test_td_loss_uses_realized_branches_only():
    sizes = [2, 3]
    q = torch.tensor([[1.0, 2.0, 0.5, NEG, 1.5], [0.0, 4.0, 3.0, 1.0, NEG]], requires_grad=True)
    action = torch.tensor([[1, 2], [0, 2]])                  # row 1's second option is infeasible, but unrealized
    realized = torch.tensor([[True, True], [True, False]])
    y = torch.tensor([1.0, 2.0])
    loss = td_loss(q, sizes, action, realized, y)
    want = 0.5 * (((1 - 2.0) ** 2 + (1 - 1.5) ** 2) / 2 + (2 - 0.0) ** 2)
    assert math.isclose(loss.item(), want, rel_tol=1e-6)
    loss.backward()
    assert torch.isfinite(q.grad).all() and q.grad[1, 2:].abs().sum() == 0


def test_sample_actions_masks_eps_and_greedy_limit():
    rng = np.random.default_rng(0)
    q = np.tile(np.array([[0.1, 0.5, NEG, 0.2, 0.0, 1.0]]), (4000, 1))
    a = sample_actions(q, [3, 3], tau=0.01, eps=0.0, rng=rng)
    assert (a[:, 0] == 1).all() and (a[:, 1] == 2).all()          # tau small: (nearly) the max
    a = sample_actions(q, [3, 3], tau=0.01, eps=1.0, rng=rng)
    assert set(a[:, 0]) == {0, 1} and abs((a[:, 0] == 0).mean() - 0.5) < 0.03   # uniform over the feasible
    assert abs(np.bincount(a[:, 1], minlength=3) / 4000 - 1 / 3).max() < 0.03
    a = sample_actions(q, [3, 3], tau=0.4, eps=0.0, rng=rng)       # softmax(Q / tau)
    p = np.exp(np.array([0.1, 0.5]) / 0.4)
    assert abs((a[:, 0] == 1).mean() - p[1] / p.sum()) < 0.03


# ---------------------------------------------------------------------------- replay
def test_replay_chains_terminals_and_eviction():
    rng = np.random.default_rng(0)
    buf = Replay(4, [2, 3])
    m, act = np.ones(5, bool), np.array([True, False])
    a = buf.add("s0", m, act)
    assert buf.ready == 0 and buf.nxt[0] == PENDING
    buf.set(a, [1, 0], [True, False], 0.5)
    buf.add_reward(a, 0.25)
    b = buf.add("s1", m, np.array([True, True]))
    buf.link(a, b)
    buf.set(b, [0, 2], [True, True], -1.0)
    assert buf.ready == 1
    s = buf.sample(8, rng)
    assert s["obs"] == ["s0"] * 8 and s["next_obs"] == ["s1"] * 8 and not s["done"].any()
    assert (s["reward"] == 0.75).all() and (s["next_active"] == [True, True]).all() and (s["action"] == [1, 0]).all()
    buf.end(b)
    assert buf.ready == 2 and buf.nxt[1] == TERMINAL
    s = buf.sample(64, rng)
    assert sum(s["done"]) == sum(o == "s1" for o in s["obs"]) and len(s["next_obs"]) == sum(~s["done"])
    # the ring keeps the newest entries; a link from an evicted entry is ignored
    for k in range(4):
        buf.add(f"n{k}", m, act)
    assert not buf.alive(a) and not buf.alive(b) and buf.ready == 0
    buf.link(a, 5)
    assert (buf.nxt[:4] == PENDING).all()


def test_vec_replay_sequence():
    rng = np.random.default_rng(0)
    v = VecReplay(8, 2, [2, 3])
    for k in range(3):
        v.add(np.full(2, k, np.float32), np.array([k % 2, k]))
        v.set_reward(float(k))
    assert v.ready == 2                                        # the newest entry has no next state yet
    v.end()
    s = v.sample(64, rng)
    assert v.ready == 3 and (s["done"] == (s["state"][:, 0] == 2)).all()
    live = ~s["done"]
    assert (s["next_state"][live, 0] == s["state"][live, 0] + 1).all() and (s["reward"] == s["state"][:, 0]).all()


# ---------------------------------------------------------------------------- DGDA-lite heads
def _one_node_graph(n):
    v = Views(node_g=np.arange(n), hop=np.zeros(n, np.int8), view=np.arange(n, dtype=np.int32), target=np.arange(n),
              src=np.zeros(0, np.int64), dst=np.zeros(0, np.int64), etype=np.zeros(0, np.int8),
              logmult=np.zeros(0, np.float32))
    return Graph(v, n_tx=n, device="cpu")


def test_da_heads_and_aux_losses():
    torch.manual_seed(0)
    det = Detector(5, 3, da=True, d_rec=5, flag_q=True)
    g, x = _one_node_graph(6), torch.randn(6, 5)
    r = det.encode(g, x, torch.zeros(0, 3))
    h = det.heads(r)
    assert h["z_s"].shape == (6, 256) and h["z_d"].shape == (6, 64) and h["z_n"].shape == (6, 128)
    domain = torch.tensor([0, 0, 0, 1, 1, 1])
    aux = det.aux_losses(h, domain, x[target_rows(g, True)], beta=0.01)
    assert set(aux) == {"kl_d", "dom", "kl_sn", "rec"} and all(torch.isfinite(v) and v >= 0 for v in aux.values())
    sum(aux.values()).backward()
    assert det.mu_dom.grad.abs().sum() > 0 and det.decoder[0].weight.grad.abs().sum() > 0
    # KL to the prior is zero at the prior, and uses the row's domain mean
    mu = torch.randn(4, 3)
    assert torch.allclose(kl_normal(mu, torch.zeros(4, 3), mu), torch.zeros(4))
    assert torch.allclose(kl_normal(mu, torch.zeros(4, 3)), 0.5 * mu.pow(2).sum(1))
    # select_tx head: dueling over (skip, flag)
    q = det.flag_q(r)
    assert q.shape == (6, 2) and torch.allclose(q.mean(1), det.flag_head(r)[:, 0], atol=1e-6)
    # eval: means, so the heads are deterministic
    det.eval()
    assert torch.equal(det.heads(r)["z_d"], det.heads(r)["zd_mu"])
    # a detector without DA heads has the T0 parameters only
    assert not any(k.startswith(("zd_", "zn_", "domain", "decoder", "mu_dom", "flag_head"))
                   for k in Detector(5, 3).state_dict())


# ---------------------------------------------------------------------------- observations
@pytest.fixture(scope="module")
def played():
    env = SimEnv(TINY, seed=3)
    env.graph = LiveGraph()
    env.start_episode(0)
    for _ in range(4):
        env.step()
    env._refresh_state()
    states = [env.launderer_state(L) for L in env.launderers]
    assert sum(st is not None and len(st.branches) == 1 for st in states) >= 1      # a tranche in flight
    assert sum(st is not None and len(st.branches) == 6 for st in states) >= 1      # funds at the source
    return env


def test_agent_observations(played):
    env, fz = played, Featurizer(identity_stats())
    gvec = world_readout(env, {"n_cand": 12, "n_flag": 3, "n_frozen": 1, "episode_committed": 40, "depth": 2})
    assert gvec.shape == (8,) and gvec[4] == 0.25 and gvec[7] == 1.0
    assert math.isclose(gvec[6], math.log1p(40) / 10, rel_tol=1e-6) and world_readout(env, {})[6] == 0.0
    Ls = {L.id: st for L in env.launderers if (st := env.launderer_state(L)) is not None}
    ids = list(Ls)
    ob = launderer_obs(env, fz, ids, gvec, seed=0)
    assert ob.x_c.shape == (len(ids), L_CTRL) and ob.cand.shape == (len(ids), K_RECIPIENTS, D_CAND)
    assert (ob.x_c[:, -8:] == gvec).all() and (ob.cand[:, env.p.n_services:] == 0).all()
    assert (ob.v.kind == KIND_CTRL).sum() == len(ids) and ob.x_w.shape[1] == 8 and ob.x_tx.shape[1] == 17
    for j, i in enumerate(ids):                              # the source is the first controlled address
        L = env.launderers[i]
        first = np.flatnonzero((ob.v.view == j) & (ob.v.kind == KIND_W))[0]
        row = (ob.v.kind[:first] == KIND_W).sum()
        assert ob.x_w[row, 6] == 1.0 and math.isclose(ob.x_w[row, 4], math.log1p(env.ledger.balance[L.source]), rel_tol=1e-5)
        assert ob.x_c[j, 0] == (L.tranche is not None)
    masks, active = pack_masks("launderer", Ls, ids)
    assert masks.shape == (len(ids), 35)
    for j, i in enumerate(ids):
        assert active[j].sum() == len(Ls[i].branches)
        if len(Ls[i].branches) == 1:
            assert masks[j].all()                            # branches outside B(s): all-true placeholders
        else:
            assert (masks[j, 6:22] == Ls[i].masks["recipient"]).all()
    Is = {i: st for i in range(env.p.n_innocents) if (st := env.innocent_state(i)) is not None}
    ib = innocent_obs(env, fz, list(Is), gvec, seed=0)
    assert ib.x_c.shape == (len(Is), I_CTRL) and ib.cand is None and len(Is) > 0
    # a view never shows another agent's controller, and the detector's view has none at all
    tids = np.arange(max(env.graph.n_tx - 20, 0), env.graph.n_tx)
    assert len(tids) >= 10
    db = detector_obs(env.graph, fz, tids, env.ts, 0, 10, 5, 256)
    assert (db.v.kind != KIND_CTRL).all() and db.x_c is None and db.x_w.shape[1] == 4
    assert (np.bincount(ob.v.view, ob.v.kind == KIND_CTRL) == 1).all()


def test_split_then_collate_is_the_same_batch(played):
    env, fz = played, Featurizer(identity_stats())
    Ls = {L.id: st for L in env.launderers if (st := env.launderer_state(L)) is not None}
    ob = launderer_obs(env, fz, list(Ls), np.zeros(8, np.float32), seed=0)
    back = collate(ob.split())
    for f in ("hop", "view", "target", "src", "dst", "etype", "kind"):
        assert (getattr(back.v, f) == getattr(ob.v, f)).all(), f
    assert (back.x_tx == ob.x_tx).all() and (back.x_w == ob.x_w).all() and (back.x_c == ob.x_c).all()
    assert (back.cand == ob.cand).all()
    # and the network accepts it
    torch.manual_seed(0)
    net = AgentQ(BRANCHES["launderer"], 17, 8, L_CTRL, pointer={"recipient": D_CAND}).eval()
    masks, _ = pack_masks("launderer", Ls, list(Ls))
    g, f = back.tensors("cpu")
    q = net(g, f["x_tx"], f["x_w"], f["x_c"], torch.from_numpy(masks), torch.from_numpy(back.cand))
    assert q.shape == masks.shape and torch.isfinite(q[torch.from_numpy(masks)]).all()


# ---------------------------------------------------------------------------- real views (T isolation)
class _FakeT:
    n_tx = 50

    def __init__(self):
        rng = np.random.default_rng(0)
        self.tx_ts = rng.integers(1, 50, 50).astype(np.int16)
        self.x_tx = rng.standard_normal((50, 17)).astype(np.float32)
        self.x_tx[:5] = np.nan                                # null-feature txs (E7)
        self.seen = []

    def batch(self, g, t, seed):
        self.seen.append((g.copy(), t.copy()))
        n = len(g)
        v = Views(node_g=np.asarray(g, np.int64), hop=np.zeros(n, np.int8), view=np.arange(n, dtype=np.int32),
                  target=np.arange(n), src=np.zeros(0, np.int64), dst=np.zeros(0, np.int64),
                  etype=np.zeros(0, np.int8), logmult=np.zeros(0, np.float32))
        return v, {"x_tx": self.x_tx[g], "x_anon": np.zeros((n, 3), np.float32), "x_w": np.zeros((0, 4), np.float32)}


def test_real_t_views_come_from_the_fitting_window_only():
    data = _FakeT()
    rv = RealViews.task_t(data)
    assert (data.tx_ts[rv.g] <= 29).all() and (rv.g >= 5).all() and len(rv.g) == ((data.tx_ts <= 29)[5:]).sum()
    ob = rv.sample(64, np.random.default_rng(0), seed=1)
    g, t = data.seen[0]
    assert (t == data.tx_ts[g]).all() and (t <= 29).all()     # each view is cut at its own time step
    assert ob.v.kind is not None and ob.x_tx.shape == (64, 17) and not hasattr(ob, "x_anon")


# ---------------------------------------------------------------------------- the loop
class _FakeW(_FakeT):
    n_tx = 50

    def batch(self, g, t, seed):
        n = len(g)
        v = Views(node_g=np.asarray(g, np.int64), hop=np.zeros(n, np.int8), view=np.arange(n, dtype=np.int32),
                  target=np.arange(n), src=np.zeros(0, np.int64), dst=np.zeros(0, np.int64),
                  etype=np.zeros(0, np.int8), logmult=np.zeros(0, np.float32))
        self.seen.append(g.copy())
        return v, {"x_tx": np.zeros((0, 17), np.float32), "x_w": np.ones((n, 58), np.float32)}


def _trainer(tmp_path, monkeypatch, arm, episodes=4, learn=mt.ROLES, fixed_eps=None, out_name=None, **kw):
    monkeypatch.setattr(mt, "EPISODES", 2)
    monkeypatch.setattr(mt, "SUBSTEPS", 12)
    monkeypatch.setattr(mt, "draw_world", lambda c, rng: replace(c, mask_rate=0.2, offset=int(rng.integers(0, 40))))
    st = identity_stats()
    cfg = MarlConfig(arm=arm, seed=0, episodes=episodes, warmup=30, batch=8, buffer=400, ckpt_worlds=1, **kw)
    real_t = real_w = None
    if arm == "full":
        ft, fw = _FakeT(), _FakeW()
        real_t = RealViews.task_t(ft)
        real_w = RealViews(fw, 50 + np.arange(20), np.full(20, -1))
    out = tmp_path / (out_name or arm)
    out.mkdir(exist_ok=True)
    return mt.Trainer(cfg, torch.device("cpu"), TINY, st, Normalizer(st), real_t, real_w, out, log=lambda *_: None,
                      learn=learn, fixed_eps=fixed_eps)


def test_nda_run_transitions_health_and_resume(tmp_path, monkeypatch):
    tr = _trainer(tmp_path, monkeypatch, "nda")
    res = tr.run()
    out = tmp_path / "nda"
    recs = [json.loads(x) for x in (out / "metrics.jsonl").read_text().splitlines()]
    epochs = [r for r in recs if r["kind"] == "epoch"]
    assert res["episodes"] == 4 and len(epochs) == 2 and sum(r["kind"] == "episode" for r in recs) == 4
    e = epochs[-1]
    assert e["updates"]["detector_t"] > 0 and e["updates"]["detector_w"] == e["updates"]["detector_t"]
    assert all(math.isfinite(v) for v in e["losses"].values())
    assert {"t_td_flag", "t_y_src", "t_adv", "w_y_src", "w_adv", "td_innocent"} <= set(e["losses"])
    assert not any(k.endswith(("kl_d", "dom", "kl_sn", "rec")) for k in e["losses"])       # NDA: no L_aux
    assert 0 <= e["detection_rate"] <= 1 and 0 <= e["success_frac"] <= 1 and e["degenerate"] == []
    assert math.isclose(sum(e["strategy_share"].values()), 1.0, rel_tol=1e-9)
    # where the proceeds went closes: cashed + frozen (change included) + fees + held
    assert abs(e["accounting_gap"]) < 1e-9 and 0 <= e["frozen_change_frac"] <= e["frozen_frac"]
    assert math.isclose(e["btc_utility"], e["success_frac"] - e["frozen_frac"], abs_tol=1e-12)
    # flag transitions: terminal, reward from the flag's outcome at the depth used
    fb = tr.det.flag_buf
    n = min(fb.n, fb.cap)
    assert (fb.nxt[:n] == TERMINAL).all()
    allowed = np.array([0.0, -1.0, 0.99, 0.98, -0.11, -0.12], np.float32)
    assert np.isclose(fb.reward[:n, None], allowed[None]).any(1).all()
    flagged = fb.action[:n, 0] == 1
    assert np.isclose(fb.reward[:n][flagged, None], allowed[None, 2:]).any(1).all()
    assert np.isclose(fb.reward[:n][~flagged, None], allowed[None, :2]).any(1).all()
    # role transitions: after a world, nothing is pending and every chain ended in a terminal
    for L in tr.roles.values():
        k = min(L.buf.n, L.buf.cap)
        assert (L.buf.nxt[:k] != PENDING).all() and (L.buf.nxt[:k] == TERMINAL).any() and not L.tail
        assert (L.buf.realized[:k].sum(1) >= 1).all() and (L.buf.realized[:k] <= L.buf.active[:k]).all()
    # resume from the checkpoint written after world 1 replays world 2 to the same state
    assert (out / "ckpt" / "final.pt").exists() and (out / "result.json").exists()
    final = {k: v.clone() for k, v in tr.det.det_t.state_dict().items()}
    ck = torch.load(out / "ckpt" / "last.pt", map_location="cpu", weights_only=False)
    assert ck["world"] == 2 and "buffers" in ck and "rng" in ck
    tr2 = _trainer(tmp_path, monkeypatch, "nda", episodes=6)                                # a different config
    with pytest.raises(ValueError):
        tr2.load(out / "ckpt" / "last.pt")
    assert mt.Trainer.run(_trainer(tmp_path, monkeypatch, "nda"))["episodes"] == 4          # done: reused
    assert all(torch.equal(final[k], v) for k, v in
               torch.load(out / "ckpt" / "final.pt", weights_only=False)["models"]["det_t"].items())


def test_resume_replays_the_killed_world_exactly(tmp_path, monkeypatch):
    a = _trainer(tmp_path, monkeypatch, "nda")
    a.run()
    out = tmp_path / "nda"
    whole = [json.loads(x) for x in (out / "metrics.jsonl").read_text().splitlines()]
    # a second run, killed after world 1: keep its world-1 checkpoint, then resume
    out2 = tmp_path / "again"
    b = _trainer(tmp_path, monkeypatch, "nda")
    b.out = out2
    (out2 / "ckpt").mkdir(parents=True)
    (out2 / "metrics.jsonl").write_text("")
    b.run_world()
    b.save(out2 / "ckpt" / "last.pt")
    with open(out2 / "metrics.jsonl", "a") as f:               # records of the killed world, the last one cut short
        f.write(json.dumps({"kind": "episode", "episode": 2, "world": 1}) + "\n")
        f.write('{"kind": "episode", "episode": 3, "wor')
    c = _trainer(tmp_path, monkeypatch, "nda")
    c.out = out2
    c.run()
    again = [json.loads(x) for x in (out2 / "metrics.jsonl").read_text().splitlines()]
    assert [r["kind"] for r in again] == [r["kind"] for r in whole]
    for x, y in zip(whole, again):
        if x["kind"] == "epoch":
            assert x["n_tx"] == y["n_tx"] and x["updates"] == y["updates"]
            assert x["detection_rate"] == y["detection_rate"] and x["success_frac"] == y["success_frac"]
            assert all(math.isclose(x["losses"][k], y["losses"][k], rel_tol=1e-4, abs_tol=1e-6) for k in x["losses"])


def test_read_records_drops_only_an_incomplete_last_line(tmp_path):
    p = tmp_path / "metrics.jsonl"
    p.write_text('{"world": 0}\n{"world": 1}\n{"wor')
    assert [r for _, r in mt.read_records(p)] == [{"world": 0}, {"world": 1}]
    assert [x for x, _ in mt.read_records(p)] == ['{"world": 0}', '{"world": 1}']
    p.write_text('{"world": 0}\n{"wor\n{"world": 1}\n')            # broken inside the file: not a cut-short append
    with pytest.raises(json.JSONDecodeError):
        mt.read_records(p)
    p.write_text("")
    assert mt.read_records(p) == []


def test_readout_counts_follow_the_step_and_the_episode(tmp_path, monkeypatch):
    from collections import defaultdict

    tr = _trainer(tmp_path, monkeypatch, "nda")
    env = SimEnv(TINY, 3, *tr.policies)
    env.graph = LiveGraph()
    tr.det.pool.begin(env.graph, 1, 0.1)
    for e in range(2):
        env.start_episode(e)
        ep, n0 = defaultdict(float), env.ledger.n_tx
        for _ in range(6):
            before = env.ledger.n_tx
            res = env.step()
            tr.after_step(env, res, ep)
            step_commits = env.ledger.n_tx - before
            # the previous sub-step's flagging ...
            assert tr.last["n_cand"] == len(res.candidates)
            assert tr.last["n_flag"] == sum(c["flagged"] for c in res.candidates)
            assert step_commits == tr.last["n_cand"] - tr.last["n_flag"] + tr.last["n_frozen"]
            # ... and the episode's running commit total, which the readout's 7th value carries
            assert tr.last["episode_committed"] == env.ledger.n_tx - n0
            assert math.isclose(world_readout(env, tr.last)[6], math.log1p(env.ledger.n_tx - n0) / 10, rel_tol=1e-6)
        assert tr.last["episode_committed"] > step_commits > 0     # a total, not the last sub-step's count


def test_full_arm_adds_l_aux_on_real_views_and_nda_refuses_them(tmp_path, monkeypatch):
    tr = _trainer(tmp_path, monkeypatch, "full", episodes=2)
    tr.run()
    e = [json.loads(x) for x in (tmp_path / "full" / "metrics.jsonl").read_text().splitlines()][-1]
    for d in ("t", "w"):
        assert {f"{d}_kl_d", f"{d}_dom", f"{d}_kl_sn", f"{d}_rec", f"{d}_y_src", f"{d}_adv"} <= set(e["losses"])
    data_t = tr.det.real_t.data
    assert data_t.seen and all((t <= 29).all() for _, t in data_t.seen)
    assert tr.det.real_w.data.seen and all((g >= 50).all() for g in tr.det.real_w.data.seen)
    with pytest.raises(ValueError):                             # NDA never samples a real batch
        st = identity_stats()
        mt.Trainer(MarlConfig(arm="nda"), torch.device("cpu"), TINY, st, Normalizer(st), tr.det.real_t, tr.det.real_w)
    with pytest.raises(ValueError):
        st = identity_stats()
        mt.Trainer(MarlConfig(arm="full"), torch.device("cpu"), TINY, st, Normalizer(st))


def test_epsilon_schedule_and_degenerate_rule(tmp_path, monkeypatch):
    tr = _trainer(tmp_path, monkeypatch, "nda")
    for ep, want in ((0, 1.0), (150, 0.525), (300, 0.05), (900, 0.05)):
        tr.episode = ep
        assert math.isclose(tr.eps, want)
    ok = [{"detection_rate": 0.5, "success_frac": 0.5}] * 25
    assert mt.degenerate(ok) == []
    stuck = ok[:5] + [{"detection_rate": 0.97, "success_frac": 0.01}] * 20
    assert mt.degenerate(stuck) == ["detection_rate > 0.95", "success_frac < 0.05"]
    assert mt.degenerate(stuck[:-1] + [{"detection_rate": 0.9, "success_frac": 0.2}]) == []
    assert mt.degenerate(stuck[-19:]) == []                    # fewer than 20 epochs: not decided yet


def test_report_places_findings_after_acceptance_and_draws_the_figures(tmp_path, monkeypatch):
    from hns.marl import report

    tr = _trainer(tmp_path, monkeypatch, "nda")
    tr.out = tmp_path / "runs" / "nda" / "0"                   # the layout the report reads: <root>/<arm>/<seed>
    tr.out.mkdir(parents=True)
    tr.run()
    notes = "## Findings\n\nsomething found\n\n## Decisions this milestone added (ours)\n\n- a choice\n"
    ok = report.write(tmp_path / "runs", tmp_path / "m5.md", tmp_path / "figs", notes)
    text = (tmp_path / "m5.md").read_text()
    order = [text.index(h) for h in ("## Acceptance", "## Findings", "## Health curves", "## Losses", "## Decisions")]
    assert order == sorted(order) and "something found" in text and "- a choice" in text
    assert "| NDA | 0 | 4 / 4 | yes |" in text and "not decided (< 20 epochs)" in text
    assert not ok                                              # the degenerate rule needs 20 epochs to pass
    # a second seed that has only started (no epoch yet, last line cut short) does not block the report
    pending = tmp_path / "runs" / "nda" / "1"
    pending.mkdir()
    (pending / "config.yaml").write_text((tr.out / "config.yaml").read_text())
    first = (tr.out / "metrics.jsonl").read_text().splitlines()[0]
    assert json.loads(first)["kind"] == "episode"
    (pending / "metrics.jsonl").write_text(first + "\n" + first[:25])
    assert not report.write(tmp_path / "runs", tmp_path / "m5.md", tmp_path / "figs", notes)
    text = (tmp_path / "m5.md").read_text()
    assert "| NDA | 1 | 1 / 4 | — | — | — | — | no epoch completed yet | — | **no** |" in text
    assert "| NDA | 0 | 4 / 4 | yes |" in text
    (tr.out / "metrics.jsonl").write_text(first + "\n")          # and neither does a report of pending runs only
    assert not report.write(tmp_path / "runs", tmp_path / "m5.md", tmp_path / "figs", notes)
    assert all((tmp_path / "figs" / f"m5_{n}.png").stat().st_size > 10_000 for n in ("health", "strategy", "losses"))


# ---------------------------------------------------------------------------- learner fixes after M5 (2026-11-10)
def test_root_value_reads_the_initiate_branch_only():
    sizes, tau = [2, 3], 0.5
    q = torch.tensor([[1.0, 2.0, 50.0, 0.0, 9.0], [0.5, NEG, 7.0, 7.0, 7.0]])
    want = torch.tensor([tau * math.log(math.exp(1 / tau) + math.exp(2 / tau)), 0.5])   # go infeasible: wait only
    assert torch.allclose(root_value(q, sizes, tau), want, atol=1e-6)
    q2 = q.clone()
    q2[:, 2:] += 100.0                                        # the launch branches do not enter the target
    assert torch.equal(root_value(q2, sizes, tau), root_value(q, sizes, tau))
    assert not torch.allclose(soft_value(q2, sizes, torch.ones(2, 2, dtype=torch.bool), tau),
                              soft_value(q, sizes, torch.ones(2, 2, dtype=torch.bool), tau))


def test_replay_records_the_sub_steps_a_transition_spans():
    buf = Replay(10, [2])
    ones = np.ones(2, bool), np.ones(1, bool)
    a = buf.add("s0", *ones, t=3)
    b = buf.add("s1", *ones, t=53)
    buf.set(a, [0], [True], 1.0)
    buf.link(a, b)
    buf.set(b, [1], [True], 0.0)
    buf.end(b)
    assert buf.t_of(a) == 3 and buf.t_of(b) == 53
    s = buf.sample(40, np.random.default_rng(0))
    assert (~s["done"]).sum() == len(s["gap"]) > 0 and (s["gap"] == 50).all()


@pytest.mark.parametrize("target", ["root", "mean"])
def test_role_targets_bootstrap_their_branches_at_gamma_per_sub_step(tmp_path, monkeypatch, target):
    from hns.marl import learner as lm

    tr = _trainer(tmp_path, monkeypatch, "nda", episodes=2, target=target)
    tr.run()
    seen = {}

    def fake_td(q, sizes, action, realized, y):
        seen["y"] = y.detach().clone()
        return (torch.where(torch.isfinite(q), q, torch.zeros_like(q)) * 0.0).sum()

    monkeypatch.setattr(lm, "td_loss", fake_td)
    cfg, spans = tr.cfg, 0
    for role in ("launderer", "innocent"):
        L = tr.roles[role]
        for seed in range(40):
            b = L.buf.sample(cfg.batch, np.random.default_rng(seed))     # the batch update() is about to draw
            y = torch.from_numpy(b["reward"]).clone()
            live = ~b["done"]
            if live.any():
                with torch.no_grad():
                    q2 = L._q(L.targ, collate(b["next_obs"]), b["next_masks"])
                f = (root_value(q2, L.sizes, cfg.tau) if target == "root"
                     else soft_value(q2, L.sizes, torch.from_numpy(b["next_active"]), cfg.tau))
                y[torch.from_numpy(live)] += torch.from_numpy(cfg.gamma ** b["gap"].astype(np.float64)).float() * f
                spans += int((b["gap"] > 1).sum())
            L.update(np.random.default_rng(seed))
            assert torch.allclose(seen["y"], y, atol=1e-5), (role, seed)
    assert spans > 0                                          # some transitions spanned more than one sub-step


def test_innocent_continuation_reward_is_discounted_to_the_decision(tmp_path, monkeypatch):
    tr = _trainer(tmp_path, monkeypatch, "nda", episodes=2)
    tr.run()
    buf = tr.roles["innocent"].buf
    r = buf.reward[:min(buf.n, buf.cap)].astype(np.float64)
    plain = np.isclose(r, 0.0, atol=1e-7) | np.isclose(r, 0.1, atol=1e-7)
    forward = (r > 0.1 + 1e-5) & (r < 0.1 + 0.1 * tr.cfg.gamma + 1e-6)      # 0.1 + 0.1 gamma^delta, delta >= 1
    assert (plain | forward).all() and forward.any()


def test_readout_commit_count_restarts_with_each_episode(tmp_path, monkeypatch):
    seen, real = [], mt.world_readout

    def spy(env, last):
        seen.append((env.episode, env.sub, last.get("episode_committed")))
        return real(env, last)

    monkeypatch.setattr(mt, "world_readout", spy)
    _trainer(tmp_path, monkeypatch, "nda", episodes=2).run()
    firsts = [c for e, s, c in seen if s == 0 and e > 0]
    assert firsts and all(c == 0 for c in firsts) and any(c for _, s, c in seen if s > 0)


class _OneDirect:
    """Launderer 0 launches one ``direct`` tranche at ``amount`` at its first idle decision; everyone else waits."""

    def __init__(self, amount, change):
        self.amount, self.change, self.done = amount, change, False

    def act(self, env, states):
        from hns.sim.entities import AMOUNTS, GO, STRATEGIES, WAIT
        out = {}
        for i, st in states.items():
            a = {b: int(np.flatnonzero(st.masks[b])[0]) for b in st.branches}
            a["initiate"] = WAIT
            if i == 0 and not self.done and len(st.branches) == 6:
                a.update(initiate=GO, amount=AMOUNTS.index(self.amount), strategy=STRATEGIES.index("direct"),
                         change=self.change)
                self.done = True
            out[i] = a
        return out


class _FlagIllicit:
    def act_global(self, env, state):
        return {"analysis_depth": 0, "update_strategy": 1}

    def score(self, env, cands):
        return np.array([float(r.spec.illicit) for r in cands])


@pytest.mark.parametrize("amount,change", [(0.1, 1), (0.1, 0), (1.0, 1)])
def test_a_flagged_setup_freezes_its_change_and_the_btc_accounting_closes(amount, change):
    env = SimEnv(TINY, 5, launderers=_OneDirect(amount, change), detector=_FlagIllicit())
    env.start_episode(0)
    env.budget = 1.0                                          # every candidate is flagged: illicit ones freeze
    L = env.launderers[0]
    bal0 = env.ledger.balance[L.source]
    res = env.step()
    tr = next(t for t in res.transitions if t.role == "launderer" and t.agent == 0)
    a = env.acct
    assert tr.realized[0] == "initiate" and len(tr.realized) >= 4
    lost = bal0 - a["fees"]                                   # the whole source, less the setup's fee
    assert math.isclose(a["frozen"], lost, rel_tol=1e-12) and a["cashed"] == 0.0
    assert math.isclose(a["frozen_change"], (1.0 - amount) * lost, rel_tol=1e-9, abs_tol=1e-15)
    assert math.isclose(tr.reward, -1.0, rel_tol=1e-9)        # the per-tranche reward still ignores the change
    f = mt.btc_fractions(env)
    assert abs(f["accounting_gap"]) < 1e-12 and math.isclose(f["frozen_frac"], lost / a["proceeds"], rel_tol=1e-12)


def test_btc_accounting_closes_over_a_static_world():
    env = SimEnv(replace(TINY, budget=0.2), 11)
    env.run_world()
    f = mt.btc_fractions(env)
    assert abs(f["accounting_gap"]) < 1e-9 and f["frozen_change_frac"] > 0 and f["success_frac"] > 0
    assert f["held_frac"] >= 0 and f["fee_frac"] > 0


def test_frozen_roles_act_but_neither_store_nor_update(tmp_path, monkeypatch):
    src = _trainer(tmp_path, monkeypatch, "nda", episodes=2)
    src.run()
    final = tmp_path / "nda" / "ckpt" / "final.pt"
    with pytest.raises(ValueError):                           # a frozen role needs a fixed epsilon
        _trainer(tmp_path, monkeypatch, "nda", learn=("launderer",), fixed_eps={"innocent": 0.05}, out_name="x")
    tr = _trainer(tmp_path, monkeypatch, "nda", episodes=2, learn=("launderer",),
                  fixed_eps={"innocent": 0.05, "detector": 0.0}, out_name="frozen")
    tr.load_models(final, ("innocent", "detector"))
    ck = torch.load(final, weights_only=False)["models"]
    assert all(torch.equal(ck["det_t"][k], v) for k, v in tr.det.det_t.state_dict().items())
    assert tr.eps_for("innocent") == 0.05 and tr.eps_for("detector") == 0.0 and tr.eps_for("launderer") == tr.eps
    rec = tr.run_world()
    assert rec["updates"]["launderer"] > 0 and rec["updates"]["innocent"] == 0 == rec["updates"]["detector_t"]
    assert rec["updates"]["detector_w"] == 0 and rec["export"] is None
    assert tr.roles["innocent"].buf.n == 0 and tr.det.flag_buf.n == 0 and tr.det.gbuf.n == 0
    assert all(torch.equal(ck["det_t"][k], v) for k, v in tr.det.det_t.state_dict().items())
    assert all(torch.equal(ck["innocent"][k], v) for k, v in tr.roles["innocent"].net.state_dict().items())
    assert rec["n_tx"] > 0 and abs(rec["accounting_gap"]) < 1e-9 and 0 <= rec["detection_rate"] <= 1


def test_scheduled_references_launch_once_per_episode():
    from hns.marl.capability import Scheduled
    from hns.sim.entities import AMOUNTS, STRATEGIES

    for never, want in ((False, TINY.n_launderers), (True, 0)):
        env = SimEnv(TINY, 2, launderers=Scheduled(np.random.default_rng(0), STRATEGIES.index("direct"),
                                                   AMOUNTS.index(0.5), never=never))
        for e in range(2):
            env.start_episode(e)
            n0 = env.acct["strategy"].copy()
            for _ in range(50):
                env.step()
            d = env.acct["strategy"] - n0
            assert d.sum() == want and d[STRATEGIES.index("direct")] == want


def test_capability_calibration_discounts_per_sub_step_to_world_end():
    from hns.marl.capability import calibration

    dec = [(10, 7, 1, 1.0, 2.0), (0, 7, 0, -0.001, 1.0), (3, 8, 0, 0.5, None)]   # (step, agent, a0, r, Q)
    c = calibration(dec, 0.9)
    g = np.array([-0.001 + 0.9 ** 10 * 1.0, 1.0])
    assert c["q_n"] == 2 and math.isclose(c["q_bias"], float(np.mean(np.array([1.0, 2.0]) - g)), rel_tol=1e-12)
    assert math.isclose(c["g_abs_mean"], float(np.abs(g).mean()), rel_tol=1e-12) and c["q_max"] == 2.0


def test_kind_values_agree():
    from hns.models import detector
    assert (detector.KIND_TX, detector.KIND_W, detector.KIND_CTRL) == (KIND_TX, KIND_W, KIND_CTRL)
    assert ObsBatch.__dataclass_fields__["cand"].default is None
