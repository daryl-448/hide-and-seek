"""OPPO-MC (SPEC §9.3 options, §10): the option policy's probability algebra, decision chains, the world-end update
with its KL guard, the trainer's integration, and the development block's reading (hns.marl.oppo_block)."""

import copy
import itertools
import json
import math
from dataclasses import replace

import numpy as np
import pytest
import torch

import hns.marl.oppo as oppo
import hns.marl.train as mt
from hns.data.norm import Normalizer
from hns.marl.learner import MarlConfig, wealth_scale
from hns.marl.obs import D_CAND, D_W_AGENT, L_CTRL, Featurizer, collate, identity_stats, launderer_obs
from hns.models import policy as P
from hns.sim.entities import AMOUNTS, GO, STRATEGIES
from hns.sim.entities import WAIT as E_WAIT
from hns.sim.env import SimEnv
from hns.sim.live import LiveGraph
from hns.sim.world import WorldParams

TINY = WorldParams(n_innocents=80, n_services=3, n_launderers=4, batch_size=4, consol_interval=3, demand_rate=0.05,
                   p_ill=0.6, p_lic=0.5, p_wunk=0.1)
DIRECT, PEEL, FAN = STRATEGIES.index("direct"), STRATEGIES.index("peel"), STRATEGIES.index("fan_out")
FULL = AMOUNTS.index(1.0)


# ---------------------------------------------------------------------------- probability algebra
def _rand_masks(rng, n, launch=True, sub=0, substeps=50):
    m = np.zeros((n, P.MASK_WIDTH), bool)
    for i in range(n):
        m[i, P.SLICES["gate"]] = (True, launch)
        m[i, P.SLICES["dur"]] = P.duration_mask(sub, substeps)
        amount = rng.random(len(AMOUNTS)) < 0.7
        amount[rng.integers(len(AMOUNTS))] = True
        m[i, P.SLICES["mode"]] = np.outer(amount, np.ones(len(STRATEGIES), bool)).ravel() & launch
        rc = rng.random(P.FACTORS["rcpt"]) < 0.5
        rc[0] = True
        m[i, P.SLICES["rcpt"]] = rc
        m[i, P.SLICES["mules"]] = (True, bool(rng.random() < 0.5))
        m[i, P.SLICES["change"]] = (True, True)
    return m


def _rand_logits(gen, n):
    return {"gate": torch.randn(n, 2, generator=gen), "dur": torch.randn(n, 5, generator=gen),
            "mode": torch.randn(n, P.N_MODES, generator=gen), "rcpt": torch.randn(n, P.N_MODES, 16, generator=gen),
            "mules": torch.randn(n, P.N_MODES, 2, generator=gen), "change": torch.randn(n, P.N_MODES, 2, generator=gen)}


def _enumerate(mask: np.ndarray, gate_free: bool):
    """Every effective action of one decision, ``ACTION``-ordered, irrelevant factors canonical."""
    f = lambda k: np.flatnonzero(mask[P.SLICES[k]])  # noqa: E731
    out = []
    if gate_free:
        out += [(P.WAIT, d, 0, 0, 0, 0) for d in f("dur")]
    if mask[P.SLICES["gate"]][P.LAUNCH]:
        for z in f("mode"):
            us = f("mules") if P.MULES_REL[z] else [0]
            cs = f("change") if P.CHANGE_REL[z] else [0]
            out += [(P.LAUNCH, 0, z, k, u, c) for k in f("rcpt") for u in us for c in cs]
    return np.array(out, np.int64)


def _logp_all(t, mask, gate_free, row):
    acts = _enumerate(mask, gate_free)
    tt = {k: v[row:row + 1].expand(len(acts), *v.shape[1:]) for k, v in t.items()}
    return P.logp_taken(tt, torch.from_numpy(acts), torch.full((len(acts),), gate_free)).double()


@pytest.mark.parametrize("gate_free", [True, False])
def test_option_probabilities_entropy_and_kl_match_enumeration(gate_free):
    rng, gen = np.random.default_rng(0), torch.Generator().manual_seed(0)
    n = 4
    masks = _rand_masks(rng, n, sub=47)                                  # durations 1 and 3 (= 5 clipped)
    t_old = P.log_tables(_rand_logits(gen, n), torch.from_numpy(masks))
    t_new = P.log_tables(_rand_logits(gen, n), torch.from_numpy(masks))
    gf = torch.full((n,), gate_free)
    ent = P.entropy(t_old, gf).double()
    joint, launch, dur = (x.double() for x in P.kl_parts(t_old, t_new, gf))
    for i in range(n):
        lo, ln = _logp_all(t_old, masks[i], gate_free, i), _logp_all(t_new, masks[i], gate_free, i)
        p = lo.exp()
        assert math.isclose(float(p.sum()), 1.0, abs_tol=1e-5)          # a distribution over effective actions
        assert math.isclose(float(-(p * lo).sum()), float(ent[i]), rel_tol=1e-4)
        assert math.isclose(float((p * (lo - ln)).sum()), float(joint[i]), rel_tol=1e-3, abs_tol=1e-6)
        # the conditionals, by enumeration of the launch actions alone and of the durations alone
        acts = _enumerate(masks[i], gate_free)
        la = acts[:, 0] == P.LAUNCH
        lo_l = lo[la] - torch.logsumexp(lo[la], 0)
        ln_l = ln[la] - torch.logsumexp(ln[la], 0)
        assert math.isclose(float((lo_l.exp() * (lo_l - ln_l)).sum()), float(launch[i]), rel_tol=1e-3, abs_tol=1e-6)
        if gate_free:
            lo_d, ln_d = lo[~la] - torch.logsumexp(lo[~la], 0), ln[~la] - torch.logsumexp(ln[~la], 0)
            assert math.isclose(float((lo_d.exp() * (lo_d - ln_d)).sum()), float(dur[i]), rel_tol=1e-3, abs_tol=1e-6)


def test_irrelevant_factors_have_no_probability_and_no_gradient():
    gen = torch.Generator().manual_seed(1)
    masks = torch.ones(4, P.MASK_WIDTH, dtype=torch.bool)
    logits = {k: v.requires_grad_() for k, v in _rand_logits(gen, 4).items()}
    t = P.log_tables(logits, masks)
    half = AMOUNTS.index(0.5)
    act = torch.tensor([[1, 0, P.mode_of(half, DIRECT), 3, 1, 1],     # direct: mules not a decision
                        [1, 0, P.mode_of(half, PEEL), 3, 1, 0],       # peel: mules not a decision
                        [1, 0, P.mode_of(FULL, FAN), 3, 0, 1],        # 100 %: change not a decision
                        [0, 2, 0, 0, 0, 0]])                          # a wait: no launch factor
    gf = torch.ones(4, dtype=torch.bool)
    lp = P.logp_taken(t, act, gf)
    lp.sum().backward()
    assert torch.isfinite(lp).all()
    g = {k: v.grad for k, v in logits.items()}
    assert (g["mules"][:2] == 0).all() and (g["mules"][2] != 0).any()
    assert (g["change"][2] == 0).all() and (g["change"][0] != 0).any() and (g["change"][1] != 0).any()
    assert all((g[f][3] == 0).all() for f in ("mode", "rcpt", "mules", "change")) and (g["dur"][3] != 0).any()
    assert all((g["dur"][i] == 0).all() for i in range(3))            # a launch has no duration
    assert (g["gate"] != 0).any(1).all()
    # an irrelevant factor's stored option does not change the probability
    act2 = act.clone()
    act2[0, 4], act2[1, 4], act2[2, 5], act2[3, 2:] = 0, 0, 0, torch.tensor([5, 7, 1, 1])
    assert torch.equal(P.logp_taken(t, act2, gf), lp.detach())


def test_masked_and_singleton_factors_stay_finite():
    rng, gen = np.random.default_rng(2), torch.Generator().manual_seed(2)
    masks = np.concatenate([_rand_masks(rng, 1, launch=False), _rand_masks(rng, 1, sub=49), _rand_masks(rng, 1)])
    masks[2, P.SLICES["mode"]] = False
    masks[2, P.SLICES["mode"].start + P.mode_of(3, 4)] = True            # one feasible mode: a singleton factor
    logits = {k: v.requires_grad_() for k, v in _rand_logits(gen, 3).items()}
    t = P.log_tables(logits, torch.from_numpy(masks))
    gf = torch.ones(3, dtype=torch.bool)
    act = torch.tensor([[0, 0, 0, 0, 0, 0], [1, 0, P.mode_of(0, 0), 0, 0, 0], [1, 0, P.mode_of(3, 4), 0, 0, 0]])
    act[1, 2] = int(np.flatnonzero(masks[1, P.SLICES["mode"]])[0])
    lp, h = P.logp_taken(t, act, gf), P.entropy(t, gf)
    assert math.isclose(float(t["gate"][0, P.WAIT].detach()), 0.0, abs_tol=1e-6)  # launch infeasible: wait is certain
    assert math.isclose(float(t["dur"][1, 0].detach()), 0.0, abs_tol=1e-6)        # one sub-step left: one duration
    assert math.isclose(float(t["mode"][2, P.mode_of(3, 4)].detach()), 0.0, abs_tol=1e-6)
    (lp.sum() + h.sum()).backward()
    assert torch.isfinite(lp).all() and torch.isfinite(h).all()
    assert all(torch.isfinite(v.grad).all() for v in logits.values())
    # sampling never picks an infeasible option, and a scheduled row always launches
    tn = {k: v.detach().numpy() for k, v in t.items()}
    for greedy in (False, True):
        a = P.sample(tn, np.array([True, True, False]), np.random.default_rng(0), greedy)
        assert a[0, 0] == P.WAIT and a[2, 0] == P.LAUNCH and a[2, 2] == P.mode_of(3, 4)
        assert torch.isfinite(P.logp_taken(t, torch.from_numpy(a), torch.tensor([True, True, False]))).all()


def test_wait_durations_clip_at_the_episode_end():
    assert P.duration_mask(0, 50).all()
    assert P.duration_mask(40, 50).tolist() == [True, True, True, False, False]     # 25 and 50 both clip to 10
    assert P.duration_mask(45, 50).tolist() == [True, True, False, False, False]
    assert P.duration_mask(49, 50).tolist() == [True, False, False, False, False]
    assert [P.effective_duration(d, 45, 50) for d in range(5)] == [1, 5, 5, 5, 5]


def test_irrelevance_tables_follow_the_simulator():
    for z in range(P.N_MODES):
        a, s = P.split_mode(z)
        assert P.mode_of(a, s) == z
        assert P.MULES_REL[z] == (STRATEGIES[s] not in ("direct", "peel"))
        assert P.CHANGE_REL[z] == (AMOUNTS[a] < 1.0)


# ---------------------------------------------------------------------------- networks
@pytest.fixture(scope="module")
def idle_batch():
    env = SimEnv(TINY, seed=17)
    env.graph = LiveGraph()
    env.start_episode(0)
    for _ in range(4):
        env.step()
    env._refresh_state()
    Ls = {L.id: st for L in env.launderers if (st := env.launderer_state(L)) is not None and len(st.branches) > 1}
    assert Ls
    ob = launderer_obs(env, Featurizer(identity_stats()), list(Ls), np.zeros(8, np.float32), seed=0)
    return ob, np.stack([oppo.option_masks(st, env.sub, 50) for st in Ls.values()])


def test_the_first_policy_is_uniform_and_the_networks_share_nothing(idle_batch):
    ob, masks = idle_batch
    torch.manual_seed(0)
    L = oppo.OppoLearner(MarlConfig(), torch.device("cpu"))
    tab, v, m = L.evaluate(ob, masks)
    for i in range(len(masks)):
        for f in P.FACTORS:
            ok = masks[i, P.SLICES[f]]
            if f == "mode" and not ok.any():
                ok = np.ones(P.N_MODES, bool)                            # the placeholder, weighted by p(launch) = 0
            x = tab[f][i]
            assert np.allclose(x[..., ok], -np.log(ok.sum()), atol=1e-5) and np.isneginf(x[..., ~ok]).all(), f
    assert np.allclose(m, wealth_scale(ob.x_c)) and np.isfinite(v).all()
    a_ids = {id(p) for p in L.actor.parameters()}
    assert not a_ids & {id(p) for p in L.critic.parameters()}
    assert all(torch.equal(x, y) for x, y in zip(L.actor.trunk.state_dict().values(), L.critic.trunk.state_dict().values()))


# ---------------------------------------------------------------------------- decision chains
def _row(agent, t, gate_free=True):
    return oppo.Row(agent, t, t, (None, None, None, None, np.zeros(L_CTRL, np.float32)), np.ones(P.MASK_WIDTH, bool),
                    gate_free, np.zeros(6, np.int64), 0.0, 0.0, 1.0)


def test_rollout_chains_each_launderer_through_its_rewards_to_world_end():
    g = 0.9
    ro = oppo.Rollout(g)
    ro.observe(7, 0, -0.5)                                       # before agent 7's first decision: nobody's
    ro.add(_row(7, 2))
    ro.add(_row(8, 3))
    for t, a, r in ((2, 7, 1.0), (3, 8, 0.25), (4, 7, -0.001), (9, 7, 2.0)):
        ro.observe(a, t, r, noop=(a == 8))
    ro.add(_row(7, 10))
    for t, a, r in ((10, 7, 0.125), (12, 8, -1.0), (30, 7, 0.5)):
        ro.observe(a, t, r)
    with pytest.raises(RuntimeError):
        ro.add(_row(8, 3))                                       # not after its previous decision
    ro.close(50)
    with pytest.raises(RuntimeError):
        ro.add(_row(7, 40))
    G = ro.returns()
    r0, r1, r2 = ro.rows
    assert (r0.delta, r1.delta, r2.delta) == (8, 47, 40) and (r0.done, r1.done, r2.done) == (False, True, True)
    assert r1.noop and not r0.noop
    g2 = 0.125 + g ** 20 * 0.5
    assert math.isclose(G[2], g2) and math.isclose(G[0], 1.0 + g ** 2 * -0.001 + g ** 7 * 2.0 + g ** 8 * g2)
    assert math.isclose(G[1], 0.25 + g ** 9 * -1.0)
    assert ro.check(G) < 1e-12
    G[0] += 1e-3
    with pytest.raises(RuntimeError):
        ro.check(G)


# ---------------------------------------------------------------------------- the trainer
def _trainer(tmp_path, monkeypatch, episodes=4, curriculum=1, out_name="oppo", **kw):
    monkeypatch.setattr(mt, "EPISODES", 2)
    monkeypatch.setattr(mt, "SUBSTEPS", 12)
    monkeypatch.setattr(mt, "CURRICULUM_WORLDS", curriculum)
    monkeypatch.setattr(oppo, "LAUNCH_LAST", 3)
    monkeypatch.setattr(mt, "draw_world", lambda c, rng: replace(c, mask_rate=0.2, offset=int(rng.integers(0, 40))))
    st = identity_stats()
    cfg = MarlConfig(arm="nda", seed=0, episodes=episodes, warmup=30, batch=8, buffer=400, ckpt_worlds=1,
                     launderer="oppo", **kw)
    out = tmp_path / out_name
    out.mkdir(exist_ok=True)
    return mt.Trainer(cfg, torch.device("cpu"), TINY, st, Normalizer(st), None, None, out, log=lambda *_: None)


class _Spy:
    """Records what the env asked of the option policy and what it answered."""

    def __init__(self, pol):
        self.pol, self.calls = pol, []

    def __getattr__(self, k):
        return getattr(self.pol, k)

    def act(self, env, states):
        out = self.pol.act(env, states)
        self.calls.append((self.pol.tr.step_no, env.sub, {i: (len(s.branches), out[i]["initiate"]) for i, s in states.items()}))
        return out


def test_options_curriculum_then_natural_decisions(tmp_path, monkeypatch):
    tr = _trainer(tmp_path, monkeypatch, episodes=4)
    spy = _Spy(tr.policies[0])
    tr.policies = (spy, *tr.policies[1:])
    rec1 = tr.run_world()
    ro1 = spy.rollout
    assert ro1.rows and all(not r.gate_free and r.action[0] == P.LAUNCH for r in ro1.rows)
    per_ep = {}
    for r in ro1.rows:                                           # at most one scheduled launch per launderer-episode
        per_ep[(r.agent, r.tw // 12)] = per_ep.get((r.agent, r.tw // 12), 0) + 1
    assert max(per_ep.values()) == 1
    st1 = rec1["learner"]["launderer"]
    assert st1["n"] == len(ro1.rows) and st1["ratio_dev_max"] < 1e-5 and st1["return_check"] < 1e-9
    assert st1["decisions"]["scheduled"] == len(ro1.rows)
    rec2 = tr.run_world()
    ro2 = spy.rollout
    assert ro2.rows and all(r.gate_free for r in ro2.rows)
    assert {int(r.action[0]) for r in ro2.rows} == {P.WAIT, P.LAUNCH}       # both kinds of decision were checked
    assert rec2["launderer"]["decisions"].get("flight_wait", 0) == 0          # in flight it always proceeds
    # a launderer decides only when idle with no wait outstanding; while waiting the env sees initiate = 0
    rows = {(r.agent, r.t): r for r in ro2.rows}
    until = {}
    for step, sub, seen in [c for c in spy.calls if c[0] >= tr.world_step]:
        for i, (nb, init) in seen.items():
            if nb == 1:                                          # in flight: proceeds, no decision
                assert init == GO and (i, step) not in rows
            elif (i, step) in rows:                              # a decision: no wait outstanding
                assert until.get(i, -1) <= step
                r = rows[(i, step)]
                if r.action[0] == P.WAIT:
                    until[i] = step + P.effective_duration(int(r.action[1]), sub, 12)
                    assert init == E_WAIT
                else:
                    until.pop(i, None)
                    assert init == GO
            else:                                                # idle without a decision: only while waiting
                assert until.get(i, -1) > step and init == E_WAIT
    st2 = rec2["learner"]["launderer"]
    assert st2["ratio_dev_max"] < 1e-5 and st2["epochs_accepted"] + st2["epochs_rejected"] >= 1
    assert abs(rec2["accounting_gap"]) < 1e-9 and all(math.isfinite(v) for v in rec2["losses"].values())
    assert rec2["updates"]["launderer"] > rec1["updates"]["launderer"] and "launderer" not in rec2["buffer"]


def test_kl_guard_undoes_the_pass_and_its_optimizer_state(tmp_path, monkeypatch):
    tr = _trainer(tmp_path, monkeypatch, episodes=6, curriculum=0)
    L = tr.roles["launderer"]
    monkeypatch.setattr(oppo, "KL_MAX", 1e9)                     # nothing is undone
    st = tr.run_world()["learner"]["launderer"]
    assert st["n"] > 0 and st["epochs_accepted"] == oppo.EPOCHS and len(st["kl"]) == oppo.EPOCHS
    assert L.opt_a.state_dict()["state"]
    real, seen = L.learn, {}

    def learn(ro):
        seen["actor"] = copy.deepcopy(L.actor.state_dict())
        seen["opt"] = copy.deepcopy(L.opt_a.state_dict())
        seen["critic"] = copy.deepcopy(L.critic.state_dict())
        return real(ro)

    L.learn = learn
    monkeypatch.setattr(oppo, "KL_MAX", -1.0)                    # every pass trips the guard
    st = tr.run_world()["learner"]["launderer"]
    assert st["n"] > 0 and st["epochs_rejected"] == 1 and st["epochs_accepted"] == 0 and len(st["kl"]) == 1
    assert all(torch.equal(v, L.actor.state_dict()[k]) for k, v in seen["actor"].items())
    now = L.opt_a.state_dict()["state"]
    assert now.keys() == seen["opt"]["state"].keys()
    assert all(torch.equal(now[k][x], seen["opt"]["state"][k][x]) for k in now for x in ("step", "exp_avg", "exp_avg_sq"))
    assert any(not torch.equal(v, L.critic.state_dict()[k]) for k, v in seen["critic"].items())  # the critic trains on


def test_oppo_resume_replays_the_killed_world_exactly(tmp_path, monkeypatch):
    a = _trainer(tmp_path, monkeypatch)
    a.run()
    whole = [json.loads(x) for x in (tmp_path / "oppo" / "metrics.jsonl").read_text().splitlines()]
    b = _trainer(tmp_path, monkeypatch, out_name="again")
    (b.out / "ckpt").mkdir(parents=True)
    (b.out / "metrics.jsonl").write_text("")
    b.run_world()
    b.save(b.out / "ckpt" / "last.pt")
    c = _trainer(tmp_path, monkeypatch, out_name="again")
    c.run()
    again = [json.loads(x) for x in (b.out / "metrics.jsonl").read_text().splitlines()]
    assert [r["kind"] for r in again] == [r["kind"] for r in whole]
    for x, y in zip([r for r in whole if r["kind"] == "epoch"], [r for r in again if r["kind"] == "epoch"]):
        assert x["n_tx"] == y["n_tx"] and x["updates"] == y["updates"] and x["btc_utility"] == y["btc_utility"]
        assert x["learner"]["launderer"]["n"] == y["learner"]["launderer"]["n"]
    assert all(torch.allclose(v, c.roles["launderer"].actor.state_dict()[k], atol=1e-5)
               for k, v in a.roles["launderer"].actor.state_dict().items())


def test_matched_evaluation_shares_the_references_schedule(monkeypatch):
    import hns.marl.capability as cap
    import hns.marl.repair2 as r2

    assert oppo.LAUNCH_LAST == cap.LAUNCH_LAST
    env = SimEnv(TINY, 3)
    env.start_episode(0)
    env._refresh_state()
    ref = r2.reference("uniform", 100)
    states = {L.id: env.launderer_state(L) for L in env.launderers}
    ref.act(env, states)

    class _Tr:
        step_no, world_step = 0, 0
        cfg = MarlConfig()

    pol = oppo.OptionPolicy(_Tr(), type("L", (), {"cfg": MarlConfig()})(), np.random.default_rng(0),
                            np.random.default_rng([cap.EVAL_SALT, 100, 3]), scheduled=True)
    env.sub = -1                                                 # nobody is due: no observation is needed
    pol.act(env, states)
    assert pol.when == ref.when


# ---------------------------------------------------------------------------- the block's reading
def _rows(u, reward=None, held=0.05, bias=0.1, g_abs=1.0):
    return [{"world": j, "btc_utility": x, "reward": x if reward is None else reward, "held_frac": held,
             "cal": {"n": 10, "bias": bias, "g_abs": g_abs}} for j, x in zip((100, 101, 102, 103), u)]


def test_block_reading_has_five_criteria():
    from hns.marl import oppo_block as ob

    refs = {"direct50": _rows([0.7] * 4, held=0.08), "uniform": _rows([-0.8] * 4), "wait": _rows([0.0] * 4),
            "repeat50": _rows([0.74] * 4)}
    a_final = _rows([0.42] * 4)
    res = {"status": "ok", "worlds_done": 30, "accounting_gap_max": 0.0, "ratio_dev_max": 1e-6}
    good = {20: _rows([0.45] * 4), 30: _rows([0.5] * 4, held=0.1)}
    matched = _rows([0.3] * 4)
    r = ob.read_block(res, good, matched, refs, a_final)
    assert r["promising"] and math.isclose(r["utility_bar"], 0.40) and math.isclose(r["reward_bar"], 0.35)
    fails = {
        "c1": dict(res=res | {"ratio_dev_max": 0.01}), "c2": dict(snaps={20: _rows([0.40] * 4), 30: _rows([0.39] * 4)}),
        "c3": dict(snaps={20: _rows([0.6] * 4), 30: good[30]}),
        "c4": dict(snaps={20: good[20], 30: _rows([0.5] * 4, held=0.1, bias=0.6)}),
        "c5": dict(snaps={20: good[20], 30: _rows([0.5] * 4, held=0.14)}),
    }
    for c, kw in fails.items():
        x = ob.read_block(kw.get("res", res), kw.get("snaps", good), matched, refs, a_final)
        assert not x[c] and not x["promising"] and all(x[k] for k in ("c1", "c2", "c3", "c4", "c5") if k != c), c
    assert not ob.read_block(res, good, _rows([-0.79] * 4), refs, a_final)["c5"]      # matched skill under 0.02
    big = {20: good[20], 30: _rows([0.5] * 4, held=0.1, bias=1.9, g_abs=8.0)}         # tolerance 0.25 mean |G|
    assert ob.read_block(res, big, matched, refs, a_final)["c4"]
    assert not ob.read_block(res | {"status": "aborted: x"}, good, matched, refs, a_final)["promising"]
