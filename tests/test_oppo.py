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
    assert st2["ratio_dev_max"] < 1e-5 and st2["steps_accepted"] + st2["steps_rejected"] >= 1
    assert abs(rec2["accounting_gap"]) < 1e-9 and all(math.isfinite(v) for v in rec2["losses"].values())
    assert rec2["updates"]["launderer"] > rec1["updates"]["launderer"] and "launderer" not in rec2["buffer"]


def _sig(m) -> tuple[float, float]:
    """A network's parameters as two sums: equal for the same weights, different after an update."""
    ps = [p.detach().double() for p in m.parameters()]
    return float(sum(p.sum() for p in ps)), float(sum((p * p).sum() for p in ps))


def test_live_play_opponents_act_with_their_world_start_networks(tmp_path, monkeypatch):
    tr = _trainer(tmp_path, monkeypatch, episodes=6, curriculum=1)
    inn, D = tr.roles["innocent"], tr.det
    same = lambda: (inn.act_net is inn.net, D.act_t is D.det_t, D.act_gq is D.gq)     # noqa: E731
    assert all(same())
    seen = []
    tr.after_update = lambda t: seen.append((t.world, same(), _sig(inn.act_net), _sig(D.act_t), _sig(D.act_gq),
                                             _sig(inn.net), _sig(D.det_t), _sig(D.gq)))
    start, end = {}, {}
    for w in range(3):
        start[w] = (_sig(inn.net), _sig(D.det_t), _sig(D.gq))
        rec = tr.run_world()
        end[w] = (_sig(inn.net), _sig(D.det_t), _sig(D.gq))
        assert all(same())                                              # a world's end drops the copies
    assert rec["updates"]["innocent"] > 0 and rec["updates"]["detector_t"] > 0
    assert len(seen) == 3 * 2 * 12
    for w, sm, ai, at, ag, *_ in seen:                                  # every sub-step acted with the world-start copies
        assert not any(sm) and (ai, at, ag) == start[w]
    last = [s for s in seen if s[0] == 2][-1]
    assert last[5] != start[2][0] and last[6] != start[2][1]            # while the learners moved within the world
    assert start[1] == end[0] and start[2] == end[1]                    # and what they learned acts from the next world on
    assert not any(p.requires_grad for p in _target_params(tr))        # the copies are never trained


def _target_params(tr):
    tr._acting(world_start=True)
    ps = [*tr.roles["innocent"].act_net.parameters(), *tr.det.act_t.parameters(), *tr.det.act_gq.parameters()]
    tr._acting(world_start=False)
    return ps


def test_acting_copies_are_for_live_oppo_play_only(tmp_path, monkeypatch):
    # frozen opponents (the capability runs): they never learn, so they act with their own networks
    tr = _trainer(tmp_path, monkeypatch, episodes=2, curriculum=1)
    tr.learn, tr.fixed_eps = frozenset({"launderer"}), {"innocent": 0.05, "detector": 0.05}
    inn, D = tr.roles["innocent"], tr.det
    flags = []
    tr.after_update = lambda t: flags.append((inn.act_net is inn.net, D.act_t is D.det_t, D.act_gq is D.gq))
    tr.run_world()
    assert flags and all(all(f) for f in flags)
    # a Q-learning launderer (M5's first runs): acting and learning networks stay the same objects
    st = identity_stats()
    cfg = MarlConfig(arm="nda", seed=0, episodes=2, warmup=30, batch=8, buffer=400, ckpt_worlds=1)
    out = tmp_path / "q"
    out.mkdir()
    tq = mt.Trainer(cfg, torch.device("cpu"), TINY, st, Normalizer(st), None, None, out, log=lambda *_: None)
    flags = []
    tq.after_update = lambda t: flags.append((tq.roles["innocent"].act_net is tq.roles["innocent"].net,
                                              tq.det.act_t is tq.det.det_t, tq.det.act_gq is tq.det.gq))
    tq.run_world()
    assert flags and all(all(f) for f in flags)


@pytest.fixture
def guard_world(tmp_path, monkeypatch):
    """A natural world's closed rollout, collected by the current actor, with several logical minibatches per pass
    (``MINIBATCH`` 4), and a learner whose Adam state is not empty. ``restore()`` returns the learner to the state
    in which the rollout was collected."""
    monkeypatch.setattr(oppo, "MINIBATCH", 4)
    monkeypatch.setattr(oppo, "MICRO", 2)
    tr = _trainer(tmp_path, monkeypatch, episodes=6, curriculum=0)
    L = tr.roles["launderer"]
    real, got = L.learn, []
    L.learn = got.append                                          # the trainer hands each world's rollout over
    tr.run_world()
    real(got[0])                                                  # one real update: Adam has moments and counters
    tr.run_world()
    L.learn = real
    ro, state0 = got[1], copy.deepcopy(L.state())
    assert len(ro.rows) >= 3 * oppo.MINIBATCH and L.opt_a.state_dict()["state"]

    def restore():
        L.load_state(copy.deepcopy(state0))

    steps = {"critic": 0}
    step_c = L.opt_c.step

    def count(*a, **k):
        steps["critic"] += 1
        return step_c(*a, **k)

    monkeypatch.setattr(L.opt_c, "step", count)
    return tr, L, ro, state0, restore, steps


def _scripted_kl(L, monkeypatch, seq):
    """Replace the KL measurement by a script: the k-th call (after the k-th Adam step) returns ``seq[k-1]`` (zeros
    after the script) and records the actor and Adam state at that moment, i.e. just before step k + 1."""
    calls, after = [], {}

    def kl(rows, old):
        calls.append(old)
        after[len(calls)] = (copy.deepcopy(L.actor.state_dict()), copy.deepcopy(L.opt_a.state_dict()))
        return seq[len(calls) - 1] if len(calls) <= len(seq) else (0.0, 0.0, 0.0)

    monkeypatch.setattr(L, "_kl", kl)
    return calls, after


def _same(L, actor: dict, opt: dict) -> bool:
    now = L.opt_a.state_dict()["state"]
    return (all(torch.equal(v, L.actor.state_dict()[k]) for k, v in actor.items()) and now.keys() == opt["state"].keys()
            and all(torch.equal(now[k][x], opt["state"][k][x]) for k in now for x in ("step", "exp_avg", "exp_avg_sq")))


def test_kl_guard_undoes_only_the_crossing_step(guard_world, monkeypatch):
    tr, L, ro, state0, restore, steps = guard_world
    n = len(ro.rows)
    p = -(-n // oppo.MINIBATCH)                                   # Adam steps in one pass
    ok = (0.001, 0.001, 0.0)
    cases = {                                                     # script -> (accepted, complete passes)
        "step 2 crosses (joint)": ([ok, (0.5, 0.0, 0.0)], 1, 0),
        "step 1 crosses (launch)": ([(0.0, 0.5, 0.0)], 0, 0),
        "a later pass crosses (duration)": ([ok] * (p + 1) + [(0.0, 0.0, 0.5)], p + 1, 1),
        "nothing crosses": ([], oppo.EPOCHS * p, oppo.EPOCHS),
    }
    for name, (seq, acc, passes) in cases.items():
        restore()
        steps["critic"], u0 = 0, L.n_updates
        calls, after = _scripted_kl(L, monkeypatch, seq)
        st = L.learn(ro)
        rej = int(acc < oppo.EPOCHS * p)
        assert (st["steps_accepted"], st["steps_rejected"], st["passes_complete"]) == (acc, rej, passes), name
        assert L.n_updates - u0 == acc + rej == len(st["kl"]) == len(calls), name     # no retry, no further step
        assert all(c is calls[0] for c in calls), name            # always against the actor that collected the world
        assert steps["critic"] == oppo.EPOCHS * p, name           # the critic completes its four passes
        if acc == 0:
            assert _same(L, state0["actor"], state0["opt_a"]), name
        else:                                                     # the state just before the crossing step, not before the world
            assert _same(L, *after[acc]) and not _same(L, state0["actor"], state0["opt_a"]), name
        assert st["kl_final"] == (list(seq[acc - 1]) if 0 < acc <= len(seq) else [0.0, 0.0, 0.0]), name


def test_kl_guard_measures_the_whole_world_against_the_collecting_actor(guard_world, monkeypatch):
    tr, L, ro, state0, restore, steps = guard_world
    # one decision with a single feasible duration, whatever the world played: a launch's durations are unused, so
    # its stored probability and value stand
    next(r for r in ro.rows if r.action[0] == P.LAUNCH).masks[P.SLICES["dur"]] = (True, False, False, False, False)
    m = np.stack([r.masks for r in ro.rows])
    support = [len(ro.rows), int(m[:, P.SLICES["gate"]][:, P.LAUNCH].sum()), int((m[:, P.SLICES["dur"]].sum(1) > 1).sum())]
    assert support[2] < support[0]                                # rows with a single feasible duration exist
    assert any(r.action[0] == P.WAIT and r.masks[P.SLICES["gate"]][P.LAUNCH] for r in ro.rows)  # sampled waits count
    seen = []
    real_kl = L._kl

    def kl(rows, old):
        seen.append(len(rows))
        return real_kl(rows, old)

    monkeypatch.setattr(L, "_kl", kl)
    restore()
    monkeypatch.setattr(oppo, "KL_MAX", 1e9)
    free = L.learn(ro)
    assert free["kl_support"] == support and set(seen) == {len(ro.rows)}
    k1, k2 = max(free["kl"][0]), max(free["kl"][1])
    assert 0 < k1 < k2                                            # drift from the collecting actor accumulates
    restore()
    monkeypatch.setattr(oppo, "KL_MAX", (k1 + k2) / 2)            # the real KL: step 1 fits, step 2 does not
    st = L.learn(ro)
    assert (st["steps_accepted"], st["steps_rejected"]) == (1, 1)
    assert np.allclose(st["kl"][1], free["kl"][1], atol=1e-6)     # measured from the same reference as before


def test_a_partial_pass_survives_save_and_load_and_non_finite_still_aborts(guard_world, monkeypatch, tmp_path):
    tr, L, ro, state0, restore, steps = guard_world
    restore()
    _scripted_kl(L, monkeypatch, [(0.001, 0.0, 0.0)] * 2 + [(1.0, 0.0, 0.0)])
    st = L.learn(ro)
    assert (st["steps_accepted"], st["passes_complete"]) == (2, 0)
    tr.save(tmp_path / "partial.pt")
    other = _trainer(tmp_path, monkeypatch, episodes=6, curriculum=0, out_name="other")
    other.load(tmp_path / "partial.pt")
    assert _same(other.roles["launderer"], L.actor.state_dict(), L.opt_a.state_dict())
    restore()
    with torch.no_grad():
        next(L.actor.parameters()).fill_(float("nan"))
    with pytest.raises(FloatingPointError):
        L.learn(ro)


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
    r = ob.read_block(res, good, matched, refs, a_final, 30)
    assert r["promising"] and math.isclose(r["utility_bar"], 0.40) and math.isclose(r["reward_bar"], 0.35)
    fails = {
        "c1": dict(res=res | {"ratio_dev_max": 0.01}), "c2": dict(snaps={20: _rows([0.40] * 4), 30: _rows([0.39] * 4)}),
        "c3": dict(snaps={20: _rows([0.6] * 4), 30: good[30]}),
        "c4": dict(snaps={20: good[20], 30: _rows([0.5] * 4, held=0.1, bias=0.6)}),
        "c5": dict(snaps={20: good[20], 30: _rows([0.5] * 4, held=0.14)}),
    }
    for c, kw in fails.items():
        x = ob.read_block(kw.get("res", res), kw.get("snaps", good), matched, refs, a_final, 30)
        assert not x[c] and not x["promising"] and all(x[k] for k in ("c1", "c2", "c3", "c4", "c5") if k != c), c
    assert not ob.read_block(res, good, _rows([-0.79] * 4), refs, a_final, 30)["c5"]  # matched skill under 0.02
    big = {20: good[20], 30: _rows([0.5] * 4, held=0.1, bias=1.9, g_abs=8.0)}         # tolerance 0.25 mean |G|
    assert ob.read_block(res, big, matched, refs, a_final, 30)["c4"]
    assert not ob.read_block(res | {"status": "aborted: x"}, good, matched, refs, a_final, 30)["promising"]
    assert not ob.read_block(res, good, matched, refs, a_final, 60)["c1"]            # attempt 60 needs 60 worlds


# ---------------------------------------------------------------------------- the confirmation's reading
def _conf_rows(u, worlds, held=0.05, reward=None, bias=0.1, g_abs=1.0):
    u = np.broadcast_to(np.asarray(u, float), (len(worlds),))
    return [{"world": j, "btc_utility": float(x), "reward": float(x if reward is None else reward), "held_frac": held,
             "cal": {"n": 10, "bias": bias, "g_abs": g_abs}} for j, x in zip(worlds, u)]


_NOISE = np.array([0.01, -0.01, 0.02, -0.02, 0.0, 0.01, -0.01, 0.0])


def _conf_refs(W):
    z = np.resize(_NOISE, len(W))
    return {"direct50": _conf_rows(0.6, W, held=0.08), "uniform": _conf_rows(-0.8 + z, W),
            "wait": _conf_rows(z, W), "repeat50": _conf_rows(0.65, W)}


def _conf_res(worlds=30):
    return {"status": "ok", "worlds_done": worlds, "accounting_gap_max": 0.0, "ratio_dev_max": 1e-6}


def _conf_ev(W, u30=0.5, u20=0.3, matched=0.5, initial=-0.8, held=0.06, bias=0.1, status="ok", snaps=(20, 30)):
    z = np.resize(_NOISE, len(W))
    return {"status": status, f"natural_{snaps[1]}": _conf_rows(u30 + z, W, held=held, bias=bias),
            f"natural_{snaps[0]}": _conf_rows(u20 + z, W), "matched": _conf_rows(matched + z, W),
            "initial": _conf_rows(initial + z, W)}


def test_confirmation_reading_has_six_criteria():
    from hns.marl import oppo_block as ob

    att = ob.ATTEMPTS["30"]
    W, SEEDS = att.conf_worlds, att.seeds
    refs, res = _conf_refs(W), {s: _conf_res() for s in SEEDS}
    ev = lambda **kw: _conf_ev(W, **kw)                                  # noqa: E731
    good = {s: ev() for s in SEEDS}
    r = ob.read_confirmation(res, good, refs, W, att)
    assert r["confirmed"] and all(r[f"c{i}"] for i in range(1, 7)), r
    assert r["matched"]["initial_seed_spread"] == 0.0
    fails = {
        "c1": dict(res=res | {1: res[1] | {"worlds_done": 29}}),
        "c2": dict(evals=good | {2: ev(bias=0.6)}),                    # one seed's critic is enough to fail
        "c3": dict(evals={0: ev(u30=0.4), 1: ev(u30=-0.1, u20=-0.1), 2: ev(u30=-0.1, u20=-0.1)}),  # 1 of 3 seeds positive
        "c4": dict(evals={s: ev(held=0.14) for s in SEEDS}),
        "c5": dict(evals={0: ev(matched=0.5), 1: ev(matched=-0.9), 2: ev(matched=-0.9)}),        # 1 of 3 seeds >= uniform
        "c6": dict(evals={s: ev(u20=0.6) for s in SEEDS}),
    }
    for c, kw in fails.items():
        x = ob.read_confirmation(kw.get("res", res), kw.get("evals", good), refs, W, att)
        assert not x[c] and not x["confirmed"], (c, x)
        assert all(x[k] for k in ("c1", "c2", "c3", "c4", "c5", "c6") if k != c), (c, x)
    x = ob.read_confirmation(res, good | {1: ev(status="aborted in the evaluation: x")}, refs, W, att)
    assert x["status"] == "seed 1 evaluation: aborted in the evaluation: x" and not x["confirmed"]
    # the initial actor is a separate bar: matched skill over uniform alone is not enough
    x = ob.read_confirmation(res, {s: ev(initial=0.49) for s in SEEDS}, refs, W, att)
    assert not x["c5"] and x["c1"] and x["c6"]
    x = ob.read_confirmation(res, good, None, W, att)                  # references missing
    assert x["status"] == "references: incomplete" and not x["confirmed"]


def test_confirmation_reads_one_opponent_when_the_other_left_no_records(tmp_path, monkeypatch):
    import argparse

    import hns.marl.repair2 as r2
    from hns.marl import oppo_block as ob

    monkeypatch.setattr(r2, "RUNS", tmp_path)
    monkeypatch.setattr(ob, "confirmation_reported", lambda *a: {"reported": True})   # not gated; needs full records
    att = ob.ATTEMPTS["30"]
    W = att.conf_worlds
    for s in att.seeds:                                                 # full0: complete; nda1: nothing at all
        d = ob.run_dir(att, "full0", s)
        d.mkdir(parents=True)
        (d / "result.json").write_text(json.dumps(_conf_res()))
        (d / "confirm.json").write_text(json.dumps(_conf_ev(W)))
    for p, rows in _conf_refs(W).items():
        r2.ref_path("full0", "confirmation", p).write_text(json.dumps({"eval": rows}))
    (ob.run_dir(att, "nda1", 1)).mkdir(parents=True)
    (ob.run_dir(att, "nda1", 1) / "result.json").write_text(json.dumps(_conf_res()))   # trained, never evaluated
    ob.cmd_confirm(argparse.Namespace(attempt="30", worlds=att.worlds, eval_worlds=None))
    out = json.loads((tmp_path / "oppo_confirmation.json").read_text())
    assert out["opponents"]["full0"]["reading"]["confirmed"] and out["opponents"]["full0"]["reported"] is not None
    nda = out["opponents"]["nda1"]["reading"]
    assert not nda["confirmed"] and out["opponents"]["nda1"]["reported"] is None
    assert "seed 0 training: incomplete" in nda["status"] and "seed 1 evaluation: incomplete" in nda["status"]
    assert "seed 1 training" not in nda["status"] and nda["status"].endswith("references: incomplete")
    assert not out["confirmed"]
    before = (tmp_path / "oppo_confirmation.json").read_bytes()
    with pytest.raises(SystemExit, match="already read"):                # a confirmation is read once
        ob.cmd_confirm(argparse.Namespace(attempt="30", worlds=att.worlds, eval_worlds=None))
    assert (tmp_path / "oppo_confirmation.json").read_bytes() == before


def test_evaluation_uses_the_references_view_stream_for_every_seed():
    from types import SimpleNamespace

    import hns.marl.capability as cap
    from hns.marl import oppo_block as ob

    ref = SimpleNamespace(cfg=MarlConfig(seed=cap.SEED), step_no=cap.EVAL_STEP + 123)
    tr = SimpleNamespace(cfg=MarlConfig(seed=2), step_no=ref.step_no)
    own = mt.Trainer.view_seed(tr, 3)
    cfg = tr.cfg
    with pytest.raises(KeyError), ob.eval_views(tr):
        assert all(mt.Trainer.view_seed(tr, s) == mt.Trainer.view_seed(ref, s) for s in (1, 3))
        assert tr.cfg.seed == cap.SEED and replace(tr.cfg, seed=2) == cfg      # nothing else changes
        raise KeyError                                                          # restored on an exception too
    assert tr.cfg is cfg and mt.Trainer.view_seed(tr, 3) == own != mt.Trainer.view_seed(ref, 3)


# ---------------------------------------------------------------------------- OPPO-MC-60 (the second attempt)
def test_attempts_keep_their_records_apart():
    import hns.marl.repair2 as r2
    from hns.marl import oppo_block as ob

    a30, a60 = ob.ATTEMPTS["30"], ob.ATTEMPTS["60"]
    assert ob.run_dir(a30, "full0", 0) == r2.RUNS / "OPPO_full0_s0"                 # attempt 30's names are unchanged
    assert ob.mark_path(a30, "nda1", 1) == r2.RUNS / "oppo_nda1_s1.json"
    assert ob.run_dir(a60, "nda1", 4) == r2.RUNS / "OPPO60_nda1_s4"
    assert ob.mark_path(a60, "full0", 3, smoke=True) == r2.RUNS / "smoke" / "oppo60_full0_s3.json"
    assert (a60.worlds, a60.snaps, a60.dev_seed, a60.seeds) == (60, (40, 60), 3, (3, 4, 5))
    assert a60.conf_worlds == tuple(range(208, 216)) and a30.conf_worlds == tuple(range(200, 208))
    used = set(ob.DEV_WORLDS) | set(r2.PANELS["selection"]) | set(a30.conf_worlds) | set(r2.TIMELINE_WORLDS) \
        | set(r2.PANEL_WORLDS)
    assert not used & set(a60.conf_worlds) and not set(a30.seeds) & set(a60.seeds)
    assert set(a60.rollouts) >= {15, 30, 40, 60}


def test_confirmation_reading_of_attempt_60_uses_its_endpoints():
    from hns.marl import oppo_block as ob

    att = ob.ATTEMPTS["60"]
    W = att.conf_worlds
    refs = _conf_refs(W)
    ev = lambda **kw: _conf_ev(W, snaps=att.snaps, **kw)                # noqa: E731
    res = {s: _conf_res(60) for s in att.seeds}
    r = ob.read_confirmation(res, {s: ev() for s in att.seeds}, refs, W, att)
    assert r["confirmed"], r
    x = ob.read_confirmation({s: _conf_res(30) for s in att.seeds}, {s: ev() for s in att.seeds}, refs, W, att)
    assert not x["c1"] and x["c4"]                                      # 30 worlds are not this attempt's budget
    x = ob.read_confirmation(res, {s: ev(u20=0.6) for s in att.seeds}, refs, W, att)
    assert not x["c6"]                                                  # retention against world 40
    x = ob.read_confirmation(res, {3: ev(), 4: ev(), 5: ev(held=0.4)}, refs, W, att)
    assert not x["c4"]                                                  # one holding seed fails the mean


@pytest.mark.parametrize("argv", [
    ["run", "--attempt", "60", "--opp", "full0", "--seed", "0"],          # development is seed 3
    ["train", "--attempt", "60", "--opp", "nda1"],                        # the default seed is the development seed
    ["train", "--attempt", "60", "--opp", "nda1", "--seed", "6"],         # not a confirmation seed
    ["confirm-eval", "--attempt", "60", "--opp", "full0", "--seed", "1"],
])
def test_attempt_60_refuses_seeds_outside_its_declaration(argv, tmp_path, monkeypatch):
    import hns.marl.repair2 as r2
    from hns.marl import oppo_block as ob

    monkeypatch.setattr(r2, "RUNS", tmp_path)
    with pytest.raises(SystemExit, match="seed"):
        ob.main(argv)
    assert not list(tmp_path.iterdir())                                 # refused before any marker or run directory


def test_snapshot_keeps_the_opponents_streams(tmp_path):
    from types import SimpleNamespace

    from hns.marl import oppo_block as ob

    g = lambda k: np.random.default_rng([7, k])                          # noqa: E731
    tr = SimpleNamespace(world=4, step_no=2000, rng=g(1), replay_rng=g(2),
                         roles={"launderer": SimpleNamespace(state=lambda: {"actor": {}})},
                         policies=[SimpleNamespace(rng=g(11), sched_rng=g(12))])
    tr.rng.random(5)                                                    # a stream that has moved
    ob.snapshot(tr, tmp_path / "w04.pt")
    d = torch.load(tmp_path / "w04.pt", weights_only=False)
    assert d["opp_rng"] == tr.rng.bit_generator.state and d["replay_rng"] == tr.replay_rng.bit_generator.state
    assert (d["world"], d["step_no"]) == (4, 2000) and d["sched"] == tr.policies[0].sched_rng.bit_generator.state


def test_block_report_compares_with_pl_at_its_world_30_for_any_endpoint(tmp_path, monkeypatch):
    import hns.marl.repair2 as r2
    from hns.marl import oppo_block as ob

    monkeypatch.setattr(r2, "RUNS", tmp_path)

    def row(j, u):
        return {"world": j, "btc_utility": u, "reward": u, "held_frac": 0.05, "success_frac": 0.5, "frozen_frac": 0.1,
                "fee_frac": 0.0, "frozen_change_frac": 0.0, "tranches": 1.0, "detection_rate": 0.1,
                "cal": {"bias_btc": 0.0, "rmse": 1.0, "rmse_norm": 0.1, "explained_variance": 0.5, "g_mean": 0.3,
                        "strata": {"low": {"n": 2, "bias": 0.1}}},
                "options": {"decisions": 10, "waits": [1, 0, 0, 0, 1], "launches": [0] * 36, "launch_noops": 0,
                            "hazard": {"low/0": [3, 1]}}}

    rows = lambda u: [row(j, u) for j in ob.DEV_WORLDS]                  # noqa: E731
    d = r2.run_dir("PL", "full0", 0)
    d.mkdir(parents=True)
    (d / "stage1.json").write_text(json.dumps({"snaps": {"20": rows(0.0), "30": rows(0.2)}}))
    refs = {p: rows(0.4) for p in ("wait", "direct50", "repeat50")}
    rep = ob.reported({40: rows(0.3), 60: rows(0.5)}, refs, "full0", None)
    assert math.isclose(rep["vs"]["PL_w30"]["utility"][0], 0.3)


# ---------------------------------------------------------------------------- OPPO-MC-60R (the third attempt)
def test_attempt_60r_is_a_confirmation_on_sixteen_unused_worlds():
    import hns.marl.repair2 as r2
    from hns.marl import oppo_block as ob

    a30, a60, a = ob.ATTEMPTS["30"], ob.ATTEMPTS["60"], ob.ATTEMPTS["60r"]
    assert (a.worlds, a.snaps, a.dev_seed, a.seeds, a.rollouts) == (60, (40, 60), None, (6, 7, 8), a60.rollouts)
    assert a.conf_worlds == tuple(range(224, 240))
    assert ob.run_dir(a, "nda1", 7) == r2.RUNS / "OPPO60R_nda1_s7"
    assert ob.mark_path(a, "full0", 6, smoke=True) == r2.RUNS / "smoke" / "oppo60r_full0_s6.json"
    pilot = set(range(216, 224)) | set(range(240, 248))                 # kept for the live pilot (SPEC §10)
    used = (set(ob.DEV_WORLDS) | set(r2.PANELS["selection"]) | set(a30.conf_worlds) | set(a60.conf_worlds)
            | set(r2.TIMELINE_WORLDS) | set(r2.PANEL_WORLDS) | pilot)
    assert not used & set(a.conf_worlds) and not set(a.seeds) & (set(a30.seeds) | set(a60.seeds))
    assert a.retain == RETAIN_60R and (a30.retain, a60.retain) == (0.05, 0.05)   # the closed attempts keep their margin


RETAIN_60R = 0.15                                                       # the declared margin (oppo_block's docstring)


@pytest.mark.parametrize("argv", [
    ["run", "--attempt", "60r", "--opp", "full0", "--seed", "6"],         # no development block
    ["read", "--attempt", "60r", "--opp", "full0", "--seed", "6"],        # so no development reading either
    ["train", "--attempt", "60r", "--opp", "nda1"],                       # no default seed
    ["train", "--attempt", "60r", "--opp", "nda1", "--seed", "3"],        # attempt 60's seed
    ["confirm-eval", "--attempt", "60r", "--opp", "full0", "--seed", "5"],
])
def test_attempt_60r_refuses_what_it_did_not_declare(argv, tmp_path, monkeypatch):
    import hns.marl.repair2 as r2
    from hns.marl import oppo_block as ob

    monkeypatch.setattr(r2, "RUNS", tmp_path)
    with pytest.raises(SystemExit, match="seed"):
        ob.main(argv)
    assert not list(tmp_path.iterdir())


def test_attempt_60r_has_no_development_reading_of_a_confirmation_run(tmp_path, monkeypatch):
    import hns.marl.repair2 as r2
    from hns.marl import oppo_block as ob

    monkeypatch.setattr(r2, "RUNS", tmp_path)
    d = ob.run_dir(ob.ATTEMPTS["60r"], "full0", 6)                       # a trained confirmation run, no oppo.json
    d.mkdir(parents=True)
    (d / "result.json").write_text(json.dumps(_conf_res(60)))
    with pytest.raises(SystemExit, match="no development block"):
        ob.main(["read", "--attempt", "60r", "--opp", "full0", "--seed", "6"])
    assert sorted(f.name for f in d.iterdir()) == ["result.json"]


def test_confirmation_reading_of_attempt_60r_uses_its_margin_and_worlds():
    from hns.marl import oppo_block as ob

    att, a60 = ob.ATTEMPTS["60r"], ob.ATTEMPTS["60"]
    W = att.conf_worlds
    refs, res = _conf_refs(W), {s: _conf_res(60) for s in att.seeds}
    ev = lambda **kw: {s: _conf_ev(W, snaps=att.snaps, **kw) for s in att.seeds}   # noqa: E731
    inside, outside = att.retain - 0.03, att.retain + 0.03
    r = ob.read_confirmation(res, ev(u20=0.5 + inside), refs, W, att)
    assert r["confirmed"] and r["retention_margin"] == att.retain and math.isclose(r["retention"][0], -inside)
    x = ob.read_confirmation(res, ev(u20=0.5 + outside), refs, W, att)
    assert not x["c6"] and all(x[f"c{i}"] for i in range(1, 6))
    # the same loss fails attempt 60's margin: the closed attempt is read as it was declared
    W60 = a60.conf_worlds
    y = ob.read_confirmation({s: _conf_res(60) for s in a60.seeds},
                             {s: _conf_ev(W60, snaps=a60.snaps, u20=0.5 + inside) for s in a60.seeds},
                             _conf_refs(W60), W60, a60)
    assert not y["c6"] and y["retention_margin"] == 0.05
    # a loss with noise across the sixteen worlds: the lower bound decides, not the mean
    noisy = ev()
    for s in att.seeds:
        for j, row in enumerate(noisy[s]["natural_40"]):
            row["btc_utility"] = 0.5 + att.retain - 0.01 + (0.2 if j % 2 else -0.2)
    z = ob.read_confirmation(res, noisy, refs, W, att)
    assert z["retention"][0] > -att.retain > z["retention"][1] and not z["c6"]


def test_confirmation_report_adds_retention_by_seed_and_in_discounted_reward():
    from hns.marl import oppo_block as ob

    att = ob.ATTEMPTS["60r"]
    W = att.conf_worlds
    evals = {}
    for k, s in enumerate(att.seeds):
        e = _conf_ev(W, snaps=att.snaps, u20=0.5 + 0.02 * k)
        for key, disc in (("natural_40", 1.5), ("natural_60", 1.7 + 0.1 * k)):
            for row in e[key]:
                row |= {"reward_discounted": disc, "success_frac": 0.7, "frozen_frac": 0.2, "fee_frac": 0.0,
                        "tranches": 5.0, "detection_rate": 0.1, "options": {"launches": [0] * 36}}
        e |= {"argmax": None, "state_use": {"60": {"n": 3}}, "actor_steps": 100 + k}
        evals[s] = e
    rep = ob.confirmation_reported(evals, _conf_refs(W), W, att)
    assert rep["retention_by_seed"] == pytest.approx({6: 0.0, 7: -0.02, 8: -0.04})
    assert rep["retention_discounted"][0] == pytest.approx(0.3)                      # +0.2, +0.3, +0.4
    assert rep["retention_discounted_by_seed"] == pytest.approx({6: 0.2, 7: 0.3, 8: 0.4})
    assert rep["state_use"][7] == {"60": {"n": 3}} and rep["actor_steps"] == {6: 100, 7: 101, 8: 102}


def test_state_use_measures_what_replacing_the_state_changes(guard_world):
    from hns.marl import oppo_block as ob

    _, L, ro, _, restore, _ = guard_world
    rows = ro.rows
    ob_ = collate([r.obs for r in rows])
    masks = np.stack([r.masks for r in rows])
    feas = masks[:, P.SLICES["gate"]][:, P.LAUNCH]
    assert feas.any() and np.ptp(ob_.x_c, axis=0).max() > 0
    with torch.no_grad():                                               # a gate that reads its body
        L.actor.gate.weight.copy_(torch.randn(L.actor.gate.weight.shape, generator=torch.Generator().manual_seed(3)))
    u = ob.state_use(L, rows)
    assert u["n"] == int(feas.sum()) and u["p_launch"][0] <= u["p_launch"][1] <= u["p_launch"][2]
    assert sum(u["band_n"].values()) == u["n"] and not L.actor.trunk.norm._forward_hooks
    with torch.no_grad():
        p = L._forward(ob_, masks)["gate"].exp()[:, P.LAUNCH].numpy()
        flat = replace(ob_, x_c=np.broadcast_to(ob_.x_c.mean(0), ob_.x_c.shape).copy())
        p_c = L._forward(flat, masks)["gate"].exp()[:, P.LAUNCH].numpy()
        g, f = ob_.tensors(L.device)
        h = L.actor.trunk(g, f["x_tx"], f["x_w"], f["x_c"]).clone()
        d = L.actor.trunk.enc.out_dim
        h[:, :d] = h[:, :d].mean(0, keepdim=True)
        p_g = torch.softmax(L.actor.gate(L.actor.gate_body(h)), 1)[:, P.LAUNCH].numpy()
    assert u["dp_launch"]["controller"] == pytest.approx(np.abs(p - p_c)[feas].mean(), rel=1e-4, abs=1e-7)
    assert u["dp_launch"]["graph"] == pytest.approx(np.abs(p - p_g)[feas].mean(), rel=1e-4, abs=1e-7)
    assert u["p_launch"] == pytest.approx(np.quantile(p[feas], [0.05, 0.5, 0.95]), abs=1e-6)
    assert u["dp_launch"]["controller"] > 0 or u["dp_launch"]["graph"] > 0
    small = ob.state_use(L, rows, chunk=3)                              # the means are over every row, not a chunk
    assert small["dp_launch"] == pytest.approx(u["dp_launch"], rel=1e-3, abs=1e-6)
    assert small["mode_tv"] == pytest.approx(u["mode_tv"], rel=1e-3, abs=1e-6)
    with torch.no_grad():                                               # a policy of biases alone reads no state
        for m in (L.actor.gate, L.actor.mode):
            m.weight.zero_()
        L.actor.gate.bias.copy_(torch.tensor([0.0, 1.0]))
    blind = ob.state_use(L, rows)
    assert blind["p_launch"] == pytest.approx([1 / (1 + math.exp(-1.0))] * 3, abs=1e-6)
    assert max(*blind["dp_launch"].values(), *blind["mode_tv"].values()) < 1e-6
    restore()


def test_retention_power_matches_a_simulation():
    from scipy import stats

    from hns.marl import oppo_block as ob

    rng = np.random.default_rng(0)
    for true, sd, n, margin in ((0.0, 0.063, 8, 0.05), (-0.036, 0.077, 16, 0.10), (-0.07, 0.07, 16, 0.10)):
        x = rng.normal(true, sd, (40000, n))
        low = x.mean(1) - stats.t.ppf(0.975, n - 1) * x.std(1, ddof=1) / math.sqrt(n)
        assert ob.retention_power(true, sd, n, margin) == pytest.approx((low > -margin).mean(), abs=0.01)
    assert ob.retention_power(-0.10, 0.07, 16, 0.10) == pytest.approx(0.025, abs=1e-6)   # at the margin: the test's size
    assert ob.retention_power(0.0, 0.063, 8, 0.05) < 0.5 < ob.retention_power(0.0, 0.063, 16, 0.10)


def test_bootstrap_power_reads_a_recorded_attempt_under_the_new_rules(tmp_path, monkeypatch):
    import hns.marl.repair2 as r2
    from hns.marl import oppo_block as ob

    monkeypatch.setattr(r2, "RUNS", tmp_path)
    src, att = ob.ATTEMPTS["60"], ob.ATTEMPTS["60r"]
    W = src.conf_worlds
    loss = {"full0": att.retain - 0.03, "nda1": att.retain + 0.03}      # inside the new margin, and outside it
    for opp in r2.OPPS:
        for s in src.seeds:
            d = ob.run_dir(src, opp, s)
            d.mkdir(parents=True)
            (d / "result.json").write_text(json.dumps(_conf_res(60)))
            (d / "confirm.json").write_text(json.dumps(_conf_ev(W, snaps=src.snaps, u20=0.5 + loss[opp])))
        for p, rows in _conf_refs(W).items():
            r2.ref_path(opp, src.panel, p).write_text(json.dumps({"eval": rows}))
    out = ob.bootstrap_power(src, att, draws=40)
    assert out["full0"] == {**{f"c{i}": 1.0 for i in range(1, 7)}, "confirmed": 1.0}
    assert out["nda1"]["c6"] == 0.0 and out["nda1"]["confirmed"] == 0.0 and out["nda1"]["c5"] == 1.0
    same = ob.bootstrap_power(src, src, draws=40)                       # under the old margin both losses fail
    assert same["full0"]["c6"] == 0.0 and same["full0"]["c4"] == 1.0
    (ob.run_dir(src, "nda1", 4) / "confirm.json").unlink()              # an incomplete record gives no estimate
    assert ob.bootstrap_power(src, att, draws=5)["nda1"] is None
    with pytest.raises(SystemExit, match="budget"):
        ob.bootstrap_power(ob.ATTEMPTS["30"], att, draws=5)
