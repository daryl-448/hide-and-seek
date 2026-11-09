"""M4: feasibility masks, realized branches and the atomic sub-step order (SPEC §9.3)."""

from collections import Counter
from dataclasses import replace

import numpy as np
import pytest

from hns.sim.entities import GO, INIT_DEFER, REUSE, STRATEGIES, WAIT, Demand
from hns.sim.env import NOOP_INNOCENT, NOOP_LAUNDERER, Proposal, SimEnv, Transition
from hns.sim.export import export_world
from hns.sim.utxo import TxSpec
from hns.sim.world import EPISODES, ROLE_INNOCENT, ROLE_LAUNDERER, SUBSTEPS, WorldParams

SMALL = WorldParams(n_innocents=300, n_services=5, n_launderers=5, batch_size=5, consol_interval=3,
                    demand_rate=0.03, p_ill=0.5, p_lic=0.3, p_wunk=0.2)


def test_launderer_branch_sets_and_masks():
    env = SimEnv(replace(SMALL, n_services=3), seed=0)
    L = env.launderers[0]
    env._refresh_state()
    assert env.launderer_state(L) is None                 # no funds: no transition
    env.ledger.genesis(L.source, 1e-4)
    L.k, L.feerate = 2, 1e-8
    st = env.launderer_state(L)
    assert st.branches == ("initiate", "amount", "recipient", "strategy", "mules", "change")
    # 0.10 x (1e-4 - fee) / k < 1e-5 BTC dust; the rest are feasible
    assert st.masks["amount"].tolist() == [False, True, True, True]
    assert st.masks["recipient"].tolist() == [True] * 3 + [False] * 13   # 3 services: empty slots
    assert st.masks["mules"].tolist() == [True, False]                   # pool 0 < k
    assert st.masks["initiate"].tolist() == [True, True]
    for _ in range(2):
        env.fresh_mule(L)                                                # pool 2 >= k
    assert env.launderer_state(L).masks["mules"].tolist() == [True, True]
    # nothing affordable: initiate = go is masked
    L2 = env.launderers[1]
    env.ledger.genesis(L2.source, 2e-5)
    L2.k, L2.feerate = 5, 1e-8
    st2 = env.launderer_state(L2)
    assert st2.masks["initiate"].tolist() == [True, False]
    # a tranche in flight: only initiate (proceed or wait)
    L.tranche = object()
    assert env.launderer_state(L).branches == ("initiate",)


def test_masks_depend_on_state_only():
    env = SimEnv(SMALL, seed=1)
    env.start_episode(0)
    env._refresh_state()
    L = env.launderers[0]
    a, b = env.launderer_state(L), env.launderer_state(L)
    assert a.branches == b.branches and all((a.masks[k] == b.masks[k]).all() for k in a.masks)


def test_innocent_split_mask():
    env = SimEnv(SMALL, seed=0)
    env.pending[0] = Demand(rcpt=env.inn_addrs[1][0], amount=1.5e-5, to_service=-1)
    st = env.innocent_state(0)
    assert st.branches == ("initiate", "split", "route", "change")
    assert st.masks["split"].tolist() == [True, False, False]
    env.pending[0].stage = 1
    assert env.innocent_state(0) is None                 # continuation: no decision, no transition
    assert env.innocent_state(1) is None                 # no demand


class _Fixed:
    """Launderer policy that plays one fixed action for every state."""

    def __init__(self, a):
        self.a = a

    def act(self, env, states):
        return {i: (dict(self.a) if len(st.branches) > 1 else {"initiate": self.a["initiate"]}) for i, st in states.items()}


@pytest.mark.parametrize("amount, strategy, dropped", [
    (3, 0, {"change", "mules"}),        # 100 %, direct
    (0, 1, set()),                      # 10 %, fan_out
    (2, 7, {"mules"}),                  # peel: always fresh mules
])
def test_realized_branches(amount, strategy, dropped):
    a = {"initiate": GO, "amount": amount, "recipient": 0, "strategy": strategy, "mules": 0, "change": 0}
    env = SimEnv(replace(SMALL, n_launderers=1), seed=2, launderers=_Fixed(a))
    env.start_episode(0)
    r = env.step()
    tr = [t for t in r.transitions if t.role == "launderer"][0]
    assert set(tr.branches) - set(tr.realized) == dropped and not tr.noop
    env2 = SimEnv(replace(SMALL, n_launderers=1), seed=2, launderers=_Fixed({**a, "initiate": WAIT}))
    env2.start_episode(0)
    tr2 = [t for t in env2.step().transitions if t.role == "launderer"][0]
    assert tr2.realized == ("initiate",)


@pytest.mark.parametrize("s", range(len(STRATEGIES)))
def test_every_template_reaches_the_recipient_within_six_hops(s):
    a = {"initiate": GO, "amount": 3, "recipient": 0, "strategy": s, "mules": 0, "change": 0}
    env = SimEnv(replace(SMALL, n_launderers=1, budget=0.0), seed=3, launderers=_Fixed(a))
    env.start_episode(0)
    L = env.launderers[0]
    funds = env.funds(L)
    hops = 0
    for _ in range(8):
        env.budget = 0.0                                 # nobody is flagged
        env.step()
        hops += 1
        if L.tranche is None:
            break
    assert L.tranche is None and hops <= 6
    led = env.ledger
    rcpt = [x for x in env.deposit if x[0] == ROLE_LAUNDERER]
    assert len(rcpt) == 1
    cashed = sum(led.u_amt[u] for u in range(len(led.u_addr)) if led.u_addr[u] == env.deposit[rcpt[0]])
    fees = sum(f for f, r in zip(led.tx_fee, led.tx_role) if r == ROLE_LAUNDERER)
    assert cashed == pytest.approx(funds - fees, rel=1e-9)  # 100 %, nobody flags: everything cashes out
    assert all(led.balance[m] == 0 for m in L.pool)


def test_cycle_revisits_first_mule_and_reuse_draws_from_pool():
    a = {"initiate": GO, "amount": 1, "recipient": 0, "strategy": STRATEGIES.index("cycle"), "mules": 0, "change": 0}
    env = SimEnv(replace(SMALL, n_launderers=1, budget=0.0), seed=4, launderers=_Fixed(a))
    env.start_episode(0)
    L = env.launderers[0]
    for _ in range(4):
        env.budget = 0.0
        env.step()
    assert L.tranche is None
    led = env.ledger
    hop_out = [[x for t, x, _ in led.out_rows if t == tid] for tid in range(led.n_tx) if led.tx_role[tid] == ROLE_LAUNDERER]
    m1 = hop_out[0][0]
    assert hop_out[2] == [m1]                                # m2 -> m1
    for _ in range(5):
        env.fresh_mule(L)                                    # pool >= any k in 2..5
    pool_before = list(L.pool)
    a2 = {**a, "strategy": STRATEGIES.index("fan_out"), "mules": REUSE}
    env.launderer_policy = _Fixed(a2)
    for _ in range(2):
        env.budget = 0.0
        env.step()
    assert any(led.tx_role[t] == ROLE_LAUNDERER and led.tx_sub[t] == env.sub - 1 for t in range(led.n_tx))
    assert set(L.pool) == set(pool_before)                  # reuse: no new mule (pool >= k)


def _two_proposals(env, specs, kinds=("demand", "demand")):
    trs = [Transition("innocent", i, ("initiate",), {}, {"initiate": 0}, realized=("initiate", "split", "route", "change"))
           for i in range(len(specs))]
    props = [Proposal(k, i, [s], trs[i]) for i, (k, s) in enumerate(zip(kinds, specs))]
    return props, trs


def test_reservation_conflict_becomes_canonical_noop():
    env = SimEnv(SMALL, seed=5)
    env.start_episode(0)
    led = env.ledger
    a = env.inn_addrs[0][0]
    utx = led.spendable([a])
    fr = 1e-8
    s1 = TxSpec(utx, pay=[(env.inn_addrs[1][0], 1e-4)], rest=[a], feerate=fr, role=ROLE_INNOCENT, entity=0,
                meta={"route": 0, "split": 1, "change": 1, "payee": env.inn_addrs[1][0], "has_change": True})
    s2 = TxSpec(list(utx), pay=[(env.inn_addrs[2][0], 1e-4)], rest=[a], feerate=fr, role=ROLE_INNOCENT, entity=1,
                meta={"route": 0, "split": 1, "change": 1, "payee": env.inn_addrs[2][0], "has_change": True})
    env.pending[0] = Demand(env.inn_addrs[1][0], 1e-4, -1)
    env.pending[1] = Demand(env.inn_addrs[2][0], 1e-4, -1)
    props, trs = _two_proposals(env, [s1, s2])
    env._propose = lambda: (props, trs)
    env.budget = 0.0
    r = env.step()
    assert r.n_noop == 1 and led.n_tx == 1
    noop = [t for t in trs if t.noop]
    ok = [t for t in trs if not t.noop]
    assert len(noop) == 1 and noop[0].action == NOOP_INNOCENT and noop[0].realized == ("initiate",)
    assert noop[0].reward == 0.0 and ok[0].reward == pytest.approx(0.1)
    assert env.pending[noop[0].agent] is not None and env.pending[ok[0].agent] is None
    # insufficient funds is also a no-op
    s3 = TxSpec(led.spendable([env.inn_addrs[3][0]]), pay=[(a, 1e6)], rest=[env.inn_addrs[3][0]], feerate=fr,
                role=ROLE_INNOCENT, entity=3, meta={"route": 0, "split": 1, "change": 1, "payee": a, "has_change": True})
    props, trs = _two_proposals(env, [s3])
    env._propose = lambda: (props, trs)
    assert env.step().n_noop == 1 and trs[0].noop and led.n_tx == 1
    assert NOOP_LAUNDERER == {"initiate": WAIT} and NOOP_INNOCENT == {"initiate": INIT_DEFER}


class _FlagLicit:
    def act_global(self, env, state):
        return {"analysis_depth": 0, "update_strategy": 1}

    def score(self, env, cands):
        return np.array([0.0 if r.spec.illicit else 1.0 for r in cands])


def test_flagged_licit_txs_are_reverted_and_leave_no_trace():
    env = SimEnv(SMALL, seed=6, detector=_FlagLicit())
    env.start_episode(0)
    for _ in range(3):
        env.budget = 0.2
        env.step()
    led = env.ledger
    n_before = led.n_tx
    pend = list(env.pending)
    queues = [list(s.queue) for s in env.services]
    bal = list(led.balance)
    n_addr = len(led.addr_role)
    env.budget = 1.0                                     # flag every candidate this sub-step
    r = env.step()
    rev = [c for c in r.candidates if c["outcome"] == "reverted"]
    assert rev and all(not c["illicit"] and c["tid"] == -1 for c in rev)
    assert all(c["outcome"] == "frozen" for c in r.candidates if c["illicit"])
    new = list(range(n_before, led.n_tx))
    assert all(led.tx_truth[t] and led.tx_frozen[t] for t in new)       # only frozen illicit txs entered
    for i, d in enumerate(pend):                                        # demands stay pending
        if d is not None and d.stage == 0:
            assert env.pending[i] is d
    assert [list(s.queue)[:len(q)] for s, q in zip(env.services, queues)] == queues
    for a in range(n_addr):                                             # reverted inputs released, no outputs
        if led.addr_role[a] != ROLE_LAUNDERER:
            assert led.balance[a] == pytest.approx(bal[a])
    assert all(t.reward == 0.0 for t in r.transitions if t.role == "innocent")
    ex = export_world(env)
    assert set(ex.txs["ltid"].to_list()) <= set(range(led.n_tx))         # exports only see committed txs
    assert r.detector_reward == pytest.approx(-0.11 * len(rev) + 0.99 * sum(c["illicit"] for c in r.candidates))


def test_static_world_runs_respects_masks_and_is_deterministic():
    envs = []
    for _ in range(2):
        env = SimEnv(SMALL, seed=7)
        env.run_world()
        envs.append(env)
    a, b = (e.ledger for e in envs)
    assert a.n_tx == b.n_tx and a.tx_fee == b.tx_fee and list(a.edges) == list(b.edges)
    sp, fr = a.unspent_total()
    assert a.genesis_total == pytest.approx(sp + fr + sum(a.tx_fee))
    assert envs[0].episode == EPISODES - 1 and envs[0].sub == SUBSTEPS
    roles = Counter(a.tx_role)
    assert all(roles[r] > 0 for r in range(3))
    assert set(a.tx_ts) == set(range(1, EPISODES + 1))


class _Route:
    """Innocent policy: pay now with a fixed route and split."""

    def __init__(self, route, split=0):
        self.route, self.split = route, split

    def act(self, env, states):
        return {i: {"initiate": 0, "split": self.split if st.masks["split"][self.split] else 0, "route": self.route,
                    "change": 0} for i, st in states.items()}


class _Checked(SimEnv):
    """Records (demand amount, BTC the recipient got) for every completed demand."""

    def _on_demand(self, pr):
        d = self.pending[pr.owner]
        super()._on_demand(pr)
        if pr.outcomes[0] == "committed" and pr.resolved[0].spec.meta["route"] == 0:
            got = sum(a for x, a in pr.resolved[0].outputs if x == d.rcpt)
            self.paid.append((d.amount, got))


@pytest.mark.parametrize("split", [0, 2])
def test_demands_are_paid_exactly(split):
    env = _Checked(replace(SMALL, demand_a=5.0, demand_b=0.5), seed=8, innocents=_Route(0, split))  # near-total spends
    env.paid = []
    for e in range(2):
        env.start_episode(e)
        for _ in range(SUBSTEPS):
            env.step()
    assert len(env.paid) > 100
    assert all(got == pytest.approx(amt, rel=1e-12) for amt, got in env.paid)


def test_unaffordable_demand_is_a_noop_and_stays_pending():
    env = SimEnv(SMALL, seed=9)
    env.start_episode(0)
    a = env.inn_addrs[0]
    bal = sum(env.ledger.balance[x] for x in a)
    d = Demand(env.inn_addrs[1][0], bal, -1)            # the whole balance leaves nothing for the fee
    env.pending[0] = d
    env.budget = 0.0
    r = env.step()
    tr = [t for t in r.transitions if t.role == "innocent" and t.agent == 0][0]
    assert tr.noop and tr.action == NOOP_INNOCENT and env.pending[0] is d and d.amount == bal


def test_forward_continuation_reward():
    env = SimEnv(SMALL, seed=10, innocents=_Route(1))
    env.start_episode(0)
    env.budget = 0.0
    env.step()                                         # hop 1 of every forward demand
    waiting = {i for i, d in enumerate(env.pending) if d is not None and d.stage == 1}
    assert waiting
    env.detector = _FlagLicit()
    env.budget = 1.0                                   # every hop 2 is blocked
    r = env.step()
    assert set(r.continuation_rewards) == waiting and set(r.continuation_rewards.values()) == {0.0}
    assert all(env.pending[i].stage == 1 for i in waiting)
    env.budget = 0.0
    r = env.step()
    assert set(r.continuation_rewards) >= waiting and set(r.continuation_rewards.values()) == {0.1}
    assert all(env.pending[i] is None or env.pending[i].stage == 0 for i in waiting)
