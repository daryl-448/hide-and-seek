"""Step-able SISTS-lite world with the atomic sub-step order (SPEC §9.2, §9.3).

One :meth:`SimEnv.step` is one sub-step:

1. **Snapshot.** Draw the stochastic parts of ``s_t`` (launderer ``k``,
   feerate and recipient slots; new innocent demands). Every agent sees the
   committed ledger.
2. **Propose.** Policies pick actions from ``B(s)`` under ``M_b(s)``;
   services and in-flight continuations are scripted.
3. **Validate and reserve**, in a seeded random order. A proposal (all its
   txs, atomically) whose inputs are already reserved or insufficient is a
   canonical **no-op**: its transition is stored with the realized action
   ``{initiate: wait/defer}`` and reward 0.
4. **Score** the tentative txs with the detector.
5. **Flag** the top ``round(b * n)``.
6. **Commit or revert.** Unflagged txs commit; flagged illicit txs commit with
   frozen outputs; flagged licit txs are reverted (never enter the ledger;
   the owner retries later).
7. **Reward**, then ``s_{t+1}`` is the committed ledger.

Static mode (calibration, NMAI): :class:`StaticLaunderers` draw every branch
uniformly over its feasible options, :class:`ScriptedInnocents` pay now,
directly, in one output, and both send change to a fresh address with
probability ``p_fresh``. :class:`RandomDetector` scores uniformly at random
at depth 1 and keeps its budget.

Learned policies (M5) observe through ``env.graph``, an optional
:class:`hns.sim.live.LiveGraph` that the env keeps equal to the committed
ledger: the tentative txs of a sub-step are added for scoring (step 4) and
popped again before the commit, so a reverted tx leaves no trace in it
either. ``env.acct`` accumulates what the §10 health checks need, in BTC:
the launderers' proceeds, what they cashed out undetected, everything frozen
in their txs (``frozen``; a flagged setup freezes its change to the source
too, which ``frozen_change`` holds), and their fees. With the launderers'
spendable balance (:meth:`SimEnv.launderer_held`) these close:
``proceeds = cashed + frozen + fees + held``.

Rewards (§9.3, magnitudes ours):

- launderer: ``(cashed out undetected - tranche funds frozen) / tranche``
  per sub-step, ``-0.001`` per sub-step while it holds funds. The frozen
  change of a flagged setup is not in this reward;
- innocent: ``+0.1`` when its decision's tx commits (a flagged tx is blocked);
  a forward route's scripted hop 2 earns its ``+0.1`` in ``continuation_rewards``.

Demand amounts are fixed for a demand's life and are paid exactly. They are
drawn from the balance minus a fee reserve at the 99.9th-percentile feerate
(ours), so an innocent tx is rarely unaffordable; when it is, it is a
canonical no-op and the demand retries with the next sub-step's feerate.
- detector, per sub-step: ``+1`` per flagged illicit tx, ``-1`` per unflagged
  illicit tx paying a service, ``-0.1`` per flagged licit tx, and
  ``-0.01 x depth`` per flagged tx (ours: "investigated" = flagged).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from hns.feats.tx_feats import tx_size
from hns.sim import labels, typologies
from hns.sim.live import tx_batch
from hns.sim.entities import (AMOUNTS, BUDGET_MULT, BUDGET_RANGE, DETECTOR_BRANCHES, FRESH, INIT_DEFER,
                              INIT_NOW, INNOCENT_BRANCHES, K_RECIPIENTS, LAUNDERER_BRANCHES, REUSE, ROUTE_DIRECT,
                              ROUTE_FORWARD, ROUTE_SERVICE, STRATEGIES, WAIT, AgentState, Demand, Launderer,
                              Service, Tranche)
from hns.sim.utxo import Ledger, Resolved, TxSpec
from hns.sim.world import (DUST, EPISODES, ROLE_INNOCENT, ROLE_LAUNDERER, ROLE_SERVICE, SUBSTEPS, WorldParams)

SPLITS = (1, 2, 3)
Z999 = 3.090232306167813  # standard normal 99.9th percentile
NOOP_LAUNDERER = {"initiate": WAIT}
NOOP_INNOCENT = {"initiate": INIT_DEFER}


@dataclass
class Transition:
    role: str
    agent: int
    branches: tuple[str, ...]
    masks: dict
    action: dict
    realized: tuple[str, ...] = ()
    reward: float = 0.0
    noop: bool = False


@dataclass
class Proposal:
    kind: str                 # setup | hop | demand | forward2 | consol | payout
    owner: int
    specs: list[TxSpec]
    tr: Transition | None = None
    tranche: Tranche | None = None
    resolved: list[Resolved] | None = None
    outcomes: list[str] = field(default_factory=list)   # per spec: committed | frozen | reverted
    tids: list[int] = field(default_factory=list)


@dataclass
class StepResult:
    transitions: list[Transition]
    candidates: list[dict]     # per tentative tx: illicit, flagged, cashout, outcome, tid
    detector_reward: float
    n_noop: int
    continuation_rewards: dict[int, float]   # innocent -> r_legit of its scripted forward hop 2 (0 if blocked)


class StaticLaunderers:
    def act(self, env: "SimEnv", states: dict[int, AgentState]) -> dict[int, dict]:
        rng, out = env.policy_rng, {}
        for i, st in states.items():
            a = {}
            for b in st.branches:
                ok = np.flatnonzero(st.masks[b])
                a[b] = int(rng.choice(ok))
            if "change" in a:
                a["change"] = FRESH if rng.random() < env.p.p_fresh else REUSE
            out[i] = a
        return out


class ScriptedInnocents:
    def act(self, env: "SimEnv", states: dict[int, AgentState]) -> dict[int, dict]:
        rng = env.policy_rng
        return {i: {"initiate": INIT_NOW, "split": 0, "route": ROUTE_DIRECT,
                    "change": FRESH if rng.random() < env.p.p_fresh else REUSE} for i in states}


class RandomDetector:
    def act_global(self, env: "SimEnv", state: AgentState) -> dict:
        return {"analysis_depth": 0, "update_strategy": 1}

    def score(self, env: "SimEnv", cands: list[Resolved]) -> np.ndarray:
        return env.detector_rng.random(len(cands))


class SimEnv:
    def __init__(self, params: WorldParams, seed: int, launderers=None, innocents=None, detector=None):
        self.p = params
        dyn, pol, lab, det = np.random.SeedSequence(seed).spawn(4)
        self.rng = np.random.default_rng(dyn)
        self.policy_rng = np.random.default_rng(pol)
        self.label_rng = np.random.default_rng(lab)
        self.detector_rng = np.random.default_rng(det)
        self.launderer_policy = launderers or StaticLaunderers()
        self.innocent_policy = innocents or ScriptedInnocents()
        self.detector = detector or RandomDetector()
        self.ledger = Ledger()
        self.budget = params.budget
        self.episode, self.sub, self.g = 0, 0, 0
        self.deposit: dict[tuple[int, int, int], int] = {}   # (role, owner, service) -> deposit address
        self.graph = None      # optional LiveGraph (M5)
        # health accounting (§10), launderer BTC: received, cashed out undetected, frozen (all outputs of their
        # flagged txs, change included), of which change, and fees; tranches per strategy
        self.acct = {"proceeds": 0.0, "cashed": 0.0, "frozen": 0.0, "frozen_change": 0.0, "fees": 0.0,
                     "strategy": np.zeros(len(STRATEGIES), np.int64)}
        self._build()

    # ------------------------------------------------------------------ world
    def _build(self):
        p, led, rng = self.p, self.ledger, self.rng
        self.services = []
        for s in range(p.n_services):
            hot = led.new_address(ROLE_SERVICE, s)
            led.genesis(hot, p.service_float)
            self.services.append(Service(s, hot, int(rng.integers(p.consol_interval))))
        n_addr = rng.geometric(1.0 / p.addr_mean, size=p.n_innocents)
        self.inn_addrs: list[list[int]] = []
        for i, n in enumerate(n_addr):
            addrs = [led.new_address(ROLE_INNOCENT, i) for _ in range(int(n))]
            for a in addrs:
                led.genesis(a, float(rng.lognormal(p.bal_mu, p.bal_sigma)))
            self.inn_addrs.append(addrs)
        self.inn_degree = np.zeros(p.n_innocents)
        self.pending: list[Demand | None] = [None] * p.n_innocents
        self.launderers = [Launderer(i, led.new_address(ROLE_LAUNDERER, i)) for i in range(p.n_launderers)]

    def start_episode(self, episode: int):
        """New time step: launderers receive this episode's proceeds (ours)."""
        self.episode, self.sub = episode, 0
        for L in self.launderers:
            amt = float(self.rng.lognormal(self.p.proceeds_mu, self.p.proceeds_sigma))
            self.ledger.genesis(L.source, amt)
            self.acct["proceeds"] += amt

    def end_episode(self):
        """The time step is over: freeze its causal wallet counts in the live graph."""
        if self.graph is not None:
            self.graph.end_step(self.ts, len(self.ledger.addr_role))

    def run_world(self):
        for e in range(EPISODES):
            self.start_episode(e)
            for _ in range(SUBSTEPS):
                self.step()
            self.end_episode()

    @property
    def ts(self) -> int:
        return self.p.ts(self.episode)

    def feerate(self) -> float:
        return float(self.rng.lognormal(self.p.fee_mu, self.p.fee_sigma))

    @property
    def reserve_feerate(self) -> float:
        """99.9th-percentile feerate: innocents keep fees at this rate out of a demand (ours)."""
        return float(np.exp(self.p.fee_mu + Z999 * self.p.fee_sigma))

    def deposit_addr(self, role: int, owner: int, service: int) -> int:
        key = (role, owner, service)
        a = self.deposit.get(key)
        if a is None:
            a = self.deposit[key] = self.ledger.new_address(ROLE_SERVICE, service)
            self.services[service].deposits.append(a)
        return a

    def fresh_mule(self, L: Launderer) -> int:
        m = self.ledger.new_address(ROLE_LAUNDERER, L.id)
        L.pool.append(m)
        return m

    def launderer_addr(self, L: Launderer) -> int:
        return self.ledger.new_address(ROLE_LAUNDERER, L.id)

    def draw_mules(self, L: Launderer, n: int, mode: int, exclude) -> list[int]:
        if mode == REUSE:
            ex = set(exclude)
            cand = [m for m in L.pool if m not in ex]
            pick = [cand[j] for j in self.rng.choice(len(cand), size=min(n, len(cand)), replace=False)] if cand else []
            return pick + [self.fresh_mule(L) for _ in range(n - len(pick))]
        return [self.fresh_mule(L) for _ in range(n)]

    def funds(self, L: Launderer) -> float:
        held = [L.source] + (L.tranche.holders if L.tranche else [])
        return sum(self.ledger.balance[a] for a in dict.fromkeys(held))

    def launderer_held(self) -> float:
        """Spendable BTC at all launderer addresses (sources and mules): the proceeds not yet cashed out, frozen or
        paid as fees."""
        bal = np.asarray(self.ledger.balance)
        return float(bal[np.asarray(self.ledger.addr_role) == ROLE_LAUNDERER].sum())

    # ------------------------------------------------------------------ state
    def _refresh_state(self):
        p, rng, led = self.p, self.rng, self.ledger
        n_slot = min(K_RECIPIENTS, p.n_services)
        for L in self.launderers:
            L.k = int(rng.integers(2, 6))
            L.feerate = self.feerate()
            L.slots = [int(s) for s in rng.choice(p.n_services, size=n_slot, replace=False)]
        w = (self.inn_degree + 1.0) ** p.pa_exp
        self._pa_cdf = np.cumsum(w / w.sum())
        idle = np.array([i for i, d in enumerate(self.pending) if d is None], dtype=np.int64)
        act = idle[rng.random(len(idle)) < p.demand_rate]
        for i in act:
            bal = sum(led.balance[a] for a in self.inn_addrs[i])
            n_utxo = sum(len(led.addr_utxos[a]) for a in self.inn_addrs[i])
            # the amount is fixed for the demand's life; keep room for the largest route's fees and a change output
            reserve = self.reserve_feerate * (tx_size(n_utxo, max(SPLITS) + 1) + tx_size(1, max(SPLITS) + 1)) + 2 * DUST
            amt = float(rng.beta(p.demand_a, p.demand_b)) * (bal - reserve)
            if amt < DUST:
                continue
            if rng.random() < p.p_ii:
                j = self._pa_draw(exclude=int(i))
                rcpt, svc = self.inn_addrs[j][int(rng.integers(len(self.inn_addrs[j])))], -1
            else:
                svc = int(rng.integers(p.n_services))
                rcpt = self.deposit_addr(ROLE_INNOCENT, int(i), svc)
            self.pending[i] = Demand(rcpt, amt, svc, born=self.g)

    def _pa_draw(self, exclude: int = -1) -> int:
        j = min(int(np.searchsorted(self._pa_cdf, self.rng.random(), side="right")), len(self._pa_cdf) - 1)
        return (j + 1) % len(self._pa_cdf) if j == exclude else j

    def launderer_state(self, L: Launderer) -> AgentState | None:
        """``B(s)``, ``M_b(s)``; ``None`` (no transition) when the launderer has no funds left."""
        if L.tranche is not None:
            return AgentState(("initiate",), {"initiate": np.ones(2, bool)})
        led = self.ledger
        utx = led.spendable([L.source])
        bal = sum(led.u_amt[u] for u in utx)
        if bal <= 0:
            return None
        fee = L.feerate * tx_size(len(utx), L.k + 1)
        amount = np.array([a * (bal - fee) / L.k >= DUST for a in AMOUNTS])
        masks = {
            "initiate": np.array([True, bool(amount.any())]),
            "amount": amount if amount.any() else np.ones(len(AMOUNTS), bool),
            "recipient": np.arange(K_RECIPIENTS) < len(L.slots),
            "strategy": np.ones(len(STRATEGIES), bool),
            "mules": np.array([True, len(L.pool) >= L.k]),
            "change": np.ones(2, bool),
        }
        return AgentState(tuple(LAUNDERER_BRANCHES), masks)

    def innocent_state(self, i: int) -> AgentState | None:
        d = self.pending[i]
        if d is None or d.stage != 0:
            return None
        masks = {"initiate": np.ones(2, bool), "split": np.array([d.amount / k >= DUST for k in SPLITS]),
                 "route": np.ones(3, bool), "change": np.ones(2, bool)}
        return AgentState(tuple(INNOCENT_BRANCHES), masks)

    def detector_state(self) -> AgentState:
        return AgentState(tuple(DETECTOR_BRANCHES), {b: np.ones(n, bool) for b, n in DETECTOR_BRANCHES.items()})

    @staticmethod
    def _check(st: AgentState, a: dict):
        for b in st.branches:
            if not st.masks[b][a[b]]:
                raise ValueError(f"infeasible action {b}={a[b]}")

    # ------------------------------------------------------------------ proposals
    def _service_proposals(self) -> list[Proposal]:
        p, led, out = self.p, self.ledger, []
        for s in self.services:
            if (self.g + s.phase) % p.consol_interval == 0:
                funded = [d for d in s.deposits if led.addr_utxos[d]]
                if funded:
                    out.append(Proposal("consol", s.id, [TxSpec(led.spendable(funded), rest=[s.hot], feerate=self.feerate(),
                                                                role=ROLE_SERVICE, entity=s.id)]))
            if s.queue and (len(s.queue) >= p.batch_size or self.g - s.queue[0][2] >= p.payout_max_wait):
                utx = led.spendable([s.hot])
                bal, fr, tot, pay = led.balance[s.hot], self.feerate(), 0.0, []
                for a, amt, _ in list(s.queue)[:p.batch_size]:
                    if tot + amt + fr * tx_size(len(utx), len(pay) + 2) + DUST > bal:
                        break
                    pay.append((a, amt))
                    tot += amt
                if pay:
                    out.append(Proposal("payout", s.id, [TxSpec(utx, pay=pay, rest=[s.hot], feerate=fr,
                                                                role=ROLE_SERVICE, entity=s.id, meta={"n": len(pay)})]))
        return out

    def _innocent_spec(self, i: int, d: Demand, k: int, route: int, change: int) -> TxSpec | None:
        led = self.ledger
        fr = self.feerate()
        if route == ROUTE_DIRECT:
            payees, send = [d.rcpt] * k, d.amount
        elif route == ROUTE_FORWARD:
            payees, send = [None], d.amount + self.reserve_feerate * tx_size(1, k + 1)
        else:
            payees, send = [None], d.amount
        funded = sorted((a for a in self.inn_addrs[i] if led.balance[a] > 0), key=lambda a: -led.balance[a])
        chosen, utx, tot = [], [], 0.0
        for a in funded:
            chosen.append(a)
            utx += led.spendable([a])
            tot += led.balance[a]
            if tot >= send + fr * tx_size(len(utx), len(payees) + 1) + DUST:
                break
        if tot < send + fr * tx_size(len(utx), len(payees) + 1) + DUST:
            return None  # insufficient: a canonical no-op; the demand (fixed amount) stays pending
        if route == ROUTE_FORWARD:
            others = [a for a in self.inn_addrs[i] if a not in chosen]
            second = min(others, key=lambda a: led.balance[a]) if others else self._innocent_addr(i)
            payees = [second]
        elif route == ROUTE_SERVICE:
            svc = int(self.rng.integers(self.p.n_services))
            payees = [self.deposit_addr(ROLE_INNOCENT, i, svc)]
        meta = {"route": route, "split": k, "change": change, "payee": payees[0]}
        if route == ROUTE_SERVICE:
            meta["via"] = svc
        ch = chosen[0] if change == REUSE else self._innocent_addr(i)
        each = send / len(payees)
        return TxSpec(utx, pay=[(a, each) for a in payees], rest=[ch], feerate=fr, role=ROLE_INNOCENT, entity=i,
                      meta=meta)

    def _innocent_addr(self, i: int) -> int:
        a = self.ledger.new_address(ROLE_INNOCENT, i)
        self.inn_addrs[i].append(a)
        return a

    def _forward2_spec(self, i: int, d: Demand) -> TxSpec | None:
        led = self.ledger
        utx = led.spendable([d.second])
        if not utx:
            return None
        # exact amount; hop 1 forwarded the hop-2 fee at the reserve feerate, so the change is what is left
        # (an unaffordable feerate draw makes it a no-op that retries next sub-step)
        ch = d.second if d.change == REUSE else self._innocent_addr(i)
        return TxSpec(utx, pay=[(d.rcpt, d.amount / d.split)] * d.split, rest=[ch], feerate=self.feerate(),
                      role=ROLE_INNOCENT, entity=i)

    def _propose(self) -> tuple[list[Proposal], list[Transition]]:
        props, trs = self._service_proposals(), []
        # launderers
        Ls = {L.id: st for L in self.launderers if (st := self.launderer_state(L)) is not None}
        la = self.launderer_policy.act(self, Ls)
        for lid, st in Ls.items():
            L, a = self.launderers[lid], la[lid]
            self._check(st, a)
            tr = Transition("launderer", lid, st.branches, st.masks, a)
            trs.append(tr)
            if a["initiate"] == WAIT:
                tr.realized = ("initiate",)
            elif L.tranche is None:
                rcpt = self.deposit_addr(ROLE_LAUNDERER, lid, L.slots[a["recipient"]])
                T = Tranche(a["strategy"], typologies.plan(a["strategy"], L.k, self.rng), [], rcpt, a["mules"])
                spec = typologies.setup_spec(self, L, T, a["amount"], a["change"])
                drop = set()
                if AMOUNTS[a["amount"]] == 1.0:
                    drop.add("change")
                if STRATEGIES[a["strategy"]] in ("direct", "peel"):
                    drop.add("mules")
                tr.realized = tuple(b for b in st.branches if b not in drop)
                props.append(Proposal("setup", lid, [spec], tr, tranche=T))
            else:
                tr.realized = ("initiate",)
                props.append(Proposal("hop", lid, typologies.body_specs(self, L, L.tranche), tr))
        # innocents
        Is = {i: st for i in range(self.p.n_innocents) if (st := self.innocent_state(i)) is not None}
        ia = self.innocent_policy.act(self, Is)
        for i, st in Is.items():
            a = ia[i]
            self._check(st, a)
            tr = Transition("innocent", i, st.branches, st.masks, a)
            trs.append(tr)
            if a["initiate"] == INIT_DEFER:
                tr.realized = ("initiate",)
                continue
            spec = self._innocent_spec(i, self.pending[i], SPLITS[a["split"]], a["route"], a["change"])
            tr.realized = st.branches  # innocent txs always have a change output
            props.append(Proposal("demand", i, [spec] if spec else [], tr))
        # scripted continuations (no transition)
        for i, d in enumerate(self.pending):
            if d is not None and d.stage == 1:
                spec = self._forward2_spec(i, d)
                props.append(Proposal("forward2", i, [spec] if spec else []))
        return props, trs

    # ------------------------------------------------------------------ step
    def step(self) -> StepResult:
        self._refresh_state()
        d_state = self.detector_state()
        d_act = self.detector.act_global(self, d_state)
        self._check(d_state, d_act)
        props, trs = self._propose()
        led = self.ledger
        self._cont_rewards: dict[int, float] = {}

        # 3. validate and reserve, in a seeded random order
        reserved: set[int] = set()
        valid: list[Proposal] = []
        n_noop = 0
        for j in self.rng.permutation(len(props)):
            pr = props[j]
            res = [led.resolve(s) for s in pr.specs]
            ins = [u for s in pr.specs for u in s.utxos]
            if pr.specs and all(res) and len(set(ins)) == len(ins) and reserved.isdisjoint(ins):
                reserved.update(ins)
                pr.resolved = res
                valid.append(pr)
            else:
                n_noop += 1
                if pr.tr is not None:
                    pr.tr.action = dict(NOOP_LAUNDERER if pr.tr.role == "launderer" else NOOP_INNOCENT)
                    pr.tr.realized, pr.tr.noop = ("initiate",), True

        # 4-5. score and flag
        cands = [(pr, r) for pr in valid for r in pr.resolved]
        batch = None
        if self.graph is not None:   # the detector scores s_t plus the tentative txs, as if committed
            batch = tx_batch(led, [r for _, r in cands], self.ts)
            self.graph.add(batch)
        scores = np.asarray(self.detector.score(self, [r for _, r in cands]), dtype=np.float64)
        if batch is not None:
            self.graph.pop(batch)
        n_flag = int(round(self.budget * len(cands)))
        flagged = np.zeros(len(cands), bool)
        flagged[np.argsort(-scores, kind="stable")[:n_flag]] = True

        # 6. commit or revert
        ts, block = self.ts, self.p.block(self.episode, self.sub)
        info, ci = [], 0
        depth = d_act["analysis_depth"] + 1
        d_reward = 0.0
        for pr in valid:
            for r in pr.resolved:
                f, ill = bool(flagged[ci]), r.spec.illicit
                cash = ill and any(led.addr_role[a] == ROLE_SERVICE for a, _ in r.outputs)
                if f and not ill:
                    out, tid = "reverted", -1
                else:
                    lab = labels.tx_label(float(self.label_rng.random()), ill, r.spec.role, self.p.p_ill, self.p.p_lic)
                    tid = led.commit(r, ts, self.sub, block, frozen=f, label=lab)
                    out = "frozen" if f else "committed"
                    owners = {led.addr_owner[a] for a in [led.u_addr[u] for u in r.spec.utxos] + [a for a, _ in r.outputs]
                              if led.addr_role[a] == ROLE_INNOCENT}
                    for o in owners:
                        self.inn_degree[o] += 1
                pr.outcomes.append(out)
                pr.tids.append(tid)
                info.append({"illicit": ill, "flagged": f, "cashout": cash, "outcome": out, "tid": tid})
                if f:
                    d_reward += (1.0 if ill else -0.1) - 0.01 * depth
                elif cash:
                    d_reward -= 1.0
                ci += 1
        if batch is not None:
            self.graph.add(batch.take(np.array([c["tid"] >= 0 for c in info], dtype=bool)))

        # 7. owner state and rewards
        for pr in valid:
            getattr(self, f"_on_{pr.kind}")(pr)
        for tr in trs:
            if tr.role == "launderer" and not tr.noop:
                tr.reward += -0.001 if self.funds(self.launderers[tr.agent]) > 0 else 0.0
        self.budget = float(np.clip(self.budget * BUDGET_MULT[d_act["update_strategy"]], *BUDGET_RANGE))
        self.sub += 1
        self.g += 1
        return StepResult(trs, info, d_reward, n_noop, self._cont_rewards)

    # ------------------------------------------------------------------ outcomes
    def _tranche_flows(self, L: Launderer, pr: Proposal) -> tuple[list[int], float, float]:
        """(next holders, BTC cashed out undetected, tranche BTC frozen) of a launderer proposal.

        ``acct`` gets everything a flagged tx froze, its change to the source included (a frozen tx freezes all
        of its outputs); the returned tranche figure, which the reward uses, leaves that change out.
        """
        nxt, cash, frozen, frozen_change = [], 0.0, 0.0, 0.0
        for r, out in zip(pr.resolved, pr.outcomes):
            change = {L.source, r.spec.meta.get("new_source", -1)}
            self.acct["fees"] += r.fee
            if out == "committed":
                nxt += r.spec.meta["next"]
                cash += sum(a for x, a in r.outputs if x == r.spec.meta["cash"])
            else:
                frozen += sum(a for x, a in r.outputs if x not in change)
                frozen_change += sum(a for x, a in r.outputs if x in change)
        self.acct["cashed"] += cash
        self.acct["frozen"] += frozen + frozen_change
        self.acct["frozen_change"] += frozen_change
        return list(dict.fromkeys(nxt)), cash, frozen

    def _on_setup(self, pr: Proposal):
        L, T, s = self.launderers[pr.owner], pr.tranche, pr.resolved[0].spec
        nxt, cash, frozen = self._tranche_flows(L, pr)
        T.amount = s.meta["tranche"]
        self.acct["strategy"][T.strategy] += 1
        if pr.outcomes[0] == "committed" and s.meta["new_source"] >= 0:
            L.source = s.meta["new_source"]
        T.plan = T.plan[1:]
        T.holders = nxt
        if STRATEGIES[T.strategy] == "cycle" and nxt:
            T.first_mule = nxt[0]
        L.tranche = T if (T.plan and T.holders) else None
        pr.tr.reward += (cash - frozen) / T.amount

    def _on_hop(self, pr: Proposal):
        L = self.launderers[pr.owner]
        T = L.tranche
        nxt, cash, frozen = self._tranche_flows(L, pr)
        T.plan = T.plan[1:]
        T.holders = nxt
        L.tranche = T if (T.plan and T.holders) else None
        pr.tr.reward += (cash - frozen) / T.amount

    def _on_demand(self, pr: Proposal):
        i, s = pr.owner, pr.resolved[0].spec
        if pr.outcomes[0] != "committed":
            return  # blocked: the demand stays pending
        pr.tr.reward += 0.1
        d = self.pending[i]
        route = s.meta["route"]
        if route == ROUTE_FORWARD:
            d.stage, d.second, d.split, d.change = 1, s.meta["payee"], s.meta["split"], s.meta["change"]
            return
        if route == ROUTE_SERVICE:
            k = s.meta["split"]
            q = self.services[s.meta["via"]].queue
            amt = sum(a for x, a in pr.resolved[0].outputs if x == s.meta["payee"])
            for _ in range(k):
                q.append((d.rcpt, amt / k, self.g))
        self._complete(i, d)

    def _on_forward2(self, pr: Proposal):
        """The forward route's hop 2 has no transition of its own; its +0.1 (or 0 if blocked) is reported per
        innocent in ``StepResult.continuation_rewards``."""
        ok = pr.outcomes[0] == "committed"
        self._cont_rewards[pr.owner] = 0.1 if ok else 0.0
        if ok:
            self._complete(pr.owner, self.pending[pr.owner])

    def _complete(self, i: int, d: Demand):
        """Demand done. A payment to a service queues a withdrawal to an innocent (ours)."""
        self.pending[i] = None
        if d.to_service >= 0:
            j = self._pa_draw()
            self.services[d.to_service].queue.append(
                (self.inn_addrs[j][int(self.rng.integers(len(self.inn_addrs[j])))], d.amount, self.g))

    def _on_consol(self, pr: Proposal):
        pass

    def _on_payout(self, pr: Proposal):
        if pr.outcomes[0] == "committed":
            q = self.services[pr.owner].queue
            for _ in range(pr.resolved[0].spec.meta["n"]):
                q.popleft()


def run_static_world(params: WorldParams, seed: int) -> SimEnv:
    env = SimEnv(params, seed)
    env.run_world()
    return env


