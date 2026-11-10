"""Frozen-opponent learner capability test (after M5; Sol consult of 2026-11-10, runs/sol_consult/CONSULT.md).

Question: can the launderer learner find a profitable policy when one exists? Against the final inspectors of
Full seed 0 and NDA seed 1, a scripted policy that launches ``direct`` at 50 % once per episode is flagged on
under 10 % of launches, yet the M5 launderers mostly chose templates that are almost always caught.

**Design.** A fresh launderer network (seed 0) learns against a finished M5 run's inspector and innocents,
loaded from its ``final.pt``; they act at a fixed epsilon of 0.05 (as at the end of M5) and never update. The
inspector's per-tx flags, budget and depth policy all act; detector-W and the export are skipped. Rewards are
the M5 rewards (§9.3, per tranche), unchanged, so that the test isolates the learner. Factors: the launderer
target (``root`` / ``mean``) x the role Q-network dropout (0.5 / 0), for each of the two opponents.

Training, per variant: 30 worlds (300 episodes); launderer epsilon 1.0 -> 0.05 over the first 150 episodes;
otherwise the M5 settings (batch 128, buffer 100k, warm-up 1,000, Polyak 0.005, lr 1e-3, gamma 0.99,
tau 0.01). Training worlds are the multiverse draws of seed 0.

Evaluation: 8 held-out multiverse worlds of 10 episodes (seeds ``[EVAL_SALT, j]``), the same for every
variant, opponent and policy, so that comparisons are paired by world. Policies:

- ``learned``: the trained launderer at epsilon 0 (``softmax(Q / tau)`` per branch);
- ``wait``: never launches;
- ``uniform``: each launderer launches once per episode, at its first idle decision on or after a sub-step drawn
  uniformly from [0, 43], with every option uniform over the feasible ones; in flight it always proceeds;
- ``direct50``: the same schedule, always ``direct`` at 50 % (other options uniform).

**Pass criteria, declared before any run.** A variant passes against an opponent if

- P1, stable: every training loss is finite; over the evaluation decisions, the eval-mode ``Q_initiate`` of every
  state is at most ``Q_MAX = (1 + tau ln 2) / (1 - gamma)`` (the per-tranche reward is at most +1 per sub-step);
  and the calibration bias ``mean(Q_initiate(s, a) - G)`` is within ``max(0.5, 0.25 mean |G|)``, where ``G``
  is the realized discounted return from the decision (gamma per sub-step, to world end);
- P2, capable: its mean launderer reward per launderer and episode is at least half of ``direct50``'s, and its
  paired difference to ``wait`` has a 95 % t-interval over the 8 worlds above 0.

It passes overall if it passes against both opponents. Reported, not gated: BTC utility (cashed - all frozen,
over proceeds), tranches, strategy and amount shares, the inspector's detection rate.

Choice for the live pilot: the root target if it passes and the mean target does not (if both pass, the root
target, which the SPEC change asks for). Dropout: the setting that passes; if both do, the one with the smaller
mean absolute calibration bias over the two opponents. If no root variant passes, the learner needs repair
before any pilot.

Usage::

    python -m hns.marl.capability train --opp runs/marl/full/0 --target root --q-dropout 0
    python -m hns.marl.capability refs --opp runs/marl/full/0
    python -m hns.marl.capability report
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import yaml

import hns.marl.train as mt
from hns.marl.learner import MarlConfig
from hns.marl.obs import launderer_obs, pack_masks
from hns.models.qnet import sample_actions
from hns.sim.entities import AMOUNTS, GO, STRATEGIES, WAIT
from hns.sim.env import SimEnv
from hns.sim.live import LiveGraph
from hns.sim.multiverse import draw_world
from hns.sim.world import EPISODES, SUBSTEPS, WorldParams

ROOT = Path("runs/capability")
REPORT = Path("reports/m5_capability.md")
OPP_EPS = 0.05
WORLDS, EPS_EPISODES, SEED = 30, 150, 0
EVAL_WORLDS, EVAL_SALT, EVAL_STEP = 8, 777_000, 10_000_000
LAUNCH_LAST = 43            # a scheduled launch's sub-step is uniform on [0, 43]
REFS = ("wait", "uniform", "direct50")


def q_max(cfg: MarlConfig) -> float:
    return (1.0 + cfg.tau * math.log(2.0)) / (1.0 - cfg.gamma)


# ---------------------------------------------------------------------------- set-up
def make(opp: Path, device, out: Path | None = None, **over) -> mt.Trainer:
    """A trainer whose launderer learns (fresh) against ``opp``'s final inspector and innocents (frozen)."""
    cfgd = yaml.safe_load((opp / "config.yaml").read_text())
    base = {k: v for k, v in cfgd.items() if k in MarlConfig.__dataclass_fields__}
    cfg = MarlConfig(**base | {"arm": "nda", "seed": SEED, "episodes": WORLDS * EPISODES, "eps_episodes": EPS_EPISODES}
                     | over)
    centre = WorldParams.from_dict(cfgd["centre"])
    tr = mt.build(cfg, device, centre, out, lambda m: print(m, flush=True))
    tr.learn = frozenset({"launderer"})
    tr.fixed_eps = {"innocent": OPP_EPS, "detector": OPP_EPS}
    tr.load_models(opp / "ckpt" / "final.pt", ("innocent", "detector"))
    return tr


# ---------------------------------------------------------------------------- launderer policies for evaluation
class Learned:
    """The trained launderer network at a fixed epsilon; logs ``Q_initiate`` of every decision."""

    def __init__(self, tr: mt.Trainer, eps: float = 0.0):
        self.tr, self.eps, self.q = tr, eps, {}

    def act(self, env, states: dict) -> dict:
        if not states:
            return {}
        tr, L = self.tr, self.tr.roles["launderer"]
        ids = list(states)
        ob = launderer_obs(env, tr.fz, ids, tr.gvec, tr.view_seed(1), tr.cfg.k2)
        masks, _ = pack_masks("launderer", states, ids)
        q = L.q_values(ob, masks)
        a = sample_actions(q, L.sizes, tr.cfg.tau, self.eps, tr.rng)
        for j, i in enumerate(ids):
            self.q[(tr.step_no, i)] = (float(q[j, 0]), float(q[j, 1]))
        names = list(L.branches)
        return {i: {b: int(a[j, bi]) for bi, b in enumerate(names) if b in states[i].branches}
                for j, i in enumerate(ids)}


class Scheduled:
    """One launch per launderer and episode, at its first idle decision on or after a uniform sub-step in
    [0, ``LAUNCH_LAST``]; in flight it always proceeds. ``strategy`` / ``amount``: a fixed option index, or
    ``None`` for uniform over the feasible ones. ``never``: always wait."""

    def __init__(self, rng: np.random.Generator, strategy=None, amount=None, never: bool = False):
        self.rng, self.strategy, self.amount, self.never = rng, strategy, amount, never
        self.when, self.launched, self.ep = {}, set(), None

    def act(self, env, states: dict) -> dict:
        if env.episode != self.ep:
            self.ep, self.launched = env.episode, set()
            self.when = {L.id: int(self.rng.integers(0, LAUNCH_LAST + 1)) for L in env.launderers}
        out = {}
        for i, st in states.items():
            if len(st.branches) == 1:
                out[i] = {"initiate": GO}
                continue
            a = {b: int(self.rng.choice(np.flatnonzero(st.masks[b]))) for b in st.branches}
            a["initiate"] = WAIT
            due = not self.never and i not in self.launched and env.sub >= self.when[i] and st.masks["initiate"][GO]
            if due and (self.amount is None or st.masks["amount"][self.amount]):
                if self.strategy is not None:
                    a["strategy"] = self.strategy
                if self.amount is not None:
                    a["amount"] = self.amount
                a["initiate"] = GO
                self.launched.add(i)
            out[i] = a
        return out


def reference(name: str, rng: np.random.Generator) -> Scheduled:
    if name == "wait":
        return Scheduled(rng, never=True)
    if name == "uniform":
        return Scheduled(rng)
    if name == "direct50":
        return Scheduled(rng, STRATEGIES.index("direct"), AMOUNTS.index(0.5))
    raise ValueError(name)


# ---------------------------------------------------------------------------- evaluation
def play_world(tr: mt.Trainer, lpol, j: int) -> dict:
    """Held-out world ``j`` with launderer policy ``lpol`` against ``tr``'s frozen opponents; nothing learns."""
    learn, tr.learn = tr.learn, frozenset()
    ss = np.random.SeedSequence([EVAL_SALT, j])
    params = draw_world(tr.centre, np.random.default_rng(ss.spawn(1)[0]))
    env = SimEnv(params, int(ss.generate_state(1)[0]), lpol, tr.policies[1], tr.policies[2])
    env.graph = LiveGraph()
    tr.rng = np.random.default_rng([EVAL_SALT, j, 1])        # the opponents' exploration, the same for every policy
    tr.step_no, tr.last = EVAL_STEP + j * EPISODES * SUBSTEPS, {}
    tot, dec = defaultdict(float), []                          # dec: (step, agent, initiate taken, reward, Q of it)
    launches = {"strategy": np.zeros(len(STRATEGIES)), "amount": np.zeros(len(AMOUNTS))}
    for e in range(EPISODES):
        env.start_episode(e)
        if tr.last:
            tr.last["episode_committed"] = 0
        ep = defaultdict(float)
        for _ in range(SUBSTEPS):
            res = env.step()
            for t in res.transitions:
                if t.role != "launderer":
                    continue
                a0 = t.action["initiate"]
                q = lpol.q.get((tr.step_no, t.agent)) if isinstance(lpol, Learned) else None
                dec.append((tr.step_no, t.agent, a0, t.reward, None if q is None else q[a0]))
                idle = len(t.branches) > 1
                tot["idle"] += idle
                if idle and a0 == GO and not t.noop:
                    tot["idle_go"] += 1
                    launches["strategy"][t.action["strategy"]] += 1
                    launches["amount"][t.action["amount"]] += 1
            tr.after_step(env, res, ep)
            tr.step_no += 1
        env.end_episode()
        for k, v in ep.items():
            tot[k] += v
    tr.learn = learn
    n_le = params.n_launderers * EPISODES
    out = {"world": j, "n_launderers": params.n_launderers, "budget0": params.budget,
           "reward": tot["r_launderer"] / n_le, "tranches": float(env.acct["strategy"].sum()) / n_le,
           "detection_rate": tot["cash_flag"] / max(tot["cash"], 1.0), "idle_go_rate": tot["idle_go"] / max(tot["idle"], 1.0),
           "budget_mean": tot["budget"] / (EPISODES * SUBSTEPS), **mt.btc_fractions(env),
           "strategy_share": dict(zip(STRATEGIES, (launches["strategy"] / max(launches["strategy"].sum(), 1)).tolist())),
           "amount_share": dict(zip(map(str, AMOUNTS), (launches["amount"] / max(launches["amount"].sum(), 1)).tolist()))}
    if isinstance(lpol, Learned):
        out |= calibration(dec, tr.cfg.gamma)
        lpol.q.clear()
    return out


def calibration(dec: list[tuple], gamma: float) -> dict:
    """``Q_initiate`` of the taken option against the realized return ``G`` from each launderer decision: discounted
    per sub-step, to world end (where §10's chains end too)."""
    by = defaultdict(list)
    for s, i, a0, r, q in dec:
        by[i].append((s, r, q))
    qs, gs = [], []
    for rows in by.values():
        rows.sort()
        g, t_next = 0.0, None
        for s, r, q in reversed(rows):
            g = r + (gamma ** (t_next - s) * g if t_next is not None else 0.0)
            t_next = s
            if q is not None and math.isfinite(q):
                qs.append(q)
                gs.append(g)
    qs, gs = np.array(qs), np.array(gs)
    if not len(qs):
        return {"q_n": 0}
    return {"q_n": int(len(qs)), "q_max": float(qs.max()), "q_mean": float(qs.mean()), "g_mean": float(gs.mean()),
            "g_abs_mean": float(np.abs(gs).mean()), "q_bias": float((qs - gs).mean())}


def evaluate(tr: mt.Trainer, policy: str, n_worlds: int = EVAL_WORLDS) -> list[dict]:
    out = []
    for j in range(n_worlds):
        lpol = Learned(tr) if policy == "learned" else reference(policy, np.random.default_rng([EVAL_SALT, j, 2]))
        t0 = time.perf_counter()
        r = play_world(tr, lpol, j)
        r["sec"] = round(time.perf_counter() - t0, 1)
        out.append(r)
        print(f"  eval {policy} world {j}: nL {r['n_launderers']} reward {r['reward']:+.3f} "
              f"utility {r['btc_utility']:+.3f} tranches {r['tranches']:.2f} {r['sec']:.0f}s", flush=True)
    return out


@torch.no_grad()
def dropout_gap(tr: mt.Trainer, n: int = 256, passes: int = 8) -> float | None:
    """Mean train-mode (dropout) minus eval-mode ``Q(wait)`` over replayed idle states: how far the values that
    updates fit sit from the values acting and the target network use."""
    from hns.marl.obs import collate
    L = tr.roles["launderer"]
    k = min(L.buf.n, L.buf.cap)
    idle = np.flatnonzero(L.buf.active[:k].sum(1) > 1)
    if not len(idle) or tr.cfg.q_dropout == 0:
        return 0.0 if len(idle) else None
    s = np.random.default_rng(0).choice(idle, size=min(n, len(idle)), replace=False)
    ob, masks = collate([L.buf.obs[i] for i in s]), L.buf.masks[s]
    L.net.eval()
    q_eval = L._q(L.net, ob, masks)[:, 0]
    L.net.train()
    q_train = torch.stack([L._q(L.net, ob, masks)[:, 0] for _ in range(passes)]).mean(0)
    L.net.eval()
    return float((q_train - q_eval).mean())


# ---------------------------------------------------------------------------- commands
def _name(opp: Path) -> str:
    return f"{opp.parent.name}{opp.name}"


def cmd_train(a) -> None:
    smoke = a.worlds != WORLDS                                 # a shortened run never counts as a result
    out = (ROOT / "smoke" if smoke else ROOT) / f"{_name(a.opp)}_{a.target}_d{a.q_dropout:g}"
    out.mkdir(parents=True, exist_ok=True)
    tr = make(a.opp, torch.device(a.device), out, target=a.target, q_dropout=a.q_dropout)
    (out / "config.yaml").write_text(yaml.safe_dump({**asdict(tr.cfg), "opponent": str(a.opp), "opp_eps": OPP_EPS,
                                                     "learn": ["launderer"], "git_sha": mt.git_sha()}, sort_keys=False))
    (out / "metrics.jsonl").write_text("")
    finite = True
    for w in range(a.worlds):
        r = tr.run_world()
        finite &= all(math.isfinite(v) for v in r["losses"].values())
        print(f"{out.name} world {w + 1}/{a.worlds} eps {tr.eps:.2f} | R_L {r['reward']['launderer']:+.3f} "
              f"utility {r['btc_utility']:+.3f} succ {r['success_frac']:.2f} det {r['detection_rate']:.2f} "
              f"tranches {r['tranches']} direct {r['strategy_share']['direct']:.2f} "
              f"td {r['losses'].get('td_launderer', float('nan')):.3f} | {r['sec']['world']:.0f}s", flush=True)
    gap = dropout_gap(tr)
    tr.save(out / "final.pt", full=False)
    res = {"variant": out.name, "opponent": str(a.opp), "target": a.target, "q_dropout": a.q_dropout,
           "worlds": a.worlds, "finite": bool(finite), "dropout_gap": gap,
           "eval": evaluate(tr, "learned", a.eval_worlds)}
    (out / "result.json").write_text(json.dumps(res, indent=1))


def cmd_refs(a) -> None:
    tr = make(a.opp, torch.device(a.device))
    root = ROOT if a.eval_worlds == EVAL_WORLDS else ROOT / "smoke"
    for p in REFS:
        f = root / f"ref_{_name(a.opp)}_{p}.json"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps({"opponent": str(a.opp), "policy": p, "eval": evaluate(tr, p, a.eval_worlds)}, indent=1))


def _t95(x: np.ndarray) -> tuple[float, float, float]:
    """Mean and two-sided 95 % t-interval."""
    from scipy import stats
    m = float(x.mean())
    h = float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / math.sqrt(len(x))) if len(x) > 1 else float("nan")
    return m, m - h, m + h


def verdicts(root: Path = ROOT) -> list[dict]:
    """Every variant's P1 / P2 verdict per opponent, from the result files."""
    cfg = MarlConfig()
    rows = []
    for f in sorted(root.glob("*_d*/result.json")):
        res = json.loads(f.read_text())
        opp = Path(res["opponent"])
        refs = {p: json.loads((root / f"ref_{_name(opp)}_{p}.json").read_text())["eval"] for p in REFS}
        ev = res["eval"]
        rew = np.array([e["reward"] for e in ev])
        ref = {p: np.array([e["reward"] for e in refs[p]]) for p in REFS}
        util = np.array([e["btc_utility"] for e in ev])
        qn = sum(e.get("q_n", 0) for e in ev)
        qmax = max(e.get("q_max", -math.inf) for e in ev)
        bias = sum(e["q_bias"] * e["q_n"] for e in ev if e.get("q_n")) / max(qn, 1)
        gabs = sum(e["g_abs_mean"] * e["q_n"] for e in ev if e.get("q_n")) / max(qn, 1)
        p1 = res["finite"] and qmax <= q_max(cfg) and abs(bias) <= max(0.5, 0.25 * gabs)
        d_wait = _t95(rew - ref["wait"])
        p2 = rew.mean() >= 0.5 * ref["direct50"].mean() and d_wait[1] > 0
        share = defaultdict(float)
        for e in ev:
            for k, v in e["strategy_share"].items():
                share[k] += v / len(ev)
        amt = defaultdict(float)
        for e in ev:
            for k, v in e["amount_share"].items():
                amt[k] += v / len(ev)
        rows.append({"variant": res["variant"], "opponent": _name(opp), "target": res["target"],
                     "q_dropout": res["q_dropout"], "reward": rew.mean(), "ref": {p: ref[p].mean() for p in REFS},
                     "d_wait": d_wait, "d_direct50": _t95(rew - ref["direct50"]), "utility": util.mean(),
                     "ref_utility": {p: np.mean([e["btc_utility"] for e in refs[p]]) for p in REFS},
                     "tranches": np.mean([e["tranches"] for e in ev]), "q_max": qmax, "q_bias": bias, "g_abs": gabs,
                     "dropout_gap": res["dropout_gap"], "finite": res["finite"], "p1": p1, "p2": p2,
                     "pass": p1 and p2, "share": dict(share), "amount": dict(amt),
                     "detection": np.mean([e["detection_rate"] for e in ev])})
    return rows


def cmd_report(a) -> None:
    rows = verdicts()
    lines = ["# M5 follow-up: frozen-opponent learner capability", "",
             "Design and pass criteria: `hns.marl.capability` (declared before the runs). Each row: a fresh launderer "
             f"trained for {WORLDS} worlds against a frozen M5 inspector and innocents (epsilon {OPP_EPS}), then "
             f"evaluated at epsilon 0 on {EVAL_WORLDS} held-out worlds shared by every row. Rewards are M5's (per "
             "tranche). Reward: per launderer and episode. Utility: (cashed - all frozen) / proceeds.", "",
             "| Opponent | Target | Q dropout | Reward | wait / uniform / direct50 | Learned - wait (95 % CI) | "
             "Utility | direct50 utility | Max Q | Q bias (mean abs G) | Dropout gap | P1 | P2 | Pass |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        dw = r["d_wait"]
        lines.append(f"| {r['opponent']} | {r['target']} | {r['q_dropout']:g} | {r['reward']:+.3f} | "
                     f"{r['ref']['wait']:+.3f} / {r['ref']['uniform']:+.3f} / {r['ref']['direct50']:+.3f} | "
                     f"{dw[0]:+.3f} ({dw[1]:+.3f}, {dw[2]:+.3f}) | {r['utility']:+.3f} | "
                     f"{r['ref_utility']['direct50']:+.3f} | {r['q_max']:.1f} | {r['q_bias']:+.2f} ({r['g_abs']:.2f}) | "
                     f"{'—' if r['dropout_gap'] is None else format(r['dropout_gap'], '+.2f')} | "
                     f"{'yes' if r['p1'] else 'no'} | {'yes' if r['p2'] else 'no'} | {'**yes**' if r['pass'] else 'no'} |")
    lines += ["", "Learned launch mix (share of launches, mean over evaluation worlds):", "",
              "| Opponent | Target | Q dropout | Tranches / L / episode | Top strategies | Amounts 10 / 25 / 50 / 100 % | "
              "Detection rate |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        top = ", ".join(f"`{k}` {v:.2f}" for k, v in sorted(r["share"].items(), key=lambda x: -x[1])[:3])
        am = " / ".join(f"{r['amount'].get(str(x), 0):.2f}" for x in AMOUNTS)
        lines.append(f"| {r['opponent']} | {r['target']} | {r['q_dropout']:g} | {r['tranches']:.2f} | {top} | {am} | "
                     f"{r['detection']:.2f} |")
    REPORT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--opp", type=Path, required=True)
    t.add_argument("--target", choices=["root", "mean"], required=True)
    t.add_argument("--q-dropout", type=float, required=True)
    r = sub.add_parser("refs")
    r.add_argument("--opp", type=Path, required=True)
    sub.add_parser("report")
    t.add_argument("--worlds", type=int, default=WORLDS, help="training worlds; anything else is a smoke run")
    for p in (t, r):
        p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
        p.add_argument("--eval-worlds", type=int, default=EVAL_WORLDS)
    a = ap.parse_args(argv)
    {"train": cmd_train, "refs": cmd_refs, "report": cmd_report}[a.cmd](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
