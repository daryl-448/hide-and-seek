"""OPPO-MC capability cycle: the development blocks (SPEC §10). Declared on 2026-11-12, before any of it ran.

Source: the fifth Sol consult (runs/sol_consult/CONSULT5.md, sections C and D), adopted by the user on 2026-11-12
together with the SPEC revision. It supersedes stage 1's remaining blocks (``hns.marl.stage1``), whose verdicts
stand. The learner is ``hns.marl.oppo``.

**Why.** No Q-learner variant of repair cycle 2 and stage 1 kept a profitable launderer policy against both frozen
opponents: the policies formed and were lost, or rich launderers held their funds. The consult traced the common
failure to valuing a launch through a gate over the learner's own changing launch options, through long
bootstrapped chains of waits and in-flight pauses. OPPO-MC changes the estimator (on-policy PPO with complete-world
returns, a separate critic that never enters a return), the credit boundary (options) and the decision structure
(automatic flight, wait durations, a joint mode). It is a bundle, not a one-factor test: a pass shows that this
package can learn the known task, not which of its parts was needed.

**Fixed conditions.** A fresh OPPO-MC launderer, seed 0, learns against the frozen final inspector and innocents of
M5's Full seed 0 (``full0``) or NDA seed 1 (``nda1``), which act at epsilon 0.05 and never update
(``hns.marl.capability.make``). 30 training worlds, the multiverse draws of seed 0 (stage 1's). Worlds 1-5 are the
options-first curriculum, worlds 6-30 the natural actor (SPEC §10). The reward, the simulator, the observations and
the opponents are unchanged. Snapshots of the actor, critic, optimizers and random streams: before training, every
two worlds, and every world from 15; the complete decision rollout of worlds 15 and 30.

**A block is one attempt**: one opponent, training and reading together, under a 45-minute cap on the local GPU
(``scripts/m5_repair2.sh oppo OPP``). ``run`` records that it started and refuses to start again; a block cut by the
cap, or aborted, is incomplete, never promising. ``read`` exists only to finish the reading of a block whose training
completed and whose reading did not run (a harness fault); it never re-reads a finished block. Full-0 runs first,
then NDA-1 as its own block, even if Full-0's reading fails, unless the failure is an implementation fault that
also invalidates the NDA run (then it is reported and a new declaration follows).

**Reading.** On development worlds 100-103 (``EVAL_SALT``; used by stage 1 already, so this is a development
reading, not a confirmation), against the same frozen opponents:

- natural: the world-20 and world-30 snapshots, the actor sampling as trained (no extra exploration);
- matched: the world-30 snapshot on the scheduled launch of the references (once per episode, the first idle
  sub-step on or after a uniform sub-step in [0, 43] where a launch is feasible, the same schedule stream as the
  references), always proceeding in flight, the actor choosing every launch factor;
- argmax: the world-30 snapshot taking every factor's most likely option (reported, never selected).

References: the scripted policies of ``hns.marl.repair2`` (``wait``, ``uniform``, ``direct50``, ``repeat50``), read
from the reference files on the selection worlds that repair2 computed on 2026-11-11 (local GPU), restricted to
100-103; legacy cell ``A``'s final network (its natural records on the same worlds). These are the files stage 1
read; they are not recomputed (the scripted policies and the opponents are unchanged; the consult proposed
recomputing them). Under them the floors below are U >= 0.401891 and R >= 0.322627 against Full-0, and
U >= 0.292472 and R >= 0.269037 against NDA-1; the formulas govern.

The block is **promising** if all five hold:

1. sound: all 30 worlds completed without an abort (a non-finite loss, gradient, probability, value or KL; an
   infeasible sampled action; a decision chain whose return differs from its rewards by more than 1e-6; a decision
   added after its world closed), the BTC accounting gap is at most 1e-6 in every world, and before every update the
   stored log-probabilities are reproduced within ``RATIO_TOL`` (the ratios start at one);
2. natural economics: the world-30 snapshot's mean BTC utility is positive and at least
   ``max(0.5 direct50's, A's - 0.02)``, and its mean reward at least half of ``direct50``'s;
3. retention: world-30 minus world-20 natural BTC utility is at least -0.05 on average over the four worlds;
4. the critic: with ``b_w = mean (V - G)`` and ``a_w = mean |G|`` over world ``w``'s decisions of the world-30
   natural rollout (``V`` before the decision, ``G`` its plain return), ``|mean_w b_w| <= max(0.5, 0.25 mean_w a_w)``;
5. utilization and option skill: the world-30 natural held share minus ``direct50``'s is at most 0.05 on average,
   and the matched BTC utility minus ``uniform``'s at least 0.02 on average.

Reported, not gated: every paired difference with its 95 % t-interval over the four worlds (to wait, ``direct50``,
``repeat50``, PL's world-30 network, retention, held, skill); where the proceeds went; launches by mode, wait
durations, launch rates by balance band and world time, launch no-ops; the critic's bias pooled, BTC-weighted, by
band and time, its RMSE and explained variance; per training world, the actor passes kept and undone, the three
KLs, entropy, clip fraction, ratio quantiles and gradient norms; the argmax policy.

**What follows** (the consult's D3, fixed now): both blocks promising -> confirmation with seeds 1 and 2 on unused
worlds (SPEC §10), not the six runs. Natural economics or held funds fail while matched skill passes -> inspect the
learned durations, the gate's support and proposed-versus-executed launches; the next declaration changes only the
timing. Matched skill fails with adequate launch data -> audit the likelihoods, masks and credit on a small
enumerated problem first. World 20 good and world 30 lost -> inspect the KLs and undone passes. Economics good and
only the critic fails -> the critic, not the game, is revised. Every negative or incomplete block is reported before
a new declaration.

**Deviations from the consult.** References are reused, not recomputed (above). Encoder saturation is not
reported; route outcomes per exposed BTC are not logged. The reading's world-20 and world-30 evaluations and the
matched and argmax ones all count inside the block's cap. The live game's per-world acting copies and NMAI's
static option launderers are not implemented yet: neither is used by these blocks.

Usage::

    scripts/m5_repair2.sh oppo full0       (oppo_block run: train, then read; the 45-minute cap)
    python -m hns.marl.oppo_block read --opp full0     (only after a harness fault in the reading)
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import yaml

import hns.marl.capability as cap
import hns.marl.repair2 as r2
import hns.marl.train as mt
from hns.marl.capability import _t95
from hns.marl.oppo import OptionPolicy, calibration, decision_summary

WORLDS, DEV_WORLDS, SNAPS = 30, (100, 101, 102, 103), (20, 30)
RATIO_TOL = 1e-3
ACCT_MAX, A_SLACK, RETAIN, HELD_SLACK, SKILL_MIN = 1e-6, 0.02, 0.05, 0.05, 0.02
ABORTS = (FloatingPointError, RuntimeError, ValueError)     # numerical, chain and legality faults (and out of memory)


def run_dir(opp: str, seed: int = 0, smoke: bool = False) -> Path:
    return (r2.RUNS / "smoke" if smoke else r2.RUNS) / f"OPPO_{opp}_s{seed}"


def mark_path(opp: str, seed: int = 0) -> Path:
    return r2.RUNS / f"oppo_{opp}_s{seed}.json"


def make(opp: str, seed: int, device, out: Path | None = None) -> mt.Trainer:
    return cap.make(r2.OPPS[opp], device, out, seed=seed, launderer="oppo")


# ---------------------------------------------------------------------------- snapshots
def snapshot(tr: mt.Trainer, path: Path) -> None:
    pol = tr.policies[0]
    torch.save({"world": tr.world, **tr.roles["launderer"].state(), "rng": pol.rng.bit_generator.state,
                "sched": pol.sched_rng.bit_generator.state, "rng_cpu": torch.get_rng_state(),
                "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}, path)


def load_snapshot(tr: mt.Trainer, path: Path) -> None:
    tr.roles["launderer"].load_state(torch.load(path, map_location=tr.device, weights_only=False), networks_only=True)


# ---------------------------------------------------------------------------- evaluation
def evaluate(tr: mt.Trainer, mode: str, worlds) -> list[dict]:
    """The loaded actor on held-out ``worlds`` against ``tr``'s frozen opponents. ``mode``: ``natural``,
    ``matched`` (the references' launch schedule) or ``argmax``. Each record adds the critic's calibration and
    what the launderers decided."""
    out = []
    for j in worlds:
        pol = OptionPolicy(tr, tr.roles["launderer"], np.random.default_rng([cap.EVAL_SALT, j, 4]),
                           np.random.default_rng([cap.EVAL_SALT, j, 3]), greedy=mode == "argmax",
                           scheduled=mode == "matched")
        t0 = time.perf_counter()
        r = r2.play(tr, pol, j)
        ro = pol.finish(tr.step_no)
        G = ro.returns()
        ro.check(G)
        r |= {"cal": calibration(ro.rows, G), "options": decision_summary(ro.rows, G) | {"waits": pol.waits.tolist()}}
        out.append(r)
        print(f"  eval {mode} world {j}: nL {r['n_launderers']} reward {r['reward']:+.3f} utility {r['btc_utility']:+.3f} "
              f"held {r['held_frac']:.3f} tranches {r['tranches']:.2f} V-G {r['cal'].get('bias', float('nan')):+.2f} "
              f"{time.perf_counter() - t0:.0f}s", flush=True)
    return out


# ---------------------------------------------------------------------------- the reading
def _dev(rows: list[dict]) -> list[dict]:
    by = {r["world"]: r for r in rows}
    return [by[j] for j in DEV_WORLDS if j in by]


def read_block(res: dict, snaps: dict | None, matched: list[dict] | None, refs: dict, a_final: list[dict]) -> dict:
    """The five criteria (module docstring). ``snaps``: world -> natural records on ``DEV_WORLDS`` (``None`` if
    the run or its evaluation aborted); ``refs``: the reference policies' records on the same worlds."""
    if res["status"] != "ok" or snaps is None or matched is None:
        return {"status": res["status"], **{f"c{i}": False for i in range(1, 6)}, "promising": False}
    col = r2._col
    first, last = min(snaps), max(snaps)
    u = {w: col(v, "btc_utility") for w, v in snaps.items()}
    rew = col(snaps[last], "reward")
    d50 = refs["direct50"]
    ud, rd, ua = col(d50, "btc_utility"), col(d50, "reward"), col(a_final, "btc_utility")
    u_bar = max(0.5 * ud.mean(), ua.mean() - A_SLACK)
    c1 = (res["worlds_done"] == WORLDS and res["accounting_gap_max"] <= ACCT_MAX
          and res["ratio_dev_max"] is not None and res["ratio_dev_max"] <= RATIO_TOL)
    c2 = u[last].mean() > 0 and u[last].mean() >= u_bar and rew.mean() >= 0.5 * rd.mean()
    keep = u[last] - u[first]
    cal = [r["cal"] for r in snaps[last]]
    bias = float(np.mean([c["bias"] for c in cal])) if all(c["n"] for c in cal) else float("nan")
    g_abs = float(np.mean([c["g_abs"] for c in cal])) if all(c["n"] for c in cal) else float("nan")
    c4 = bool(np.isfinite(bias) and abs(bias) <= max(0.5, 0.25 * g_abs))
    held = col(snaps[last], "held_frac") - col(d50, "held_frac")
    skill = col(matched, "btc_utility") - col(refs["uniform"], "btc_utility")
    c5 = held.mean() <= HELD_SLACK and skill.mean() >= SKILL_MIN
    out = {"status": "ok", "c1": bool(c1), "c2": bool(c2), "c3": bool(keep.mean() >= -RETAIN), "c4": c4, "c5": bool(c5),
           "utility": {str(w): float(x.mean()) for w, x in u.items()}, "reward": float(rew.mean()),
           "utility_bar": float(u_bar), "reward_bar": float(0.5 * rd.mean()),
           "direct50": {"utility": float(ud.mean()), "reward": float(rd.mean()),
                        "held": float(col(d50, "held_frac").mean())},
           "a_final_utility": float(ua.mean()),
           "retention": {"differences": keep.tolist(), "interval": list(_t95(keep))},
           "critic": {"bias": bias, "g_abs": g_abs, "tolerance": max(0.5, 0.25 * g_abs) if np.isfinite(g_abs) else None},
           "held_over_direct50": {"differences": held.tolist(), "interval": list(_t95(held))},
           "matched": {"utility": float(col(matched, "btc_utility").mean()),
                       "uniform": float(col(refs["uniform"], "btc_utility").mean()),
                       "skill": {"differences": skill.tolist(), "interval": list(_t95(skill))}},
           "worlds_done": res["worlds_done"], "accounting_gap_max": res["accounting_gap_max"],
           "ratio_dev_max": res["ratio_dev_max"]}
    return out | {"promising": all(out[f"c{i}"] for i in range(1, 6))}


def reported(snaps: dict, refs: dict, opp: str, argmax: list[dict] | None) -> dict:
    """Not gated: paired comparisons, where the proceeds went, the critic's strata and the launch behaviour."""
    col, last = r2._col, snaps[max(snaps)]
    u, rw = col(last, "btc_utility"), col(last, "reward")
    out = {"vs": {}}
    for name in ("wait", "direct50", "repeat50"):
        out["vs"][name] = {"utility": list(_t95(u - col(refs[name], "btc_utility"))),
                           "reward": list(_t95(rw - col(refs[name], "reward")))}
    f = r2.run_dir("PL", opp, 0) / "stage1.json"
    if f.exists() and json.loads(f.read_text())["snaps"] is not None:
        pl = _dev(json.loads(f.read_text())["snaps"][str(SNAPS[-1])])
        pl = [r for r in pl if r["world"] in {x["world"] for x in last}]
        out["vs"]["PL_w30"] = {"utility": list(_t95(u - col(pl, "btc_utility"))),
                               "held": float(col(pl, "held_frac").mean())}
    out["proceeds"] = {k: float(col(last, k).mean()) for k in ("success_frac", "frozen_frac", "fee_frac", "held_frac",
                                                                 "frozen_change_frac", "tranches", "detection_rate")}
    cal = [r["cal"] for r in last]
    out["critic"] = {k: float(np.mean([c[k] for c in cal if c.get(k) is not None]))
                     for k in ("bias_btc", "rmse", "rmse_norm", "explained_variance", "g_mean")}
    strata: dict = {}
    for c in cal:
        for k, v in c["strata"].items():
            s = strata.setdefault(k, [0, 0.0])
            s[0] += v["n"]
            s[1] += v["bias"] * v["n"]
    out["critic"]["strata"] = {k: {"n": n, "bias": b / n} for k, (n, b) in sorted(strata.items())}
    opts = [r["options"] for r in last]
    hz: dict = {}
    for o in opts:
        for k, v in o["hazard"].items():
            h = hz.setdefault(k, [0, 0])
            h[0], h[1] = h[0] + v[0], h[1] + v[1]
    out["options"] = {"decisions": int(sum(o["decisions"] for o in opts)),
                      "waits": np.sum([o["waits"] for o in opts], axis=0).tolist(),
                      "launches_by_mode": np.sum([o["launches"] for o in opts], axis=0).tolist(),
                      "launch_noops": int(sum(o["launch_noops"] for o in opts)), "hazard": hz}
    if argmax is not None:
        out["argmax_utility"] = float(col(argmax, "btc_utility").mean())
    return out


def training_summary(out: Path) -> dict:
    """Per training world, from ``metrics.jsonl``: the actor passes kept and undone, the KLs and the learner's
    other statistics, and the world's economics."""
    rows = []
    for _, e in mt.read_records(out / "metrics.jsonl"):
        if e["kind"] != "epoch":
            continue
        st = e["learner"].get("launderer", {})
        rows.append({"world": e["world"] + 1, "utility": e["btc_utility"], "held": e["held_frac"],
                     **{k: st.get(k) for k in ("n", "epochs_accepted", "epochs_rejected", "kl", "entropy", "clip_frac",
                                               "ratio_q", "adv_scale", "value_bias", "grad_actor", "grad_critic",
                                               "ratio_dev_max")}})
    return {"worlds": rows}


# ---------------------------------------------------------------------------- commands
def train(a) -> dict:
    smoke = a.worlds != WORLDS
    out = run_dir(a.opp, a.seed, smoke)
    if out.exists():
        if not smoke:
            raise SystemExit(f"{out}: a block's run is not restarted")
        shutil.rmtree(out)
    (out / "snap").mkdir(parents=True)
    tr = make(a.opp, a.seed, torch.device(a.device), out)
    (out / "config.yaml").write_text(yaml.safe_dump(
        {**asdict(tr.cfg), "opponent": str(r2.OPPS[a.opp]), "opp_eps": cap.OPP_EPS,
         "learn": ["launderer"], "git_sha": mt.git_sha()}, sort_keys=False))
    (out / "metrics.jsonl").write_text("")
    snapshot(tr, out / "snap" / "w00.pt")
    status, t0 = "ok", time.perf_counter()
    try:
        for w in range(1, a.worlds + 1):
            r = tr.run_world()
            if w % 2 == 0 or w >= 15 or w == a.worlds:
                snapshot(tr, out / "snap" / f"w{w:02d}.pt")
            if w in (15, 30):
                torch.save(tr.policies[0].rollout, out / "snap" / f"rollout_w{w:02d}.pt", pickle_protocol=5)
            st = r["learner"]["launderer"]
            kl = max((max(k) for k in st.get("kl", [])), default=float("nan"))
            print(f"{out.name} world {w}/{a.worlds} | R_L {r['reward']['launderer']:+.3f} utility {r['btc_utility']:+.3f} "
                  f"held {r['held_frac']:.2f} det {r['detection_rate']:.2f} tranches {r['tranches']} "
                  f"direct {r['strategy_share']['direct']:.2f} | decisions {st.get('n')} passes "
                  f"{st.get('epochs_accepted')}+{st.get('epochs_rejected')} kl {kl:.4f} H {st.get('entropy') or float('nan'):.2f} "
                  f"| {r['sec']['world']:.0f}s, {(time.perf_counter() - t0) / 60:.1f} min", flush=True)
    except ABORTS as e:
        status = f"aborted: {type(e).__name__}: {e}"
        print(f"{out.name}: {status}", flush=True)
    st = [e["learner"].get("launderer", {}) for e in tr.epochs]
    res = {"opp": a.opp, "seed": a.seed, "worlds": a.worlds, "worlds_done": tr.world, "status": status,
           "git_sha": mt.git_sha(), "minutes": (time.perf_counter() - t0) / 60,
           "accounting_gap_max": max((abs(e["accounting_gap"]) for e in tr.epochs), default=0.0),
           "ratio_dev_max": max((s["ratio_dev_max"] for s in st if s.get("n")), default=None),
           "value_dev_max": max((s["value_dev_max"] for s in st if s.get("n")), default=None),
           "return_check_max": max((float(s["return_check"]) for s in st if "return_check" in s), default=None)}
    (out / "result.json").write_text(json.dumps(res, indent=1))
    return res


def read(a) -> dict:
    smoke = a.worlds != WORLDS or a.eval_worlds is not None
    out = run_dir(a.opp, a.seed, smoke)
    res = json.loads((out / "result.json").read_text())
    worlds = DEV_WORLDS[:a.eval_worlds]
    snap_worlds = SNAPS if not smoke else (0, a.worlds)          # a smoke run reads its first and last snapshot
    snaps = matched = argmax = None
    if res["status"] == "ok":
        tr = make(a.opp, a.seed, torch.device(a.device))
        try:
            snaps = {}
            for w in snap_worlds:
                load_snapshot(tr, out / "snap" / f"w{w:02d}.pt")
                snaps[w] = evaluate(tr, "natural", worlds)
            matched = evaluate(tr, "matched", worlds)                  # the world-30 snapshot is loaded
        except ABORTS as e:
            res, snaps = res | {"status": f"aborted in the reading: {type(e).__name__}: {e}"}, None
        if snaps is not None:                                          # reported only
            try:
                argmax = evaluate(tr, "argmax", worlds)
            except ABORTS:
                argmax = None
    refs = {p: [r for r in r2._refs(a.opp, "selection")[p] if r["world"] in worlds] for p in r2.REFS}
    a_final = [r for r in json.loads((r2.run_dir("A", a.opp, 0) / "result.json").read_text())["eval"]["boltzmann"]
               if r["world"] in worlds]
    reading = read_block(res, snaps, matched, refs, a_final)
    rep = reported(snaps, refs, a.opp, argmax) if snaps is not None else None
    (out / "oppo.json").write_text(json.dumps(
        {"opp": a.opp, "seed": a.seed, "reading": reading, "reported": rep, "training": training_summary(out),
         "snaps": None if snaps is None else {str(w): v for w, v in snaps.items()}, "matched": matched,
         "argmax": argmax}, indent=1))
    print(json.dumps(reading, indent=1))
    if rep is not None:
        print(json.dumps(rep, indent=1))
    return reading


def cmd_run(a) -> None:
    """One block, one attempt: train, then read. The marker written first makes a second start refuse."""
    smoke = a.worlds != WORLDS or a.eval_worlds is not None
    mark = mark_path(a.opp, a.seed) if not smoke else r2.RUNS / "smoke" / f"oppo_{a.opp}_s{a.seed}.json"
    if mark.exists() and not smoke:
        raise SystemExit(f"{mark}: this block was already attempted ({json.loads(mark.read_text())['state']}); "
                         "it is not run again")
    mark.parent.mkdir(parents=True, exist_ok=True)
    info = {"opp": a.opp, "seed": a.seed, "started": time.strftime("%Y-%m-%d %H:%M:%S"), "state": "started"}
    mark.write_text(json.dumps(info, indent=1))
    train(a)
    reading = read(a)
    mark.write_text(json.dumps(info | {"state": "done", "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
                                       "promising": reading["promising"]}, indent=1))


def cmd_read(a) -> None:
    smoke = a.worlds != WORLDS or a.eval_worlds is not None
    out = run_dir(a.opp, a.seed, smoke)
    if not (out / "result.json").exists() or (out / "oppo.json").exists():
        raise SystemExit(f"{out}: read finishes a block whose training completed and whose reading did not run")
    read(a)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "read"):
        p = sub.add_parser(name)
        p.add_argument("--opp", choices=list(r2.OPPS), required=True)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
        p.add_argument("--worlds", type=int, default=WORLDS, help="anything else is a smoke run")
        p.add_argument("--eval-worlds", type=int, help="a smoke run's evaluation worlds")
    a = ap.parse_args(argv)
    {"run": cmd_run, "read": cmd_read}[a.cmd](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
