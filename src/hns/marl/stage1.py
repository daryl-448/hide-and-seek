"""Stage 1 of the launderer learner's repair: which one-factor change to train first, and how its run is read.

Declared on 2026-11-11, before any of it ran. Source: the third Sol consult (runs/sol_consult/CONSULT3.md,
sections B, C0, C1 and C4), after stage 0 (``hns.marl.repair2``).

**What is known.** Stage 0 trained the legacy learner (cell ``A``) against both frozen opponents. Against NDA-1
the policy was lost while the trunk was still active. The consult then found that rich launderers there stop
launching because the launch options they would take are bad (their launch value matches what those launches
return), that the inspector does not flag a rich ``direct`` launch at 50 % more often, and that entries of
launderers holding over 19 BTC are 1 to 4 % of the replay buffer but 38 to 49 % of the squared TD error. Two
one-factor changes to the legacy learner are candidates, each a cell of ``repair2.STAGE_CELLS``:

- ``P``, the wealth scale (``hns.marl.learner``): the network outputs ``q`` with ``Q = m(s) q``,
  ``m(s) = 1 + (source balance + tranche holders' funds) / exp(1.5) BTC``, and the loss fits ``q`` to ``y / m(s)``.
  The reward, the soft root target and the behaviour policy ``softmax(Q / tau)`` are unchanged;
- ``I``, a value stream per branch (``baseline = "independent"``).

**Step 1, offline head fits** (``offline``; CPU, no simulator learning, no GPU). Per opponent, from the legacy
run's world-30 replay buffer: every finalized entry whose source holds at least ``expm1(3)`` = 19.09 BTC, plus
6,000 other finalized entries drawn uniformly without replacement (seed 0). An entry's weight is 1 if rich, else
(other entries) / 6,000, so weighted statistics describe the buffer. Inputs: the world-30 online encoder's
readout of each entry's observation, frozen. Targets: the legacy learner's one-step target ``y`` from the world-30
target network, fixed. Held out: the entries observed in training worlds whose 0-based index is divisible by 3;
the others train. Four fresh heads from the same initial weights (seed 0), shared or independent baseline x raw
or wealth scale, each trained for 1,000 updates of Adam at 1e-3 on batches of 128 drawn with probability
proportional to the weights (as uniform replay would draw them), gradient clip 1.0, the realized-branch squared
loss. No early stopping: the last step is what is measured.

Held-out measures, all weighted: ``root``, the RMSE of the root's taken value in reward units; ``root_norm``, the
same error divided by ``m(s)``; ``root_btc``, ``root`` weighted also by the launderer's funds; ``child``, the RMSE
over the realized launch branches' taken values.

- The wealth scale is useful if, against both opponents, shared/wealth has ``root_norm`` at least 20 % below
  shared/raw's and ``root`` at most 10 % above it. It is down-weighted if ``root_btc`` is above shared/raw's
  against either opponent.
- The independent baseline is useful if, against both, independent/raw has ``child`` at least 20 % below
  shared/raw's and ``root`` at most 10 % above it.
- The cell to train first (``choose``): ``P`` if the wealth scale is useful and not down-weighted; else ``I`` if
  the independent baseline is useful; else ``P`` if the wealth scale is useful (down-weighted, and said so); else
  none, and no training run follows: the result is reported instead.

These fits say how well each head form can fit fixed historical targets on a frozen representation. They do not
show that a variant learns a good policy.

**Step 2, one training run.** The chosen cell, 30 worlds against NDA-1, seed 0: the same worlds, opponents,
panel, snapshots and logs as stage 0's ``A`` run, trained without the screen's evaluation. One block of at most
45 minutes of local GPU for training and reading together (``run``, under ``scripts/m5_repair2.sh stage1``'s
timeout). A block is one attempt: ``run`` records that it started and refuses to start again, so a block cut by
the cap, or failed, stays incomplete and is reported as such. Full-0 follows only after a promising NDA-1
reading, as its own block.

The reading plays the world-20 and world-30 snapshots on development worlds 100 to 103 at epsilon 0 with the
sampled policy. References on the same worlds: the legacy ``A`` run's final network and scripted ``direct`` at 50 % once
per episode. The run is **promising** if all four hold:

1. it completed its 30 worlds without a numerical abort, the BTC accounting gap is at most 1e-6, and at world 30
   at most 1 % of the panel's states have an all-zero trunk output;
2. the world-30 snapshot's mean BTC utility and mean reward are each at least half of ``direct50``'s, and its
   mean BTC utility is positive and at most 0.02 below ``A``'s;
3. the world-30 snapshot's BTC utility minus the world-20 snapshot's is at least -0.05 on average over the four
   worlds (all four differences and the paired interval are reported);
4. on the world-30 rollouts, the bias of ``Q_initiate`` against the entropy-matched return is within
   ``max(0.5, 0.25 mean |G|)``, in reward units (also reported by balance band and BTC-weighted).

Only worlds 20 and 30 are read; no other snapshot is evaluated or chosen. Worlds 100 to 103 were already used by
stage 0, and this is one seed: a promising run is a reason to run the next block, not an M5 acceptance and not
the gates P1 to P3 of ``hns.marl.repair2``. Not promising: a healthier trunk and better errors without the
economics points to the other one-factor cell; a trunk that still collapses points to a LeakyReLU-only test.
Whatever the reading, it is reported before the next block runs.

Beyond stage 0's logs the run records, per world, the per-branch squared errors both in reward units and divided
by ``m(s)^2`` and the scale's mean and maximum over the sampled rows; on the panel, the largest ``|q|`` and the
modal row of the network's own output (so the scale cannot hide a blind network); and per evaluation world, by
balance band, how often a launch happens and which strategy and amount the network prefers where it waits. The
ratio ``m(s') / m(s)`` is not logged.

**Outcome of step 1** (2026-11-11): neither form met its bar, so by the rule no cell was chosen
(runs/repair2/stage1_choice.json). The user then decided the next two blocks, in this order.

**The joint launch critic, cell ``J``** (declared 2026-11-11 before its run; the consult's section C5, with
departures listed). What changes: the launderer's critic (``hns.models.qnet.JointHeads``) and its target
(``hns.marl.learner``); it is a bundle, not one factor.

- Values: ``Q(wait)``; in flight ``Q(proceed)``; for a launch ``U(mode) + A(recipient) - max A``, with one output
  per mode (amount x strategy x mules x change, 144) and the recipient scored by attention; ``Q(go) = max_mode U``.
  So the gate compares waiting with the best launch, not with an average over launch options, and a launch trains
  the value of the action taken, with the loss's whole weight.
- Target: hard and double. The online network picks wait or go at ``s'`` (and the best mode and recipient); the
  target network values that choice. ``tau`` shapes behaviour only.
- Behaviour: the gate is ``softmax((Q(wait), Q(go)) / tau)``; a launch samples mode and recipient from
  ``softmax(U / tau)`` and ``softmax(A / tau)``; epsilon replaces the gate's choice and each launch branch's
  option, each on its own, by a uniform feasible one, so every option is explored as often as before.
- Trunk: LeakyReLU(0.01). Everything else is the legacy learner's: the reward, FIFO replay, optimizer, schedule.
- Departures from C5: the critic is joint over the four table branches and additive in the recipient, not one
  table of 2,304 tuples; the leaky trunk is part of the bundle.

One block against NDA-1 (``run``, 45-minute cap, one attempt), read like step 2, with two differences that follow
from the design: criterion 4 compares ``Q`` with the plain return of the argmax policy (the target is hard), on
the world-30 snapshot's argmax rollouts of the same four worlds; and criterion 1, since a leaky trunk cannot be
all-zero, also requires that at world 30 the modal row of the network's values covers at most 5 % of the panel's
idle states (a blind network gives one row to most of them). Promising leads to the same cell against Full-0 as
its own block.

**Cell ``P`` anyway** (the user's decision, outside step 1's rule): one block against NDA-1, read exactly as
step 2 declares. It is exploratory: step 1 did not select it.

Usage::

    python -m hns.marl.stage1 offline --opp full0        (and nda1)
    python -m hns.marl.stage1 choose
    scripts/m5_repair2.sh stage1 P nda1                  (stage1 run: train, then read; 45-minute cap)
"""

from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np
import torch

import hns.marl.repair2 as r2
from hns.marl.buffer import PENDING
from hns.marl.capability import _t95
from hns.marl.learner import WEALTH_REF, wealth_scale
from hns.marl.obs import BRANCHES, D_CAND, collate
from hns.models.qnet import BranchHeads, taken, td_loss
from hns.sim.entities import GO
from hns.sim.world import EPISODES, SUBSTEPS

RICH, N_OTHER, HOLD_EVERY = 3.0, 6000, 3            # log1p(source balance); other entries sampled; held-out worlds
UPDATES, BATCH, LR, CLIP = 1000, 128, 1e-3, 1.0
VARIANTS = {"shared/raw": ("shared", False), "shared/wealth": ("shared", True),
            "independent/raw": ("independent", False), "independent/wealth": ("independent", True)}
BETTER, WORSE = 0.8, 1.1                            # a measure at least 20 % lower; at most 10 % higher
DEV_WORLDS, SNAPS = (100, 101, 102, 103), (20, 30)
A_SLACK, RETAIN, MODAL_MAX = 0.02, 0.05, 0.05


# ---------------------------------------------------------------------------- step 1: offline head fits
def offline_data(opp: str, device, n_other: int = N_OTHER) -> dict:
    """The fit's fixed inputs and targets from the legacy run's world-30 buffer and networks."""
    out = r2.run_dir("A", opp, 0)
    tr = r2.make("A", opp, 0, device)
    r2.load_snapshot(tr, out / "snap" / f"w{r2.WORLDS:02d}.pt")
    L = tr.roles["launderer"]
    buf = torch.load(out / "snap" / f"replay_w{r2.WORLDS:02d}.pt", weights_only=False)
    fin = np.flatnonzero(buf.nxt[:min(buf.n, buf.cap)] != PENDING)
    bal = np.array([buf.obs[s][4][1] for s in fin])
    rich, other = fin[bal >= RICH], fin[bal < RICH]
    pick = np.random.default_rng(0).choice(other, min(n_other, len(other)), replace=False)
    slots = np.concatenate([rich, pick])
    w = np.concatenate([np.ones(len(rich)), np.full(len(pick), len(other) / len(pick))])
    keys = ("h", "y", "masks", "action", "realized", "cand", "x_c", "t")
    parts = {k: [] for k in keys}
    L.net.eval()
    for i in range(0, len(slots), 256):
        b = buf.take(slots[i:i + 256])
        ob = collate(b["obs"])
        with torch.no_grad():
            g, f = ob.tensors(device)
            parts["h"].append(L.net.enc.encode(g, f["x_tx"], f["x_w"], x_c=f["x_c"]).cpu())
            parts["y"].append(L.batch_target(b).cpu())
        for k, v in (("masks", b["masks"]), ("action", b["action"]), ("realized", b["realized"]), ("cand", ob.cand),
                     ("x_c", ob.x_c), ("t", b["t"])):
            parts[k].append(torch.from_numpy(np.ascontiguousarray(v)))
    d = {k: torch.cat(v) for k, v in parts.items()}
    d["w"] = torch.from_numpy(w)
    d["m"] = torch.from_numpy(wealth_scale(d["x_c"].numpy())).unsqueeze(1)
    d["held"] = (d["t"] // (EPISODES * SUBSTEPS)) % HOLD_EVERY == 0
    d["n_rich"], d["n_other"] = len(rich), len(other)
    return d


def fit(d: dict, baseline: str, scaled: bool, updates: int = UPDATES, seed: int = 0) -> torch.Tensor:
    """Train one fresh head on the training entries; returns its taken values in reward units, ``(N, branches)``
    (0 at unrealized branches)."""
    torch.manual_seed(seed)                                    # the same initial weights for every variant
    heads = BranchHeads(d["h"].shape[1], BRANCHES["launderer"], pointer={"recipient": D_CAND}, baseline=baseline)
    opt = torch.optim.Adam(heads.parameters(), lr=LR)
    rng = np.random.default_rng(seed)                          # and the same batches
    idx = np.flatnonzero(~d["held"].numpy())
    p = d["w"].numpy()[idx] / d["w"].numpy()[idx].sum()
    for _ in range(updates):
        i = torch.from_numpy(idx[rng.choice(len(idx), BATCH, p=p)])
        q = heads(d["h"][i], d["masks"][i], {"recipient": d["cand"][i]})
        y = d["y"][i] / d["m"][i, 0] if scaled else d["y"][i]
        loss = td_loss(q, heads.sizes, d["action"][i], d["realized"][i], y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(heads.parameters(), CLIP)
        opt.step()
    with torch.no_grad():
        q = heads(d["h"], d["masks"], {"recipient": d["cand"]})
        return taken(q * d["m"] if scaled else q, heads.sizes, d["action"], d["realized"])


def _rmse(e: np.ndarray, w: np.ndarray) -> float:
    return float(math.sqrt((w * e ** 2).sum() / w.sum())) if w.sum() > 0 else float("nan")


def measures(d: dict, pred: torch.Tensor) -> dict:
    """The held-out, weighted errors of one fit (the module docstring defines them)."""
    h = d["held"].numpy()
    w, y, m = d["w"].numpy()[h], d["y"].numpy()[h].astype(np.float64), d["m"].numpy()[h, 0].astype(np.float64)
    p, r = pred.numpy()[h].astype(np.float64), d["realized"].numpy()[h]
    e, funds = y - p[:, 0], (m - 1.0) * WEALTH_REF
    ec, wc = (y[:, None] - p[:, 1:])[r[:, 1:]], np.broadcast_to(w[:, None], r[:, 1:].shape)[r[:, 1:]]
    mc = np.broadcast_to(m[:, None], r[:, 1:].shape)[r[:, 1:]]
    out = {"n": int(h.sum()), "root": _rmse(e, w), "root_norm": _rmse(e / m, w), "root_btc": _rmse(e, w * funds),
           "child": _rmse(ec, wc), "child_norm": _rmse(ec / mc, wc)}
    x_c, a0 = d["x_c"].numpy()[h], d["action"].numpy()[h, 0]
    kind = np.where(x_c[:, 0] > 0, "flight", np.where(a0 == GO, "go", "wait"))
    band = np.where(x_c[:, 1] < r2.BANDS[0], "low", np.where(x_c[:, 1] < r2.BANDS[1], "mid", "high"))
    out["by"] = {f"{b}/{k}": {"n": int(s.sum()), "root": _rmse(e[s], w[s]), "root_norm": _rmse(e[s] / m[s], w[s])}
                 for b in ("low", "mid", "high") for k in ("wait", "go", "flight") if (s := (band == b) & (kind == k)).any()}
    return out


def offline_reading(res: dict) -> dict:
    """Step 1's reading and the cell to train first. ``res``: ``opponent -> variant -> measures``."""
    def ratio(opp, variant, key):
        return res[opp][variant][key] / res[opp]["shared/raw"][key]

    wealth = all(ratio(o, "shared/wealth", "root_norm") <= BETTER and ratio(o, "shared/wealth", "root") <= WORSE for o in res)
    down = wealth and any(ratio(o, "shared/wealth", "root_btc") > 1.0 for o in res)
    indep = all(ratio(o, "independent/raw", "child") <= BETTER and ratio(o, "independent/raw", "root") <= WORSE for o in res)
    cell = "P" if wealth and not down else "I" if indep else "P" if wealth else None
    return {"wealth_useful": bool(wealth), "wealth_down_weighted": bool(down), "independent_useful": bool(indep),
            "cell": cell,
            "ratios": {o: {v: {k: ratio(o, v, k) for k in ("root", "root_norm", "root_btc", "child")} for v in VARIANTS}
                       for o in res}}


def offline_path(opp: str):
    return r2.RUNS / f"offline_{opp}.json"


def cmd_offline(a) -> None:
    smoke = a.updates != UPDATES or a.others != N_OTHER
    d = offline_data(a.opp, torch.device("cpu"), a.others)
    res = {"opp": a.opp, "n": len(d["y"]), "n_rich": d["n_rich"], "n_other": d["n_other"],
           "held_out": int(d["held"].sum()), "variants": {}}
    print(f"{a.opp}: {res['n']} entries ({d['n_rich']} rich of {d['n_rich'] + d['n_other']} finalized), "
          f"{res['held_out']} held out")
    for name, (baseline, scaled) in VARIANTS.items():
        res["variants"][name] = m = measures(d, fit(d, baseline, scaled, a.updates))
        print(f"  {name:18s} root {m['root']:.4f}  root/m {m['root_norm']:.4f}  root BTC-weighted {m['root_btc']:.4f}  "
              f"child {m['child']:.4f}  child/m {m['child_norm']:.4f}", flush=True)
    f = r2.RUNS / "smoke" / f"offline_{a.opp}.json" if smoke else offline_path(a.opp)
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(res, indent=1))


def cmd_choose(a) -> None:
    res = {o: json.loads(offline_path(o).read_text())["variants"] for o in r2.OPPS}
    out = offline_reading(res)
    (r2.RUNS / "stage1_choice.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


# ---------------------------------------------------------------------------- step 2: the run's reading
def read_run(res: dict, snaps: dict | None, a_final: list[dict], direct50: list[dict], argmax: list[dict] | None = None) -> dict:
    """The four criteria of one run. ``res``: its ``result.json``; ``snaps``: ``world -> world records`` of the
    world-20 and world-30 snapshots on ``DEV_WORLDS`` (``None`` if the run or its evaluation aborted);
    ``a_final`` and ``direct50``: the references' records on the same worlds; ``argmax``: the world-30 snapshot's
    argmax-policy records, for a cell with a hard target."""
    if res["status"] != "ok" or snaps is None:
        return {"status": res["status"], "c1": False, "c2": False, "c3": False, "c4": False, "promising": False}
    u = {w: r2._col(v, "btc_utility") for w, v in snaps.items()}
    last, first = SNAPS[-1], SNAPS[0]
    rew, ua = r2._col(snaps[last], "reward"), r2._col(a_final, "btc_utility")
    ud, rd = r2._col(direct50, "btc_utility"), r2._col(direct50, "reward")
    cell = r2.ALL_CELLS[res["cell"]]
    c1 = res["accounting_gap_max"] <= r2.ACCT_MAX and res["zero_trunk_share"] <= r2.DEAD_MAX
    if cell.get("critic") == "joint":
        c1 = c1 and res["modal_row_share"] <= MODAL_MAX
    c2 = (u[last].mean() >= 0.5 * ud.mean() and rew.mean() >= 0.5 * rd.mean() and u[last].mean() > 0
          and u[last].mean() >= ua.mean() - A_SLACK)
    keep = u[last] - u[first]
    cal = r2.calibration_gate({"boltzmann": snaps[last], "argmax": argmax}, cell["backup"])
    out = {"status": "ok", "c1": bool(c1), "c2": bool(c2), "c3": bool(keep.mean() >= -RETAIN), "c4": bool(cal["ok"]),
           "utility": {str(w): float(x.mean()) for w, x in u.items()}, "reward": float(rew.mean()),
           "a_final_utility": float(ua.mean()), "direct50_utility": float(ud.mean()), "direct50_reward": float(rd.mean()),
           "retention": {"differences": keep.tolist(), "interval": list(_t95(keep))}, "calibration": cal,
           "zero_trunk_share": res["zero_trunk_share"], "modal_row_share": res.get("modal_row_share"),
           "accounting_gap_max": res["accounting_gap_max"]}
    return out | {"promising": bool(out["c1"] and out["c2"] and out["c3"] and out["c4"])}


def _dev(rows: list[dict]) -> list[dict]:
    by = {r["world"]: r for r in rows}
    return [by[j] for j in DEV_WORLDS]


def block_path(cell: str, opp: str):
    return r2.RUNS / f"stage1_{cell}_{opp}.json"


def cmd_run(a) -> None:
    """One block, one attempt: train the cell (no screen evaluation), then read it. The marker written first makes
    a second start refuse, whatever became of the first (done, failed, or cut by the script's time cap)."""
    mark = block_path(a.cell, a.opp)
    if mark.exists():
        raise SystemExit(f"{mark}: this block was already attempted ({json.loads(mark.read_text())['state']}); "
                         "it is not run again")
    mark.parent.mkdir(parents=True, exist_ok=True)
    info = {"cell": a.cell, "opp": a.opp, "seed": a.seed, "started": time.strftime("%Y-%m-%d %H:%M:%S"), "state": "started"}
    mark.write_text(json.dumps(info, indent=1))
    r2.cmd_train(argparse.Namespace(cell=a.cell, opp=a.opp, seed=a.seed, device=a.device, worlds=r2.WORLDS,
                                    eval_worlds=None, no_eval=True))
    reading = read(a)
    mark.write_text(json.dumps(info | {"state": "done", "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
                                       "promising": reading["promising"]}, indent=1))


def read(a) -> dict:
    out = r2.run_dir(a.cell, a.opp, a.seed)
    res = json.loads((out / "result.json").read_text())
    snaps = argmax = None
    if res["status"] == "ok":
        tr = r2.make(a.cell, a.opp, a.seed, torch.device(a.device))
        try:
            snaps = {}
            for w in SNAPS:
                r2.load_snapshot(tr, out / "snap" / f"w{w:02d}.pt")
                snaps[w] = r2.evaluate(tr, "boltzmann", DEV_WORLDS)
            if r2.ALL_CELLS[a.cell]["backup"] == "hard":       # the last snapshot is loaded: its argmax policy
                argmax = r2.evaluate(tr, "argmax", DEV_WORLDS)
        except FloatingPointError as e:                        # a non-finite Q while playing: not promising
            res, snaps = res | {"status": f"aborted: {e}"}, None
    a_final = _dev(json.loads((r2.run_dir("A", a.opp, 0) / "result.json").read_text())["eval"]["boltzmann"])
    reading = read_run(res, snaps, a_final, _dev(r2._refs(a.opp, "selection")["direct50"]), argmax)
    if snaps is not None and argmax is not None:
        reading["argmax_utility"] = float(np.mean([r["btc_utility"] for r in argmax]))
    (out / "stage1.json").write_text(json.dumps({"cell": a.cell, "opp": a.opp, "seed": a.seed, "reading": reading,
                                                  "snaps": None if snaps is None else {str(w): v for w, v in snaps.items()},
                                                  "argmax": argmax}, indent=1))
    print(json.dumps(reading, indent=1))
    if snaps is not None:
        shown = {f"world {w}, sampled policy": (rows, "soft") for w, rows in snaps.items()}
        if argmax is not None:                                 # a hard target is calibrated on these, not on the above
            shown[f"world {SNAPS[-1]}, argmax policy"] = (argmax, "hard")
        for name, (rows, k) in shown.items():
            gate = {b: np.sum([r["gate"][b] for r in rows], axis=0).tolist() for b in ("low", "mid", "high")}
            cal = {b: round(float(np.mean([r["cal"][b].get(f"bias_{k}", np.nan) for r in rows])), 3)
                   for b in ("bal_low", "bal_mid", "bal_high")}
            print(f"{name}: per band [idle decisions, launches, prefers direct, prefers 50 %] {gate}; Q - G ({k} "
                  f"return) by band {cal}; BTC-weighted {np.mean([r['cal'][f'bias_{k}_btc'] for r in rows]):+.3f}")
    return reading


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("offline")
    p.add_argument("--opp", choices=list(r2.OPPS), required=True)
    p.add_argument("--updates", type=int, default=UPDATES, help="anything else is a smoke run")
    p.add_argument("--others", type=int, default=N_OTHER, help="anything else is a smoke run")
    sub.add_parser("choose")
    p = sub.add_parser("run")
    p.add_argument("--cell", choices=list(r2.STAGE_CELLS), required=True)
    p.add_argument("--opp", choices=list(r2.OPPS), required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args(argv)
    {"offline": cmd_offline, "choose": cmd_choose, "run": cmd_run}[a.cmd](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
