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

**Outcome of the two blocks** (2026-11-11; runs/repair2/J_nda1_s0/stage1.json, P_nda1_s0/stage1.json). Neither
is promising. ``J`` keeps its trunk active (about 76 of 128 units fire, no blind state) but its values inflate
(``Q`` above the return by 3.4) and its world-30 utility is -0.26. ``P`` reaches +0.48 at world 30 (``direct50``:
+0.58, the legacy final: +0.07) and improves from world 20, but its ReLU trunk decays (7 units fire, 4.7 % of
the panel's states are all-zero, the bar is 1 %) and ``Q`` is 1.09 below the return (tolerance 0.5).

**Cell ``PL``: the wealth scale with a LeakyReLU trunk** (declared 2026-11-11 before its run; the user's
decision). The legacy learner plus the two changes that each did their part: ``scale = "wealth"`` as in ``P``
and ``trunk_act = "leaky"`` as in ``J``'s head. Two factors away from the legacy learner, one from ``P``. One block
against NDA-1 (``run``, 45-minute cap, one attempt), read as step 2 declares, with one difference that follows
from the trunk, as for ``J``: a leaky trunk cannot be all-zero, so criterion 1 also requires that at world 30 the
modal row of the network's own output covers at most 5 % of the panel's idle states. Criterion 4 is unchanged,
although ``P`` failed it: the soft target is compared with the sampled policy's entropy-matched return at
epsilon 0. To tell a pessimistic critic from the cost of exploration, the reading also plays the world-30
snapshot on the same four worlds at the training epsilon of 0.05 and reports its utility and ``Q - G`` there
(reported, not gated; under epsilon the entropy-matched return is not an exact identity). Promising leads to
``PL`` against Full-0 as its own block.

**Outcome of ``PL``'s blocks** (2026-11-11; runs/repair2/PL_*_s0/stage1.json). Against NDA-1 it is promising, the
first learner to meet all four criteria, but thinly: world-30 utility +0.33 (bar +0.29), reward +0.283 (bar
+0.269). Against Full-0 it is not (criteria 2 and 4): utility +0.23 where the bar is +0.40 (``A``'s final minus
0.02), and ``Q`` above the return by 1.23. Both hold about half of the proceeds at world end (NDA 49 %, Full 56 %;
``direct50`` 7 to 9 %): launderers over 19 BTC launch at under 1 % of their idle decisions. A fourth Sol consult
(runs/sol_consult/CONSULT4.md, read-only probes) found that against Full-0 rich waits are valued 2.2 above their
realized return with a small one-step error, while against NDA-1 they are about calibrated; that launches are
under-valued in both (in three same-state forks ``Q_go`` was 0.3 to 0.75 where the launch returned 3.0 to 4.2,
with ``direct`` at 50 % already sampled); and that rich ``go`` rows carry 24 % / 33 % of the root's squared error
in reward units but 3 % / 5 % of it under the wealth scale's ``1 / m(s)^2``. Its replays also showed that four
GPU evaluation worlds can move by up to 0.08 utility between numerically equivalent paths.

**Cell ``PLR1``: ``PL`` with the root's error divided by ``m(s)``** (declared 2026-11-12 before its run; the
consult's section B1, approved by the user the same day). One factor away from ``PL``: ``root_weight = "m"``
(``hns.marl.learner``), so a transition's loss is
``[(y - Q_root)^2 / m(s) + sum_{realized launch branches b} (y - Q_b)^2 / m(s)^2] / |R(s, a)|``. Everything else is
``PL``'s: ``Q = m(s) q``, the behaviour policy ``softmax(Q / 0.01)``, the one-step soft root target, the LeakyReLU
trunk, the shared value stream, the root's ``1 / |R|``, FIFO replay, optimizer, schedule, seed 0 and its common
initial weights, the same 30 worlds and frozen opponents. Expected mechanism (an inference, not a result): the
change acts mostly through rich launch rows, whose errors are of order 2 to 3; a rich wait's one-step error is of
order 0.03, so its larger weight changes little. The consult's offline fit on frozen features improved Full-0's
BTC-weighted root error by 28 % and NDA-1's by 3 %: a bet, not a verified repair.

One block against **Full-0 first** (the opponent ``PL`` fails), one attempt (``run``), with a 25-minute cap for
training and reading together (``scripts/m5_repair2.sh stage1``; the four earlier blocks took 16 to 18 minutes, and
a cut block is not run again). Read exactly as ``PL``'s block: the four criteria with the modal-row check, and
the reported replay at epsilon 0.05. Against Full-0, criterion 2 needs utility at least +0.402 and reward at least
+0.323. Also reported, not gated (``reported``, in stage1.json):

- utilization of the world-30 snapshot, per world and on average, next to ``direct50``'s on the same worlds: where
  the proceeds went (cashed, frozen, fees, held), held minus ``direct50``'s held (with its paired interval),
  launches per launderer and episode, and the share of launderer-episodes with a launch;
- per source-balance band and world-time bin (sub-steps 0-249, 250-399, 400-449, 450-489, 490-499): idle decisions
  with a feasible launch, how many tried to launch and how many of those were no-ops; the same for in-flight
  decisions and proceeding;
- calibration (``Q - G``, soft) by stratum: idle and in flight, the three bands, the world's halves, BTC-weighted;
- the world-30 utility minus ``PL``'s on the same opponent, per world, with its paired interval. With four worlds
  and the numerical spread above, a difference of a few hundredths is not a finding.

After the block, outside its cap, on CPU (``diag``; reported only): for ``PLR1`` and ``PL`` on the same opponent,
the world-30 network's error against its own one-step target on the world-30 buffer (every rich entry plus 6,000
others, weighted as in step 1), for the root by root action (wait, go, in flight) and band, in reward units, divided
by ``m(s)`` and BTC-weighted, and for the launch branches; the buffer's finalized rich entries that end the world
(by root action); and the training clip share per world.

The held-funds requirement of the confirmation (``hns.marl.repair2``, added the same day) is not part of this
progress label; held funds are reported so that it can be anticipated.

What follows (agreed with the user before the run). Promising against Full-0: ``PLR1`` against NDA-1 as its own
block, read the same way. Not promising: if the world-30 held share against Full-0 is still at least ``PL``'s
minus 0.10 (rich launderers still wait), the next block is ``PL`` with a hard root target, Full-0 first; if it fell
by more than that (they launch, and what they launch is what fails), ``PL`` with independent value streams,
NDA-1 first. A numerical failure (criterion 1) is reported first and decides neither. Each next block is declared
here before it runs. If it fails too, the next step is the user's decision; a structural change of the learner
(one launch-timing decision per episode, or an actor-critic) would depart from SPEC's Algorithm 1.

Usage::

    python -m hns.marl.stage1 offline --opp full0        (and nda1)
    python -m hns.marl.stage1 choose
    scripts/m5_repair2.sh stage1 P nda1                  (stage1 run: train, then read; the block's cap)
    python -m hns.marl.stage1 diag --cell PLR1 --opp full0   (after the block, CPU, reported only)
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
TRAIN_EPS = 0.05                                    # the exploration floor of training, for the reported replay
A_SLACK, RETAIN, MODAL_MAX = 0.02, 0.05, 0.05


# ---------------------------------------------------------------------------- step 1: offline head fits
def offline_data(opp: str, device, n_other: int = N_OTHER, cell: str = "A") -> dict:
    """The fit's fixed inputs and targets from a run's world-30 buffer and networks (step 1: the legacy run's);
    ``pred``: that network's own taken values in reward units (0 at unrealized branches)."""
    out = r2.run_dir(cell, opp, 0)
    tr = r2.make(cell, opp, 0, device)
    r2.load_snapshot(tr, out / "snap" / f"w{r2.WORLDS:02d}.pt")
    L = tr.roles["launderer"]
    buf = torch.load(out / "snap" / f"replay_w{r2.WORLDS:02d}.pt", weights_only=False)
    fin = np.flatnonzero(buf.nxt[:min(buf.n, buf.cap)] != PENDING)
    bal = np.array([buf.obs[s][4][1] for s in fin])
    rich, other = fin[bal >= RICH], fin[bal < RICH]
    pick = np.random.default_rng(0).choice(other, min(n_other, len(other)), replace=False)
    slots = np.concatenate([rich, pick])
    w = np.concatenate([np.ones(len(rich)), np.full(len(pick), len(other) / len(pick))])
    keys = ("h", "y", "pred", "masks", "action", "realized", "cand", "x_c", "t", "done")
    parts = {k: [] for k in keys}
    L.net.eval()
    for i in range(0, len(slots), 256):
        b = buf.take(slots[i:i + 256])
        ob = collate(b["obs"])
        with torch.no_grad():
            g, f = ob.tensors(device)
            parts["h"].append(L.net.enc.encode(g, f["x_tx"], f["x_w"], x_c=f["x_c"]).cpu())
            parts["y"].append(L.batch_target(b).cpu())
            act, rz = torch.from_numpy(b["action"]).to(device), torch.from_numpy(b["realized"]).to(device)
            parts["pred"].append(L.taken_q(L.net, ob, b["masks"], act, rz).cpu())
        for k, v in (("masks", b["masks"]), ("action", b["action"]), ("realized", b["realized"]), ("cand", ob.cand),
                     ("x_c", ob.x_c), ("t", b["t"]), ("done", b["done"])):
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
    if cell["trunk_act"] == "leaky":                           # never all-zero: a blind network shows as one row
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


BASE = {"PLR1": "PL"}                                # a cell's paired comparison in the reported part


def utilization(rows: list[dict], d50: list[dict]) -> dict:
    """Reported: where the proceeds went and how often launderers launched, per world, next to ``direct50``'s."""
    out = {"worlds": [r["world"] for r in rows]}
    for k in ("success_frac", "frozen_frac", "fee_frac", "held_frac", "tranches"):
        out[k] = {"policy": r2._col(rows, k).tolist(), "direct50": r2._col(d50, k).tolist()}
    over = r2._col(rows, "held_frac") - r2._col(d50, "held_frac")
    out["held_over_direct50"] = {"differences": over.tolist(), "interval": list(_t95(over))}
    out["ep_launch_share"] = [r.get("ep_launch_share") for r in rows]
    return out


def strata(rows: list[dict]) -> dict:
    """Reported: the hazard table summed over the worlds, and the soft calibration bias by stratum pooled over
    decisions (the BTC-weighted one averaged over worlds)."""
    hz = {k: np.sum([r["hazard"][k] for r in rows if "hazard" in r], axis=0).tolist()
          for k in (rows[0].get("hazard") or {})}
    cal = {}
    for k in ("idle", "flight", "bal_low", "bal_mid", "bal_high", "early", "late"):
        n = sum(r["cal"][k]["n"] for r in rows if k in r["cal"])
        b = sum(r["cal"][k]["bias_soft"] * r["cal"][k]["n"] for r in rows if k in r["cal"] and r["cal"][k]["n"])
        cal[k] = {"n": int(n), "bias_soft": b / n if n else None}
    cal["btc_weighted"] = float(np.mean([r["cal"]["bias_soft_btc"] for r in rows if r["cal"]["n"]]))
    return {"hazard": hz, "calibration": cal}


def versus(rows: list[dict], cell: str, opp: str) -> dict | None:
    """Reported: the world-30 utility minus that of the base cell's block on the same opponent, per world."""
    f = r2.run_dir(BASE[cell], opp, 0) / "stage1.json" if cell in BASE else None
    if f is None or not f.exists() or json.loads(f.read_text())["snaps"] is None:
        return None
    base = _dev(json.loads(f.read_text())["snaps"][str(SNAPS[-1])])
    d = r2._col(rows, "btc_utility") - r2._col(base, "btc_utility")
    return {"base": BASE[cell], "differences": d.tolist(), "interval": list(_t95(d)),
            "base_held": float(r2._col(base, "held_frac").mean())}


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
    snaps = argmax = explore = None
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
        if snaps is not None:                                  # reported only, so its failure changes no reading
            try:
                explore = r2.evaluate(tr, "boltzmann", DEV_WORLDS, eps=TRAIN_EPS)
            except FloatingPointError:
                explore = None
    a_final = _dev(json.loads((r2.run_dir("A", a.opp, 0) / "result.json").read_text())["eval"]["boltzmann"])
    d50 = _dev(r2._refs(a.opp, "selection")["direct50"])
    reading = read_run(res, snaps, a_final, d50, argmax)
    if snaps is not None and argmax is not None:
        reading["argmax_utility"] = float(np.mean([r["btc_utility"] for r in argmax]))
    reported = None
    if snaps is not None:                                      # reported only: none of it changes the reading
        last = snaps[SNAPS[-1]]
        reported = {"utilization": utilization(last, d50), **strata(last), "versus": versus(last, a.cell, a.opp)}
    (out / "stage1.json").write_text(json.dumps({"cell": a.cell, "opp": a.opp, "seed": a.seed, "reading": reading,
                                                  "snaps": None if snaps is None else {str(w): v for w, v in snaps.items()},
                                                  "argmax": argmax, "explore": explore, "reported": reported}, indent=1))
    print(json.dumps(reading, indent=1))
    if reported is not None:
        u, v = reported["utilization"], reported["versus"]
        print(f"world {SNAPS[-1]} (reported): held {np.mean(u['held_frac']['policy']):.3f} (direct50 "
              f"{np.mean(u['held_frac']['direct50']):.3f}; over it by {u['held_over_direct50']['interval'][0]:+.3f}, "
              f"95 % {u['held_over_direct50']['interval'][1]:+.3f} to {u['held_over_direct50']['interval'][2]:+.3f}); "
              f"cashed {np.mean(u['success_frac']['policy']):.3f}, frozen {np.mean(u['frozen_frac']['policy']):.3f}; "
              f"launches per launderer-episode {np.mean(u['tranches']['policy']):.2f}; launderer-episodes with a "
              f"launch {np.mean([x for x in u['ep_launch_share'] if x is not None] or [np.nan]):.3f}")
        print(f"calibration by stratum (soft, pooled): {reported['calibration']}")
        print(f"hazard [idle feasible, tried to launch, no-op, in flight, tried to proceed, no-op] by band/time bin: "
              f"{reported['hazard']}")
        if v is not None:
            print(f"minus {v['base']} at world {SNAPS[-1]}: {[round(x, 3) for x in v['differences']]}, mean "
                  f"{v['interval'][0]:+.3f} (95 % {v['interval'][1]:+.3f} to {v['interval'][2]:+.3f}); "
                  f"{v['base']} held {v['base_held']:.3f}")
    if snaps is not None:
        shown = {f"world {w}, sampled policy": (rows, "soft") for w, rows in snaps.items()}
        if argmax is not None:                                 # a hard target is calibrated on these, not on the above
            shown[f"world {SNAPS[-1]}, argmax policy"] = (argmax, "hard")
        if explore is not None:
            print(f"world {SNAPS[-1]} at the training epsilon {TRAIN_EPS} (reported): BTC utility "
                  f"{np.mean([r['btc_utility'] for r in explore]):+.3f}")
            shown[f"world {SNAPS[-1]}, sampled policy at epsilon {TRAIN_EPS}"] = (explore, "soft")
        for name, (rows, k) in shown.items():
            gate = {b: np.sum([r["gate"][b] for r in rows], axis=0).tolist() for b in ("low", "mid", "high")}
            cal = {b: round(float(np.mean([r["cal"][b].get(f"bias_{k}", np.nan) for r in rows])), 3)
                   for b in ("bal_low", "bal_mid", "bal_high")}
            print(f"{name}: per band [idle decisions, launches, prefers direct, prefers 50 %] {gate}; Q - G ({k} "
                  f"return) by band {cal}; BTC-weighted {np.mean([r['cal'][f'bias_{k}_btc'] for r in rows]):+.3f}")
    return reading


# ---------------------------------------------------------------------------- after a block: reported only
def buffer_errors(cell: str, opp: str, device) -> dict:
    """The world-30 network's error against its own one-step target on its world-30 buffer (every rich entry and
    ``N_OTHER`` others, weighted as in step 1, all of them measured), the rich entries that end the world, and the
    training clip share per world."""
    d = offline_data(opp, device, cell=cell)
    d["held"] = torch.ones(len(d["y"]), dtype=torch.bool)       # measure every sampled entry
    out = measures(d, d["pred"])
    x_c, a0, done = d["x_c"].numpy(), d["action"].numpy()[:, 0], d["done"].numpy()
    rich = x_c[:, 1] >= RICH
    kind = np.where(x_c[:, 0] > 0, "flight", np.where(a0 == GO, "go", "wait"))
    out["rich_terminal"] = {k: int((rich & done & (kind == k)).sum()) for k in ("wait", "go", "flight")}
    out["rich_entries"], out["entries"] = int(d["n_rich"]), int(d["n_rich"] + d["n_other"])
    clip = {}
    for line in (r2.run_dir(cell, opp, 0) / "metrics.jsonl").read_text().splitlines():
        e = json.loads(line)
        st = (e.get("learner") or {}).get("launderer") or {}
        if st.get("n"):
            clip.setdefault(e["world"] + 1, []).append(st["clip_share"])
    out["clip_share"] = {w: float(np.mean(v)) for w, v in sorted(clip.items())}
    return out


def cmd_diag(a) -> None:
    res = {}
    for cell in (a.cell, BASE.get(a.cell)):
        if cell is None or not (r2.run_dir(cell, a.opp, 0) / "snap" / f"replay_w{r2.WORLDS:02d}.pt").exists():
            continue
        res[cell] = m = buffer_errors(cell, a.opp, torch.device(a.device))
        print(f"{cell} {a.opp}: root {m['root']:.3f}  root/m {m['root_norm']:.3f}  root BTC-weighted {m['root_btc']:.3f}  "
              f"child {m['child']:.3f}; rich entries ending the world {m['rich_terminal']} of {m['rich_entries']}; "
              f"clip share at worlds 20 / 30 {m['clip_share'].get(20)} / {m['clip_share'].get(30)}")
        for k, v in m["by"].items():
            print(f"    {k:12s} n {v['n']:6d}  root {v['root']:.3f}  root/m {v['root_norm']:.3f}")
    (r2.run_dir(a.cell, a.opp, 0) / "stage1_diag.json").write_text(json.dumps(res, indent=1))


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
    p = sub.add_parser("diag")
    p.add_argument("--cell", choices=list(r2.STAGE_CELLS), required=True)
    p.add_argument("--opp", choices=list(r2.OPPS), required=True)
    p.add_argument("--device", default="cpu")
    a = ap.parse_args(argv)
    {"offline": cmd_offline, "choose": cmd_choose, "run": cmd_run, "diag": cmd_diag}[a.cmd](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
