"""M4 calibration: label-blind random search against T-safe real data (SPEC §9.6).

Random search over the §9.7 bounds (``hns.sim.world.CALIBRATED``), static
mode, CPU, no learning. Each configuration runs ``worlds`` worlds and is
scored by the mean JS (§7) to **T-safe real data only** (ts 1-29, pooled over
classes and unknown, no labels read):

- each of the 17 shared tx features (non-null txs), on the D5 bins of ``T.json``;
- each D3 causal wallet count at t = 29 (each wallet's last state at or
  before ts 29), on the D5 bins; the simulator's counts are each wallet's
  state at its world's last time step (ours: a world spans 10 steps, the
  real window 29, so ``c_steps`` cannot match in range);
- the per-step tx count (a count: ``log1p`` with the D5 range rule, 50 bins);
- the wallet-tx link-degree distribution (distinct linked txs per wallet,
  txs in ts 1-29 only), ``log1p`` with the D5 range rule.

Configuration 0 is the ``WorldParams`` default. The best configuration is
frozen as the multiverse centre (``configs/sim_centre.yaml``). The label
rates are then solved from the public README totals on that centre
(``hns.sim.labels``), and the shortcut guard is checked. Lifetime wallet
features are never calibrated against (§5, T isolation).

Writes ``configs/sim_centre.yaml``, ``reports/m4_calibration.md`` and
``runs/calib/results.jsonl``.

Usage: ``uv run python -m hns.sim.calibrate [configs/sim.yaml]``
"""

from __future__ import annotations

import json
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
import polars as pl
import yaml

from hns.data.causal import COUNTS
from hns.data.graph import GRAPH_DIR
from hns.data.load import AUG_FEATURES, PARQUET_DIR
from hns.data.norm import NORM_DIR, T_FIT_MAX_TS
from hns.eval import js as jsd
from hns.eval.rf_anchor import git_sha
from hns.feats import manifest
from hns.sim import labels
from hns.sim.env import run_static_world
from hns.sim.export import export_world
from hns.sim.multiverse import draw_world
from hns.sim.world import CALIBRATED, ROLE_SERVICE, WorldParams

CENTRE = Path("configs/sim_centre.yaml")
REPORT = Path("reports/m4_calibration.md")
RUNS = Path("runs/calib")


# ---------------------------------------------------------------------------- real side
def real_targets(parquet_dir: Path = PARQUET_DIR, graph_dir: Path = GRAPH_DIR, norm_dir: Path = NORM_DIR) -> dict:
    """Bins and real histograms per term. Reads no class column."""
    norm = json.loads((norm_dir / "T.json").read_text())
    txs = pl.read_parquet(parquet_dir / "txs.parquet", columns=["tid", "ts", *AUG_FEATURES])
    txs = txs.filter(pl.col("ts") <= T_FIT_MAX_TS)
    bins, real, anchors = {}, {}, {}
    nn = txs.filter(pl.col("size").is_not_null())
    for f in AUG_FEATURES:
        s = norm["tx"][f]
        bins[f"tx:{f}"] = (s["transform"], s["js_lo"], s["js_hi"], s["js_bins"])
    cw = pl.read_parquet(graph_dir / "causal_wallet.parquet").filter(pl.col("ts") <= T_FIT_MAX_TS)
    last = cw.sort("wid", "ts").group_by("wid").last()
    for c in COUNTS:
        s = norm["wallet_T"][c]
        bins[f"wallet_T:{c}"] = (s["transform"], s["js_lo"], s["js_hi"], s["js_bins"])
    per_step = txs.group_by("ts").len()["len"].to_numpy()
    bins["steps:n_tx"] = ("log1p", *jsd.range_rule(np.log1p(per_step)), 50)
    links = pl.concat([pl.read_parquet(parquet_dir / f, columns=["wid", "tid"]) for f in ("addr_tx.parquet", "tx_addr.parquet")])
    deg = links.join(txs.select("tid"), on="tid").unique().group_by("wid").len()["len"].to_numpy()
    bins["links:degree"] = ("log1p", *jsd.range_rule(np.log1p(deg)), 50)

    values = {f"tx:{f}": nn[f].to_numpy() for f in AUG_FEATURES}
    values |= {f"wallet_T:{c}": last[c].to_numpy() for c in COUNTS}
    values |= {"steps:n_tx": per_step, "links:degree": deg}
    for k, v in values.items():
        real[k] = hist(v, bins[k])
    anchors = {"per_step": [int(per_step.min()), float(np.median(per_step)), int(per_step.max())],
               "link_degree": [float(np.median(deg)), float(np.percentile(deg, 99)), int(deg.max())],
               "n_tx": txs.height, "n_wallets_t29": last.height}
    return {"bins": bins, "real": real, "anchors": anchors}


def hist(v: np.ndarray, b: tuple) -> np.ndarray:
    transform, lo, hi, n = b
    return jsd.histogram(manifest.apply(transform, v), lo, hi, n)


# ---------------------------------------------------------------------------- sim side
def sim_values(ex) -> dict[str, np.ndarray]:
    t = ex.txs
    last = ex.causal.sort("wid", "ts").group_by("wid").last()
    deg = (pl.concat([ex.addr_tx.select("wid", "tid"), ex.tx_addr.select("wid", "tid")]).unique()
           .group_by("wid").len()["len"].to_numpy())
    v = {f"tx:{f}": t[f].to_numpy() for f in AUG_FEATURES}
    v |= {f"wallet_T:{c}": last[c].to_numpy() for c in COUNTS}
    v |= {"steps:n_tx": ex.steps["n_retained"].to_numpy(), "links:degree": deg}
    return v


def world_summary(ex) -> dict:
    t = ex.txs
    return {"n_retained": ex.steps["n_retained"].to_list(), "n_live": ex.steps["n_live"].to_list(),
            "illicit_share": float(t["truth"].mean()), "n_illicit": int(t["truth"].sum()),
            "n_service": int((t["role"] == ROLE_SERVICE).sum()), "n_tx": t.height,
            "n_wallets": ex.wallets.height}


def evaluate(params: WorldParams, seeds: list[int], bins: dict) -> dict:
    """Pooled histograms over ``seeds`` worlds, plus per-world summaries."""
    h = {k: np.zeros(b[3]) for k, b in bins.items()}
    worlds = []
    for s in seeds:
        ex = export_world(run_static_world(params, s))
        for k, v in sim_values(ex).items():
            h[k] += hist(v, bins[k])
        worlds.append(world_summary(ex))
    return {"hist": h, "worlds": worlds}


def score(h: dict, real: dict) -> dict[str, float]:
    return {k: jsd.js(h[k], real[k]) for k in real}


def _job(args):
    i, params, seeds, bins = args
    t0 = time.time()
    r = evaluate(params, seeds, bins)
    r["i"], r["sec"] = i, time.time() - t0
    return r


def sample_config(base: WorldParams, rng: np.random.Generator) -> WorldParams:
    upd = {}
    for name, (lo, hi, is_int) in CALIBRATED.items():
        upd[name] = int(rng.integers(lo, hi + 1)) if is_int else float(rng.uniform(lo, hi))
    return replace(base, **upd)


# ---------------------------------------------------------------------------- driver
def run(cfg_path: Path = Path("configs/sim.yaml")) -> None:
    cfg = yaml.safe_load(cfg_path.read_text())
    c = cfg["calibration"]
    base = WorldParams.from_dict(cfg["fixed"])
    tgt = real_targets()
    bins, real = tgt["bins"], tgt["real"]
    rng = np.random.default_rng(c["seed"])
    configs = [base] + [sample_config(base, rng) for _ in range(c["n_configs"] - 1)]
    for i in range(len(configs)):
        configs[i] = replace(configs[i], offset=int(rng.integers(0, 40)))
    seeds = [[c["seed"] * 100_000 + i * 100 + w for w in range(c["worlds"])] for i in range(len(configs))]
    t0 = time.time()
    # spawn, not fork: a forked child of a process running polars' thread pool can deadlock
    os.environ.setdefault("POLARS_MAX_THREADS", "2")
    with ProcessPoolExecutor(c["workers"], mp_context=multiprocessing.get_context("spawn")) as pool:
        res = sorted(pool.map(_job, [(i, p, seeds[i], bins) for i, p in enumerate(configs)]), key=lambda r: r["i"])
    for r in res:
        r["js"] = score(r["hist"], real)
        r["mean_js"] = float(np.mean(list(r["js"].values())))
    search_sec = time.time() - t0
    RUNS.mkdir(parents=True, exist_ok=True)
    with open(RUNS / "results.jsonl", "w") as f:
        for r, p in zip(res, configs):
            f.write(json.dumps({"i": r["i"], "mean_js": r["mean_js"], "js": r["js"], "params": p.to_dict(),
                                "worlds": r["worlds"], "sec": r["sec"]}) + "\n")
    best = min(res, key=lambda r: r["mean_js"])
    centre = replace(configs[best["i"]], offset=base.offset)

    # label rates from README totals, on the centre (labels do not change static dynamics)
    lab = c["label_worlds"]
    lseeds = [c["seed"] * 100_000 + 90_000 + w for w in range(lab)]
    exs = [export_world(run_static_world(replace(centre, offset=int(o)), s))
           for s, o in zip(lseeds, np.random.default_rng(c["seed"] + 1).integers(0, 40, lab))]
    n = sum(e.txs.height for e in exs)
    n_ill = sum(int(e.txs["truth"].sum()) for e in exs)
    n_svc = sum(int((e.txs["role"] == ROLE_SERVICE).sum()) for e in exs)
    tx_rates = labels.solve_tx_rates(n, n_ill, n_svc, n - n_ill - n_svc)
    centre = replace(centre, p_ill=tx_rates["p_ill"], p_lic=tx_rates["p_lic"], p_wunk=0.0)
    exs = [export_world(run_static_world(replace(centre, offset=int(o)), s))
           for s, o in zip(lseeds, np.random.default_rng(c["seed"] + 1).integers(0, 40, lab))]
    nw = sum(e.wallets.height for e in exs)
    nw_ill = sum(int((e.wallets["class"] == labels.ILLICIT).sum()) for e in exs)
    nw_rule = sum(int(e.wallets["licit_rule"].sum()) for e in exs)
    w_rates = labels.solve_p_wunk(nw, nw_ill, nw_rule)
    centre = replace(centre, p_wunk=w_rates["p_wunk"])
    exs = [export_world(run_static_world(replace(centre, offset=int(o)), s))
           for s, o in zip(lseeds, np.random.default_rng(c["seed"] + 1).integers(0, 40, lab))]
    achieved = achieved_rates(exs)
    guard = [labels.shortcut_auc(e.txs, seed=i) for i, e in enumerate(exs)]

    # centre JS on fresh seeds, and multiverse draws (§16: look at them before M5)
    cres = evaluate(centre, [c["seed"] * 100_000 + 95_000 + w for w in range(lab)], bins)
    cjs = score(cres["hist"], real)
    mrng = np.random.default_rng(c["seed"] + 2)
    mv = []
    for w in range(c["multiverse_draws"]):
        p = draw_world(centre, mrng)
        ex = export_world(run_static_world(p, c["seed"] * 100_000 + 97_000 + w))
        mv.append({"params": p, **world_summary(ex), **achieved_rates([ex])})

    CENTRE.write_text("# Multiverse centre, frozen by `python -m hns.sim.calibrate` (SPEC §9.6). Do not hand-edit.\n"
                      f"# code: {git_sha()}\n" + yaml.safe_dump(centre.to_dict(), sort_keys=False))
    REPORT.write_text(report(cfg, tgt, res, configs, best, centre, tx_rates, w_rates, achieved, guard, cjs, cres,
                             mv, search_sec))
    print(f"M4: best mean JS {best['mean_js']:.3f} (config {best['i']}), centre JS {np.mean(list(cjs.values())):.3f}; "
          f"wrote {CENTRE}, {REPORT}")
    if max(guard) > labels.GUARD_MAX_AUC:
        raise SystemExit(f"shortcut guard failed: AUC {max(guard):.3f} > {labels.GUARD_MAX_AUC}")


def achieved_rates(exs) -> dict:
    n = sum(e.txs.height for e in exs)
    t_lab = sum(int((e.txs["class"] != labels.UNKNOWN).sum()) for e in exs)
    t_ill = sum(int((e.txs["class"] == labels.ILLICIT).sum()) for e in exs)
    nw = sum(e.wallets.height for e in exs)
    w_lab = sum(int((e.wallets["class"] != labels.UNKNOWN).sum()) for e in exs)
    w_ill = sum(int((e.wallets["class"] == labels.ILLICIT).sum()) for e in exs)
    return {"tx_labelled": t_lab / n, "tx_illicit_share": t_ill / max(t_lab, 1),
            "w_labelled": w_lab / nw, "w_illicit_share": w_ill / max(w_lab, 1)}


# ---------------------------------------------------------------------------- report
def _f(x, d=3):
    return f"{x:.{d}f}"


def report(cfg, tgt, res, configs, best, centre, tx_rates, w_rates, achieved, guard, cjs, cres, mv, search_sec) -> str:
    c = cfg["calibration"]
    order = sorted(res, key=lambda r: r["mean_js"])
    a = tgt["anchors"]
    status = {f"tx:{k}": v["status"] for k, v in manifest.load()["tx"].items()}
    status |= {f"wallet_T:{k}": v["status"] for k, v in manifest.load()["wallet_T"].items()}
    L = [
        "# M4: simulator calibration (static mode)", "",
        "SPEC §9.6, §15 M4. Label-blind random search over the §9.7 bounds against T-safe real data "
        "(ts 1–29, pooled over classes and unknown; no class column is read). Objective: mean JS (base 2) over "
        f"{len(tgt['real'])} terms. Lifetime wallet features are not calibrated against (T isolation).", "",
        f"Code: `{git_sha()}`. Generated by `uv run python -m hns.sim.calibrate`. {c['n_configs']} configurations × "
        f"{c['worlds']} worlds each (config 0 = defaults), {c['workers']} workers, {search_sec / 60:.1f} min. "
        "Fixed during the search: " + ", ".join(f"`{k}` = {v}" for k, v in cfg["fixed"].items()) + ".", "",
        "## Real anchors (ts 1–29)", "",
        f"- txs: {a['n_tx']:,}; per-step tx count min / median / max: {a['per_step'][0]:,} / {a['per_step'][1]:,.0f} / "
        f"{a['per_step'][2]:,}",
        f"- wallet–tx link degree (distinct linked txs) median / p99 / max: {a['link_degree'][0]:.0f} / "
        f"{a['link_degree'][1]:.0f} / {a['link_degree'][2]:,}",
        f"- wallets with a link by ts 29: {a['n_wallets_t29']:,}", "",
        "## Search", "",
        f"Best: config {best['i']}, mean JS {_f(best['mean_js'])}. Default (config 0): {_f(res[0]['mean_js'])}. "
        f"Median over configs: {_f(float(np.median([r['mean_js'] for r in res])))}.", "",
        "| Rank | Config | Mean JS | Median retained txs / step | Illicit share (truth) |", "|---|---|---|---|---|",
    ]
    for k, r in enumerate(order[:5]):
        steps = [x for w in r["worlds"] for x in w["n_retained"]]
        ill = np.mean([w["illicit_share"] for w in r["worlds"]])
        L.append(f"| {k + 1} | {r['i']} | {_f(r['mean_js'])} | {np.median(steps):,.0f} | {_f(ill)} |")
    L += ["", "## Centre (frozen in `configs/sim_centre.yaml`)", "", "| Parameter | Bounds | Value |", "|---|---|---|"]
    for name, (lo, hi, is_int) in CALIBRATED.items():
        v = getattr(centre, name)
        L.append(f"| `{name}` | {lo}–{hi} | {v if is_int else _f(v)} |")
    L += [f"| `p_ill` / `p_lic` / `p_wunk` | solved (§9.4) | {_f(centre.p_ill)} / {_f(centre.p_lic)} / "
          f"{_f(centre.p_wunk)} |", ""]
    L += [f"## Per-term JS at the centre ({c['label_worlds']} fresh worlds)", "",
          f"Mean JS {_f(float(np.mean(list(cjs.values()))))}. Assumed definitions (§7) are marked; they cannot be "
          "checked against recomputation on the real data.", "",
          "| Term | Status | JS |", "|---|---|---|"]
    for k, v in sorted(cjs.items(), key=lambda kv: -kv[1]):
        L.append(f"| `{k}` | {status.get(k, 'derived')} | {_f(v)} |")
    steps = [x for w in cres["worlds"] for x in w["n_retained"]]
    live = [x for w in cres["worlds"] for x in w["n_live"]]
    L += ["", f"Centre retained txs per step: min / median / max {min(steps):,} / {np.median(steps):,.0f} / "
          f"{max(steps):,} (live, before pruning: median {np.median(live):,.0f}).", ""]
    L += ["## Observed labels (§9.4, README totals)", "",
          "| Quantity | Target | Achieved |", "|---|---|---|",
          f"| tx labelled fraction | {_f(labels.TX_LABELLED)} | {_f(achieved['tx_labelled'])} |",
          f"| tx illicit share of labelled | {_f(labels.TX_ILLICIT_SHARE)} | {_f(achieved['tx_illicit_share'])} |",
          f"| wallet labelled fraction | {_f(labels.W_LABELLED)} | {_f(achieved['w_labelled'])} |",
          f"| wallet illicit share of labelled | {_f(labels.W_ILLICIT_SHARE)} (reported, not fitted) | "
          f"{_f(achieved['w_illicit_share'])} |", "",
          f"Solved: `p_ill` = {_f(tx_rates['p_ill'])} (unclipped {_f(tx_rates['p_ill_unclipped'])}), "
          f"`p_lic` = {_f(tx_rates['p_lic'])} (unclipped {_f(tx_rates['p_lic_unclipped'])}), "
          f"`p_wunk` = {_f(w_rates['p_wunk'])} (unclipped {_f(w_rates['p_wunk_unclipped'])}; clip [0, 0.5]). "
          f"Tx labelled-fraction excess after clipping: {_f(tx_rates['labelled_frac'] - labels.TX_LABELLED)} "
          "(service-initiated txs are always labelled licit, §9.4, so their share is a floor on the licit labels). "
          f"Wallet labelled-fraction shortfall after clipping: {_f(w_rates['shortfall'])}.", "",
          f"Shortcut guard (labelled vs unknown among innocent licit txs, held-out ROC-AUC of a logistic regression on "
          f"the 17 shared features): {', '.join(_f(g) for g in guard)}; limit {labels.GUARD_MAX_AUC}. "
          f"{'Pass' if max(guard) <= labels.GUARD_MAX_AUC else 'FAIL'}.", ""]
    L += ["## Multiverse draws (§9.5, §16)", "",
          "| Draw | n_L | b | Offset | Innocents | Services | Retained txs / step (min–max) | Illicit share | "
          "tx labelled | wallet labelled |", "|---|---|---|---|---|---|---|---|---|---|"]
    for w, m in enumerate(mv):
        p = m["params"]
        L.append(f"| {w} | {p.n_launderers} | {p.budget:.2f} | {p.offset} | {p.n_innocents:,} | {p.n_services} | "
                 f"{min(m['n_retained']):,}–{max(m['n_retained']):,} | {_f(m['illicit_share'])} | "
                 f"{_f(m['tx_labelled'])} | {_f(m['w_labelled'])} |")
    L += ["", "## Notes (ours)", "",
          "- Static mode: launderers draw every branch uniformly over its feasible options; innocents pay now, "
          "directly, in one output; both send change to a fresh address with probability `p_fresh`. The detector "
          "flags uniformly at random at the world's budget.",
          "- Launderers receive new proceeds at the start of every episode, so laundering spans the world.",
          "- Services queue a withdrawal to an innocent for every completed innocent payment to a service, so money "
          "keeps circulating; payouts wait at most `payout_max_wait` sub-steps for a full batch.",
          "- The simulator's causal counts are each wallet's state at its world's last step; the real ones are at "
          "ts 29. A world spans 10 steps, so `c_steps` cannot reach the real range.",
          "- `num_input_addresses` / `num_output_addresses` count inputs and outputs (spent and created outputs), "
          "which is never below the distinct-address link count, as audited on the real data.", ""]
    return "\n".join(L)


if __name__ == "__main__":
    run(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("configs/sim.yaml"))
