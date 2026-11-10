"""M5 report: acceptance checks and health curves of the MARL runs (SPEC §10, §15).

Reads ``runs/marl/<arm>/<seed>/{metrics.jsonl, result.json, config.yaml}`` and
writes ``reports/m5_marl.md`` with ``reports/figs/m5_*.png``. The health
curves replace the paper's Figs. 4-5: one panel per measure (one y-axis
each), one colour per arm, the seed mean as the 2 px line and the seeds as
thin lines behind it. Every plotted series is also summarized in a table.

Usage: ``uv run python -m hns.marl.report [runs/marl]``
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import yaml

from hns.eval.rf_anchor import git_sha
from hns.marl.train import DEGENERATE_HI, DEGENERATE_LO, DEGENERATE_WINDOW, RUNS, degenerate, read_records
from hns.sim.entities import STRATEGIES

REPORT = Path("reports/m5_marl.md")
FIGS = Path("reports/figs")
ARMS = {"full": "Full", "nda": "NDA"}
SPLIT = "## Decisions"   # reports/m5_notes.md: what precedes this heading is placed right after the acceptance table
# categorical slots 1-2 of the validated palette, in fixed order (colour follows the arm); light surface
COLOR = {"full": "#2a78d6", "nda": "#eb6834"}
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"


def load(root: Path) -> dict:
    """``{arm: {seed: {"epochs": [...], "episodes": [...], "result": dict | None, "config": dict}}}``.

    A run that is still going (or was killed) is loaded too: it may have no epoch yet, and its last metrics line
    may be incomplete.
    """
    runs: dict = {}
    for arm in ARMS:
        for d in sorted((root / arm).glob("*")) if (root / arm).exists() else []:
            m = d / "metrics.jsonl"
            if not m.exists():
                continue
            recs = [r for _, r in read_records(m)]
            res = d / "result.json"
            runs.setdefault(arm, {})[int(d.name)] = {
                "epochs": [r for r in recs if r["kind"] == "epoch"],
                "episodes": [r for r in recs if r["kind"] == "episode"],
                "result": json.loads(res.read_text()) if res.exists() else None,
                "config": yaml.safe_load((d / "config.yaml").read_text()),
            }
    return runs


def series(run: dict, get) -> np.ndarray:
    return np.array([get(e) for e in run["epochs"]], dtype=float)


PANELS = [
    ("Detection rate", "share of illicit cash-out txs flagged", lambda e: e["detection_rate"], True),
    ("Launderer success fraction", "BTC cashed out undetected / proceeds", lambda e: e["success_frac"], True),
    ("Illicit / licit tx ratio", "executed txs per episode, epoch mean", lambda e: e["illicit_licit_ratio"], False),
    ("Launderer reward", "mean episode reward per launderer", lambda e: e["reward"]["launderer"], False),
    ("Innocent reward", "mean episode reward per innocent", lambda e: e["reward"]["innocent"], False),
    ("Detector reward", "mean episode reward", lambda e: e["reward"]["detector"], False),
]
LOSS_PANELS = [
    ("detector-T: L_y src", "t_y_src"), ("detector-T: L_adv", "t_adv"), ("detector-T: TD, per-tx flags", "t_td_flag"),
    ("detector-W: L_y src", "w_y_src"), ("detector-W: L_adv", "w_adv"), ("launderer: TD (log scale)", "td_launderer"),
    ("detector-T: L_rec (Full)", "t_rec"), ("detector-T: L_dom (Full)", "t_dom"), ("innocent: TD", "td_innocent"),
    ("detector-T: TD, global branches", "t_td_global"),
]


def _style(ax, title: str, sub: str = "") -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=10, color=INK, fontweight="bold", pad=14 if sub else 6)
    if sub:
        ax.text(0, 1.02, sub, transform=ax.transAxes, fontsize=8, color=INK2, va="bottom")
    ax.grid(True, axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(which="both", colors=MUTED, labelsize=8, length=0)


def _lines(ax, runs: dict, get) -> None:
    for arm, by_seed in runs.items():
        ys = [series(r, get) for r in by_seed.values() if r["epochs"]]
        if not ys:
            continue
        n = min(len(y) for y in ys)
        x = np.arange(1, n + 1)
        for y in ys:
            ax.plot(np.arange(1, len(y) + 1), y, color=COLOR[arm], linewidth=0.8, alpha=0.35, solid_capstyle="round")
        ax.plot(x, np.mean([y[:n] for y in ys], axis=0), color=COLOR[arm], linewidth=2, solid_capstyle="round",
                solid_joinstyle="round", label=ARMS[arm])


def _figure(runs: dict, panels: list, path: Path, ncol: int = 3, ylim: tuple | None = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nrow = -(-len(panels) // ncol)
    banded = False
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.1 * ncol, 2.9 * nrow), facecolor=SURFACE, squeeze=False)
    for ax, p in zip(axes.flat, panels):
        title, sub, get = p[0], p[1], p[2]
        _style(ax, title, sub)
        if len(p) > 3 and p[3]:            # the degenerate bands of §10
            ax.axhspan(DEGENERATE_HI, 1.0, color=GRID, alpha=0.6, linewidth=0)
            ax.axhspan(0.0, DEGENERATE_LO, color=GRID, alpha=0.6, linewidth=0)
            ax.set_ylim(0, 1)
            banded = True
        if len(p) > 4 and p[4]:
            ax.set_yscale("log")
        _lines(ax, runs, get)
        if ylim is not None:               # small multiples of one measure share the scale
            ax.set_ylim(*ylim)
    for ax in axes.flat[len(panels):]:
        ax.set_visible(False)
    for c in range(ncol):                  # the lowest visible panel of every column carries the x label
        col = [axes[r, c] for r in range(nrow) if r * ncol + c < len(panels)]
        if col:
            col[-1].set_xlabel("epoch (world of 10 episodes)", fontsize=8, color=INK2)
    h, lab = axes.flat[0].get_legend_handles_labels()
    if len(h) > 1:                         # one colour needs no legend box: the title names it
        fig.legend(h, lab, loc="upper right", ncol=len(h), frameon=False, fontsize=9, labelcolor=INK2,
                   bbox_to_anchor=(0.995, 1.0))
    note = "Thick line: mean over seeds. Thin lines: seeds."
    if banded:
        note += "  Shaded: the §10 bands (degenerate if a series stays above 0.95 or below 0.05 for 20 epochs)."
    fig.text(0.008, 0.995, note, fontsize=8, color=INK2, va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def _ms(xs) -> str:
    xs = [x for x in xs if x is not None]
    return f"{np.mean(xs):.3f} ± {np.std(xs):.3f}" if xs else "—"


def _tail(run: dict, get, k: int = DEGENERATE_WINDOW):
    y = series(run, get)
    return float(np.mean(y[-k:])) if len(y) else None


def write(root: Path = RUNS, path: Path = REPORT, figs: Path = FIGS, notes: str = "") -> bool:
    runs = load(root)
    if not runs:
        raise SystemExit(f"no MARL runs under {root}")
    _figure(runs, PANELS, figs / "m5_health.png")
    _figure(runs, [(s, "share of started tranches", (lambda e, s=s: e["strategy_share"][s])) for s in STRATEGIES],
            figs / "m5_strategy.png", ylim=(0, 1))
    _figure(runs, [(t, "epoch mean", (lambda e, k=k: e["losses"].get(k, np.nan)), False, k == "td_launderer")
                   for t, k in LOSS_PANELS], figs / "m5_losses.png")
    cfg = next(iter(next(iter(runs.values())).values()))["config"]
    target = cfg["episodes"]
    L = [
        "# M5: MARL pretraining (Full and NDA)", "",
        "SPEC §10, §15 M5. Launderers, innocents and the inspector (detector-T) co-evolve in the SISTS-lite "
        "multiverse; detector-W trains on each finished world's export. Full adds in-loop `L_aux` on unlabelled real "
        "fitting-split views; NDA has no `L_aux` and reads no real batch. No real label is read in either arm.", "",
        f"Report code: `{git_sha()}`. Generated by `uv run python -m hns.marl.report`. Run code and device are in each "
        "run's `config.yaml`.", "",
        "## Acceptance (§15)", "",
        f"Health curves present; no NaNs (a run stops on any non-finite epoch loss); {target} episodes complete; "
        f"non-degenerate (§10: over the last {DEGENERATE_WINDOW} epochs, neither the detection rate nor the launderer "
        f"success fraction stays above {DEGENERATE_HI} or below {DEGENERATE_LO}).", "",
        "| Arm | Seed | Episodes | Finite losses | Detection rate, last 20 epochs (min–max) | "
        "Success fraction, last 20 epochs (min–max) | Epochs of the last 20 with success < 0.05 | Degenerate (§10 rule) | "
        "Wall time | Pass |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    ok_all = True
    for arm, by_seed in runs.items():
        for seed, r in sorted(by_seed.items()):
            ep = r["epochs"]
            n_ep = len(r["episodes"])
            if not ep:                         # started, no world finished yet: nothing to summarize
                ok_all = False
                L.append(f"| {ARMS[arm]} | {seed} | {n_ep} / {target} | — | — | — | — | no epoch completed yet | — | "
                         "**no** |")
                continue
            finite = all(np.isfinite(list(e["losses"].values())).all() for e in ep)
            deg = degenerate(ep) if len(ep) >= DEGENERATE_WINDOW else None
            det, suc = series(r, lambda e: e["detection_rate"])[-20:], series(r, lambda e: e["success_frac"])[-20:]
            done = r["result"] is not None and n_ep == target
            ok = bool(done and finite and deg == [])
            ok_all &= ok
            hours = sum(e["sec"]["world"] for e in ep) / 3600
            L.append(f"| {ARMS[arm]} | {seed} | {n_ep} / {target} | {'yes' if finite else '**no**'} | "
                     f"{det.min():.2f}–{det.max():.2f} | {suc.min():.2f}–{suc.max():.2f} | {int((suc < DEGENERATE_LO).sum())} | "
                     f"{'not decided (< 20 epochs)' if deg is None else ('no' if not deg else '**' + '; '.join(deg) + '**')} | "
                     f"{hours:.1f} h | {'yes' if ok else '**no**'} |")
    L += ["", "The §10 rule only calls a run degenerate if a series stays in a band for all 20 epochs. Read the pass "
          "together with the findings below."]
    head, sep, tail = notes.partition(SPLIT)     # findings go under the acceptance table, the rest at the end
    if head.strip():
        L += ["", head.rstrip()]
    L += ["", "## Health curves (§10; they replace Figs. 4–5)", "", "![Health curves](figs/m5_health.png)", "",
          "Mean over the last 20 epochs, then mean ± sd over seeds:", "",
          "| Arm | Detection rate | Success fraction | Frozen fraction | Illicit / licit ratio | Launderer reward | "
          "Innocent reward | Detector reward | Budget `b` | Depth-2 share | Flag precision |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    cols = [lambda e: e["detection_rate"], lambda e: e["success_frac"], lambda e: e["frozen_frac"],
            lambda e: e["illicit_licit_ratio"], lambda e: e["reward"]["launderer"], lambda e: e["reward"]["innocent"],
            lambda e: e["reward"]["detector"], lambda e: e["budget_mean"], lambda e: e["depth2_share"],
            lambda e: e["flag_precision"]]
    for arm, by_seed in runs.items():
        L.append(f"| {ARMS[arm]} | " + " | ".join(_ms([_tail(r, c) for r in by_seed.values()]) for c in cols) + " |")
    L += ["", "First 10 epochs (epsilon ≈ 1: near-random play), for comparison:", "",
          "| Arm | Detection rate | Success fraction | Illicit / licit ratio | Launderer reward | Detector reward |",
          "|---|---|---|---|---|---|"]
    for arm, by_seed in runs.items():
        first = lambda c: _ms([float(np.mean(series(r, c)[:10])) for r in by_seed.values() if r["epochs"]])  # noqa: E731
        L.append(f"| {ARMS[arm]} | " + " | ".join(first(cols[i]) for i in (0, 1, 3, 4, 6)) + " |")
    L += ["", "## Strategy usage (share of started tranches)", "", "![Strategy usage](figs/m5_strategy.png)", "",
          "Mean over the last 20 epochs, then mean ± sd over seeds:", "",
          "| Arm | " + " | ".join(f"`{s}`" for s in STRATEGIES) + " |", "|---|" + "---|" * len(STRATEGIES)]
    for arm, by_seed in runs.items():
        L.append(f"| {ARMS[arm]} | " + " | ".join(
            _ms([_tail(r, lambda e, s=s: e["strategy_share"][s]) for r in by_seed.values()]) for s in STRATEGIES) + " |")
    L += ["", "## Losses", "", "![Losses](figs/m5_losses.png)", "",
          "Mean over the last 20 epochs, then mean ± sd over seeds:", "",
          "| Arm | " + " | ".join(f"`{k}`" for _, k in LOSS_PANELS) + " |", "|---|" + "---|" * len(LOSS_PANELS)]
    for arm, by_seed in runs.items():
        L.append(f"| {ARMS[arm]} | " + " | ".join(
            _ms([_tail(r, lambda e, k=k: e["losses"][k]) for r in by_seed.values()
                 if r["epochs"] and k in r["epochs"][-1]["losses"]])
            for _, k in LOSS_PANELS) + " |")
    if sep:
        L += ["", (sep + tail).strip()]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L) + "\n")
    return ok_all


if __name__ == "__main__":
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else RUNS
    notes = Path("reports/m5_notes.md")
    ok = write(root, notes=notes.read_text() if notes.exists() else "")
    print(f"M5: wrote {REPORT}; acceptance {'passed' if ok else 'NOT passed'}")
    sys.exit(0 if ok else 1)
