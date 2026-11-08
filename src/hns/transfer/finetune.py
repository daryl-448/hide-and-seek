"""Fine-tune one detector on one task (SPEC §6 training protocol, §11). Only the target-only arm, T0.

T0: random init, ``L = L_y^tgt + λ L_adv`` on the label fraction ``f`` of the
task's fitting split. Adam, batch 128 targets (split into micro-batches with
gradient accumulation), gradient clip 1.0, at most 50 epochs, early stopping
on val illicit-F1 (at the val-optimal threshold) with patience 5, restoring
the best epoch. BCE with ``pos_weight = sqrt(n_neg / n_pos)`` of the training
targets. Test is never scored here.

Each epoch draws fresh views (seed from the run seed and epoch); val views use
one fixed seed. The run resumes from ``ckpt/last.pt`` if it exists with the
same config.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
import yaml

from hns.eval.metrics import best_f1_threshold, binary_metrics
from hns.models.detector import Detector, Graph
from hns.models.hrg import TaskData
from hns.transfer.adv import adv_loss

EVAL_SEED = 0x5EED


@dataclass
class RunConfig:
    task: str
    seed: int
    fraction: float
    lam: float
    arm: str = "t0"
    lr: float = 1e-3
    batch: int = 128
    micro_batch: int = 128
    eval_batch: int = 128
    max_epochs: int = 50
    patience: int = 5
    clip: float = 1.0
    eps: float = 0.1
    p_miss: float = 0.1
    dropout: float = 0.5

    def run_dir(self, root: Path) -> Path:
        return root / self.task / f"{self.arm}_f{self.fraction}_lam{self.lam}" / str(self.seed)


def _tensors(feats: dict[str, np.ndarray], device) -> dict[str, torch.Tensor]:
    return {k: torch.from_numpy(v).to(device) for k, v in feats.items()}


@torch.no_grad()
def predict(model: Detector, data: TaskData, tg: dict, device, batch: int) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(tg["g"]), batch):
        v, feats = data.batch(tg["g"][i:i + batch], tg["t"][i:i + batch], EVAL_SEED)
        out.append(torch.sigmoid(model(Graph(v, data.n_tx, device), **_tensors(feats, device))).cpu().numpy())
    return np.concatenate(out)


def val_metrics(y: np.ndarray, score: np.ndarray) -> dict:
    return binary_metrics(y, score, threshold=best_f1_threshold(y, score))


def train(cfg: RunConfig, data: TaskData, root: Path, device, sha: str = "unknown", log=print) -> dict:
    """Train one run; returns the best epoch's val metrics. Writes ``config.yaml``, ``metrics.jsonl``,
    ``ckpt/{last,best}.pt`` and ``preds_val.parquet`` under ``cfg.run_dir(root)``."""
    if cfg.arm != "t0":
        raise NotImplementedError(cfg.arm)
    out = cfg.run_dir(root)
    done = out / "result.json"
    if done.exists():
        res = json.loads(done.read_text())
        if res["config"] == asdict(cfg):
            log(f"{out}: done, reusing")
            return res["val"]
    (out / "ckpt").mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed)
    tr, va = data.targets("fit", cfg.fraction), data.targets("val")
    n_pos = float(tr["y"].sum())
    pos_weight = torch.tensor(math.sqrt((len(tr["y"]) - n_pos) / n_pos), device=device)
    model = Detector(data.d_tx, data.d_w, data.d_anon, dropout=cfg.dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    state = {"epoch": 0, "best_f1": -1.0, "best_epoch": -1, "bad": 0, "best_state": None}
    last = out / "ckpt" / "last.pt"
    if last.exists():
        ck = torch.load(last, map_location="cpu", weights_only=False)  # RNG states must stay on CPU
        if ck["config"] != asdict(cfg):
            raise ValueError(f"{last} was written by a different config")
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); state = ck["state"]
        torch.set_rng_state(ck["rng_cpu"])
        if ck["rng_cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state(ck["rng_cuda"])
        log(f"{out}: resuming after epoch {state['epoch']}")
    else:
        (out / "config.yaml").write_text(yaml.safe_dump(
            {**asdict(cfg), "git_sha": sha, "n_train": len(tr["y"]), "n_train_illicit": int(n_pos),
             "n_val": len(va["y"]), "pos_weight": float(pos_weight)}, sort_keys=False))
        (out / "metrics.jsonl").write_text("")

    while state["epoch"] < cfg.max_epochs and state["bad"] < cfg.patience:
        ep = state["epoch"]
        model.train()
        perm = np.random.default_rng([cfg.seed, ep]).permutation(len(tr["y"]))
        view_seed = (cfg.seed << 20) + ep + 1
        tot, n_b = 0.0, 0
        for b in range(0, len(perm), cfg.batch):
            idx = perm[b:b + cfg.batch]
            opt.zero_grad(set_to_none=True)
            for m in range(0, len(idx), cfg.micro_batch):
                mi = idx[m:m + cfg.micro_batch]
                v, feats = data.batch(tr["g"][mi], tr["t"][mi], view_seed)
                g = Graph(v, data.n_tx, device)
                y = torch.from_numpy(tr["y"][mi]).to(device)

                def nll(f, g=g, y=y):
                    return F.binary_cross_entropy_with_logits(model(g, **f), y, pos_weight=pos_weight,
                                                              reduction="sum") / len(idx)

                x = _tensors(feats, device)
                loss = nll(x) + cfg.lam * adv_loss(nll, x, cfg.eps, cfg.p_miss)
                loss.backward()
                tot += loss.detach().item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip)
            opt.step()
            n_b += 1
        if not math.isfinite(tot):
            raise FloatingPointError(f"{out}: non-finite loss in epoch {ep}")
        score = predict(model, data, va, device, cfg.eval_batch)
        m = val_metrics(va["y"], score)
        improved = m["illicit_f1"] > state["best_f1"]
        if improved:
            state.update(best_f1=m["illicit_f1"], best_epoch=ep, bad=0,
                         best_state={k: t.detach().cpu().clone() for k, t in model.state_dict().items()})
            pl.DataFrame({"g": va["g"], "y": va["y"], "score": score}).write_parquet(out / "preds_val.parquet")
        else:
            state["bad"] += 1
        state["epoch"] = ep + 1
        rec = {"epoch": ep, "train_loss": tot / max(n_b, 1), "split": "val", **m}
        with open(out / "metrics.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        log(f"{cfg.task} f={cfg.fraction} λ={cfg.lam} seed {cfg.seed} epoch {ep}: loss {rec['train_loss']:.4f} "
            f"val F1 {m['illicit_f1']:.3f} PR-AUC {m['pr_auc']:.3f}{' *' if improved else ''}")
        torch.save({"config": asdict(cfg), "model": model.state_dict(), "opt": opt.state_dict(), "state": state,
                    "rng_cpu": torch.get_rng_state(),
                    "rng_cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None}, last)

    torch.save({"config": asdict(cfg), "model": state["best_state"], "epoch": state["best_epoch"]},
               out / "ckpt" / "best.pt")
    model.load_state_dict(state["best_state"])
    best = val_metrics(va["y"], predict(model, data, va, device, cfg.eval_batch))
    best["best_epoch"] = state["best_epoch"]
    done.write_text(json.dumps({"config": asdict(cfg), "val": best}, indent=1))
    return best
