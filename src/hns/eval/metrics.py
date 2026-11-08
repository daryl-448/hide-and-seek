"""Binary illicit-vs-licit metrics (SPEC §6). Illicit is the positive class (1)."""

from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support, roc_auc_score)


def recall_at_k(y: np.ndarray, score: np.ndarray, frac: float) -> float:
    """Share of all illicit nodes found among the top ``ceil(frac * n)`` scores."""
    k = math.ceil(frac * len(y))
    top = np.argsort(-score, kind="stable")[:k]
    return float(y[top].sum() / max(y.sum(), 1))


def binary_metrics(y: np.ndarray, score: np.ndarray, threshold: float = 0.5,
                   ts: np.ndarray | None = None) -> dict:
    """Metrics at ``pred = score > threshold``, plus ranking metrics on ``score``.

    ``score > 0.5`` on a two-class probability is exactly sklearn's
    ``predict`` (argmax, ties to class 0), which the S0 notebook uses.
    """
    y = np.asarray(y).astype(np.int64)
    score = np.asarray(score, dtype=np.float64)
    pred = (score > threshold).astype(np.int64)
    p, r, f, _ = precision_recall_fscore_support(y, pred, labels=[0, 1], zero_division=0)
    out = {
        "n": int(len(y)), "n_illicit": int(y.sum()), "threshold": float(threshold),
        "illicit_precision": float(p[1]), "illicit_recall": float(r[1]), "illicit_f1": float(f[1]),
        "pr_auc": float(average_precision_score(y, score)),
        "roc_auc": float(roc_auc_score(y, score)),
        "recall_at_1pct": recall_at_k(y, score, 0.01),
        "recall_at_5pct": recall_at_k(y, score, 0.05),
        # comparison-only readings of the paper's undefined "F1" (E10)
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y, pred, average="weighted", zero_division=0)),
        "micro_f1": float(f1_score(y, pred, average="micro", zero_division=0)),
        "confusion": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
    }
    if ts is not None:
        ts = np.asarray(ts)
        out["illicit_f1_per_ts"] = {
            int(t): float(f1_score(y[ts == t], pred[ts == t], labels=[1], average=None, zero_division=0)[0])
            for t in np.unique(ts)
        }
    return out


def best_f1_threshold(y: np.ndarray, score: np.ndarray) -> float:
    """Threshold maximizing illicit F1 of ``score > threshold`` (SPEC §6: chosen on val).

    Cuts only between distinct scores, at their midpoint; ties go to the
    smaller threshold set (fewer positives).
    """
    y = np.asarray(y).astype(np.int64)
    score = np.asarray(score, dtype=np.float64)
    order = np.argsort(-score, kind="stable")
    s, yy = score[order], y[order]
    tp = np.cumsum(yy)
    k = np.arange(1, len(s) + 1)
    f1 = 2 * tp / (k + yy.sum())
    valid = np.r_[s[1:] < s[:-1], True]
    i = int(np.argmax(np.where(valid, f1, -1.0)))
    return float((s[i] + s[i + 1]) / 2) if i + 1 < len(s) else float(np.nextafter(s[-1], -np.inf))
