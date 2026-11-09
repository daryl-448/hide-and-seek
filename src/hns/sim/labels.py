"""Simulator labels, mirroring Elliptic++'s labelling (SPEC §9.4, E2, E6).

- Ground truth: a tx is illicit iff a launderer initiates it.
- Observed tx labels, drawn once at commit: an illicit tx is labelled illicit
  with probability ``p_ill``; a licit tx is labelled licit if a service
  initiates it, else with probability ``p_lic``; everything else is unknown.
  The draw depends on the initiator's role only, never on a feature (the
  shortcut guard).
- Wallet labels, at export over the retained txs: illicit iff linked to an
  illicit-labelled tx; licit iff linked to a licit-labelled tx and not
  illicit, except that a ``p_wunk`` share of those is set to unknown; else
  unknown.

``p_ill`` and ``p_lic`` are solved from the public README totals (22.9 % of
txs labelled, 9.8 % of those illicit), and ``p_wunk`` from the wallet totals
(32.2 % labelled, 5.4 % of those illicit), never from per-node real labels.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from hns.sim.world import ROLE_SERVICE

ILLICIT, LICIT, UNKNOWN = 1, 2, 3

# README class totals (public statistics, not labels).
TX_TOTAL, TX_ILLICIT, TX_LICIT = 203_769, 4_545, 42_019
W_TOTAL, W_ILLICIT, W_LICIT = 822_942, 14_266, 251_088
TX_LABELLED = (TX_ILLICIT + TX_LICIT) / TX_TOTAL           # 0.2285
TX_ILLICIT_SHARE = TX_ILLICIT / (TX_ILLICIT + TX_LICIT)    # 0.0976
W_LABELLED = (W_ILLICIT + W_LICIT) / W_TOTAL               # 0.3224
W_ILLICIT_SHARE = W_ILLICIT / (W_ILLICIT + W_LICIT)        # 0.0538
P_WUNK_MAX = 0.5


def tx_label(u: float, illicit: bool, role: int, p_ill: float, p_lic: float) -> int:
    """Observed class from a uniform draw ``u`` and the initiator's role only."""
    if illicit:
        return ILLICIT if u < p_ill else UNKNOWN
    if role == ROLE_SERVICE:
        return LICIT
    return LICIT if u < p_lic else UNKNOWN


def wallet_labels(links: pl.DataFrame, tx_class: pl.DataFrame, p_wunk: float, rng: np.random.Generator) -> pl.DataFrame:
    """``links``: ``wid, tid``; ``tx_class``: ``tid, class``. Returns ``wid, class, licit_rule`` sorted by wid."""
    lc = (links.join(tx_class, on="tid").group_by("wid")
          .agg((pl.col("class") == ILLICIT).any().alias("ill"), (pl.col("class") == LICIT).any().alias("lic"))
          .sort("wid"))
    rule = (~lc["ill"] & lc["lic"]).to_numpy()
    drop = rng.random(lc.height) < p_wunk
    cls = np.where(lc["ill"].to_numpy(), ILLICIT, np.where(rule & ~drop, LICIT, UNKNOWN)).astype(np.int8)
    return pl.DataFrame({"wid": lc["wid"], "class": cls, "licit_rule": rule})


def solve_tx_rates(n: int, n_illicit: int, n_service: int, n_other: int) -> dict:
    """``p_ill``, ``p_lic`` that hit the README tx totals in expectation, clipped to [0, 1].

    ``n`` txs, of which ``n_illicit`` launderer, ``n_service`` service-initiated
    and ``n_other`` other licit txs.
    """
    want_ill = TX_LABELLED * TX_ILLICIT_SHARE * n
    want_lic = TX_LABELLED * (1 - TX_ILLICIT_SHARE) * n
    p_ill = want_ill / n_illicit if n_illicit else 1.0
    p_lic = (want_lic - n_service) / n_other if n_other else 0.0
    p_ill_c, p_lic_c = float(np.clip(p_ill, 0, 1)), float(np.clip(p_lic, 0, 1))
    got_ill = p_ill_c * n_illicit
    got_lic = n_service + p_lic_c * n_other
    return {"p_ill": p_ill_c, "p_lic": p_lic_c, "p_ill_unclipped": p_ill, "p_lic_unclipped": p_lic,
            "labelled_frac": (got_ill + got_lic) / n, "illicit_share": got_ill / max(got_ill + got_lic, 1e-12),
            "target_labelled_frac": TX_LABELLED, "target_illicit_share": TX_ILLICIT_SHARE}


def solve_p_wunk(n_wallets: int, n_illicit: int, n_licit_rule: int) -> dict:
    """``p_wunk`` that brings the labelled-wallet fraction to 32.2 %, clipped to [0, 0.5]."""
    want = W_LABELLED * n_wallets
    p = 1 - (want - n_illicit) / n_licit_rule if n_licit_rule else 0.0
    p_c = float(np.clip(p, 0, P_WUNK_MAX))
    got = n_illicit + (1 - p_c) * n_licit_rule
    return {"p_wunk": p_c, "p_wunk_unclipped": p, "labelled_frac": got / n_wallets,
            "illicit_share": n_illicit / max(got, 1e-12), "shortfall": W_LABELLED - got / n_wallets,
            "target_labelled_frac": W_LABELLED, "target_illicit_share": W_ILLICIT_SHARE}


GUARD_MAX_AUC = 0.6


def shortcut_auc(txs: pl.DataFrame, seed: int = 0) -> float:
    """Shortcut guard (§9.4): held-out ROC-AUC of a logistic regression on the
    shared tx features predicting "labelled vs unknown" among licit-by-truth
    txs that innocents initiate. Must be <= 0.6.

    ``txs``: an export's tx table (``truth``, ``role``, ``class``, 17 features).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    from hns.data.load import AUG_FEATURES
    from hns.feats import manifest
    from hns.sim.world import ROLE_INNOCENT

    d = txs.filter(~pl.col("truth") & (pl.col("role") == ROLE_INNOCENT))
    spec = manifest.load()["tx"]
    x = np.stack([manifest.apply(spec[f]["transform"], d[f].to_numpy(), f) for f in AUG_FEATURES], axis=1)
    x = (x - x.mean(0)) / np.where(x.std(0) > 0, x.std(0), 1.0)
    y = (d["class"] == LICIT).to_numpy()
    if y.all() or not y.any():
        return 0.5
    perm = np.random.default_rng(seed).permutation(len(y))
    tr, te = perm[: len(y) // 2], perm[len(y) // 2:]
    lr = LogisticRegression(max_iter=1000).fit(x[tr], y[tr])
    return float(roc_auc_score(y[te], lr.predict_proba(x[te])[:, 1]))
