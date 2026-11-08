"""HRGATConv, detector, adversarial term and val threshold (SPEC §6, §7, §8)."""

import math

import numpy as np
import torch

from hns.eval.metrics import best_f1_threshold
from hns.models.detector import Detector, Graph
from hns.models.hrg import TX_INW, W_INTX, Views
from hns.models.hrgat import LevelInfluence
from hns.transfer.adv import adv_loss, fgsm, mask_features


def _views():
    # two views: tx 0 with wallet 10 (n_tx = 5 -> global 15), and wallet 16 alone
    return Views(node_g=np.array([0, 15, 16]), hop=np.array([0, 1, 0], np.int8), view=np.array([0, 0, 1], np.int32),
                 target=np.array([0, 2]), src=np.array([1, 0]), dst=np.array([0, 1]),
                 etype=np.array([TX_INW, W_INTX], np.int8), logmult=np.zeros(2, np.float32))


def test_influence_at_init():
    inf = LevelInfluence(0.5)
    lvl_d, lvl_s = torch.tensor([0, 0, 1, 1]), torch.tensor([0, 1, 0, 1])
    assert torch.allclose(inf(lvl_d, lvl_s).exp(), torch.tensor([1.0, 0.5, 0.5, 1.0]))


def test_zero_init_adapter_is_shared_only():
    torch.manual_seed(0)
    g = Graph(_views(), n_tx=5, device="cpu")
    m = Detector(d_tx=3, d_w=2, d_anon=4).eval()
    x_tx, x_w = torch.randn(1, 3), torch.randn(2, 2)
    a = m(g, x_tx, x_w, torch.randn(1, 4))
    b = m(g, x_tx, x_w, torch.zeros(1, 4))
    assert a.shape == (2,) and torch.allclose(a, b)


def test_fgsm_step_and_mask():
    torch.manual_seed(0)
    x = {"a": torch.randn(50, 7)}
    w = torch.randn(7)
    loss = lambda f: (f["a"] @ w).sum()  # noqa: E731
    adv = fgsm(loss, x, eps=0.1)
    assert torch.allclose(adv["a"] - x["a"], 0.1 * w.sign().expand(50, 7))
    m = mask_features({"a": torch.ones(20000, 5)}, 0.1)["a"]
    assert abs(float((m == 0).float().mean()) - 0.1) < 0.01
    assert math.isfinite(float(adv_loss(loss, x)))


def test_best_f1_threshold_matches_brute_force():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    s = np.round(rng.random(300) * 0.5 + 0.3 * y, 2)  # ties
    thr = best_f1_threshold(y, s)

    def f1(c):
        p = s > c
        tp = (p & (y == 1)).sum()
        return 2 * tp / (p.sum() + y.sum())
    assert math.isclose(f1(thr), max(f1(c) for c in np.r_[np.unique(s), -1.0]))


def test_train_forward_finite_with_huge_logvar():
    torch.manual_seed(0)
    g = Graph(_views(), n_tx=5, device="cpu")
    m = Detector(d_tx=3, d_w=2).train()
    torch.nn.init.constant_(m.zs_logvar.bias, 500.0)  # exp(250) overflows without the bound
    out = m(g, torch.randn(1, 3), torch.randn(2, 2))
    out.sum().backward()
    assert torch.isfinite(out).all()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
