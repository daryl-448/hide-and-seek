"""Adversarial term L_adv (SPEC §8.4, Eq. 21 with the sign fixed, G5).

During fine-tuning, per batch: zero each standardized input feature with
probability ``p_miss`` (the paper's "mimic missing values"), then take one
FGSM step of size ``eps`` (L-inf) on the masked inputs in the direction that
*increases* the label NLL. ``L_adv`` is the NLL at that perturbed copy, which
the model minimizes. Graph structure is never perturbed.
"""

from __future__ import annotations

import torch


def mask_features(feats: dict[str, torch.Tensor], p_miss: float, gen: torch.Generator | None = None) -> dict:
    return {k: x * (torch.rand(x.shape, generator=gen, device=x.device) >= p_miss).to(x.dtype)
            for k, x in feats.items()}


def fgsm(loss_fn, feats: dict[str, torch.Tensor], eps: float) -> dict[str, torch.Tensor]:
    """One signed-gradient ascent step on ``loss_fn(feats)`` w.r.t. the inputs only (no parameter grads)."""
    leaves = {k: x.detach().requires_grad_(True) for k, x in feats.items()}
    grads = torch.autograd.grad(loss_fn(leaves), list(leaves.values()))
    return {k: (x + eps * g.sign()).detach() for (k, x), g in zip(leaves.items(), grads)}


def adv_loss(loss_fn, feats: dict[str, torch.Tensor], eps: float = 0.1, p_miss: float = 0.1,
             gen: torch.Generator | None = None) -> torch.Tensor:
    """``loss_fn(feats) -> scalar NLL``; returns the NLL at the masked-plus-FGSM inputs."""
    return loss_fn(fgsm(loss_fn, mask_features(feats, p_miss, gen), eps))
