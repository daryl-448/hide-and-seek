"""OPPO-MC's launderer actor and critic, and the option policy's probability algebra (SPEC §9.3, §10).

A launderer decides only when idle with funds and no wait outstanding. A decision is

- ``wait`` for a duration ``d`` in ``DURATIONS`` sub-steps (clipped at the episode's end; durations that clip to the
  same length are offered once, the shortest original), or
- ``launch``: a joint mode ``z = amount * 9 + strategy`` (36), then a recipient ``k`` (the pointer over ``K``
  candidate services), ``mules`` ``u`` and ``change`` ``c``, each conditioned on ``z``. ``mules`` is irrelevant
  for ``direct`` and ``peel``, ``change`` at amount 100 %: an irrelevant factor is canonical (index 0) and adds
  neither log-probability nor entropy.

So ``pi(wait, d) = pi_g(wait) pi_d(d)`` and ``pi(launch, z, k, u, c) = pi_g(launch) pi_z(z) pi_k(k | z)
pi_u(u | z)^I_u(z) pi_c(c | z)^I_c(z)``, every factor a softmax over its feasible options. On a scheduled
(curriculum or matched) decision the gate is exogenous: the action is a launch, and only the launch factors count.

Masks are one boolean row per decision, the factors side by side in ``FACTORS`` order (:data:`SLICES`). A
factor whose options are all infeasible never contributes: the gate's ``wait`` is always feasible, a duration of
one sub-step always is, and the mode mask is all false only when a launch is infeasible, in which case the launch
factors are evaluated on a placeholder mask and weighted by ``p(launch) = 0``.

:class:`OptionActor` and :class:`ValueNet` each own an HRG-GAT encoder (``hns.models.detector.Encoder``, dropout 0),
a LayerNorm on its readout, and the controller features concatenated after it. The actor's output layers start at
zero, so its first policy is uniform over the feasible choices.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from hns.models.detector import Encoder, Graph
from hns.sim.entities import AMOUNTS, K_RECIPIENTS, STRATEGIES
from hns.sim.live import N_AGENT_EDGE_TYPES

DURATIONS = (1, 5, 10, 25, 50)
N_MODES = len(AMOUNTS) * len(STRATEGIES)
FACTORS = {"gate": 2, "dur": len(DURATIONS), "mode": N_MODES, "rcpt": K_RECIPIENTS, "mules": 2, "change": 2}
_o = np.concatenate([[0], np.cumsum(list(FACTORS.values()))])
SLICES = {f: slice(int(_o[i]), int(_o[i + 1])) for i, f in enumerate(FACTORS)}
MASK_WIDTH = int(_o[-1])
ACTION = ("gate", "dur", "mode", "rcpt", "mules", "change")       # an action row's columns
WAIT, LAUNCH = 0, 1
# which modes make mules / change a decision: mules for every strategy but direct and peel, change below 100 %
MULES_REL = np.array([STRATEGIES[z % len(STRATEGIES)] not in ("direct", "peel") for z in range(N_MODES)])
CHANGE_REL = np.array([AMOUNTS[z // len(STRATEGIES)] < 1.0 for z in range(N_MODES)])
HIDDEN, ATT = 128, 64


def mode_of(amount: int, strategy: int) -> int:
    return amount * len(STRATEGIES) + strategy


def split_mode(z: int) -> tuple[int, int]:
    """``(amount, strategy)`` option indices of mode ``z``."""
    return z // len(STRATEGIES), z % len(STRATEGIES)


def duration_mask(sub: int, substeps: int) -> np.ndarray:
    """The durations offered at sub-step ``sub`` of an episode of ``substeps``: each clipped at the episode's end,
    and a clipped length offered once, by the shortest original duration that gives it."""
    eff = [min(d, substeps - sub) for d in DURATIONS]
    return np.array([eff[i] not in eff[:i] for i in range(len(eff))])


def effective_duration(d: int, sub: int, substeps: int) -> int:
    return min(DURATIONS[d], substeps - sub)


# ---------------------------------------------------------------------------- networks
class _Trunk(nn.Module):
    """HRG-GAT readout through a LayerNorm, with the controller features concatenated (a direct skip)."""

    def __init__(self, d_tx: int, d_w: int, d_ctrl: int):
        super().__init__()
        self.enc = Encoder(d_tx, d_w, d_ctrl=d_ctrl, dropout=0.0, n_edge_types=N_AGENT_EDGE_TYPES)
        self.norm = nn.LayerNorm(self.enc.out_dim)
        self.out_dim = self.enc.out_dim + d_ctrl

    def forward(self, g: Graph, x_tx, x_w, x_c) -> torch.Tensor:
        return torch.cat([self.norm(self.enc.encode(g, x_tx, x_w, x_c=x_c)), x_c], dim=1)


def _body(d: int) -> nn.Module:
    return nn.Sequential(nn.Linear(d, HIDDEN), nn.LeakyReLU(0.01))


class OptionActor(nn.Module):
    """Logits of every factor: ``gate (B, 2)``, ``dur (B, 5)``, ``mode (B, 36)``, and per mode ``rcpt (B, 36, K)``,
    ``mules (B, 36, 2)``, ``change (B, 36, 2)``. The gate and duration share one body, the launch factors another.
    The recipient is scored by additive attention between the launch state, a mode embedding and each candidate."""

    def __init__(self, d_tx: int, d_w: int, d_ctrl: int, d_cand: int):
        super().__init__()
        self.trunk = _Trunk(d_tx, d_w, d_ctrl)
        d = self.trunk.out_dim
        self.gate_body, self.launch_body = _body(d), _body(d)
        self.gate, self.dur = nn.Linear(HIDDEN, 2), nn.Linear(HIDDEN, len(DURATIONS))
        self.mode = nn.Linear(HIDDEN, N_MODES)
        self.mules, self.change = nn.Linear(HIDDEN, N_MODES * 2), nn.Linear(HIDDEN, N_MODES * 2)
        self.ptr_s, self.ptr_z, self.ptr_c = nn.Linear(HIDDEN, ATT), nn.Embedding(N_MODES, ATT), nn.Linear(d_cand, ATT)
        self.ptr_v = nn.Linear(ATT, 1)
        for m in (self.gate, self.dur, self.mode, self.mules, self.change, self.ptr_v):    # a uniform first policy
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, g: Graph, x_tx, x_w, x_c, cand) -> dict[str, torch.Tensor]:
        h = self.trunk(g, x_tx, x_w, x_c)
        hg, hl = self.gate_body(h), self.launch_body(h)
        n = len(h)
        e = torch.tanh(self.ptr_s(hl)[:, None, None, :] + self.ptr_z.weight[None, :, None, :]
                       + self.ptr_c(cand)[:, None, :, :])
        return {"gate": self.gate(hg), "dur": self.dur(hg), "mode": self.mode(hl), "rcpt": self.ptr_v(e).squeeze(-1),
                "mules": self.mules(hl).view(n, N_MODES, 2), "change": self.change(hl).view(n, N_MODES, 2)}


class ValueNet(nn.Module):
    """The critic's ``v(o)``, ``(B,)``; the learner scales it by ``m(o)``."""

    def __init__(self, d_tx: int, d_w: int, d_ctrl: int):
        super().__init__()
        self.trunk = _Trunk(d_tx, d_w, d_ctrl)
        self.body, self.out = _body(self.trunk.out_dim), nn.Linear(HIDDEN, 1)

    def forward(self, g: Graph, x_tx, x_w, x_c) -> torch.Tensor:
        return self.out(self.body(self.trunk(g, x_tx, x_w, x_c))).squeeze(1)


# ---------------------------------------------------------------------------- probability algebra
def _masks(masks: torch.Tensor) -> dict[str, torch.Tensor]:
    m = {f: masks[:, s] for f, s in SLICES.items()}
    m["mode"] = torch.where(m["mode"].any(1, keepdim=True), m["mode"], torch.ones_like(m["mode"]))  # placeholder
    return m


def log_tables(logits: dict[str, torch.Tensor], masks: torch.Tensor) -> dict[str, torch.Tensor]:
    """Log-probabilities of every factor's options, ``-inf`` where infeasible (the conditional factors per mode)."""
    m = _masks(masks)
    lsm = lambda x, k: x.masked_fill(~k, float("-inf")).log_softmax(-1)  # noqa: E731
    return {"gate": lsm(logits["gate"], m["gate"]), "dur": lsm(logits["dur"], m["dur"]),
            "mode": lsm(logits["mode"], m["mode"]), "rcpt": lsm(logits["rcpt"], m["rcpt"][:, None, :]),
            "mules": lsm(logits["mules"], m["mules"][:, None, :]), "change": lsm(logits["change"], m["change"][:, None, :])}


def _rel(ref: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return (torch.as_tensor(MULES_REL, device=ref.device), torch.as_tensor(CHANGE_REL, device=ref.device))


def _ent(lp: torch.Tensor) -> torch.Tensor:
    """Entropy over the last dimension of a log-probability table with ``-inf`` at infeasible options."""
    return -(lp.exp() * lp.masked_fill(~torch.isfinite(lp), 0.0)).sum(-1)


def _kl(old: torch.Tensor, new: torch.Tensor) -> torch.Tensor:
    """``KL(old || new)`` over the last dimension; both ``-inf`` at the same infeasible options."""
    ok = torch.isfinite(old)
    return (old.exp() * (old - new).masked_fill(~ok, 0.0)).sum(-1)


def logp_taken(t: dict[str, torch.Tensor], action: torch.Tensor, gate_free: torch.Tensor) -> torch.Tensor:
    """Log-probability of each row's complete effective action, ``(B,)``. ``action``: ``(B, 6)`` in ``ACTION``
    order (unused columns ignored); ``gate_free``: false on a scheduled decision, whose gate is exogenous."""
    rel_u, rel_c = _rel(action)
    g, d, z, k, u, c = action.unbind(1)
    r = torch.arange(len(action), device=action.device)
    launch = t["mode"][r, z] + t["rcpt"][r, z, k]
    launch = launch + torch.where(rel_u[z], t["mules"][r, z, u], torch.zeros_like(launch))
    launch = launch + torch.where(rel_c[z], t["change"][r, z, c], torch.zeros_like(launch))
    natural = t["gate"][r, g] + torch.where(g == LAUNCH, launch, t["dur"][r, d])
    return torch.where(gate_free, natural, launch)


def _launch_tail(t: dict[str, torch.Tensor], per_mode) -> torch.Tensor:
    """``E_{z ~ pi_z}`` of a per-mode quantity ``(B, 36)``."""
    return (t["mode"].exp() * per_mode).sum(1)


def entropy(t: dict[str, torch.Tensor], gate_free: torch.Tensor) -> torch.Tensor:
    """The joint policy's entropy by the chain rule, ``(B,)``: ``H_g + p(wait) H_d + p(launch) [H_z +
    E_z(H_k|z + I_u H_u|z + I_c H_c|z)]``; on a scheduled decision, the bracket alone."""
    rel_u, rel_c = _rel(gate_free)
    zero = torch.zeros((), device=gate_free.device)
    cond = _ent(t["rcpt"]) + torch.where(rel_u, _ent(t["mules"]), zero) + torch.where(rel_c, _ent(t["change"]), zero)
    h_launch = _ent(t["mode"]) + _launch_tail(t, cond)
    p = t["gate"].exp()
    natural = _ent(t["gate"]) + p[:, WAIT] * _ent(t["dur"]) + p[:, LAUNCH] * h_launch
    return torch.where(gate_free, natural, h_launch)


def kl_parts(old: dict[str, torch.Tensor], new: dict[str, torch.Tensor], gate_free: torch.Tensor):
    """``(joint, launch | launch, duration | wait)`` KL from the old to the new policy per row, by the chain rule.
    The joint is the policy that acted: on a scheduled decision, the launch factors alone."""
    rel_u, rel_c = _rel(gate_free)
    zero = torch.zeros((), device=gate_free.device)
    cond = (_kl(old["rcpt"], new["rcpt"]) + torch.where(rel_u, _kl(old["mules"], new["mules"]), zero)
            + torch.where(rel_c, _kl(old["change"], new["change"]), zero))
    k_launch = _kl(old["mode"], new["mode"]) + _launch_tail(old, cond)
    k_dur = _kl(old["dur"], new["dur"])
    p = old["gate"].exp()
    natural = _kl(old["gate"], new["gate"]) + p[:, WAIT] * k_dur + p[:, LAUNCH] * k_launch
    return torch.where(gate_free, natural, k_launch), k_launch, k_dur


def sample(t: dict[str, np.ndarray], gate_free: np.ndarray, rng: np.random.Generator, greedy: bool = False) -> np.ndarray:
    """One action per row from numpy log-probability tables (``(B, 6)``, ``ACTION`` order; unused columns 0;
    irrelevant mules / change canonical). A scheduled row launches. ``greedy``: each factor's most likely option."""
    n = len(gate_free)
    pick = (lambda lp: lp.argmax(-1)) if greedy else (lambda lp: (lp + rng.gumbel(size=lp.shape)).argmax(-1))
    out = np.zeros((n, len(ACTION)), np.int64)
    r = np.arange(n)
    gate = pick(t["gate"])
    out[:, 0] = np.where(gate_free, gate, LAUNCH)
    out[:, 1] = pick(t["dur"])
    out[:, 2] = z = pick(t["mode"])
    out[:, 3] = pick(t["rcpt"][r, z])
    out[:, 4] = np.where(MULES_REL[z], pick(t["mules"][r, z]), 0)
    out[:, 5] = np.where(CHANGE_REL[z], pick(t["change"][r, z]), 0)
    launch = out[:, 0] == LAUNCH
    out[launch, 1] = 0
    out[~launch, 2:] = 0
    return out
