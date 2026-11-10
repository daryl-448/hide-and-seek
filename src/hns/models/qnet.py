"""Branching dueling Q-networks for the three roles (SPEC §10; BDQ, Tavakoli et al. 2018).

For every branch ``b`` of a role's structured action,

    Q_b(s, a_b) = V(s) + A_b(s, a_b) - mean_{a in M_b(s)} A_b(s, a),

with the mean over the options that are feasible in ``s``. Infeasible
options come back as ``-inf``. The launderer's ``recipient`` branch is a
pointer: its advantages score the ``K`` candidate services by additive
attention between the state embedding and each candidate's features.

- :class:`AgentQ`: an HRG-GAT encoder over the agent view (its own
  parameters; ``hns.models.detector.Encoder``) and the branch heads on the
  512-d readout. One per role, shared by the role's agents (§9.3).
- :class:`GlobalQ`: the inspector's global branches (``analysis_depth``,
  ``update_strategy``) on the world's global readout, which is a plain
  vector, so the state embedding is an MLP.

The inspector's per-tx ``select_tx`` head lives on the detector itself
(``Detector.flag_q``).

The soft-Q pieces of §10 are here too: :func:`soft_value` is the mean
target's ``(1/|B(s')|) sum_b tau logsumexp_a Q_b(s', a) / tau`` (the
inspector's global branches, which are always all realized),
:func:`root_value` is the root target ``tau logsumexp_a Q_initiate(s', a) /
tau`` (launderers and innocents), :func:`td_loss` is the mean squared error
over the realized branches ``R(s, a)``, and :func:`sample_actions` is the
behaviour policy ``softmax(Q_b / tau)`` over ``M_b`` mixed with
epsilon-greedy.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from hns.models.detector import Encoder, Graph
from hns.sim.live import N_AGENT_EDGE_TYPES


class BranchHeads(nn.Module):
    def __init__(self, d_in: int, branches: dict[str, int], hidden: int = 128, pointer: dict[str, int] | None = None,
                 att: int = 64):
        super().__init__()
        self.names, self.sizes = list(branches), list(branches.values())
        self.pointer = dict(pointer or {})
        self.trunk = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU())
        self.value = nn.Linear(hidden, 1)
        self.adv = nn.ModuleDict({b: nn.Linear(hidden, n) for b, n in branches.items() if b not in self.pointer})
        self.ptr_s = nn.ModuleDict({b: nn.Linear(hidden, att) for b in self.pointer})
        self.ptr_c = nn.ModuleDict({b: nn.Linear(d, att) for b, d in self.pointer.items()})
        self.ptr_v = nn.ModuleDict({b: nn.Linear(att, 1) for b in self.pointer})

    def forward(self, h: torch.Tensor, masks: torch.Tensor, cands: dict[str, torch.Tensor] | None = None) -> torch.Tensor:
        """``Q`` of every option of every branch, ``(B, sum sizes)``; ``-inf`` where ``masks`` is false."""
        z = self.trunk(h)
        v = self.value(z)
        out, o = [], 0
        for b, n in zip(self.names, self.sizes):
            if b in self.pointer:
                a = self.ptr_v[b](torch.tanh(self.ptr_s[b](z).unsqueeze(1) + self.ptr_c[b](cands[b]))).squeeze(-1)
            else:
                a = self.adv[b](z)
            m = masks[:, o:o + n]
            mean = (a * m).sum(1, keepdim=True) / m.sum(1, keepdim=True).clamp_min(1)
            out.append((v + a - mean).masked_fill(~m, float("-inf")))
            o += n
        return torch.cat(out, dim=1)


class AgentQ(nn.Module):
    def __init__(self, branches: dict[str, int], d_tx: int, d_w: int, d_ctrl: int, pointer: dict[str, int] | None = None,
                 dropout: float = 0.5, **enc):
        super().__init__()
        self.enc = Encoder(d_tx, d_w, d_ctrl=d_ctrl, dropout=dropout, n_edge_types=N_AGENT_EDGE_TYPES, **enc)
        self.heads = BranchHeads(self.enc.out_dim, branches, pointer=pointer)
        self.sizes = self.heads.sizes

    def forward(self, g: Graph, x_tx, x_w, x_c, masks, cand=None) -> torch.Tensor:
        cands = None if cand is None else {b: cand for b in self.heads.pointer}
        return self.heads(self.enc.encode(g, x_tx, x_w, x_c=x_c), masks, cands)


class GlobalQ(nn.Module):
    def __init__(self, branches: dict[str, int], d_in: int, hidden: int = 128):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU())
        self.heads = BranchHeads(hidden, branches)
        self.sizes = self.heads.sizes

    def forward(self, x: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        return self.heads(self.body(x), masks)


# ---------------------------------------------------------------------------- soft-Q pieces
def soft_value(q: torch.Tensor, sizes: list[int], active: torch.Tensor, tau: float) -> torch.Tensor:
    """``(1/|B(s)|) sum_{b in B(s)} tau logsumexp_{a in M_b(s)} Q_b(s, a) / tau`` per row.

    ``q``: ``(B, sum sizes)`` with ``-inf`` at infeasible options. ``active``:
    ``(B, n_branches)``, which branches are in ``B(s)``.
    """
    vals = torch.stack([tau * torch.logsumexp(qb / tau, dim=1) for qb in q.split(sizes, dim=1)], dim=1)
    act = active.to(vals.dtype)
    return (vals * act).sum(1) / act.sum(1).clamp_min(1)


def root_value(q: torch.Tensor, sizes: list[int], tau: float) -> torch.Tensor:
    """``tau logsumexp_{a in M_0(s)} Q_0(s, a) / tau`` per row: the soft value of the first (root) branch only.

    For launderers and innocents the root is ``initiate``: it is in ``B(s)`` and realized under every action, so
    it is the only branch whose values every transition trains. The other branches only rank the options of a
    launch (§10, ours). Infeasible options are ``-inf`` in ``q`` and drop out.
    """
    return tau * torch.logsumexp(q[:, :sizes[0]] / tau, dim=1)


def td_loss(q: torch.Tensor, sizes: list[int], action: torch.Tensor, realized: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """``mean_batch (1/|R(s,a)|) sum_{b in R(s,a)} (y - Q_b(s, a_b))^2``."""
    qa = torch.stack([qb.gather(1, action[:, i:i + 1]).squeeze(1) for i, qb in enumerate(q.split(sizes, dim=1))], dim=1)
    qa = torch.where(realized, qa, torch.zeros_like(qa))   # an unrealized branch's stored option may be infeasible (-inf)
    r = realized.to(qa.dtype)
    return ((((y.unsqueeze(1) - qa) ** 2) * r).sum(1) / r.sum(1).clamp_min(1)).mean()


def sample_actions(q: np.ndarray, sizes: list[int], tau: float, eps: float, rng: np.random.Generator) -> np.ndarray:
    """Behaviour policy: per branch, ``softmax(Q_b / tau)`` over the feasible options, or uniform over them with
    probability ``eps`` (independently per row and branch). ``q``: ``(B, sum sizes)`` with ``-inf`` at infeasible
    options. Returns ``(B, n_branches)`` option indices."""
    out = np.zeros((q.shape[0], len(sizes)), np.int64)
    o = 0
    for i, n in enumerate(sizes):
        qb = q[:, o:o + n].astype(np.float64)
        feas = np.isfinite(qb)
        explore = rng.random(q.shape[0]) < eps
        logits = np.where(explore[:, None], 0.0, qb / tau)
        z = np.where(feas, logits + rng.gumbel(size=qb.shape), -np.inf)   # Gumbel-max: a softmax sample
        out[:, i] = z.argmax(1)
        o += n
    return out
