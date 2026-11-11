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

Repair cycle 2 (SPEC §10, ``hns.marl.repair2``) tests variants of the role
heads, each off by default: ``baseline="independent"`` gives every branch but
the root its own value stream, ``Q_b = V_b(s) + A_b - mean A_b``, so the
branches' means are no longer forced equal; ``act="leaky"`` makes the trunk
LeakyReLU(0.01); ``norm=True`` puts a LayerNorm on the readout before the
trunk. :func:`root_max` is the hard root target, and :func:`td_loss` has a
Huber and a branch-weighted form.

:class:`JointHeads` (stage 1, ``hns.marl.stage1``) is a different critic for
the launderer: it values complete actions, not branches. Its values are

- ``Q(s, wait)``, and with a tranche in flight ``Q(s, proceed)``;
- for a launch, ``Q(s, (m, k)) = U(s, m) + A(s, k) - max_k' A(s, k')``, where the
  mode ``m`` is one option of every launch branch but the pointer (amount x
  strategy x mules x change, 144 modes) with its own output ``U(s, m)``, and
  ``k`` is the recipient, scored by the pointer's attention ``A``;
- ``Q(s, go) = max_{m, k} Q(s, (m, k)) = max_m U(s, m)``: the best launch.

So the value of launching never averages over launch options, and every
launch trains the value of exactly the action taken.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from hns.models.detector import Encoder, Graph
from hns.sim.live import N_AGENT_EDGE_TYPES


class BranchHeads(nn.Module):
    def __init__(self, d_in: int, branches: dict[str, int], hidden: int = 128, pointer: dict[str, int] | None = None,
                 att: int = 64, baseline: str = "shared", act: str = "relu", norm: bool = False):
        super().__init__()
        if baseline not in ("shared", "independent") or act not in ("relu", "leaky"):
            raise ValueError(f"unknown head option {baseline} / {act}")
        self.names, self.sizes = list(branches), list(branches.values())
        self.pointer = dict(pointer or {})
        self.trunk = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU() if act == "relu" else nn.LeakyReLU(0.01))
        self.value = nn.Linear(hidden, 1)
        self.adv = nn.ModuleDict({b: nn.Linear(hidden, n) for b, n in branches.items() if b not in self.pointer})
        self.ptr_s = nn.ModuleDict({b: nn.Linear(hidden, att) for b in self.pointer})
        self.ptr_c = nn.ModuleDict({b: nn.Linear(d, att) for b, d in self.pointer.items()})
        self.ptr_v = nn.ModuleDict({b: nn.Linear(att, 1) for b in self.pointer})
        # the variants draw no random numbers and come last, so every variant starts from the same common weights;
        # a branch's own value stream starts as a copy of the shared one, so Q starts the same too
        self.norm = nn.LayerNorm(d_in) if norm else None
        self.value_b = None
        if baseline == "independent":
            self.value_b = nn.ModuleDict({b: copy.deepcopy(self.value) for b in self.names[1:]})

    def forward(self, h: torch.Tensor, masks: torch.Tensor, cands: dict[str, torch.Tensor] | None = None) -> torch.Tensor:
        """``Q`` of every option of every branch, ``(B, sum sizes)``; ``-inf`` where ``masks`` is false."""
        z = self.trunk(h if self.norm is None else self.norm(h))
        root = self.value(z)
        out, o = [], 0
        for i, (b, n) in enumerate(zip(self.names, self.sizes)):
            v = root if i == 0 or self.value_b is None else self.value_b[b](z)
            if b in self.pointer:
                a = self.ptr_v[b](torch.tanh(self.ptr_s[b](z).unsqueeze(1) + self.ptr_c[b](cands[b]))).squeeze(-1)
            else:
                a = self.adv[b](z)
            m = masks[:, o:o + n]
            mean = (a * m).sum(1, keepdim=True) / m.sum(1, keepdim=True).clamp_min(1)
            out.append((v + a - mean).masked_fill(~m, float("-inf")))
            o += n
        return torch.cat(out, dim=1)


class JointHeads(nn.Module):
    """The launderer's critic over complete actions (module docstring). ``branches``: the role's branches with the
    root first; ``pointer``: the one pointer branch and its candidates' feature width. The trunk is LeakyReLU(0.01).

    :meth:`parts` gives the critic's own values. :meth:`forward` lays them out like :class:`BranchHeads` does, for
    everything that reads values per branch: the root holds ``Q(wait)`` and ``Q(go)`` (``Q(proceed)`` in flight),
    the pointer branch ``Q(go) + A - max A``, and every other branch the best launch that uses each of its options
    (``max`` of ``U`` over the modes with that option). The best option of every branch together is the best
    launch.
    """

    def __init__(self, d_in: int, branches: dict[str, int], pointer: dict[str, int], hidden: int = 128, att: int = 64):
        super().__init__()
        self.names, self.sizes = list(branches), list(branches.values())
        (self.ptr, d_cand), = pointer.items()
        self.table = [b for b in self.names[1:] if b != self.ptr]            # the mode's branches, in order
        self.dims = [branches[b] for b in self.table]
        self.n_modes = int(np.prod(self.dims))
        self.trunk = nn.Sequential(nn.Linear(d_in, hidden), nn.LeakyReLU(0.01))
        self.root = nn.Linear(hidden, 2)                                     # wait; proceed (a tranche in flight)
        self.modes = nn.Linear(hidden, self.n_modes)
        self.ptr_s, self.ptr_c, self.ptr_v = nn.Linear(hidden, att), nn.Linear(d_cand, att), nn.Linear(att, 1)

    def _slices(self) -> dict[str, slice]:
        o = np.concatenate([[0], np.cumsum(self.sizes)])
        return {b: slice(int(o[i]), int(o[i + 1])) for i, b in enumerate(self.names)}

    def mode_of(self, action: torch.Tensor) -> torch.Tensor:
        """The mode index of each row's per-branch option indices ``(B, n_branches)``."""
        m = torch.zeros_like(action[:, 0])
        for b, n in zip(self.table, self.dims):
            m = m * n + action[:, self.names.index(b)]
        return m

    def parts(self, h: torch.Tensor, masks: torch.Tensor, cands: torch.Tensor, idle: torch.Tensor):
        """``(root, U, A)``: ``root`` ``(B, 2)`` is ``Q(wait), Q(go)`` with ``Q(go) = max U`` for an idle state and
        ``Q(proceed)`` in flight, ``-inf`` where the root option is infeasible; ``U`` ``(B, modes)`` with ``-inf`` at
        infeasible modes; ``A`` ``(B, K)``, the recipient's advantage with its best feasible option at 0, ``-inf``
        at infeasible candidates."""
        z = self.trunk(h)
        sl = self._slices()
        u = self.modes(z).masked_fill(~self.mode_mask(masks), float("-inf"))
        mk = masks[:, sl[self.ptr]]
        a = self.ptr_v(torch.tanh(self.ptr_s(z).unsqueeze(1) + self.ptr_c(cands))).squeeze(-1).masked_fill(~mk, float("-inf"))
        a = a - a.max(dim=1, keepdim=True).values
        r = self.root(z)
        root = torch.stack([r[:, 0], torch.where(idle, u.max(dim=1).values, r[:, 1])], dim=1)
        return root.masked_fill(~masks[:, sl[self.names[0]]], float("-inf")), u, a

    def mode_mask(self, masks: torch.Tensor) -> torch.Tensor:
        """``(B, modes)``: a mode is feasible if each of its options is."""
        sl = self._slices()
        ok = torch.ones(len(masks), 1, dtype=torch.bool, device=masks.device)
        for b in self.table:
            ok = (ok.unsqueeze(2) & masks[:, sl[b]].unsqueeze(1)).flatten(1)
        return ok

    def best(self, parts) -> torch.Tensor:
        """The greedy complete action as per-branch option indices ``(B, n_branches)``: the root's best option,
        and one best mode and one best recipient (the first of each among ties), decoded into its options."""
        root, u, a = parts
        out = torch.zeros(len(u), len(self.names), dtype=torch.long, device=u.device)
        out[:, 0], m = root.argmax(1), u.argmax(1)
        for b, n in zip(reversed(self.table), reversed(self.dims)):
            out[:, self.names.index(b)] = m % n
            m = m // n
        out[:, self.names.index(self.ptr)] = a.argmax(1)
        return out

    def check(self, parts, masks: torch.Tensor, idle: torch.Tensor) -> torch.Tensor:
        """``(all finite, largest |Q|)`` over the feasible complete actions: wait, proceed in flight, and for an
        idle state with a feasible launch every mode with every recipient."""
        root, u, a = parts
        sl = self._slices()
        ok, mk, mr = self.mode_mask(masks), masks[:, sl[self.ptr]], masks[:, sl[self.names[0]]]
        finite = torch.isfinite(root[mr]).all() & torch.isfinite(u[ok]).all() & torch.isfinite(a[mk]).all()
        lo = (u.masked_fill(~ok, float("inf")).min(1).values + a.masked_fill(~mk, float("inf")).min(1).values).abs()
        launch = torch.where(idle & mr[:, 1], torch.maximum(lo, u.max(1).values.abs()), torch.zeros_like(lo))
        return finite, torch.maximum(root.masked_fill(~mr, 0.0).abs().max(1).values, launch).max()

    def taken(self, parts, action: torch.Tensor, idle: torch.Tensor) -> torch.Tensor:
        """``Q(s, a)`` of each row's complete action: wait, proceed, or the launch ``(mode, recipient)``."""
        root, u, a = parts
        launch = (u.gather(1, self.mode_of(action).unsqueeze(1)) + a.gather(1, action[:, self.names.index(self.ptr)].unsqueeze(1))).squeeze(1)
        go = torch.where(idle, launch, root[:, 1])
        return torch.where(action[:, 0] == 0, root[:, 0], go)

    def layout(self, parts, masks: torch.Tensor) -> torch.Tensor:
        root, u, a = parts
        sl, best = self._slices(), u.max(dim=1, keepdim=True).values
        cube = u.view(len(u), *self.dims)
        out = []
        for b in self.names:
            if b == self.names[0]:
                q = root
            elif b == self.ptr:
                q = best + a
            else:
                d = self.table.index(b) + 1
                q = cube.amax(dim=[i for i in range(1, cube.dim()) if i != d])
            out.append(q.masked_fill(~masks[:, sl[b]], float("-inf")))
        return torch.cat(out, dim=1)

    def forward(self, h: torch.Tensor, masks: torch.Tensor, cands: torch.Tensor, idle: torch.Tensor) -> torch.Tensor:
        return self.layout(self.parts(h, masks, cands, idle), masks)


class AgentQ(nn.Module):
    """``head``: options of :class:`BranchHeads`, or ``{"critic": "joint"}`` for :class:`JointHeads` (launderers:
    its idle states are those whose first controller feature, "tranche in flight", is 0; ``hns.marl.obs``)."""

    def __init__(self, branches: dict[str, int], d_tx: int, d_w: int, d_ctrl: int, pointer: dict[str, int] | None = None,
                 dropout: float = 0.5, head: dict | None = None, **enc):
        super().__init__()
        head = dict(head or {})
        self.joint = head.pop("critic", "branch") == "joint"
        self.enc = Encoder(d_tx, d_w, d_ctrl=d_ctrl, dropout=dropout, n_edge_types=N_AGENT_EDGE_TYPES, **enc)
        self.heads = (JointHeads(self.enc.out_dim, branches, pointer) if self.joint
                      else BranchHeads(self.enc.out_dim, branches, pointer=pointer, **head))
        self.sizes = self.heads.sizes

    def parts(self, g: Graph, x_tx, x_w, x_c, masks, cand):
        """The joint critic's ``(root, U, A)`` and which rows are idle."""
        idle = x_c[:, 0] == 0
        return self.heads.parts(self.enc.encode(g, x_tx, x_w, x_c=x_c), masks, cand, idle), idle

    def forward(self, g: Graph, x_tx, x_w, x_c, masks, cand=None) -> torch.Tensor:
        if self.joint:
            return self.heads.layout(self.parts(g, x_tx, x_w, x_c, masks, cand)[0], masks)
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


def root_max(q: torch.Tensor, sizes: list[int]) -> torch.Tensor:
    """``max_{a in M_0(s)} Q_0(s, a)`` per row: the hard root target (repair cycle 2), Algorithm 1's max on the root."""
    return q[:, :sizes[0]].max(dim=1).values


def taken(q: torch.Tensor, sizes: list[int], action: torch.Tensor, realized: torch.Tensor) -> torch.Tensor:
    """``Q_b(s, a_b)`` of the taken option of every branch, ``(B, n_branches)``; 0 where the branch is unrealized (its
    stored option may be infeasible, ``-inf``)."""
    qa = torch.stack([qb.gather(1, action[:, i:i + 1]).squeeze(1) for i, qb in enumerate(q.split(sizes, dim=1))], dim=1)
    return torch.where(realized, qa, torch.zeros_like(qa))


def td_loss(q: torch.Tensor, sizes: list[int], action: torch.Tensor, realized: torch.Tensor, y: torch.Tensor,
            kind: str = "mse") -> torch.Tensor:
    """``mean_batch (1/|R(s,a)|) sum_{b in R(s,a)} e(y - Q_b(s, a_b))`` with ``e(d) = d^2`` (``mse``, §10).

    Repair cycle 2: ``huber`` keeps that average with ``e(d) = d^2`` for ``|d| <= 1`` and ``2|d| - 1`` beyond (twice
    the Huber loss of delta 1, so only the tail changes). ``branch`` weights by branch, not by transition:
    ``mean_batch d_root^2 + 0.25 sum_{b != root} mean_{rows that realize b} d_b^2``; a branch that no row of the
    batch realizes adds 0.
    """
    d = y.unsqueeze(1) - taken(q, sizes, action, realized)
    r = realized.to(d.dtype)
    e = (2.0 * F.huber_loss(d, torch.zeros_like(d), reduction="none", delta=1.0) if kind == "huber" else d ** 2) * r
    if kind == "branch":
        return e[:, 0].mean() + 0.25 * (e[:, 1:].sum(0) / r[:, 1:].sum(0).clamp_min(1)).sum()
    if kind not in ("mse", "huber"):
        raise ValueError(f"unknown loss {kind}")
    return (e.sum(1) / r.sum(1).clamp_min(1)).mean()


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
