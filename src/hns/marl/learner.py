"""Per-role learners of the MARL loop: Algorithm 1 made implementable (SPEC §10, G7).

**Launderers and innocents** (:class:`RoleLearner`): one branching dueling
Q-network per role, shared by the role's agents, trained on the TD loss
only. Target (one per transition, soft-Q with temperature ``tau``):

    y = r + gamma ** gap (1 - d) F(s')

with the root target (``target = "root"``, ours)

    F(s') = tau logsumexp_{a in M_initiate(s')} Q_initiate^targ(s', a) / tau,

or the mean target as SPEC §10 first wrote it (``target = "mean"``, kept
for the M5 diagnosis)

    F(s') = (1/|B(s')|) sum_{b in B(s')} tau logsumexp_{a in M_b(s')} Q_b^targ(s', a) / tau,

and loss ``(1/|R(s,a)|) sum_{b in R(s,a)} (y - Q_b(s, a_b))^2``. ``s'`` is
the agent's next decision point (ours: a launderer without funds and an
innocent without a demand have no state, so the chain skips those
sub-steps), and ``gap`` is the number of sub-steps between ``s`` and ``s'``,
so ``gamma`` discounts per sub-step. ``d = 1`` only at world end. Why the
root: when an agent waits, only ``initiate`` is realized, yet the mean
target bootstraps through all of ``B(s')``, so branches trained only on
launches carry most of every idle state's target (reports/m5_marl.md).

The role networks have their own dropout, ``q_dropout``: acting and the
target network run in eval mode, updates in train mode. It is 0: at 0.5 the
values that updates fit and the values the targets use differ, and the
learner diverged (reports/m5_capability.md).

**Repair cycle 2** (SPEC §10, ``hns.marl.repair2``) tests learner variants
for the roles in ``design_roles``; every other role keeps the settings
above. ``baseline``, ``trunk_act`` and ``head_norm`` are the head variants of
``hns.models.qnet``. ``backup = "hard"`` replaces the root target's soft
value by ``max_{a in M_initiate(s')} Q_initiate^targ(s', a)``, so ``tau``
only shapes the behaviour policy. ``replay = "archive"`` gives a fifth of the
buffer to a lifetime reservoir (``hns.marl.buffer``). ``td`` is the loss
form (``hns.models.qnet.td_loss``). Each role's update counts what it
sampled, its per-branch squared errors and its gradient norms in
``RoleLearner.stats`` (read and reset by :meth:`RoleLearner.pop_stats`), and
raises ``FloatingPointError`` on a non-finite loss or gradient.

**detector-T** (:class:`DetectorLearner`, the acting inspector):
``TD + L_DA + lambda_MARL L_adv`` in one gradient step.

- TD, per-tx flags: a terminal one-step transition per candidate tx, so
  ``y = r`` for the taken option of ``(Q_skip, Q_flag)``. The observation is
  the candidate's T view at decision time: ``s_t`` plus the sub-step's
  tentative txs, at the chosen analysis depth.
- TD, global branches (``analysis_depth``, ``update_strategy``): the
  sub-step's summed detector reward, bootstrapping from the next sub-step's
  global readout (:class:`hns.models.qnet.GlobalQ`). A sub-step's summed
  reward is of order 1 to 10, so these values are of order ``r / (1 - gamma)``,
  hundreds, which Adam at lr 1e-3 cannot track. The branch is therefore
  learned in units of ``global_scale`` reward (ours): ``r``, ``Q`` and ``tau``
  are all divided by it, which leaves the objective and the behaviour policy
  ``softmax(Q / tau)`` unchanged.
- ``L_y^src``: BCE on ``batch`` committed simulator txs with observed
  labels, drawn from the :class:`SourcePool` (the current world's ledger and
  the last ``pool_worlds`` finished ones; ours), each in its T view: the
  committed graph restricted to ``ts <= t`` with causal counts as of ``t``.
  ``pos_weight = sqrt(n_neg / n_pos)`` over the pool, as in §6.
- ``L_aux`` (Full only): ``L_kl_d + L_dom + L_kl_sn + L_rec`` on those
  simulator txs and on ``batch`` real T-view txs from ts 1-29, labels unread.
- ``L_adv``: the label NLL of the same simulator txs at a perturbed copy:
  the episode's hashed feature mask at the world's mask rate, then one FGSM
  step (§8.4). Acting always uses clean features.

**detector-W** (:func:`da_step` on a :class:`SimW`): ``L_DA + lambda L_adv``
on labelled wallets of the finished world's export, in the W view, plus
unlabelled real W-fitting-split wallets for ``L_aux``. Its mask is drawn
once per world (ours: a W view spans the world's episodes).

The NDA arm drops ``L_aux`` and never samples a real batch.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F

from hns.data.norm import T_FIT_MAX_TS, Normalizer
from hns.marl.buffer import ARCHIVE_SHARE, Replay, VecReplay
from hns.marl.obs import (BRANCHES, D_CAND, D_W_AGENT, G_DIM, I_CTRL, L_CTRL, Featurizer, ObsBatch, collate,
                          detector_obs, merge, with_kind)
from hns.models.detector import Detector, target_rows
from hns.models.hrg import N_REL, ViewSampler, build_relations
from hns.models.qnet import AgentQ, GlobalQ, root_max, root_value, soft_value, taken, td_loss
from hns.sim.labels import ILLICIT, UNKNOWN
from hns.sim.live import keep_mask
from hns.transfer.adv import fgsm


@dataclass
class MarlConfig:
    arm: str = "full"            # full | nda
    seed: int = 0
    episodes: int = 1000         # MAX_EPISODES (paper); 10 per world
    lr: float = 1e-3
    gamma: float = 0.99
    tau: float = 0.01            # ENTROPY_COEF, used as the soft-Q temperature
    batch: int = 128
    clip: float = 1.0
    dropout: float = 0.5         # the detectors' GAT (paper)
    q_dropout: float = 0.0       # the launderer and innocent Q-networks (ours)
    target: str = "root"         # root | mean: the launderer and innocent bootstrap (ours, see above)
    # repair cycle 2 (ours): learner variants, for the roles in design_roles; the first value of each is M5's
    baseline: str = "shared"     # shared | independent: one V for all branches, or one per branch
    backup: str = "soft"         # soft | hard: the root target's soft value, or its max
    trunk_act: str = "relu"      # relu | leaky
    head_norm: bool = False      # LayerNorm on the readout, before the head's trunk
    replay: str = "fifo"         # fifo | archive
    td: str = "mse"              # mse | huber | branch
    design_roles: str = "launderer,innocent"
    beta: float = 0.01
    lam: float = 0.1             # lambda_MARL, preset (§8.4)
    fgsm_eps: float = 0.1
    buffer: int = 100_000
    warmup: int = 1000
    polyak: float = 0.005
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_episodes: int = 300
    k1: int = 10
    k2: int = 5
    cap: int = 512               # nodes per T / W view (§8.1)
    obs_cap: int = 256           # nodes per replayed observation (§10)
    pool_worlds: int = 2         # finished worlds kept as L_y^src sources, besides the current one (ours)
    global_scale: float = 100.0  # reward unit of the inspector's global branches: 1 / (1 - gamma) (ours)
    ckpt_worlds: int = 2         # checkpoint every 2 worlds = 20 episodes (<= 25, §10)
    amp: str = "none"            # none | bf16 (GAT layers only, sm_80)

    def __post_init__(self):
        if self.arm not in ("full", "nda"):
            raise ValueError(f"unknown arm {self.arm}")
        if self.target not in ("root", "mean"):
            raise ValueError(f"unknown target {self.target}")
        if (self.baseline not in ("shared", "independent") or self.backup not in ("soft", "hard")
                or self.trunk_act not in ("relu", "leaky") or self.replay not in ("fifo", "archive")
                or self.td not in ("mse", "huber", "branch")
                or not set(self.design_roles.split(",")) <= {"launderer", "innocent", ""}):
            raise ValueError("unknown learner variant")
        if self.backup == "hard" and self.target != "root":
            raise ValueError("the hard backup is defined for the root target")

    def design(self, role: str) -> dict:
        """The learner variant of ``role``: the configured one inside ``design_roles``, M5's otherwise."""
        keys = ("baseline", "backup", "trunk_act", "head_norm", "replay", "td")
        if role in self.design_roles.split(","):
            return {k: getattr(self, k) for k in keys}
        return {k: MarlConfig.__dataclass_fields__[k].default for k in keys}

    @property
    def da(self) -> bool:
        return self.arm == "full"


def _t(a, device) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(a)).to(device)


def _polyak(net: torch.nn.Module, targ: torch.nn.Module, rate: float) -> None:
    with torch.no_grad():
        for p, tp in zip(net.parameters(), targ.parameters()):
            tp.lerp_(p, rate)


def _target_of(net: torch.nn.Module) -> torch.nn.Module:
    t = copy.deepcopy(net).eval()
    for p in t.parameters():
        p.requires_grad_(False)
    return t


def _amp(cfg: MarlConfig, *encoders) -> None:
    if cfg.amp == "bf16":
        for e in encoders:
            e.amp_dtype = torch.bfloat16
    elif cfg.amp != "none":
        raise ValueError(cfg.amp)


# ---------------------------------------------------------------------------- launderers, innocents
class RoleLearner:
    def __init__(self, role: str, cfg: MarlConfig, device):
        self.role, self.cfg, self.device = role, cfg, device
        self.branches = BRANCHES[role]
        if next(iter(self.branches)) != "initiate":
            raise ValueError("the root target needs initiate as the first branch")
        ptr = {"recipient": D_CAND} if role == "launderer" else None
        self.design = d = cfg.design(role)
        self.net = AgentQ(self.branches, 17, D_W_AGENT, L_CTRL if role == "launderer" else I_CTRL, pointer=ptr,
                          dropout=cfg.q_dropout,
                          head={"baseline": d["baseline"], "act": d["trunk_act"], "norm": d["head_norm"]}).to(device)
        self.targ = _target_of(self.net)
        _amp(cfg, self.net.enc, self.targ.enc)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=cfg.lr)
        self.sizes = self.net.sizes
        arch = int(ARCHIVE_SHARE * cfg.buffer) if d["replay"] == "archive" else 0
        self.buf = Replay(cfg.buffer - arch, self.sizes, arch, seed=[cfg.seed, 3, int(role == "launderer")])
        self.tail: dict[int, int] = {}     # agent -> id of its newest buffer entry (this world)
        self.n_updates = 0
        self.pop_stats()

    def _q(self, net, ob: ObsBatch, masks: np.ndarray) -> torch.Tensor:
        g, f = ob.tensors(self.device)
        cand = None if ob.cand is None else _t(ob.cand, self.device)
        return net(g, f["x_tx"], f["x_w"], f["x_c"], _t(masks, self.device), cand)

    @torch.no_grad()
    def q_values(self, ob: ObsBatch, masks: np.ndarray) -> np.ndarray:
        """``Q`` of every option for acting; a non-finite value at a feasible option raises."""
        self.net.eval()
        q = self._q(self.net, ob, masks).cpu().numpy()
        if not np.isfinite(q[masks]).all():
            raise FloatingPointError(f"{self.role}: non-finite Q at a feasible option")
        return q

    def next_value(self, q2: torch.Tensor, next_active: torch.Tensor) -> torch.Tensor:
        """``F(s')`` of the configured target, from the target network's ``Q(s', .)``."""
        if self.cfg.target == "root":
            return root_max(q2, self.sizes) if self.design["backup"] == "hard" else root_value(q2, self.sizes, self.cfg.tau)
        return soft_value(q2, self.sizes, next_active, self.cfg.tau)

    def batch_target(self, b: dict) -> torch.Tensor:
        """``y = r + gamma ** gap (1 - d) F(s')`` of a replay batch."""
        cfg, dev = self.cfg, self.device
        y = _t(b["reward"], dev)
        if b["next_obs"]:
            with torch.no_grad():
                q2 = self._q(self.targ, collate(b["next_obs"]), b["next_masks"])
                v2 = self.next_value(q2, _t(b["next_active"], dev))
            y = y.clone()
            y[_t(~b["done"], dev)] += cfg.gamma ** _t(b["gap"], dev).to(y.dtype) * v2
        return y

    def pop_stats(self) -> dict:
        """The update statistics since the last call, as plain numbers; resets them."""
        st, nb = getattr(self, "stats", None), len(self.branches)
        self.stats = {"n": 0, "rows": 0, "archive_rows": 0, "clipped": 0, "grad": 0.0, "grad_max": 0.0, "grad_enc": 0.0,
                      "grad_head": 0.0, "sq": np.zeros(nb), "cnt": np.zeros(nb), "coef": np.zeros(nb),
                      "options": np.zeros(sum(self.sizes), np.int64)}
        if not st or not st["n"]:
            return {"n": 0}
        n, o = st["n"], np.concatenate([[0], np.cumsum(self.sizes)])
        return {"n": n, "archive_share": st["archive_rows"] / st["rows"], "clip_share": st["clipped"] / n,
                "grad_norm": st["grad"] / n, "grad_norm_max": st["grad_max"], "grad_norm_encoder": st["grad_enc"] / n,
                "grad_norm_head": st["grad_head"] / n,
                # per branch: realized rows per batch, their mean squared error, the weight of the branch's squared
                # errors in the loss (summed over a batch's rows), and how often each option was sampled
                "branch_rows": dict(zip(self.branches, (st["cnt"] / n).tolist())),
                "loss_coef": dict(zip(self.branches, (st["coef"] / n).tolist())),
                "branch_mse": dict(zip(self.branches, (st["sq"] / np.maximum(st["cnt"], 1)).tolist())),
                "sampled_options": {b: st["options"][o[i]:o[i + 1]].tolist() for i, b in enumerate(self.branches)}}

    def update(self, rng: np.random.Generator) -> float | None:
        """One TD step on a replay batch; ``None`` during warm-up."""
        cfg, dev = self.cfg, self.device
        if self.buf.ready < cfg.warmup:
            return None
        b = self.buf.sample(cfg.batch, rng)
        y = self.batch_target(b)
        self.net.train()
        q = self._q(self.net, collate(b["obs"]), b["masks"])
        action, realized = _t(b["action"], dev), _t(b["realized"], dev)
        loss = td_loss(q, self.sizes, action, realized, y, self.design["td"])
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        with torch.no_grad():
            sq = ((y.unsqueeze(1) - taken(q, self.sizes, action, realized)) ** 2 * realized).sum(0)
            g_enc = torch.sqrt(sum((p.grad ** 2).sum() for p in self.net.enc.parameters() if p.grad is not None))
        g = torch.nn.utils.clip_grad_norm_(self.net.parameters(), cfg.clip)
        loss_v, g_v = float(loss.detach()), float(g)
        if not (math.isfinite(loss_v) and math.isfinite(g_v)):
            raise FloatingPointError(f"{self.role}: non-finite loss or gradient at update {self.n_updates}")
        self.opt.step()
        _polyak(self.net, self.targ, cfg.polyak)
        self.n_updates += 1
        st, o = self.stats, np.concatenate([[0], np.cumsum(self.sizes)])
        st["n"] += 1
        st["rows"] += len(y)
        st["archive_rows"] += b["n_archive"]
        st["clipped"] += g_v > cfg.clip
        st["grad"] += g_v
        st["grad_max"] = max(st["grad_max"], g_v)
        st["grad_enc"] += float(g_enc)
        st["grad_head"] += math.sqrt(max(g_v ** 2 - float(g_enc) ** 2, 0.0))
        r = b["realized"].astype(np.float64)
        st["coef"] += (np.concatenate([[1.0], 0.25 * r[:, 1:].any(0)]) if self.design["td"] == "branch"
                       else (r / r.sum(1, keepdims=True)).mean(0))
        st["sq"] += sq.cpu().numpy()
        st["cnt"] += b["realized"].sum(0)
        for i in range(len(self.sizes)):
            st["options"][o[i]:o[i + 1]] += np.bincount(b["action"][b["realized"][:, i], i], minlength=self.sizes[i])
        return loss_v

    def end_world(self) -> None:
        """World end: every agent's last transition is terminal."""
        for i in self.tail.values():
            self.buf.end(i)
        self.tail.clear()


# ---------------------------------------------------------------------------- sources of labelled simulator nodes
class _Grow:
    def __init__(self, dtype):
        self.a, self.n = np.zeros(1024, dtype), 0

    def extend(self, x) -> None:
        x = np.asarray(x, self.a.dtype)
        if self.n + len(x) > len(self.a):
            self.a = np.concatenate([self.a, np.zeros(max(len(self.a), len(x)), self.a.dtype)])
        self.a[self.n:self.n + len(x)] = x
        self.n += len(x)

    @property
    def v(self) -> np.ndarray:
        return self.a[:self.n]


class SourcePool:
    """Committed simulator txs with observed labels (§10), over the current and the last few worlds."""

    def __init__(self, keep: int):
        self.keep, self.worlds = keep, []

    def begin(self, graph, mask_seed: int, mask_rate: float) -> None:
        """A new world starts; ``graph`` is its live graph."""
        self.worlds = self.worlds[-self.keep:] if self.keep else []
        self.worlds.append({"g": graph, "tid": _Grow(np.int64), "y": _Grow(np.float32), "seed": int(mask_seed),
                            "rate": float(mask_rate)})

    def add(self, tids, y) -> None:
        w = self.worlds[-1]
        w["tid"].extend(tids)
        w["y"].extend(y)

    @property
    def n(self) -> int:
        return sum(w["tid"].n for w in self.worlds)

    def pos_weight(self) -> float:
        pos = sum(float(w["y"].v.sum()) for w in self.worlds)
        return math.sqrt((self.n - pos) / pos) if 0 < pos < self.n else 1.0

    def sample(self, n: int, rng: np.random.Generator, fz: Featurizer, seed: int, k1: int, k2: int, cap: int):
        """``n`` labelled txs in their T views: ``(batch, y, keep_tx, keep_w)``; ``keep_*`` are the episodes'
        hashed feature masks of the batch's tx and wallet nodes."""
        sizes = np.array([w["tid"].n for w in self.worlds], np.float64)
        counts = rng.multinomial(n, sizes / sizes.sum())
        parts, ys, kt, kw = [], [], [], []
        for w, c in zip(self.worlds, counts):
            if c == 0:
                continue
            g = w["g"]
            idx = rng.integers(0, w["tid"].n, c)
            tids = w["tid"].v[idx]
            ts = g.tx_ts[tids].astype(np.int64)
            ob = detector_obs(g, fz, tids, ts, seed, k1, k2, cap)
            ep_seed = w["seed"] * 1000 + ts[ob.v.view]            # the mask of the target's episode
            is_tx = ob.v.kind == 0
            kt.append(keep_mask(ep_seed[is_tx], ob.v.node_g[is_tx], ob.x_tx.shape[1], w["rate"]))
            kw.append(keep_mask(ep_seed[~is_tx], ob.v.node_g[~is_tx], ob.x_w.shape[1], w["rate"]))
            parts.append(ob)
            ys.append(w["y"].v[idx])
        return merge(parts), np.concatenate(ys), np.concatenate(kt), np.concatenate(kw)


class SimW:
    """One exported world in the W view (§9.2), for detector-W: lifetime wallet features, AddrAddr."""

    def __init__(self, ex, norm_w: Normalizer, cfg: MarlConfig, mask_seed: int, mask_rate: float):
        self.n_tx, n_w = ex.txs.height, ex.wallets.height
        self.x_tx = norm_w.standardize("tx", ex.txs)
        self.x_w = norm_w.standardize("wallet_W", ex.wallets)
        rel = build_relations(self.n_tx, n_w, ex.graphs["tx_flow"], ex.graphs["addr_tx"], ex.graphs["addr_addr"],
                              ex.txs["ts"].to_numpy().astype(np.int16))
        self.sampler = ViewSampler(rel, tuple(range(N_REL)), False, cfg.k1, cfg.k2, cfg.cap)
        cls = ex.wallets["class"].to_numpy()
        lab = np.flatnonzero(cls != UNKNOWN)
        self.g, self.y = lab + self.n_tx, (cls[lab] == ILLICIT).astype(np.float32)
        self.seed, self.rate = int(mask_seed), float(mask_rate)

    @property
    def n(self) -> int:
        return len(self.g)

    def pos_weight(self) -> float:
        pos = float(self.y.sum())
        return math.sqrt((self.n - pos) / pos) if 0 < pos < self.n else 1.0

    def sample(self, n: int, rng: np.random.Generator, seed: int):
        idx = rng.integers(0, self.n, n)
        v = with_kind(self.sampler.sample(self.g[idx], None, seed), self.n_tx)
        is_tx = v.kind == 0
        ob = ObsBatch(v, self.x_tx[v.node_g[is_tx]], self.x_w[v.node_g[~is_tx] - self.n_tx])
        s = np.full(len(v.node_g), self.seed, np.int64)
        return (ob, self.y[idx], keep_mask(s[is_tx], v.node_g[is_tx], ob.x_tx.shape[1], self.rate),
                keep_mask(s[~is_tx], v.node_g[~is_tx], ob.x_w.shape[1], self.rate))


class RealViews:
    """Unlabelled real nodes of one task's fitting split, for ``L_aux`` (§8.3). Labels are never read."""

    def __init__(self, data, g: np.ndarray, t: np.ndarray):
        self.data, self.g, self.t = data, np.asarray(g, np.int64), np.asarray(t, np.int64)

    @classmethod
    def task_t(cls, data) -> "RealViews":
        """Non-null txs of ts 1-29 (T isolation, §5)."""
        ok = (data.tx_ts <= T_FIT_MAX_TS) & np.isfinite(data.x_tx[:, 0])
        g = np.flatnonzero(ok)
        return cls(data, g, data.tx_ts[g])

    @classmethod
    def task_w(cls, data) -> "RealViews":
        """Official-train wallets minus val, plus the unknown wallets."""
        ids = data.split.filter(pl.col("split").is_in(["fit", "unl"]))["id"].to_numpy().astype(np.int64)
        return cls(data, ids + data.n_tx, np.full(len(ids), -1))

    def sample(self, n: int, rng: np.random.Generator, seed: int) -> ObsBatch:
        idx = rng.integers(0, len(self.g), n)
        v, f = self.data.batch(self.g[idx], self.t[idx], seed)
        return ObsBatch(with_kind(v, self.data.n_tx), f["x_tx"], f["x_w"])


# ---------------------------------------------------------------------------- detectors
def da_losses(det: Detector, cfg: MarlConfig, device, src, real: ObsBatch | None, of_tx: bool, pos_weight: float,
              flag: ObsBatch | None = None) -> tuple[dict[str, torch.Tensor], torch.Tensor | None]:
    """``L_y^src``, the ``L_aux`` terms (if ``real`` is given) and ``L_adv`` of one detector update.

    ``src``: ``(batch, y, keep_tx, keep_w)`` of labelled simulator targets.
    ``flag``: an extra batch that shares the clean forward pass; its readouts
    are returned (detector-T's per-tx TD batch).
    """
    ob_s, y, keep_tx, keep_w = src
    parts = ([flag] if flag is not None else []) + [ob_s] + ([real] if real is not None else [])
    g, f = merge(parts).tensors(device)
    det.train()
    r = det.encode(g, f["x_tx"], f["x_w"])
    n_f, n_s = (len(flag) if flag is not None else 0), len(ob_s)
    h = det.heads(r[n_f:])
    y_t = _t(y, device)
    pw = torch.tensor(pos_weight, device=device)
    out = {"y_src": F.binary_cross_entropy_with_logits(h["logit"][:n_s], y_t, pos_weight=pw)}
    if real is not None:
        domain = torch.cat([torch.zeros(n_s, dtype=torch.long, device=device),
                            torch.ones(len(real), dtype=torch.long, device=device)])
        x_all = f["x_tx"] if of_tx else f["x_w"]
        out |= det.aux_losses(h, domain, x_all[target_rows(g, of_tx)[n_f:]], cfg.beta)
    # L_adv: NLL at the masked-plus-FGSM copy of the simulator batch
    g_s, f_s = ob_s.tensors(device)
    masked = {"x_tx": f_s["x_tx"] * _t(keep_tx, device), "x_w": f_s["x_w"] * _t(keep_w, device)}

    def nll(x):
        return F.binary_cross_entropy_with_logits(det(g_s, x["x_tx"], x["x_w"]), y_t, pos_weight=pw)

    out["adv"] = nll(fgsm(nll, masked, cfg.fgsm_eps))
    return out, (r[:n_f] if flag is not None else None)


def da_total(losses: dict[str, torch.Tensor], lam: float) -> torch.Tensor:
    """``L_DA + lambda L_adv`` with every ``L_DA`` term at weight 1."""
    return sum(v for k, v in losses.items() if k != "adv") + lam * losses["adv"]


class DetectorLearner:
    """detector-T (acting, ``TD + L_DA + lambda L_adv``) and detector-W (passive, ``L_DA + lambda L_adv``)."""

    def __init__(self, cfg: MarlConfig, device, fz: Featurizer, d_w_lifetime: int, real_t: RealViews | None = None,
                 real_w: RealViews | None = None):
        if cfg.da and (real_t is None or real_w is None):
            raise ValueError("the Full arm needs real fitting-split views for L_aux")
        if not cfg.da and (real_t is not None or real_w is not None):
            raise ValueError("NDA never samples a real batch")
        self.cfg, self.device, self.fz, self.real_t, self.real_w = cfg, device, fz, real_t, real_w
        self.det_t = Detector(17, 4, dropout=cfg.dropout, da=True, d_rec=17, flag_q=True).to(device)
        self.det_w = Detector(17, d_w_lifetime, dropout=cfg.dropout, da=True, d_rec=d_w_lifetime).to(device)
        self.gq = GlobalQ(BRANCHES["detector"], G_DIM).to(device)
        self.gq_targ = _target_of(self.gq)
        _amp(cfg, self.det_t, self.det_w)
        self.opt_t = torch.optim.Adam([*self.det_t.parameters(), *self.gq.parameters()], lr=cfg.lr)
        self.opt_w = torch.optim.Adam(self.det_w.parameters(), lr=cfg.lr)
        self.flag_buf = Replay(cfg.buffer, [2])
        self.gbuf = VecReplay(cfg.buffer, G_DIM, self.gq.sizes)
        self.pool = SourcePool(cfg.pool_worlds)
        self.g_tau = cfg.tau / cfg.global_scale      # the global branches' temperature in their reward unit
        self.n_updates = 0           # detector-T updates, total
        self.n_w_updates = 0

    # --- acting
    @torch.no_grad()
    def global_q(self, gvec: np.ndarray) -> np.ndarray:
        """Q of the global branches in units of ``global_scale`` reward: sample with ``g_tau``."""
        self.gq.eval()
        m = torch.ones(1, sum(self.gq.sizes), dtype=torch.bool, device=self.device)
        return self.gq(_t(gvec[None], self.device), m).cpu().numpy()

    @torch.no_grad()
    def flag_margin(self, ob: ObsBatch) -> np.ndarray:
        """``Q_flag - Q_skip`` per candidate, on clean features."""
        self.det_t.eval()
        g, f = ob.tensors(self.device)
        q = self.det_t.flag_q(self.det_t.encode(g, f["x_tx"], f["x_w"]))
        return (q[:, 1] - q[:, 0]).cpu().numpy()

    # --- learning
    def update(self, rng: np.random.Generator, seed: int) -> dict[str, float] | None:
        cfg, dev = self.cfg, self.device
        if self.flag_buf.ready < cfg.warmup:
            return None
        fb = self.flag_buf.sample(cfg.batch, rng)
        flag = collate(fb["obs"])
        losses: dict[str, torch.Tensor] = {}
        if self.pool.n >= cfg.batch:
            src = self.pool.sample(cfg.batch, rng, self.fz, seed, cfg.k1, cfg.k2, cfg.cap)
            real = self.real_t.sample(cfg.batch, rng, seed) if cfg.da else None
            losses, r_f = da_losses(self.det_t, cfg, dev, src, real, True, self.pool.pos_weight(), flag)
        else:
            g, f = flag.tensors(dev)
            self.det_t.train()
            r_f = self.det_t.encode(g, f["x_tx"], f["x_w"])
        q = self.det_t.flag_q(r_f)
        qa = q.gather(1, _t(fb["action"], dev)).squeeze(1)
        losses["td_flag"] = F.mse_loss(qa, _t(fb["reward"], dev))       # terminal: y = r
        if self.gbuf.ready >= cfg.batch:
            gb = self.gbuf.sample(cfg.batch, rng)
            ones = torch.ones(cfg.batch, sum(self.gq.sizes), dtype=torch.bool, device=dev)
            act = torch.ones(cfg.batch, len(self.gq.sizes), dtype=torch.bool, device=dev)
            with torch.no_grad():
                v2 = soft_value(self.gq_targ(_t(gb["next_state"], dev), ones), self.gq.sizes, act, self.g_tau)
                y = _t(gb["reward"], dev) / cfg.global_scale + cfg.gamma * (~_t(gb["done"], dev)).float() * v2
            self.gq.train()
            losses["td_global"] = td_loss(self.gq(_t(gb["state"], dev), ones), self.gq.sizes, _t(gb["action"], dev),
                                          act, y)
        total = sum(v for k, v in losses.items() if k != "adv") + cfg.lam * losses.get("adv", 0.0)
        self.opt_t.zero_grad(set_to_none=True)
        total.backward()
        # two networks without a shared parameter: clipped separately (ours), so the global branches' value scale
        # (summed rewards) cannot shrink the detector's own gradient
        torch.nn.utils.clip_grad_norm_(self.det_t.parameters(), cfg.clip)
        torch.nn.utils.clip_grad_norm_(self.gq.parameters(), cfg.clip)
        self.opt_t.step()
        _polyak(self.gq, self.gq_targ, cfg.polyak)
        self.n_updates += 1
        return {k: float(v.detach()) for k, v in losses.items()} | {"total": float(total.detach())}

    def update_w(self, sim: SimW, rng: np.random.Generator, seed: int) -> dict[str, float]:
        cfg = self.cfg
        src = sim.sample(cfg.batch, rng, seed)
        real = self.real_w.sample(cfg.batch, rng, seed) if cfg.da else None
        losses, _ = da_losses(self.det_w, cfg, self.device, src, real, False, sim.pos_weight())
        total = da_total(losses, cfg.lam)
        self.opt_w.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(self.det_w.parameters(), cfg.clip)
        self.opt_w.step()
        self.n_w_updates += 1
        return {k: float(v.detach()) for k, v in losses.items()} | {"total": float(total.detach())}
