"""HRG-GAT encoder and detector: node-state projections, 2 HRGATConv layers, readout, DGDA-lite heads (SPEC §8).

:class:`Encoder` is the HRG-GAT that every policy shares in architecture
(never in parameters): the detectors here and the role Q-networks in
``hns.models.qnet``.

- Node state (Eq. 10): ``Linear_tx(shared) + adapter(anon)`` for txs (the
  adapter exists only when ``d_anon > 0``, i.e. T fine-tuning, and starts at
  zero, §7), ``Linear_w(wallet features)`` for wallets, ``Linear_c`` for an
  agent view's controller node (§9.3), plus a learned hop marker
  ``{target, hop1, hop2}``.
- Edge state (Eq. 11): ``MLP([one-hot message type; log1p(multiplicity)])``.
- Self-loops (type ``SELF_LOOP``) are added so a node attends to itself.
- Readout: ``[O_local || O_global]``, the target's embedding and the mean over
  its view (512 dims).

:class:`Detector` adds the ``Z_s`` Gaussian head and the label head
``p(y | Z_s)``, which is all the target-only arm (T0) trains. With
``da=True`` it also has the §8.3 heads: Gaussian ``Z_d`` (64) and ``Z_n``
(128), a domain head on ``Z_d``, learnable domain means for the ``Z_d``
prior, and a decoder of the target's standardized shared features. With
``flag_q=True`` it has the acting inspector's ``select_tx`` Q head (§9.3,
§10): dueling ``Q_skip, Q_flag`` on the readout.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from hns.models.hrg import N_EDGE_TYPES, SELF_LOOP, Views
from hns.models.hrgat import HRGATConv, LevelInfluence

LOGVAR_MAX = 10.0  # ours: |log σ²| bound for the latent samples
KIND_TX, KIND_W, KIND_CTRL = 0, 1, 2


class Graph:
    """``Views`` as tensors on ``device``, with self-loops appended.

    Node kinds come from ``v.kind`` if set, else from ``node_g >= n_tx``
    (wallet). A controller node (kind 2) is a level-2 node, like a wallet.
    """

    def __init__(self, v: Views, n_tx: int, device):
        n = len(v.hop)
        loops = np.arange(n, dtype=np.int64)
        src = np.concatenate([v.src, loops])
        dst = np.concatenate([v.dst, loops])
        typ = np.concatenate([v.etype.astype(np.int64), np.full(n, SELF_LOOP, np.int64)])
        lm = np.concatenate([v.logmult, np.zeros(n, np.float32)])
        kind = (v.node_g >= n_tx).astype(np.int64) if v.kind is None else v.kind.astype(np.int64)
        level = np.minimum(kind, 1)
        t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(device, non_blocking=True)  # noqa: E731
        self.n, self.n_views = n, len(v.target)
        self.is_tx = t(kind == KIND_TX)
        self.is_w = t(kind == KIND_W)
        self.is_ctrl = t(kind == KIND_CTRL)
        self.hop = t(v.hop.astype(np.int64))
        self.view = t(v.view.astype(np.int64))
        self.target = t(v.target)
        self.edge_index = t(np.stack([src, dst]))
        self.etype, self.logmult = t(typ), t(lm)
        self.level_src, self.level_dst = t(level[src]), t(level[dst])


class Encoder(nn.Module):
    def __init__(self, d_tx: int, d_w: int, d_anon: int = 0, hidden: int = 256, heads: int = 4, head_dim: int = 64,
                 layers: int = 2, edge_dim: int = 32, dropout: float = 0.5, hrg_alpha: float = 0.5, d_ctrl: int = 0,
                 n_edge_types: int = N_EDGE_TYPES):
        super().__init__()
        self.dropout, self.n_edge_types = dropout, n_edge_types
        self.amp_dtype = None  # torch.bfloat16 on an sm_80 GPU: autocast for the GAT layers only (§3)
        self.proj_tx = nn.Linear(d_tx, hidden)
        self.proj_w = nn.Linear(d_w, hidden)
        self.adapter = None
        if d_anon:
            self.adapter = nn.Linear(d_anon, hidden)
            nn.init.zeros_(self.adapter.weight)
            nn.init.zeros_(self.adapter.bias)
        self.marker = nn.Embedding(3, hidden)
        self.psi = nn.Sequential(nn.Linear(n_edge_types + 1, edge_dim), nn.ELU(), nn.Linear(edge_dim, edge_dim))
        self.influence = LevelInfluence(hrg_alpha)
        dims = [hidden] + [heads * head_dim] * layers
        self.convs = nn.ModuleList(HRGATConv(dims[i], head_dim, heads, edge_dim, dropout) for i in range(layers))
        self.out_dim = 2 * dims[-1]
        if d_ctrl:
            self.proj_c = nn.Linear(d_ctrl, hidden)

    def encode(self, g: Graph, x_tx, x_w, x_anon=None, x_c=None) -> torch.Tensor:
        """Readout ``[O_local || O_global]`` per view, ``(n_views, 2 * out)``."""
        s_tx = self.proj_tx(x_tx)
        if self.adapter is not None:
            s_tx = s_tx + self.adapter(x_anon)
        h = s_tx.new_empty(g.n, s_tx.shape[1])
        h[g.is_tx] = s_tx
        h[g.is_w] = self.proj_w(x_w)
        if x_c is not None:
            h[g.is_ctrl] = self.proj_c(x_c)
        h = h + self.marker(g.hop)
        e = self.psi(torch.cat([F.one_hot(g.etype, self.n_edge_types).to(h.dtype), g.logmult.unsqueeze(1)], dim=1))
        log_infl = self.influence(g.level_dst, g.level_src)
        with torch.autocast(h.device.type, dtype=self.amp_dtype, enabled=self.amp_dtype is not None):
            for conv in self.convs:
                h = F.dropout(h, p=self.dropout, training=self.training)
                h = F.elu(conv(h, g.edge_index, e, log_infl))
        h = h.float()
        pooled = h.new_zeros(g.n_views, h.shape[1]).index_add_(0, g.view, h)
        counts = torch.bincount(g.view, minlength=g.n_views).clamp_min(1).unsqueeze(1).to(h.dtype)
        return torch.cat([h[g.target], pooled / counts], dim=1)


def _sample(mu: torch.Tensor, logvar: torch.Tensor, sample: bool) -> torch.Tensor:
    return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar) if sample else mu


def kl_normal(mu: torch.Tensor, logvar: torch.Tensor, prior_mu: torch.Tensor | None = None) -> torch.Tensor:
    """``KL(N(mu, diag σ²) || N(prior_mu, I))`` per row, summed over dimensions."""
    d = mu if prior_mu is None else mu - prior_mu
    return 0.5 * (logvar.exp() + d.pow(2) - 1.0 - logvar).sum(1)


class Detector(Encoder):
    def __init__(self, d_tx: int, d_w: int, d_anon: int = 0, hidden: int = 256, heads: int = 4, head_dim: int = 64,
                 layers: int = 2, edge_dim: int = 32, z_s: int = 256, dropout: float = 0.5, hrg_alpha: float = 0.5,
                 da: bool = False, d_rec: int = 0, z_d: int = 64, z_n: int = 128, flag_q: bool = False):
        super().__init__(d_tx, d_w, d_anon, hidden, heads, head_dim, layers, edge_dim, dropout, hrg_alpha)
        out = self.out_dim
        self.zs_mu = nn.Linear(out, z_s)
        self.zs_logvar = nn.Linear(out, z_s)
        self.label = nn.Sequential(nn.Linear(z_s, 128), nn.ReLU(), nn.Linear(128, 1))
        self.da = da
        if da:
            self.zd_mu, self.zd_logvar = nn.Linear(out, z_d), nn.Linear(out, z_d)
            self.zn_mu, self.zn_logvar = nn.Linear(out, z_n), nn.Linear(out, z_n)
            self.domain = nn.Linear(z_d, 1)                      # domain head, MLP 64 -> 1
            self.mu_dom = nn.Parameter(torch.zeros(2, z_d))      # prior means of Z_d: row 0 source, row 1 target
            self.decoder = nn.Sequential(nn.Linear(z_s + z_d + z_n, 256), nn.ReLU(), nn.Linear(256, d_rec))
        if flag_q:
            self.flag_head = nn.Sequential(nn.Linear(out, 128), nn.ReLU(), nn.Linear(128, 3))

    def forward(self, g: Graph, x_tx, x_w, x_anon=None) -> torch.Tensor:
        """Illicit logit per view. ``Z_s`` is a reparameterized sample in training, its mean in eval."""
        r = self.encode(g, x_tx, x_w, x_anon)
        z = self.zs_mu(r)
        if self.training:
            # T0 has no KL on Z_s, so nothing else bounds log σ²; unbounded, it overflowed exp() in training
            z = z + torch.randn_like(z) * torch.exp(0.5 * self.zs_logvar(r).clamp(-LOGVAR_MAX, LOGVAR_MAX))
        return self.label(z).squeeze(1)

    def heads(self, r: torch.Tensor) -> dict[str, torch.Tensor]:
        """Label logit and the three Gaussian heads on a readout ``r`` (samples in training, means in eval)."""
        s = self.training
        mu, lv = self.zs_mu(r), self.zs_logvar(r).clamp(-LOGVAR_MAX, LOGVAR_MAX)
        z_s = _sample(mu, lv, s)
        out = {"logit": self.label(z_s).squeeze(1), "zs_mu": mu, "zs_logvar": lv, "z_s": z_s}
        if self.da:
            for k in ("zd", "zn"):
                mu = getattr(self, f"{k}_mu")(r)
                lv = getattr(self, f"{k}_logvar")(r).clamp(-LOGVAR_MAX, LOGVAR_MAX)
                out |= {f"{k}_mu": mu, f"{k}_logvar": lv, f"z_{k[1]}": _sample(mu, lv, s)}
        return out

    def aux_losses(self, h: dict[str, torch.Tensor], domain: torch.Tensor, x_target: torch.Tensor,
                   beta: float = 0.01) -> dict[str, torch.Tensor]:
        """The four ``L_aux`` terms (§8.3) on the output of :meth:`heads`.

        ``domain``: 0 for simulator rows, 1 for real rows. ``x_target``: each
        row's target node's standardized shared features.
        """
        return {
            "kl_d": beta * kl_normal(h["zd_mu"], h["zd_logvar"], self.mu_dom[domain]).mean(),
            "dom": F.binary_cross_entropy_with_logits(self.domain(h["z_d"]).squeeze(1), domain.to(h["z_d"].dtype)),
            "kl_sn": beta * (kl_normal(h["zs_mu"], h["zs_logvar"]) + kl_normal(h["zn_mu"], h["zn_logvar"])).mean(),
            "rec": F.mse_loss(self.decoder(torch.cat([h["z_s"], h["z_d"], h["z_n"]], dim=1)), x_target),
        }

    def flag_q(self, r: torch.Tensor) -> torch.Tensor:
        """``(Q_skip, Q_flag)`` per view: ``V + A - mean(A)`` (both options are always feasible)."""
        o = self.flag_head(r)
        a = o[:, 1:]
        return o[:, :1] + a - a.mean(1, keepdim=True)


def target_rows(g: Graph, of_tx: bool) -> torch.Tensor:
    """Row of each view's target in ``x_tx`` (``of_tx``) or ``x_w``: features are stacked per kind in node order."""
    m = g.is_tx if of_tx else g.is_w
    return (torch.cumsum(m.long(), 0) - 1)[g.target]
