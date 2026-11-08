"""HRG-GAT detector: node-state projections, 2 HRGATConv layers, readout, Z_s and label head (SPEC §8).

Only the parts the target-only arm (T0) trains are here: the encoder, the
``Z_s`` Gaussian head and the label head ``p(y | Z_s)``. The ``Z_d``/``Z_n``
heads, decoder and domain head of §8.3 are added with the DA arms.

- Node state (Eq. 10): ``Linear_tx(shared) + adapter(anon)`` for txs (the
  adapter exists only when ``d_anon > 0``, i.e. T, and starts at zero, §7),
  ``Linear_w(wallet features)`` for wallets, plus a learned hop marker
  ``{target, hop1, hop2}``.
- Edge state (Eq. 11): ``MLP([one-hot message type; log1p(multiplicity)])``.
- Self-loops (type ``SELF_LOOP``) are added so a node attends to itself.
- Readout: ``[O_local || O_global]``, the target's embedding and the mean over
  its view (512 dims).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from hns.models.hrg import N_EDGE_TYPES, SELF_LOOP, Views
from hns.models.hrgat import HRGATConv, LevelInfluence

LOGVAR_MAX = 10.0  # ours: |log σ²| bound for the Z_s sample


class Graph:
    """``Views`` as tensors on ``device``, with self-loops appended."""

    def __init__(self, v: Views, n_tx: int, device):
        n = len(v.node_g)
        loops = np.arange(n, dtype=np.int64)
        src = np.concatenate([v.src, loops])
        dst = np.concatenate([v.dst, loops])
        typ = np.concatenate([v.etype.astype(np.int64), np.full(n, SELF_LOOP, np.int64)])
        lm = np.concatenate([v.logmult, np.zeros(n, np.float32)])
        level = (v.node_g >= n_tx).astype(np.int64)
        t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(device, non_blocking=True)  # noqa: E731
        self.n, self.n_views = n, len(v.target)
        self.is_tx = t(level == 0)
        self.hop = t(v.hop.astype(np.int64))
        self.view = t(v.view.astype(np.int64))
        self.target = t(v.target)
        self.edge_index = t(np.stack([src, dst]))
        self.etype, self.logmult = t(typ), t(lm)
        self.level_src, self.level_dst = t(level[src]), t(level[dst])


class Detector(nn.Module):
    def __init__(self, d_tx: int, d_w: int, d_anon: int = 0, hidden: int = 256, heads: int = 4, head_dim: int = 64,
                 layers: int = 2, edge_dim: int = 32, z_s: int = 256, dropout: float = 0.5, hrg_alpha: float = 0.5):
        super().__init__()
        self.dropout = dropout
        self.proj_tx = nn.Linear(d_tx, hidden)
        self.proj_w = nn.Linear(d_w, hidden)
        self.adapter = None
        if d_anon:
            self.adapter = nn.Linear(d_anon, hidden)
            nn.init.zeros_(self.adapter.weight)
            nn.init.zeros_(self.adapter.bias)
        self.marker = nn.Embedding(3, hidden)
        self.psi = nn.Sequential(nn.Linear(N_EDGE_TYPES + 1, edge_dim), nn.ELU(), nn.Linear(edge_dim, edge_dim))
        self.influence = LevelInfluence(hrg_alpha)
        dims = [hidden] + [heads * head_dim] * layers
        self.convs = nn.ModuleList(HRGATConv(dims[i], head_dim, heads, edge_dim, dropout) for i in range(layers))
        out = dims[-1]
        self.zs_mu = nn.Linear(2 * out, z_s)
        self.zs_logvar = nn.Linear(2 * out, z_s)
        self.label = nn.Sequential(nn.Linear(z_s, 128), nn.ReLU(), nn.Linear(128, 1))

    def encode(self, g: Graph, x_tx, x_w, x_anon=None) -> torch.Tensor:
        """Readout ``[O_local || O_global]`` per view, ``(n_views, 2 * out)``."""
        s_tx = self.proj_tx(x_tx)
        if self.adapter is not None:
            s_tx = s_tx + self.adapter(x_anon)
        h = s_tx.new_empty(g.n, s_tx.shape[1])
        h[g.is_tx] = s_tx
        h[~g.is_tx] = self.proj_w(x_w)
        h = h + self.marker(g.hop)
        e = self.psi(torch.cat([F.one_hot(g.etype, N_EDGE_TYPES).to(h.dtype), g.logmult.unsqueeze(1)], dim=1))
        log_infl = self.influence(g.level_dst, g.level_src)
        for conv in self.convs:
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = F.elu(conv(h, g.edge_index, e, log_infl))
        pooled = h.new_zeros(g.n_views, h.shape[1]).index_add_(0, g.view, h)
        counts = torch.bincount(g.view, minlength=g.n_views).clamp_min(1).unsqueeze(1).to(h.dtype)
        return torch.cat([h[g.target], pooled / counts], dim=1)

    def forward(self, g: Graph, x_tx, x_w, x_anon=None) -> torch.Tensor:
        """Illicit logit per view. ``Z_s`` is a reparameterized sample in training, its mean in eval."""
        r = self.encode(g, x_tx, x_w, x_anon)
        z = self.zs_mu(r)
        if self.training:
            # T0 has no KL on Z_s, so nothing else bounds log σ²; unbounded, it overflowed exp() in training
            z = z + torch.randn_like(z) * torch.exp(0.5 * self.zs_logvar(r).clamp(-LOGVAR_MAX, LOGVAR_MAX))
        return self.label(z).squeeze(1)
