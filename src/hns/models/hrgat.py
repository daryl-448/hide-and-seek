"""HRGATConv: GAT attention plus a hierarchical-influence prior (SPEC §8.1-8.2, Eqs. 2, 5, 9, 19).

``e_ij = LeakyReLU(a^T [W h_i || W h_j || W_e e_ij]) + log I_ij``, softmax over
the incoming messages of ``i``, ``h'_i = sum_j alpha_ij W h_j`` (heads
concatenated; the caller applies ELU). ``I_ij = hrg_alpha^|L_i - L_j| *
softplus(F[L_i, L_j])`` with a learnable 2x2 level-pair table ``F`` (G4: the
exponent is ``|ΔL|``).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax


class LevelInfluence(nn.Module):
    """``log I_ij`` per message, from the destination and source levels (0 = tx, 1 = wallet)."""

    def __init__(self, hrg_alpha: float = 0.5):
        super().__init__()
        self.log_alpha = math.log(hrg_alpha)
        # softplus(F) = 1 at init, so I_ij = alpha^|dL| at init
        self.table = nn.Parameter(torch.full((2, 2), math.log(math.e - 1)))

    def forward(self, level_dst: torch.Tensor, level_src: torch.Tensor) -> torch.Tensor:
        s = F.softplus(self.table[level_dst, level_src])
        return (level_dst - level_src).abs().to(s.dtype) * self.log_alpha + torch.log(s)


class HRGATConv(MessagePassing):
    def __init__(self, in_dim: int, head_dim: int, heads: int, edge_dim: int, dropout: float = 0.5):
        super().__init__(aggr="add", node_dim=0)
        self.heads, self.head_dim, self.dropout = heads, head_dim, dropout
        self.lin = nn.Linear(in_dim, heads * head_dim, bias=False)
        self.lin_edge = nn.Linear(edge_dim, heads * head_dim, bias=False)
        self.att_dst = nn.Parameter(torch.empty(1, heads, head_dim))
        self.att_src = nn.Parameter(torch.empty(1, heads, head_dim))
        self.att_edge = nn.Parameter(torch.empty(1, heads, head_dim))
        self.bias = nn.Parameter(torch.zeros(heads * head_dim))
        for p in (self.att_dst, self.att_src, self.att_edge):
            nn.init.xavier_uniform_(p)

    def forward(self, x, edge_index, edge_attr, log_infl):
        h = self.lin(x).view(-1, self.heads, self.head_dim)
        e = self.lin_edge(edge_attr).view(-1, self.heads, self.head_dim)
        out = self.propagate(edge_index, h=h, e=e, log_infl=log_infl)
        return out.reshape(-1, self.heads * self.head_dim) + self.bias

    def message(self, h_i, h_j, e, log_infl, index, ptr, size_i):
        logit = (h_i * self.att_dst).sum(-1) + (h_j * self.att_src).sum(-1) + (e * self.att_edge).sum(-1)
        logit = F.leaky_relu(logit, 0.2) + log_infl.unsqueeze(-1)
        a = softmax(logit, index, ptr, size_i)
        a = F.dropout(a, p=self.dropout, training=self.training)
        return h_j * a.unsqueeze(-1)
