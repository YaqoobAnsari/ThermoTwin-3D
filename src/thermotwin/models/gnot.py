"""GNOT — General Neural Operator Transformer, as a gridless Block-2 operator.

GNOT (Hao et al., ICML 2023, arXiv:2302.14376) is the strongest general-purpose neural-operator
transformer and the most direct competitor to Transolver: where Transolver attends over learned
*physics slices*, GNOT keeps **per-point** tokens but makes attention affordable with **normalised
linear attention** (O(N) instead of O(N²)) and injects geometry through a **mixture-of-experts
FFN gated by the point coordinates** (its "heterogeneous"/geometric-gating idea). If a strong
general operator beats our physics-primed `delta_transolver`, we need to know — so GNOT is the
existential baseline, not a courtesy one.

This is a self-contained, faithful-in-spirit port (pure ``torch`` — no DGL/vendored CUDA): the two
load-bearing GNOT ingredients are (1) linear attention and (2) geometry-gated MoE. It is wrapped in
the same Block-2 point-cloud contract as :mod:`thermotwin.models.transolver`:

* :class:`GNOTOperator` — ``forward(input_geom, x, latent_queries, sdf, output_queries) -> (B, n, 1)``;
  ``latent_queries`` / ``sdf`` / ``output_queries`` are accepted for GINO-contract compatibility
  but unused (GNOT is gridless; ``output_queries == input_geom`` for v1).
* :class:`DeltaGNOT` — predicts the correction on the analytic 1-D clear-wall prior, mirroring
  :class:`~thermotwin.models.transolver.DeltaTransolver`.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.init import trunc_normal_

__all__ = ["GNOTOperator", "DeltaGNOT", "build_gnot", "build_delta_gnot"]


class _LinearAttention(nn.Module):
    """Normalised linear (kernel) self-attention — O(N) in the number of points.

    Uses the ``elu(·)+1`` feature map (Katharopoulos et al. 2020): with ``φ(q), φ(k) ≥ 0``,
    ``attn(q) = φ(q) (Σ_k φ(k)ᵀ v) / (φ(q) · Σ_k φ(k))`` reproduces softmax-like attention at
    linear cost — GNOT's mechanism for scaling per-point attention to dense fields.
    """

    def __init__(self, dim: int, heads: int = 8, dim_head: int = 32) -> None:
        super().__init__()
        inner = heads * dim_head
        self.heads = heads
        self.dim_head = dim_head
        self.to_qkv = nn.Linear(dim, inner * 3, bias=False)
        self.to_out = nn.Linear(inner, dim)

    def forward(self, x: Tensor) -> Tensor:
        b, n, _ = x.shape
        qkv = self.to_qkv(x).reshape(b, n, 3, self.heads, self.dim_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (b, h, n, d)
        q = F.elu(q) + 1.0
        k = F.elu(k) + 1.0
        kv = torch.einsum("bhnd,bhne->bhde", k, v)  # (b, h, d, d)
        z = 1.0 / (torch.einsum("bhnd,bhd->bhn", q, k.sum(dim=2)) + 1e-6)
        out = torch.einsum("bhnd,bhde,bhn->bhne", q, kv, z)  # (b, h, n, d)
        out = out.permute(0, 2, 1, 3).reshape(b, n, self.heads * self.dim_head)
        return self.to_out(out)


class _GeoMoE(nn.Module):
    """Mixture-of-experts FFN gated by point geometry (GNOT's heterogeneous gating).

    ``n_experts`` independent MLPs; the per-point mixture weights are a softmax over a linear
    function of the raw coordinates, so different geometric regions (corner, bulk, bridge) can be
    routed to different experts.
    """

    def __init__(self, dim: int, hidden: int, space_dim: int = 3, n_experts: int = 4) -> None:
        super().__init__()
        self.gate = nn.Linear(space_dim, n_experts)
        self.experts = nn.ModuleList(
            nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
            for _ in range(n_experts)
        )

    def forward(self, x: Tensor, coords: Tensor) -> Tensor:
        w = F.softmax(self.gate(coords), dim=-1)  # (b, n, e)
        outs = torch.stack([e(x) for e in self.experts], dim=-1)  # (b, n, dim, e)
        return torch.einsum("bnde,bne->bnd", outs, w)


class _GNOTBlock(nn.Module):
    """Pre-norm block: linear self-attention + geometry-gated MoE FFN, both residual."""

    def __init__(self, dim: int, heads: int, space_dim: int, n_experts: int, mlp_ratio: int = 4) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = _LinearAttention(dim, heads=heads, dim_head=max(dim // heads, 16))
        self.ln2 = nn.LayerNorm(dim)
        self.moe = _GeoMoE(dim, dim * mlp_ratio, space_dim=space_dim, n_experts=n_experts)

    def forward(self, x: Tensor, coords: Tensor) -> Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.moe(self.ln2(x), coords)
        return x


class GNOTOperator(nn.Module):
    """GNOT under the Block-2 point-cloud contract.

    Args:
        in_channels: per-point feature width (3 data-only, 4 with the theta1d prior channel).
        out_channels: output channels (1 = theta).
        space_dim: coordinate dimensionality (3).
        n_layers / n_hidden / n_head / n_experts: GNOT capacity.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 1,
        space_dim: int = 3,
        n_layers: int = 6,
        n_hidden: int = 128,
        n_head: int = 8,
        n_experts: int = 4,
    ) -> None:
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(space_dim + int(in_channels), n_hidden * 2), nn.GELU(), nn.Linear(n_hidden * 2, n_hidden)
        )
        self.blocks = nn.ModuleList(
            _GNOTBlock(n_hidden, n_head, space_dim, n_experts) for _ in range(n_layers)
        )
        self.ln = nn.LayerNorm(n_hidden)
        self.head = nn.Linear(n_hidden, int(out_channels))
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @staticmethod
    def _bn_(t: Tensor) -> Tensor:
        return t.unsqueeze(0) if t.dim() == 2 else t

    def forward(
        self,
        input_geom: Tensor,
        x: Tensor,
        latent_queries: Tensor | None = None,
        sdf: Tensor | None = None,
        output_queries: Tensor | None = None,
    ) -> Tensor:
        coords = self._bn_(input_geom)  # (1, n, 3)
        feats = self._bn_(x)  # (1, n, F)
        h = self.embed(torch.cat((coords, feats), dim=-1))
        for block in self.blocks:
            h = block(h, coords)
        return self.head(self.ln(h))  # (1, n, out)


class DeltaGNOT(nn.Module):
    """GNOT that predicts a correction added to the analytic 1-D theta prior."""

    def __init__(self, operator: GNOTOperator) -> None:
        super().__init__()
        self.operator = operator

    def forward(
        self,
        input_geom: Tensor,
        x: Tensor,
        latent_queries: Tensor | None,
        sdf: Tensor | None,
        output_queries: Tensor | None,
        query_prior: Tensor,
    ) -> Tensor:
        corr = self.operator(input_geom, x, latent_queries, sdf, output_queries)  # (1, n, 1)
        prior = query_prior
        if prior.dim() == 1:
            prior = prior.unsqueeze(0).unsqueeze(-1)
        elif prior.dim() == 2:
            prior = prior.unsqueeze(0) if (prior.shape[-1] == 1 and prior.shape[0] != corr.shape[0]) else prior.unsqueeze(-1)
        if prior.shape[-2:] != corr.shape[-2:]:
            raise ValueError(
                f"query_prior shape {tuple(query_prior.shape)} incompatible with correction {tuple(corr.shape)}."
            )
        return prior + corr


def build_gnot(in_channels: int, **kwargs) -> GNOTOperator:
    """Construct a :class:`GNOTOperator`. See the class for argument semantics."""
    return GNOTOperator(in_channels=in_channels, **kwargs)


def build_delta_gnot(in_channels: int, out_channels: int = 1, **kwargs) -> DeltaGNOT:
    """Construct a :class:`DeltaGNOT` wrapping a fresh :class:`GNOTOperator`."""
    if out_channels != 1:
        raise ValueError(f"delta_gnot predicts scalar theta; out_channels must be 1, got {out_channels}.")
    return DeltaGNOT(build_gnot(in_channels=in_channels, out_channels=1, **kwargs))
