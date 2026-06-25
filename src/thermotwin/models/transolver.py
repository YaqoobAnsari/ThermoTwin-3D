"""Transolver — a gridless physics-attention operator for irregular geometry.

The Block-2 null result (Exp 2.2) was diagnosed as an *equal-resolution* artefact:
GINO encodes the cloud onto a regular latent grid and runs an FNO there, so on
box-like geometry it inherits the *same* spectral resolution ceiling as the voxel-FNO
baseline and can only add encode/decode error on top. Transolver (Wu et al., ICML
2024, arXiv:2402.02366) removes the grid entirely: it learns a soft assignment of
each point to ``slice_num`` *physics slices* (points in the same physical state — a
thermal bridge, a material interior, a corner — cluster into one token), runs
attention over the ``M ≪ N`` slice tokens (linear cost), then "deslices" back to the
points. Its ablation shows learned slices beat a fixed grid *even on gridded data*,
which is exactly why it is the strongest candidate to break our grid-bottleneck null.

This module vendors the irregular-mesh Transolver core (``Physics_Attention_Irregular_Mesh``
+ the transformer stack from ``vendored/Transolver``) self-contained — pure ``torch`` +
``einops``, no ``timm``/``Time_Input``/``unified_pos``/CUDA-grid paths — and wraps it in
our Block-2 point-cloud contract:

* :class:`TransolverOperator` — ``forward(input_geom, x, latent_queries, sdf,
  output_queries) -> (B, n, 1)``. Geometry enters through the point coordinates and the
  learned physics-slices; ``latent_queries`` / ``sdf`` / ``output_queries`` are accepted
  for drop-in compatibility with the GINO contract but unused (Transolver has no latent
  grid, and ``output_queries == input_geom`` for v1). Dropping the SDF is an honest
  architectural difference to report, not a bug.
* :class:`DeltaTransolver` — predicts the correction and adds the per-query analytic
  1-D clear-wall prior back, mirroring :class:`~thermotwin.models.gino.DeltaGino`.
"""

from __future__ import annotations

import torch
from einops import rearrange
from torch import Tensor, nn
from torch.nn.init import trunc_normal_

__all__ = [
    "TransolverOperator",
    "DeltaTransolver",
    "build_transolver",
    "build_delta_transolver",
]


class _PhysicsAttentionIrregular(nn.Module):
    """Physics-Attention for irregular meshes/point clouds (1-D/2-D/3-D).

    Verbatim port of ``Physics_Attention_Irregular_Mesh`` from the official Transolver
    (vendored), which is pure ``torch`` + ``einops``. Softmax-assigns each of ``N``
    points to ``slice_num`` slices, attends over the slice tokens, then deslices back.
    """

    def __init__(
        self, dim: int, heads: int = 8, dim_head: int = 64, dropout: float = 0.0, slice_num: int = 64
    ) -> None:
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head**-0.5
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.temperature = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)

        self.in_project_x = nn.Linear(dim, inner_dim)
        self.in_project_fx = nn.Linear(dim, inner_dim)
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        torch.nn.init.orthogonal_(self.in_project_slice.weight)  # principled init
        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x: Tensor) -> Tensor:
        b, n, _ = x.shape
        # (1) Slice: soft-assign each point to slice tokens.
        fx_mid = (
            self.in_project_fx(x).reshape(b, n, self.heads, self.dim_head).permute(0, 2, 1, 3).contiguous()
        )
        x_mid = (
            self.in_project_x(x).reshape(b, n, self.heads, self.dim_head).permute(0, 2, 1, 3).contiguous()
        )
        slice_weights = self.softmax(self.in_project_slice(x_mid) / self.temperature)  # B H N G
        slice_norm = slice_weights.sum(2)  # B H G
        slice_token = torch.einsum("bhnc,bhng->bhgc", fx_mid, slice_weights)
        slice_token = slice_token / ((slice_norm + 1e-5)[:, :, :, None].repeat(1, 1, 1, self.dim_head))
        # (2) Attention among slice tokens.
        q = self.to_q(slice_token)
        k = self.to_k(slice_token)
        v = self.to_v(slice_token)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.dropout(self.softmax(dots))
        out_slice_token = torch.matmul(attn, v)  # B H G D
        # (3) Deslice back to points.
        out_x = torch.einsum("bhgc,bhng->bhnc", out_slice_token, slice_weights)
        out_x = rearrange(out_x, "b h n d -> b n (h d)")
        return self.to_out(out_x)


class _MLP(nn.Module):
    """Residual MLP (GELU), matching the vendored Transolver MLP with ``n_layers=0``."""

    def __init__(self, n_in: int, n_hidden: int, n_out: int, n_layers: int = 0) -> None:
        super().__init__()
        self.linear_pre = nn.Sequential(nn.Linear(n_in, n_hidden), nn.GELU())
        self.linear_post = nn.Linear(n_hidden, n_out)
        self.linears = nn.ModuleList(
            nn.Sequential(nn.Linear(n_hidden, n_hidden), nn.GELU()) for _ in range(n_layers)
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.linear_pre(x)
        for layer in self.linears:
            x = layer(x) + x
        return self.linear_post(x)


class _TransolverBlock(nn.Module):
    """One Transolver block: physics-attention + MLP, both residual; optional head."""

    def __init__(
        self,
        n_head: int,
        hidden_dim: int,
        dropout: float,
        mlp_ratio: int = 4,
        last_layer: bool = False,
        out_dim: int = 1,
        slice_num: int = 64,
    ) -> None:
        super().__init__()
        self.last_layer = last_layer
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.attn = _PhysicsAttentionIrregular(
            hidden_dim, heads=n_head, dim_head=hidden_dim // n_head, dropout=dropout, slice_num=slice_num
        )
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = _MLP(hidden_dim, hidden_dim * mlp_ratio, hidden_dim, n_layers=0)
        if last_layer:
            self.ln_3 = nn.LayerNorm(hidden_dim)
            self.head = nn.Linear(hidden_dim, out_dim)

    def forward(self, fx: Tensor) -> Tensor:
        fx = self.attn(self.ln_1(fx)) + fx
        fx = self.mlp(self.ln_2(fx)) + fx
        if self.last_layer:
            return self.head(self.ln_3(fx))
        return fx


class _TransolverNet(nn.Module):
    """The irregular-mesh Transolver (Time_Input / unified_pos paths stripped).

    ``forward(coords (B,N,space_dim), feats (B,N,fun_dim)) -> (B, N, out_dim)``.
    """

    def __init__(
        self,
        space_dim: int = 3,
        fun_dim: int = 4,
        out_dim: int = 1,
        n_layers: int = 8,
        n_hidden: int = 128,
        n_head: int = 8,
        dropout: float = 0.0,
        mlp_ratio: int = 2,
        slice_num: int = 64,
    ) -> None:
        super().__init__()
        self.preprocess = _MLP(fun_dim + space_dim, n_hidden * 2, n_hidden, n_layers=0)
        self.blocks = nn.ModuleList(
            _TransolverBlock(
                n_head=n_head,
                hidden_dim=n_hidden,
                dropout=dropout,
                mlp_ratio=mlp_ratio,
                out_dim=out_dim,
                slice_num=slice_num,
                last_layer=(i == n_layers - 1),
            )
            for i in range(n_layers)
        )
        self.placeholder = nn.Parameter((1 / n_hidden) * torch.rand(n_hidden, dtype=torch.float))
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

    def forward(self, coords: Tensor, feats: Tensor) -> Tensor:
        fx = self.preprocess(torch.cat((coords, feats), dim=-1))
        fx = fx + self.placeholder[None, None, :]
        for block in self.blocks:
            fx = block(fx)
        return fx


class TransolverOperator(nn.Module):
    """Transolver under the Block-2 point-cloud contract.

    Args:
        in_channels: per-point feature width fed as ``fx`` (e.g. 3 for data-only
            ``[logk_std, r_si, r_se]`` or 4 with the ``theta1d`` prior channel).
        out_channels: output channels (1 = theta).
        space_dim: coordinate dimensionality (3).
        n_layers / n_hidden / n_head / slice_num / mlp_ratio: Transolver capacity.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 1,
        space_dim: int = 3,
        n_layers: int = 8,
        n_hidden: int = 128,
        n_head: int = 8,
        slice_num: int = 64,
        mlp_ratio: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.net = _TransolverNet(
            space_dim=space_dim,
            fun_dim=int(in_channels),
            out_dim=int(out_channels),
            n_layers=n_layers,
            n_hidden=n_hidden,
            n_head=n_head,
            dropout=dropout,
            mlp_ratio=mlp_ratio,
            slice_num=slice_num,
        )

    @staticmethod
    def _bn_(t: Tensor) -> Tensor:
        """Add the leading batch dim if a bare ``(n, c)`` tensor is passed."""
        return t.unsqueeze(0) if t.dim() == 2 else t

    def forward(
        self,
        input_geom: Tensor,
        x: Tensor,
        latent_queries: Tensor | None = None,
        sdf: Tensor | None = None,
        output_queries: Tensor | None = None,
    ) -> Tensor:
        """Predict theta (or its correction) at the input points.

        ``latent_queries`` / ``sdf`` / ``output_queries`` are accepted for contract
        compatibility with GINO but unused — Transolver is gridless and predicts at the
        input points (v1 sets ``output_queries == input_geom``).
        """
        coords = self._bn_(input_geom)  # (1, n, 3)
        feats = self._bn_(x)  # (1, n, F)
        return self.net(coords, feats)  # (1, n, out)


class DeltaTransolver(nn.Module):
    """Transolver that predicts a correction added to the analytic 1-D theta prior.

    Mirrors :class:`~thermotwin.models.gino.DeltaGino`: the smooth bulk is the
    closed-form 1-D clear-wall prior (supplied per output query by the dataset) and the
    operator spends its capacity only on the bridge-driven residual.
    """

    def __init__(self, operator: TransolverOperator) -> None:
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
        if prior.dim() == 1:  # (n,)
            prior = prior.unsqueeze(0).unsqueeze(-1)
        elif prior.dim() == 2:  # (B, n) or (n, 1)
            if prior.shape[-1] == 1 and prior.shape[0] != corr.shape[0]:
                prior = prior.unsqueeze(0)
            else:
                prior = prior.unsqueeze(-1)
        if prior.shape[-2:] != corr.shape[-2:]:
            raise ValueError(
                f"query_prior shape {tuple(query_prior.shape)} incompatible with "
                f"correction shape {tuple(corr.shape)}."
            )
        return prior + corr


def build_transolver(in_channels: int, **kwargs) -> TransolverOperator:
    """Construct a :class:`TransolverOperator`. See the class for argument semantics."""
    return TransolverOperator(in_channels=in_channels, **kwargs)


def build_delta_transolver(in_channels: int, out_channels: int = 1, **kwargs) -> DeltaTransolver:
    """Construct a :class:`DeltaTransolver` wrapping a fresh :class:`TransolverOperator`."""
    if out_channels != 1:
        raise ValueError(
            f"delta_transolver predicts scalar theta; out_channels must be 1, got {out_channels}."
        )
    return DeltaTransolver(build_transolver(in_channels=in_channels, out_channels=1, **kwargs))
