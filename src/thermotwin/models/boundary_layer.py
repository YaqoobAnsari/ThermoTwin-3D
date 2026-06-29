"""Boundary-layer (matched-asymptotic) residual operator with optional calibrated uncertainty.

The H1 mechanism. Instead of an *additive* correction on the analytic prior (`θ̂ = prior + Δ`,
which is the now-preempted DeltaPhi/Δ-ML design), we reframe the prior as the **outer** solution
of a matched-asymptotic expansion and the operator's correction as a learned **inner** solution
that lives in a thin layer around material/geometric interfaces:

    θ̂(x) = θ_prior(x) + w(d(x); ε) · Δ(x)

where ``d(x)`` is the distance to the nearest conductivity interface (the stretched inner
coordinate — the thermal bridge is at ``d=0``) and ``w(d; ε) = exp(-(d/ε)²)`` is a learnable
decay/matching window that forces the correction to vanish far from interfaces (``w → 0`` as
``d ≫ ε``) and act only in the bridge halo. ``ε`` (the layer width) is learned.

With ``uncertainty=True`` the operator additionally emits a per-point log-variance, trained with
a heteroscedastic Gaussian NLL, giving (a) a **calibrated reliability map** (large variance where
the prior fails = a thermal-bridge localizer) and (b) the per-point uncertainty that the inverse
twin consumes as a measurement weighting (the operator→inverse UQ hand-off).

Ablation contract (the pre-registered A/B): a plain additive delta is exactly this module with
``w ≡ 1`` (``ε → ∞``); so the *same* code, with the window enabled vs disabled, isolates whether
the boundary-layer structure earns its keep over additive residual learning.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from thermotwin.models.pointnet2 import build_pointnet2

__all__ = [
    "interface_distance",
    "boundary_window",
    "heteroscedastic_loss",
    "BoundaryLayerOperator",
    "build_bl_pointnet2",
    "build_blu_pointnet2",
]


def interface_distance(points: Tensor, logk: Tensor, *, jump: float = 0.5, max_d: float = 1.0) -> Tensor:
    """Per-point distance to the nearest *conductivity interface* (the inner coordinate).

    For each point, the distance to the nearest point whose (log-)conductivity differs by more
    than ``jump`` — i.e. the distance to the nearest material discontinuity (thermal bridge edge).
    Points with no interface within ``max_d`` (clear wall far from any bridge) get ``max_d``, so
    the decay window switches their correction off. Clamped to ``max_d``.

    Args:
        points: ``(N, 3)`` coordinates.
        logk: ``(N,)`` per-point (standardised) log-conductivity.
        jump: conductivity-difference threshold that defines an "interface".
        max_d: cap / fill for points with no interface in range.

    Returns:
        ``(N,)`` interface distance (0 at a bridge edge, growing into clear wall and bridge bulk).
    """
    pts = points.reshape(-1, points.shape[-1])
    lk = logk.reshape(-1)
    dist = torch.cdist(pts, pts)  # (N, N)
    differs = (lk.unsqueeze(0) - lk.unsqueeze(1)).abs() > jump  # (N, N) True where conductivity jumps
    dist = dist.masked_fill(~differs, float("inf"))
    d = dist.min(dim=1).values
    d = torch.where(torch.isinf(d), torch.full_like(d, max_d), d)
    return d.clamp(max=max_d)


def boundary_window(d: Tensor, eps: Tensor | float) -> Tensor:
    """Matched-asymptotic decay window ``w = exp(-(d/ε)²)`` — 1 at the interface, →0 far away."""
    return torch.exp(-((d / eps) ** 2))


def heteroscedastic_loss(pred: Tensor, log_var: Tensor, target: Tensor) -> Tensor:
    """Gaussian negative log-likelihood with learned per-point variance (Kendall & Gal 2017).

    ``0.5·exp(−s)·(pred−target)² + 0.5·s`` averaged, with ``s`` = log-variance clamped for
    stability. Lets the model express *where* it is unreliable (large ``s`` at the bridges the
    prior misses) instead of being penalised uniformly.
    """
    s = log_var.reshape_as(pred).clamp(-10.0, 10.0)
    return (0.5 * torch.exp(-s) * (pred - target) ** 2 + 0.5 * s).mean()


class BoundaryLayerOperator(nn.Module):
    """Wrap any point-operator as a boundary-layer inner correction on an analytic outer prior.

    The wrapped ``operator`` consumes the per-point features **plus the interface distance** as an
    extra channel (so it sees the stretched inner coordinate) and outputs the raw correction
    (``out_channels=1``) or correction+log-variance (``out_channels=2``). This module applies the
    learnable decay window and adds the prior back.

    Args:
        operator: a base operator with signature
            ``(input_geom, x, latent_queries, sdf, output_queries) -> (B, n, out_channels)``.
        uncertainty: if True, expects ``out_channels=2`` and returns ``(pred, log_var)``.
        eps_init: initial boundary-layer width ε.
        window: if False, disables the window (``w ≡ 1``) — the additive-residual ablation.
    """

    def __init__(self, operator: nn.Module, *, uncertainty: bool = False, eps_init: float = 0.1, window: bool = True) -> None:
        super().__init__()
        self.operator = operator
        self.uncertainty = bool(uncertainty)
        self.use_window = bool(window)
        # ε = softplus(raw_eps) + 1e-3 keeps the width strictly positive and stable.
        inv = float(torch.log(torch.expm1(torch.tensor(max(eps_init - 1e-3, 1e-3)))))
        self.raw_eps = nn.Parameter(torch.tensor(inv))

    @property
    def eps(self) -> Tensor:
        return torch.nn.functional.softplus(self.raw_eps) + 1e-3

    def forward(
        self,
        input_geom: Tensor,
        feats: Tensor,
        latent_queries: Tensor | None,
        sdf: Tensor | None,
        output_queries: Tensor | None,
        query_prior: Tensor,
        interface_d: Tensor,
    ):
        d = interface_d
        if d.dim() == 1:
            d = d.unsqueeze(0)
        d3 = d.unsqueeze(-1)  # (B, n, 1) — appended as the inner-coordinate feature
        x = torch.cat([feats, d3.to(feats.dtype)], dim=-1)
        out = self.operator(input_geom, x, latent_queries, sdf, output_queries)  # (B, n, 1|2)
        corr = out[..., :1]
        w = boundary_window(d3, self.eps) if self.use_window else torch.ones_like(d3)
        prior = query_prior.reshape(corr.shape)
        pred = prior + w * corr
        if self.uncertainty:
            return pred, out[..., 1:2]
        return pred


def build_bl_pointnet2(feat_channels: int = 4, *, k: int = 16, width: int = 128,
                       eps_init: float = 0.1, window: bool = True) -> BoundaryLayerOperator:
    """Boundary-layer PointNet++ (deterministic). ``feat_channels`` excludes the appended distance."""
    op = build_pointnet2(in_channels=feat_channels + 1, out_channels=1, k=k, width=width)
    return BoundaryLayerOperator(op, uncertainty=False, eps_init=eps_init, window=window)


def build_blu_pointnet2(feat_channels: int = 4, *, k: int = 16, width: int = 128,
                        eps_init: float = 0.1, window: bool = True) -> BoundaryLayerOperator:
    """Boundary-layer PointNet++ with a heteroscedastic uncertainty head (out_channels=2)."""
    op = build_pointnet2(in_channels=feat_channels + 1, out_channels=2, k=k, width=width)
    return BoundaryLayerOperator(op, uncertainty=True, eps_init=eps_init, window=window)
