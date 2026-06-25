"""PointNet++ — the point-based baseline, under the Block-2 contract.

PointNet++ (Qi et al., NeurIPS 2017) is the canonical hierarchical point-cloud network: **set
abstraction** layers downsample and aggregate local neighbourhoods (group k-NN, shared MLP,
max-pool), and **feature propagation** layers interpolate the learned features back to every point
for dense per-point prediction. It represents the point-segmentation family in the bake-off — a
geometry-aware architecture that is *not* an operator, so it tests whether plain hierarchical point
features rival the physics operators on these conduction fields. Self-contained pure ``torch``
(random-sampling SSG variant; k-NN via ``cdist`` is fine for the ~2k-point clouds here). Same
contract as :mod:`thermotwin.models.transolver`:

* :class:`PointNet2Operator` — ``forward(input_geom, x, latent_queries, sdf, output_queries) ->
  (B, n, 1)``.
* :class:`DeltaPointNet2` — predicts the correction on the analytic 1-D clear-wall prior.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

__all__ = ["PointNet2Operator", "DeltaPointNet2", "build_pointnet2", "build_delta_pointnet2"]


def _knn(query: Tensor, points: Tensor, k: int) -> tuple[Tensor, Tensor]:
    """k nearest ``points`` for each ``query`` -> ``(idx (b,m,k), dist (b,m,k))``."""
    d = torch.cdist(query, points)  # (b, m, n)
    dist, idx = d.topk(min(k, points.shape[1]), dim=-1, largest=False)
    return idx, dist


def _gather(feats: Tensor, idx: Tensor) -> Tensor:
    """Gather ``feats (b,n,c)`` at ``idx (b,m,k)`` -> ``(b,m,k,c)``."""
    b, _, c = feats.shape
    m, k = idx.shape[1], idx.shape[2]
    flat = idx.reshape(b, m * k, 1).expand(-1, -1, c)
    return torch.gather(feats, 1, flat).reshape(b, m, k, c)


def _shared_mlp(n_in: int, dims: tuple[int, ...]) -> nn.Sequential:
    layers: list[nn.Module] = []
    c = n_in
    for d in dims:
        layers += [nn.Linear(c, d), nn.GELU()]
        c = d
    return nn.Sequential(*layers)


class _SetAbstraction(nn.Module):
    """Sample centroids, group their k-NN, shared-MLP + max-pool to centroid features."""

    def __init__(self, in_c: int, dims: tuple[int, ...], k: int, ratio: float) -> None:
        super().__init__()
        self.k = k
        self.ratio = ratio
        self.mlp = _shared_mlp(in_c + 3, dims)

    def forward(self, xyz: Tensor, feats: Tensor) -> tuple[Tensor, Tensor]:
        b, n, _ = xyz.shape
        m = max(1, int(n * self.ratio))
        # Random subsampling of centroids (cheaper than FPS; fine for a dense uniform cloud).
        sel = torch.stack([torch.randperm(n, device=xyz.device)[:m] for _ in range(b)])  # (b, m)
        new_xyz = torch.gather(xyz, 1, sel.unsqueeze(-1).expand(-1, -1, 3))
        idx, _ = _knn(new_xyz, xyz, self.k)
        grouped_xyz = _gather(xyz, idx) - new_xyz.unsqueeze(2)  # local coords
        grouped = torch.cat([grouped_xyz, _gather(feats, idx)], dim=-1)
        out = self.mlp(grouped).max(dim=2).values  # (b, m, dims[-1])
        return new_xyz, out


class _FeaturePropagation(nn.Module):
    """Interpolate coarse features to finer points (3-NN inverse-distance) + skip MLP."""

    def __init__(self, in_c: int, dims: tuple[int, ...]) -> None:
        super().__init__()
        self.mlp = _shared_mlp(in_c, dims)

    def forward(self, xyz_fine: Tensor, xyz_coarse: Tensor, feats_fine: Tensor | None, feats_coarse: Tensor) -> Tensor:
        idx, dist = _knn(xyz_fine, xyz_coarse, 3)
        w = 1.0 / (dist + 1e-8)
        w = w / w.sum(dim=-1, keepdim=True)  # (b, n, 3)
        interp = (_gather(feats_coarse, idx) * w.unsqueeze(-1)).sum(dim=2)  # (b, n, c)
        if feats_fine is not None:
            interp = torch.cat([interp, feats_fine], dim=-1)
        return self.mlp(interp)


class PointNet2Operator(nn.Module):
    """PointNet++ (2-level SSG) under the Block-2 point-cloud contract."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 1,
        k: int = 16,
        width: int = 128,
    ) -> None:
        super().__init__()
        self.sa1 = _SetAbstraction(int(in_channels), (width // 2, width), k=k, ratio=0.25)
        self.sa2 = _SetAbstraction(width, (width, width * 2), k=k, ratio=0.25)
        self.fp2 = _FeaturePropagation(width * 2 + width, (width, width))
        self.fp1 = _FeaturePropagation(width + int(in_channels), (width, width))
        self.head = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, int(out_channels)))

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
        xyz0 = self._bn_(input_geom)
        f0 = self._bn_(x)
        xyz1, f1 = self.sa1(xyz0, f0)
        xyz2, f2 = self.sa2(xyz1, f1)
        f1u = self.fp2(xyz1, xyz2, f1, f2)
        f0u = self.fp1(xyz0, xyz1, f0, f1u)
        return self.head(f0u)  # (b, n, out)


class DeltaPointNet2(nn.Module):
    """PointNet++ that predicts a correction added to the analytic 1-D theta prior."""

    def __init__(self, operator: PointNet2Operator) -> None:
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
        corr = self.operator(input_geom, x, latent_queries, sdf, output_queries)
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


def build_pointnet2(in_channels: int, **kwargs) -> PointNet2Operator:
    """Construct a :class:`PointNet2Operator`. See the class for argument semantics."""
    return PointNet2Operator(in_channels=in_channels, **kwargs)


def build_delta_pointnet2(in_channels: int, out_channels: int = 1, **kwargs) -> DeltaPointNet2:
    """Construct a :class:`DeltaPointNet2` wrapping a fresh :class:`PointNet2Operator`."""
    if out_channels != 1:
        raise ValueError(f"delta_pointnet2 predicts scalar theta; out_channels must be 1, got {out_channels}.")
    return DeltaPointNet2(build_pointnet2(in_channels=in_channels, out_channels=1, **kwargs))
