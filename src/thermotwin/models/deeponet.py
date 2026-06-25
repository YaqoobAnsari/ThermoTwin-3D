"""DeepONet — the canonical operator-learning baseline, under the Block-2 contract.

DeepONet (Lu et al., Nat. Mach. Intell. 2021) is the reference neural operator a reviewer expects
to see: a **branch** net encodes the input function into ``p`` coefficients, a **trunk** net
encodes the query location into ``p`` basis functions, and the prediction is their inner product —
``u(y) = Σ_i b_i(input) · t_i(y) + b₀``. It is included as the honest low-bias baseline: a global
branch encoding deliberately discards local geometric detail, so it tests how far a *non*-geometry-
resolved operator gets on these fields. Wrapped in the same point-cloud contract as
:mod:`thermotwin.models.transolver`:

* :class:`DeepONetOperator` — branch = mean-pooled encoding of the per-point input features
  (the discretised input function); trunk = per-query coordinate encoding;
  ``forward(input_geom, x, latent_queries, sdf, output_queries) -> (B, n, 1)``.
* :class:`DeltaDeepONet` — predicts the correction on the analytic 1-D clear-wall prior.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

__all__ = ["DeepONetOperator", "DeltaDeepONet", "build_deeponet", "build_delta_deeponet"]


def _mlp(n_in: int, hidden: int, n_out: int, depth: int = 3) -> nn.Sequential:
    layers: list[nn.Module] = [nn.Linear(n_in, hidden), nn.GELU()]
    for _ in range(depth - 1):
        layers += [nn.Linear(hidden, hidden), nn.GELU()]
    layers += [nn.Linear(hidden, n_out)]
    return nn.Sequential(*layers)


class DeepONetOperator(nn.Module):
    """DeepONet under the Block-2 point-cloud contract.

    Args:
        in_channels: per-point feature width (the input function; 3 data-only / 4 with prior).
        out_channels: output channels (1 = theta).
        space_dim: query coordinate dimensionality (3).
        p: number of branch/trunk basis coefficients.
        hidden / depth: branch/trunk MLP capacity.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 1,
        space_dim: int = 3,
        p: int = 128,
        hidden: int = 256,
        depth: int = 4,
    ) -> None:
        super().__init__()
        self.out_channels = int(out_channels)
        self.p = p
        # Branch sees the input function at each point (coords + feats), then pools to a global code.
        self.branch = _mlp(space_dim + int(in_channels), hidden, p * self.out_channels, depth)
        self.trunk = _mlp(space_dim, hidden, p, depth)
        self.bias = nn.Parameter(torch.zeros(self.out_channels))

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
        coords = self._bn_(input_geom)  # (b, n, 3)
        feats = self._bn_(x)  # (b, n, F)
        b, n, _ = coords.shape
        branch = self.branch(torch.cat((coords, feats), dim=-1)).mean(dim=1)  # (b, p*out) global code
        branch = branch.reshape(b, self.out_channels, self.p)
        trunk = self.trunk(coords)  # (b, n, p)
        out = torch.einsum("bop,bnp->bno", branch, trunk) + self.bias  # (b, n, out)
        return out


class DeltaDeepONet(nn.Module):
    """DeepONet that predicts a correction added to the analytic 1-D theta prior."""

    def __init__(self, operator: DeepONetOperator) -> None:
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


def build_deeponet(in_channels: int, **kwargs) -> DeepONetOperator:
    """Construct a :class:`DeepONetOperator`. See the class for argument semantics."""
    return DeepONetOperator(in_channels=in_channels, **kwargs)


def build_delta_deeponet(in_channels: int, out_channels: int = 1, **kwargs) -> DeltaDeepONet:
    """Construct a :class:`DeltaDeepONet` wrapping a fresh :class:`DeepONetOperator`."""
    if out_channels != 1:
        raise ValueError(f"delta_deeponet predicts scalar theta; out_channels must be 1, got {out_channels}.")
    return DeltaDeepONet(build_deeponet(in_channels=in_channels, out_channels=1, **kwargs))
