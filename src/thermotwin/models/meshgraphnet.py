"""MeshGraphNet — the graph-neural-network baseline, under the Block-2 contract.

MeshGraphNet (Pfaff et al., ICLR 2021) is the reference GNN for physics on irregular meshes:
encode nodes and (relative-displacement) edges, run ``n_steps`` of message passing where each edge
is updated from its endpoints and each node from its aggregated incident edges, then decode per
node. It represents the message-passing / GNN family in the bake-off — local, permutation-
equivariant, geometry-aware, but with a fixed receptive field that grows only with depth (unlike
the global attention of GNOT/Transolver). Here the mesh is a symmetric **k-NN graph** over the
point cloud (no connectivity is given). Self-contained pure ``torch`` (scatter via ``index_add``).
Same contract as :mod:`thermotwin.models.transolver`:

* :class:`MeshGraphNetOperator` — ``forward(input_geom, x, latent_queries, sdf, output_queries) ->
  (B, n, 1)``.
* :class:`DeltaMeshGraphNet` — predicts the correction on the analytic 1-D clear-wall prior.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

__all__ = [
    "MeshGraphNetOperator",
    "DeltaMeshGraphNet",
    "build_meshgraphnet",
    "build_delta_meshgraphnet",
]


def _mlp(n_in: int, hidden: int, n_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(n_in, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(),
        nn.Linear(hidden, n_out), nn.LayerNorm(n_out),
    )


def _knn_edges(xyz: Tensor, k: int) -> Tensor:
    """Directed k-NN edge index ``(2, E)`` for a single cloud ``xyz (n, 3)`` (excludes self)."""
    d = torch.cdist(xyz, xyz)  # (n, n)
    idx = d.topk(k + 1, dim=-1, largest=False).indices[:, 1:]  # drop self
    n = xyz.shape[0]
    dst = torch.arange(n, device=xyz.device).unsqueeze(1).expand(-1, k).reshape(-1)
    src = idx.reshape(-1)
    return torch.stack([src, dst], dim=0)  # (2, E)


class _GraphBlock(nn.Module):
    """One MeshGraphNet message-passing step: edge update then node update, both residual."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.edge_mlp = _mlp(3 * width, width, width)
        self.node_mlp = _mlp(2 * width, width, width)

    def forward(self, node: Tensor, edge: Tensor, ei: Tensor) -> tuple[Tensor, Tensor]:
        src, dst = ei[0], ei[1]
        edge_new = edge + self.edge_mlp(torch.cat([node[src], node[dst], edge], dim=-1))
        agg = torch.zeros_like(node).index_add_(0, dst, edge_new)
        node_new = node + self.node_mlp(torch.cat([node, agg], dim=-1))
        return node_new, edge_new


class MeshGraphNetOperator(nn.Module):
    """MeshGraphNet over a k-NN graph, under the Block-2 point-cloud contract."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 1,
        k: int = 12,
        width: int = 128,
        n_steps: int = 8,
    ) -> None:
        super().__init__()
        self.k = k
        self.node_enc = _mlp(3 + int(in_channels), width, width)
        self.edge_enc = _mlp(4, width, width)  # relative xyz (3) + distance (1)
        self.blocks = nn.ModuleList(_GraphBlock(width) for _ in range(n_steps))
        self.decoder = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, int(out_channels)))

    @staticmethod
    def _dn_(t: Tensor) -> Tensor:
        """Drop a leading batch dim of 1 -> ``(n, c)`` (this op processes one cloud)."""
        return t.squeeze(0) if t.dim() == 3 else t

    def _forward_single(self, xyz: Tensor, feats: Tensor) -> Tensor:
        ei = _knn_edges(xyz, self.k)
        src, dst = ei[0], ei[1]
        rel = xyz[src] - xyz[dst]
        edge = self.edge_enc(torch.cat([rel, rel.norm(dim=-1, keepdim=True)], dim=-1))
        node = self.node_enc(torch.cat([xyz, feats], dim=-1))
        for block in self.blocks:
            node, edge = block(node, edge, ei)
        return self.decoder(node)  # (n, out)

    def forward(
        self,
        input_geom: Tensor,
        x: Tensor,
        latent_queries: Tensor | None = None,
        sdf: Tensor | None = None,
        output_queries: Tensor | None = None,
    ) -> Tensor:
        xyz = self._dn_(input_geom)  # (n, 3)
        feats = self._dn_(x)  # (n, F)
        out = self._forward_single(xyz, feats)  # (n, out)
        return out.unsqueeze(0)  # (1, n, out)


class DeltaMeshGraphNet(nn.Module):
    """MeshGraphNet that predicts a correction added to the analytic 1-D theta prior."""

    def __init__(self, operator: MeshGraphNetOperator) -> None:
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


def build_meshgraphnet(in_channels: int, **kwargs) -> MeshGraphNetOperator:
    """Construct a :class:`MeshGraphNetOperator`. See the class for argument semantics."""
    return MeshGraphNetOperator(in_channels=in_channels, **kwargs)


def build_delta_meshgraphnet(in_channels: int, out_channels: int = 1, **kwargs) -> DeltaMeshGraphNet:
    """Construct a :class:`DeltaMeshGraphNet` wrapping a fresh :class:`MeshGraphNetOperator`."""
    if out_channels != 1:
        raise ValueError(f"delta_meshgraphnet predicts scalar theta; out_channels must be 1, got {out_channels}.")
    return DeltaMeshGraphNet(build_meshgraphnet(in_channels=in_channels, out_channels=1, **kwargs))
