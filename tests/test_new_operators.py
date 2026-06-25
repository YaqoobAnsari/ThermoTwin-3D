"""Contract tests for the added gridless operators (GNOT / DeepONet / PointNet++ / MeshGraphNet).

Each must honour the Block-2 point-cloud contract: a data-only operator with a 5-arg forward
``(input_geom, x, latent_queries, sdf, output_queries) -> (B, n, 1)`` and a delta variant with a
6th ``query_prior`` arg that adds the prior back. We check output shape, a clean backward, and
that the delta wrapper equals operator-correction + prior.
"""

from __future__ import annotations

import pytest
import torch

from thermotwin.models.deeponet import build_deeponet, build_delta_deeponet
from thermotwin.models.gnot import build_delta_gnot, build_gnot
from thermotwin.models.meshgraphnet import build_delta_meshgraphnet, build_meshgraphnet
from thermotwin.models.pointnet2 import build_delta_pointnet2, build_pointnet2

FAMILIES = {
    "gnot": (build_gnot, build_delta_gnot),
    "deeponet": (build_deeponet, build_delta_deeponet),
    "pointnet2": (build_pointnet2, build_delta_pointnet2),
    "meshgraphnet": (build_meshgraphnet, build_delta_meshgraphnet),
}


@pytest.fixture
def cloud():
    torch.manual_seed(0)
    n = 256
    return torch.rand(1, n, 3), torch.rand(1, n, 4), torch.rand(1, n, 3), torch.rand(1, n)


@pytest.mark.parametrize("name", list(FAMILIES))
def test_data_only_shape_and_backward(name, cloud):
    geom, f4, f3, _prior = cloud
    op = FAMILIES[name][0](in_channels=3)
    out = op(geom, f3, None, None, geom)
    assert out.shape == (1, geom.shape[1], 1)
    out.mean().backward()
    assert any(p.grad is not None for p in op.parameters())


@pytest.mark.parametrize("name", list(FAMILIES))
def test_delta_adds_prior(name, cloud):
    geom, f4, _f3, prior = cloud
    dop = FAMILIES[name][1](in_channels=4)
    dop.eval()
    # PointNet++ samples centroids randomly, so seed identically before each forward to make
    # the operator's stochastic draws match between the standalone and delta-wrapped calls.
    with torch.no_grad():
        torch.manual_seed(1)
        corr = dop.operator(geom, f4, None, None, geom)
        torch.manual_seed(1)
        full = dop(geom, f4, None, None, geom, prior)
    assert full.shape == (1, geom.shape[1], 1)
    assert torch.allclose(full, corr + prior.unsqueeze(-1), atol=1e-5)


@pytest.mark.parametrize("name", list(FAMILIES))
def test_accepts_unbatched_input(name, cloud):
    # the harness sometimes passes bare (n, c); operators add the batch dim themselves
    geom, _f4, f3, _prior = cloud
    op = FAMILIES[name][0](in_channels=3)
    out = op(geom[0], f3[0], None, None, geom[0])
    assert out.shape[-1] == 1 and out.shape[-2] == geom.shape[1]
