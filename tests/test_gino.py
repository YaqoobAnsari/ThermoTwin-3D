"""GINO geometry-conditioned operator: shape contract, PINN-ready gradients, delta.

These are CPU smokes — a few hundred input points, a tiny ``G = 8`` latent grid, a
few hundred output queries — so they run in well under a second and never touch a GPU.
They lock the three properties Block-2 depends on:

* the operator maps an irregular point cloud + SDF to ``(B, n_out, 1)``;
* gradients flow through ``output_queries`` (so the same forward can drive a PDE
  residual at arbitrary collocation points — PINN-ready);
* ``delta_gino`` is *exactly* additive: its output is ``query_prior + correction``,
  so perturbing the supplied prior shifts the output by the same amount and leaves the
  network's correction untouched.
"""

from __future__ import annotations

import pytest
import torch

torch.manual_seed(1337)

from thermotwin.models.gino import (  # noqa: E402
    DeltaGino,
    GinoOperator,
    build_delta_gino,
    build_gino,
)
from thermotwin.models.registry import build_model  # noqa: E402

# Tiny problem sizes shared across tests.
G = 8  # latent grid side; modes must stay < G // 2 == 4
N_IN = 256
N_OUT = 200
FEAT = 4


def _make_inputs(batch: int = 1, n_in: int = N_IN, n_out: int = N_OUT, *, requires_grad=False):
    """Build a synthetic irregular cloud, latent grid, SDF, and queries in [0,1]^3."""
    input_geom = torch.rand(1, n_in, 3)
    x = torch.rand(batch, n_in, FEAT)
    lin = torch.linspace(0.0, 1.0, G)
    gx, gy, gz = torch.meshgrid(lin, lin, lin, indexing="ij")
    latent_queries = torch.stack([gx, gy, gz], dim=-1).unsqueeze(0)  # (1, G, G, G, 3)
    sdf = torch.rand(batch, G, G, G, 1)
    output_queries = torch.rand(1, n_out, 3, requires_grad=requires_grad)
    return input_geom, x, latent_queries, sdf, output_queries


def _build_operator() -> GinoOperator:
    return build_gino(
        in_channels=FEAT,
        out_channels=1,
        fno_in_channels=16,
        fno_n_modes=(3, 3, 3),
        fno_hidden_channels=16,
        fno_n_layers=2,
        in_gno_radius=0.2,  # generous radius for the sparse smoke cloud
        out_gno_radius=0.2,
        latent_grid=G,
        latent_feature_channels=1,
    )


def test_operator_output_shape():
    op = _build_operator()
    input_geom, x, latent_queries, sdf, output_queries = _make_inputs()
    out = op(input_geom, x, latent_queries, sdf, output_queries)
    assert out.shape == (1, N_OUT, 1)


def test_operator_batched_features():
    """Geometry is shared (leading 1); the feature batch and SDF batch drive B."""
    op = _build_operator()
    b = 2
    input_geom, x, latent_queries, sdf, output_queries = _make_inputs(batch=b)
    out = op(input_geom, x, latent_queries, sdf, output_queries)
    assert out.shape == (b, N_OUT, 1)


def test_operator_accepts_unbatched_geometry():
    """Bare (n, 3) / (G, G, G, 3) / (B, G, G, G) inputs get their leading dims added."""
    op = _build_operator()
    input_geom, x, latent_queries, sdf, output_queries = _make_inputs()
    out = op(
        input_geom.squeeze(0),  # (n_in, 3)
        x,
        latent_queries.squeeze(0),  # (G, G, G, 3)
        sdf.squeeze(-1),  # (B, G, G, G)
        output_queries.squeeze(0),  # (n_out, 3)
    )
    assert out.shape == (1, N_OUT, 1)


def test_gradient_flows_through_output_queries():
    """PINN-readiness: d(theta)/d(query coords) must exist and be non-trivial."""
    op = _build_operator()
    input_geom, x, latent_queries, sdf, output_queries = _make_inputs(requires_grad=True)
    out = op(input_geom, x, latent_queries, sdf, output_queries)
    out.sum().backward()
    assert output_queries.grad is not None
    assert torch.isfinite(output_queries.grad).all()
    assert output_queries.grad.abs().sum() > 0


def test_delta_gino_is_additive_in_prior():
    """delta_gino output == query_prior + correction; perturbing the prior shifts it."""
    torch.manual_seed(0)
    dg = build_delta_gino(
        in_channels=FEAT,
        fno_in_channels=16,
        fno_n_modes=(3, 3, 3),
        fno_hidden_channels=16,
        fno_n_layers=2,
        in_gno_radius=0.2,
        out_gno_radius=0.2,
        latent_grid=G,
    )
    assert isinstance(dg, DeltaGino)
    input_geom, x, latent_queries, sdf, output_queries = _make_inputs()

    prior = torch.rand(1, N_OUT)
    out = dg(input_geom, x, latent_queries, sdf, output_queries, prior)
    assert out.shape == (1, N_OUT, 1)

    # The bare correction equals output minus prior (same weights, deterministic).
    correction = dg.operator(input_geom, x, latent_queries, sdf, output_queries)
    assert torch.allclose(out, prior.unsqueeze(-1) + correction, atol=1e-6)

    # Shifting the prior by a constant shifts the output by the same constant, and the
    # correction is unchanged — the additive structure is exact.
    shift = 0.37
    out_shifted = dg(input_geom, x, latent_queries, sdf, output_queries, prior + shift)
    assert torch.allclose(out_shifted, out + shift, atol=1e-6)


def test_delta_gino_prior_shape_variants():
    """query_prior may be (n_out,), (n_out, 1), (B, n_out), or (B, n_out, 1)."""
    dg = build_delta_gino(
        in_channels=FEAT,
        fno_in_channels=16,
        fno_n_modes=(3, 3, 3),
        fno_hidden_channels=16,
        fno_n_layers=2,
        in_gno_radius=0.2,
        out_gno_radius=0.2,
        latent_grid=G,
    )
    input_geom, x, latent_queries, sdf, output_queries = _make_inputs()
    base = torch.rand(N_OUT)
    refs = []
    for prior in (base, base.unsqueeze(-1), base.unsqueeze(0), base.unsqueeze(0).unsqueeze(-1)):
        out = dg(input_geom, x, latent_queries, sdf, output_queries, prior)
        assert out.shape == (1, N_OUT, 1)
        refs.append(out)
    for r in refs[1:]:
        assert torch.allclose(r, refs[0], atol=1e-6)


def test_build_delta_gino_rejects_multichannel():
    with pytest.raises(ValueError, match="out_channels must be 1"):
        build_delta_gino(in_channels=FEAT, out_channels=2, latent_grid=G)


def test_mode_validation_rejects_too_high_modes():
    """Each retained mode must stay below the latent-grid Nyquist limit G // 2."""
    with pytest.raises(ValueError, match="must be <"):
        build_gino(in_channels=FEAT, fno_n_modes=(4, 4, 4), latent_grid=G)  # 4 == G // 2


def test_mode_validation_rejects_wrong_dim():
    with pytest.raises(ValueError, match="length gno_coord_dim"):
        build_gino(in_channels=FEAT, fno_n_modes=(3, 3), latent_grid=G)


@pytest.mark.parametrize("name", ["gino", "delta_gino"])
def test_grid_path_raises_with_pointer(name: str):
    """Routing a geometry model through the grid build_model must fail loudly."""
    with pytest.raises(NotImplementedError, match="build_"):
        build_model({"name": name})
