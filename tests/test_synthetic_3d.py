"""3-D synthetic conduction corpus: physics sanity, prior match, SDF, shapes."""

from __future__ import annotations

import numpy as np

from thermotwin.data.synthetic_3d import (
    FEATURE_DIM,
    BlockSample,
    Bridge3D,
    box_sdf_grid,
    build_k_field_3d,
    clear_block_u,
    generate_corpus_3d,
    point_theta1d,
    sample_block,
    solve_block,
)
from thermotwin.data.synthetic_fem import _BASE_WALLS

WALL = _BASE_WALLS["mass_insulated"]


def _block(bridges=()):
    return BlockSample(
        layers=WALL,
        width_y_m=0.6,
        width_z_m=0.6,
        t_indoor=20.0,
        t_outdoor=-5.0,
        cells_per_layer=4,
        cells_y=16,
        cells_z=16,
        bridges=bridges,
    )


def test_field_shape_and_k_values():
    s = _block()
    k, spacing = build_k_field_3d(s)
    assert k.ndim == 3
    assert k.shape[1] == 16 and k.shape[2] == 16
    assert len(spacing) == 3
    # Base k holds only the four layer conductivities of the mass wall.
    assert set(np.round(np.unique(k), 6)) == {0.72, 1.95, 0.035, 0.16}


def test_no_bridge_theta_matches_1d_prior():
    """A no-bridge block has no in-plane gradient: the sampled θ equals the analytic
    1-D prior everywhere except a thin interpolation band at layer interfaces."""
    s = _block()
    k, spacing = build_k_field_3d(s)
    res = solve_block(s)
    rng = np.random.default_rng(0)
    smp = sample_block(s, res, spacing, n_points=3000, grid=16, rng=rng)
    err = np.abs(smp["theta"] - smp["prior"])
    # θ and the prior are interpolated from grids that agree at every cell centre to
    # machine precision for a clear column, so they match everywhere (the interface
    # kink is smeared identically in both). Only float32 round-off remains.
    assert err.max() < 1e-5
    assert err.mean() < 1e-6


def test_no_bridge_field_has_no_in_plane_gradient():
    """The FV temperature of a clear block is constant across each through-wall plane."""
    s = _block()
    res = solve_block(s)
    th = (res.temperature - s.t_outdoor) / (s.t_indoor - s.t_outdoor)
    # std across the two in-plane axes, at every through-wall slice, is ~0.
    assert th.std(axis=(1, 2)).max() < 1e-6


def test_bridge_creates_in_plane_variation():
    """A finite bridge prism through the insulation makes θ vary in-plane."""
    bridge = Bridge3D(0.225, 0.325, 0.20, 0.30, 0.20, 0.30, conductivity_w_mk=50.0)
    s = _block(bridges=(bridge,))
    res = solve_block(s)
    th = (res.temperature - s.t_outdoor) / (s.t_indoor - s.t_outdoor)
    assert th.std(axis=(1, 2)).max() > 1e-3


def test_bridge_raises_u_above_clear():
    """A steel prism through the insulation raises effective U over the clear-wall U."""
    bridge = Bridge3D(0.225, 0.325, 0.20, 0.30, 0.20, 0.30, conductivity_w_mk=50.0)
    s = _block(bridges=(bridge,))
    assert solve_block(s).u_value > clear_block_u(s) * 1.02


def test_point_prior_endpoints_and_monotonicity():
    s = _block()
    k, spacing = build_k_field_3d(s)
    dx0 = np.asarray(spacing[0])
    xf = np.linspace(0.0, 1.0, 50)
    p = point_theta1d(k[:, 0, 0], dx0, xf, s.r_si, s.r_se)
    # θ=1 at the indoor (lo) face, 0 at the outdoor (hi) face, monotone decreasing.
    assert p[0] > 0.9 and p[-1] < 0.05
    assert np.all(np.diff(p) <= 1e-9)


def test_box_sdf_sign():
    """SDF < 0 strictly inside the block (grid centres), ≈0 at the faces."""
    _, spacing = build_k_field_3d(_block())
    sdf = box_sdf_grid(spacing, grid=16)
    assert sdf.shape == (16, 16, 16)
    assert sdf.max() <= 0.0 + 1e-6  # nothing is outside the unit cube
    assert sdf.min() < -0.1  # the centre is well inside
    # The cell nearest each face is the least-negative (closest to 0).
    assert sdf[0].max() > sdf[sdf.shape[0] // 2].min()


def test_sample_shapes_and_dtypes():
    s = _block()
    k, spacing = build_k_field_3d(s)
    res = solve_block(s)
    rng = np.random.default_rng(3)
    smp = sample_block(s, res, spacing, n_points=512, grid=12, rng=rng)
    assert smp["points"].shape == (512, 3)
    assert smp["feats"].shape == (512, FEATURE_DIM)
    assert smp["theta"].shape == (512,)
    assert smp["prior"].shape == (512,)
    assert smp["sdf"].shape == (12, 12, 12)
    for key in ("points", "feats", "theta", "prior", "sdf"):
        assert smp[key].dtype == np.float32
    # Normalised points live in the unit cube.
    assert smp["points"].min() >= 0.0 and smp["points"].max() <= 1.0


def test_corpus_deterministic_and_bridges_genuine():
    a = generate_corpus_3d(4, seed=1337, grid=12, n_points=256, cells_per_layer=3)
    b = generate_corpus_3d(4, seed=1337, grid=12, n_points=256, cells_per_layer=3)
    assert len(a) == 4
    for ra, rb in zip(a, b, strict=True):
        assert np.array_equal(ra["points"], rb["points"])
        assert np.array_equal(ra["theta"], rb["theta"])
        # Bridges never lower U below the clear-wall value (ADR 0006).
        assert float(ra["u_value"]) >= float(ra["u_clear"]) - 1e-4
        assert ra["feats"].shape[1] == FEATURE_DIM
