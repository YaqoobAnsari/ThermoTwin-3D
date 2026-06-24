"""Irregular (rotated, off-lattice) 3-D corpus: off-grid-ness, prior match, U, loading."""

from __future__ import annotations

import json

import numpy as np

from thermotwin.data.synthetic_3d import (
    FEATURE_DIM,
    BlockSample,
    Bridge3D,
    build_k_field_3d,
    clear_block_u,
    solve_block,
)
from thermotwin.data.synthetic_3d_irreg import (
    generate_corpus_irregular,
    halton_unit_cube,
    random_rotation,
    rotated_box_sdf_grid,
    sample_block_irregular,
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


def test_random_rotation_is_proper_orthogonal():
    rng = np.random.default_rng(0)
    for _ in range(5):
        r = random_rotation(rng)
        assert r.shape == (3, 3)
        np.testing.assert_allclose(r @ r.T, np.eye(3), atol=1e-10)
        assert abs(np.linalg.det(r) - 1.0) < 1e-10


def test_halton_cloud_is_off_lattice():
    """A genuinely irregular cloud: no axis has its coordinates lying on a few planes."""
    rng = np.random.default_rng(1)
    pts = halton_unit_cube(2048, rng)
    assert pts.shape == (2048, 3)
    assert pts.min() >= 0.0 and pts.max() <= 1.0
    # On a lattice each axis takes only ~N^(1/3) distinct values; an off-grid cloud
    # takes nearly N distinct values per axis (Halton + random shift => all distinct).
    for a in range(3):
        n_unique = len(np.unique(np.round(pts[:, a], 6)))
        assert n_unique > 0.9 * len(pts)


def test_sample_points_are_off_grid():
    """The stored (rotated) cloud is not lattice-structured along any axis."""
    s = _block()
    k, spacing = build_k_field_3d(s)
    res = solve_block(s)
    rng = np.random.default_rng(2)
    smp = sample_block_irregular(s, res, spacing, n_points=1500, grid=12, rng=rng)
    pts = smp["points"]
    # No coordinate plane is shared by many points: each axis has ~all-distinct values.
    for a in range(3):
        n_unique = len(np.unique(np.round(pts[:, a], 5)))
        assert n_unique > 0.9 * len(pts)
    # And the cloud is genuinely rotated: the per-axis principal directions are not the
    # canonical basis (a lattice-aligned cloud would have ~axis-aligned principal axes).
    centred = pts - pts.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centred, full_matrices=False)
    # The dominant singular vector should not be (nearly) a coordinate axis.
    assert np.max(np.abs(vh[0])) < 0.999


def test_no_bridge_theta_matches_prior():
    """A clear block's sampled θ equals its per-point 1-D prior (rotation moves only the
    stored coordinates, not the physics)."""
    s = _block()
    k, spacing = build_k_field_3d(s)
    res = solve_block(s)
    rng = np.random.default_rng(3)
    smp = sample_block_irregular(s, res, spacing, n_points=3000, grid=16, rng=rng)
    err = np.abs(smp["theta"] - smp["prior"])
    assert err.max() < 1e-5
    # prior channel inside feats matches the standalone prior array.
    np.testing.assert_array_equal(smp["feats"][:, 3], smp["prior"])


def test_bridge_raises_u_above_clear():
    bridge = Bridge3D(0.225, 0.325, 0.20, 0.30, 0.20, 0.30, conductivity_w_mk=50.0)
    s = _block(bridges=(bridge,))
    assert solve_block(s).u_value > clear_block_u(s) * 1.02


def test_rotated_sdf_sign_and_off_axis():
    """Rotated-box SDF is negative inside, positive at protruding corners — the
    axis-aligned-grid mismatch that motivates GINO."""
    rng = np.random.default_rng(4)
    rot = random_rotation(rng)
    sdf = rotated_box_sdf_grid(rot, grid=20)
    assert sdf.shape == (20, 20, 20)
    assert sdf.min() < 0.0  # interior of the tilted box
    # A rotated unit box poking out of the unit lattice leaves some grid cells OUTSIDE
    # the solid (sdf > 0) — exactly what an axis-aligned box SDF never has.
    assert sdf.max() > 0.0
    # Identity rotation reduces to the axis-aligned box: nothing outside the cube.
    sdf_id = rotated_box_sdf_grid(np.eye(3), grid=20)
    assert sdf_id.max() <= 1e-6


def test_sample_shapes_and_dtypes():
    s = _block()
    k, spacing = build_k_field_3d(s)
    res = solve_block(s)
    rng = np.random.default_rng(5)
    smp = sample_block_irregular(s, res, spacing, n_points=512, grid=12, rng=rng)
    assert smp["points"].shape == (512, 3)
    assert smp["feats"].shape == (512, FEATURE_DIM)
    assert smp["theta"].shape == (512,)
    assert smp["prior"].shape == (512,)
    assert smp["sdf"].shape == (12, 12, 12)
    assert smp["rotation"].shape == (3, 3)
    for key in ("points", "feats", "theta", "prior", "sdf", "rotation"):
        assert smp[key].dtype == np.float32


def test_corpus_deterministic_and_bridges_genuine():
    a = generate_corpus_irregular(4, seed=1337, grid=12, n_points=256, cells_per_layer=3)
    b = generate_corpus_irregular(4, seed=1337, grid=12, n_points=256, cells_per_layer=3)
    assert len(a) == 4
    for ra, rb in zip(a, b, strict=True):
        assert np.array_equal(ra["points"], rb["points"])
        assert np.array_equal(ra["theta"], rb["theta"])
        assert float(ra["u_value"]) >= float(ra["u_clear"]) - 1e-4
        assert ra["feats"].shape[1] == FEATURE_DIM
        assert "rotation" in ra


def test_loads_through_pointcloud_dataset(tmp_path):
    """The irregular corpus round-trips through the existing PointCloudDataset +
    collate — same keys/shapes as the box corpus, so the benchmark is unchanged."""
    from thermotwin.data.pointcloud_dataset import (
        FEATURE_DIM as DS_FEATURE_DIM,
    )
    from thermotwin.data.pointcloud_dataset import (
        PointCloudDataset,
        collate_pointcloud,
    )

    records = generate_corpus_irregular(3, seed=7, grid=10, n_points=200, cells_per_layer=2)
    arrays = ("points", "feats", "theta", "prior", "sdf", "grid_shape", "rotation")
    scalars = ("u_value", "u_clear", "heat_flux", "t_indoor", "t_outdoor", "r_si", "r_se")
    rows = []
    for r in records:
        fname = f"sample_{r['id']:05d}.npz"
        np.savez_compressed(
            tmp_path / fname,
            **{k: r[k] for k in arrays},
            **{s: r[s] for s in scalars},
        )
        rows.append({"file": fname})
    (tmp_path / "manifest.json").write_text(json.dumps({"samples": rows}))

    ds = PointCloudDataset(tmp_path, voxelise=True, voxel_grid=10)
    assert len(ds) == 3
    item = ds[0]
    assert item["input_geom"].shape == (200, 3)
    assert item["feats"].shape == (200, DS_FEATURE_DIM)
    assert item["gino_feats"].shape == (200, DS_FEATURE_DIM - 1)
    assert item["sdf"].shape == (10, 10, 10)
    assert item["voxel_feats"].shape == (DS_FEATURE_DIM, 10, 10, 10)
    assert item["voxel_theta"].shape == (10, 10, 10)

    batch = collate_pointcloud([item])
    assert batch["input_geom"].shape == (1, 200, 3)
    assert batch["feats"].shape == (1, 200, DS_FEATURE_DIM)
    assert batch["theta"].shape == (1, 200)
