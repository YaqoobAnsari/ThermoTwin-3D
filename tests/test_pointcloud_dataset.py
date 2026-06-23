"""Block-2 point-cloud dataset: shapes, prior-channel handling, voxelisation, collate.

These are fast CPU checks over the real ``data/processed/block2_*`` corpus (a single
sample is enough), plus a pure-numpy check of the indoor-face U-value estimator on the
ground-truth field (it must recover the stored U-value to within the estimator's known
bridge-driven slack and be **exact** on a clear column).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from thermotwin.data.pointcloud_dataset import (
    FEATURE_DIM,
    PRIOR_CHANNEL,
    PointCloudDataset,
    collate_pointcloud,
    latent_grid_coords,
    voxelise_sample,
)
from thermotwin.eval.building import u_from_indoor_face_cloud

_REPO = Path(__file__).resolve().parents[1]
_TRAIN = _REPO / "data" / "processed" / "block2_train"
_HAVE_CORPUS = (_TRAIN / "manifest.json").exists()
pytestmark = pytest.mark.skipif(not _HAVE_CORPUS, reason="block2 corpus not generated")

G = 16


def test_latent_grid_coords_shape_and_range():
    q = latent_grid_coords(G)
    assert q.shape == (G, G, G, 3)
    # cell centres live strictly inside the unit cube
    assert float(q.min()) > 0.0 and float(q.max()) < 1.0


def test_item_keys_and_shapes():
    ds = PointCloudDataset(_TRAIN, latent_grid=G)
    item = ds[0]
    n = item["input_geom"].shape[0]
    assert item["input_geom"].shape == (n, 3)
    assert item["feats"].shape == (n, FEATURE_DIM)
    # the data-only gino view has the prior channel removed
    assert item["gino_feats"].shape == (n, FEATURE_DIM - 1)
    assert item["sdf"].shape == (G, G, G)
    assert item["latent_queries"].shape == (G, G, G, 3)
    assert item["output_queries"].shape == (n, 3)
    assert item["theta"].shape == (n,) and item["prior"].shape == (n,)
    assert item["u_value"].ndim == 0


def test_gino_feats_drops_prior_channel():
    ds = PointCloudDataset(_TRAIN, latent_grid=G)
    item = ds[0]
    full = item["feats"].numpy()
    kept = np.delete(full, PRIOR_CHANNEL, axis=1)
    assert np.allclose(item["gino_feats"].numpy(), kept)
    # the dropped channel is exactly the per-point prior
    assert np.allclose(full[:, PRIOR_CHANNEL], item["prior"].numpy())


def test_output_queries_equal_input_points_v1():
    ds = PointCloudDataset(_TRAIN, latent_grid=G)
    item = ds[0]
    assert torch.allclose(item["output_queries"], item["input_geom"])


def test_collate_single_adds_leading_dims():
    ds = PointCloudDataset(_TRAIN, latent_grid=G)
    batch = collate_pointcloud([ds[0]])
    n = ds[0]["input_geom"].shape[0]
    assert batch["input_geom"].shape == (1, n, 3)
    assert batch["feats"].shape == (1, n, FEATURE_DIM)
    assert batch["gino_feats"].shape == (1, n, FEATURE_DIM - 1)
    assert batch["sdf"].shape == (1, G, G, G)
    assert batch["latent_queries"].shape == (1, G, G, G, 3)
    assert batch["prior"].shape == (1, n)
    assert batch["u_value"].shape == (1,)


def test_collate_distinct_geometry_rejects_stacking():
    ds = PointCloudDataset(_TRAIN, latent_grid=G)
    if len(ds) < 2:
        pytest.skip("need >=2 samples")
    with pytest.raises(ValueError, match="distinct geometry"):
        collate_pointcloud([ds[0], ds[1]])


def test_voxelise_fills_full_grid():
    ds = PointCloudDataset(_TRAIN, latent_grid=G, voxelise=True, voxel_grid=G)
    item = ds[0]
    assert item["voxel_feats"].shape == (FEATURE_DIM, G, G, G)
    assert item["voxel_theta"].shape == (G, G, G)
    # no holes: every voxel filled (linear inside hull, nearest at the boundary)
    assert torch.isfinite(item["voxel_theta"]).all()
    assert torch.isfinite(item["voxel_feats"]).all()
    # voxelised theta stays in the dimensionless range
    assert float(item["voxel_theta"].min()) >= -0.05
    assert float(item["voxel_theta"].max()) <= 1.05


def test_voxelise_sample_multichannel_shape():
    pts = np.random.default_rng(0).uniform(0, 1, size=(500, 3))
    vals = np.random.default_rng(1).uniform(0, 1, size=(500, 4))
    out = voxelise_sample(pts, vals, grid=8)
    assert out.shape == (8, 8, 8, 4)
    assert np.isfinite(out).all()


def test_u_estimator_recovers_groundtruth_u():
    """On GT theta the indoor-face estimator tracks the stored U (exact on clear walls)."""
    ds = PointCloudDataset(_TRAIN, latent_grid=G)
    rel_errs, clear_errs = [], []
    for i in range(min(len(ds), 24)):
        item = ds[i]
        u_true = float(item["u_value"])
        u_clear = float(item["u_clear"])
        u_pred = u_from_indoor_face_cloud(
            item["theta"].numpy(), item["prior"].numpy(), item["points"].numpy(), u_clear
        )
        rel = abs(u_pred - u_true) / u_true
        rel_errs.append(rel)
        if abs(u_true - u_clear) < 1e-4:  # clear column -> must be exact
            clear_errs.append(rel)
    assert np.mean(rel_errs) < 0.10  # estimator floor on GT
    if clear_errs:
        assert max(clear_errs) < 1e-4
