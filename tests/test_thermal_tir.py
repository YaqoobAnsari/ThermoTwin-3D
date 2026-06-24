"""TIR ingestion: loader shapes/dtypes, pose table, tone map and saliency.

Two layers:

* **Unit** — on synthetic frames built in-memory, so they always run: tone-map range,
  saliency mask shape/dtype, and that saliency targets *warm* pixels.
* **Integration** — on the real TUM2TWIN TIR sample if present, else skipped: 73 frames
  of the right uint16 480x640 shape, a (73, 7) pose table aligned to the frames, a 4x4
  ENU->ECEF matrix with the expected homogeneous bottom row.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from thermotwin.data.thermal_tir import (
    DEFAULT_TIR_DIR,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    enu_to_ecef_matrix,
    heat_loss_saliency,
    list_frame_paths,
    load_pose_table,
    load_sequence,
    tone_map,
    warm_area_fraction,
)

_REPO = Path(__file__).resolve().parents[1]
_TIR_DIR = _REPO / DEFAULT_TIR_DIR
_HAVE_TIR = (_TIR_DIR / "camera_ir").is_dir() and bool(list_frame_paths(_TIR_DIR))

_N_EXPECTED = 73


def _synthetic_frame(seed: int = 0) -> np.ndarray:
    """A flat warm-ish field (counts ~15000) with a few hot blobs and a cold border."""
    rng = np.random.default_rng(seed)
    f = rng.normal(15000.0, 30.0, size=(FRAME_HEIGHT, FRAME_WIDTH))
    # a couple of warm anomalies (windows / thermal bridges)
    f[100:130, 200:230] += 600.0
    f[300:320, 400:440] += 500.0
    # fixed cold border sentinel, like the real frames (179)
    f[:5, :] = 179.0
    return f.astype(np.uint16)


# --- unit (always run) ------------------------------------------------------------


def test_tone_map_in_byte_range():
    out = tone_map(_synthetic_frame())
    assert out.dtype == np.uint8
    assert out.shape == (FRAME_HEIGHT, FRAME_WIDTH)
    assert out.min() >= 0
    assert out.max() <= 255


def test_tone_map_flat_frame_is_midgrey():
    flat = np.full((16, 16), 12345, dtype=np.uint16)
    out = tone_map(flat)
    assert out.dtype == np.uint8
    assert np.all(out == 128)


def test_saliency_returns_bool_mask():
    f = _synthetic_frame()
    mask = heat_loss_saliency(f)
    assert mask.dtype == np.bool_
    assert mask.shape == f.shape


def test_saliency_targets_warm_pixels():
    f = _synthetic_frame()
    mask = heat_loss_saliency(f)
    assert mask.any(), "expected the planted warm blobs to be flagged"
    # salient pixels must be warmer than the non-salient background
    assert f[mask].mean() > f[~mask].mean()


def test_warm_area_fraction_in_unit_interval():
    frac = warm_area_fraction(_synthetic_frame())
    assert 0.0 <= frac <= 1.0


# --- integration (real sample) ----------------------------------------------------


@pytest.mark.skipif(not _HAVE_TIR, reason="TUM2TWIN TIR sample not present")
def test_sequence_frames_shape_dtype():
    seq = load_sequence(_TIR_DIR)
    assert seq.n_frames == _N_EXPECTED
    assert seq.frames.shape == (_N_EXPECTED, FRAME_HEIGHT, FRAME_WIDTH)
    assert seq.frames.dtype == np.uint16


@pytest.mark.skipif(not _HAVE_TIR, reason="TUM2TWIN TIR sample not present")
def test_pose_table_shape():
    pose = load_pose_table(_TIR_DIR)
    assert pose.shape == (_N_EXPECTED, 7)


@pytest.mark.skipif(not _HAVE_TIR, reason="TUM2TWIN TIR sample not present")
def test_pose_aligns_to_frames():
    seq = load_sequence(_TIR_DIR)
    assert np.array_equal(seq.pose[:, 0].astype(np.int64), seq.frame_ids)


@pytest.mark.skipif(not _HAVE_TIR, reason="TUM2TWIN TIR sample not present")
def test_enu_to_ecef_is_homogeneous_4x4():
    mat = enu_to_ecef_matrix(_TIR_DIR)
    assert mat.shape == (4, 4)
    assert np.allclose(mat[3], [0.0, 0.0, 0.0, 1.0])
    # rotation block columns are unit-norm (proper ENU->ECEF rotation)
    rot = mat[:3, :3]
    assert np.allclose(np.linalg.norm(rot, axis=0), 1.0, atol=1e-6)


@pytest.mark.skipif(not _HAVE_TIR, reason="TUM2TWIN TIR sample not present")
def test_real_saliency_returns_mask():
    seq = load_sequence(_TIR_DIR)
    mask = heat_loss_saliency(seq.frames[seq.n_frames // 2])
    assert mask.dtype == np.bool_
    assert mask.shape == (FRAME_HEIGHT, FRAME_WIDTH)
