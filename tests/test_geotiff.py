"""Tests for the minimal stdlib GeoTIFF reader."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from thermotwin.geometry.geotiff import GeoTransform, read_geotiff, read_geotiff_transform

_REPO = Path(__file__).resolve().parents[1]
_ORTHO = _REPO / "data/raw/tum2twin-datasets/tif/TUM_flipped_georeferenced.tif"


def test_world_pixel_roundtrip():
    gt = GeoTransform(sx=2.0, sy=2.0, x0=1000.0, y0=5000.0)
    col, row = gt.world_to_pixel(np.array([1000.0, 1010.0]), np.array([5000.0, 4980.0]))
    assert col[0] == pytest.approx(0.0)
    assert row[0] == pytest.approx(0.0)
    assert col[1] == pytest.approx(5.0)  # +10 m east / 2 m per px
    assert row[1] == pytest.approx(10.0)  # -20 m north / 2 m per px (row grows southward)
    x, y = gt.pixel_to_world(col, row)
    assert x[1] == pytest.approx(1010.0)
    assert y[1] == pytest.approx(4980.0)


@pytest.mark.skipif(not _ORTHO.exists(), reason="TUM2TWIN orthophoto not present")
def test_read_real_orthophoto():
    img, gt = read_geotiff(_ORTHO)
    assert img.shape[0] == 734 and img.shape[1] == 769
    assert gt.sx == pytest.approx(4.022, abs=0.01)
    assert gt.sy == pytest.approx(4.022, abs=0.01)
    # tie-point is the campus origin in UTM32N; buildings sit east+south of it
    assert 689_000 < gt.x0 < 690_000
    assert 5_337_000 < gt.y0 < 5_338_000


@pytest.mark.skipif(not _ORTHO.exists(), reason="TUM2TWIN orthophoto not present")
def test_transform_only_matches_full_read():
    gt = read_geotiff_transform(_ORTHO)
    _, gt2 = read_geotiff(_ORTHO)
    assert gt.sx == gt2.sx and gt.x0 == gt2.x0
