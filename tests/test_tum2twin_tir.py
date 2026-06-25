"""Tests for the TUM2TWIN orthophoto ⟷ footprint enrichment adapter."""

from __future__ import annotations

import numpy as np

from thermotwin.data.tum2twin_tir import footprint_masks, score
from thermotwin.geometry.geotiff import GeoTransform


def test_footprint_masks_rasterise_in_bounds():
    gt = GeoTransform(sx=1.0, sy=1.0, x0=0.0, y0=50.0)  # 50x50 raster, 1 m/px
    fp = [("b1", np.array([[10.0, 40.0], [20.0, 40.0], [20.0, 30.0], [10.0, 30.0]]))]
    masks = footprint_masks(fp, gt, (50, 50))
    assert len(masks) == 1
    _, m = masks[0]
    assert m.shape == (50, 50)
    assert 50 < m.sum() < 150  # ~10x10 px square


def test_enrichment_high_when_anomaly_on_footprint():
    rng = np.random.default_rng(0)
    thermal = rng.normal(100.0, 1.0, size=(80, 80))
    valid = np.ones((80, 80), dtype=bool)
    # a hot heat-loss blob sitting under the building footprint
    thermal[35:45, 35:45] += 40.0
    foot = np.zeros((80, 80), dtype=bool)
    foot[33:47, 33:47] = True
    res = score(thermal, valid, [("b1", foot)])
    assert res["enrichment"] > 3.0
    assert res["anomaly_rate_on_buildings"] > res["anomaly_rate_off_buildings"]
    assert res["mean_building_thermal_contrast_dn"] > 0.0


def test_enrichment_near_baseline_when_anomaly_off_footprint():
    rng = np.random.default_rng(1)
    thermal = rng.normal(100.0, 1.0, size=(80, 80))
    valid = np.ones((80, 80), dtype=bool)
    thermal[5:12, 5:12] += 40.0  # anomaly far from the footprint
    foot = np.zeros((80, 80), dtype=bool)
    foot[50:60, 50:60] = True
    res = score(thermal, valid, [("b1", foot)])
    assert res["enrichment"] < 1.0  # no measured heat-loss on this building
