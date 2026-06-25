"""TUM2TWIN airborne thermal-IR ⟷ CityGML geometry — the *measured-thermal* cross-task rung.

Every other geometry corpus is scored on **simulated** physics; this is the one place we put
measured airborne thermal next to the *same* real buildings our LoD2/LoD3 corpus models, and ask
a falsifiable question: **do the measured heat-loss anomalies actually land on the building
envelopes we reconstruct?** It is the substrate the inverse twin (H2) will later calibrate
against, validated here as a localisation task.

Honest scope. The usable TUM2TWIN thermal artefact is the **georeferenced orthophoto**
(``tif/TUM_flipped_georeferenced.tif``): a north-up GeoTIFF in EPSG:25832 (4 m/px), tone-mapped
(3 near-grayscale bands, values ~1–204) — so the thermal is **relative**, not calibrated °C like
ThermoScenes. (The raw ``camera_ir`` frames carry only *vehicle* poses and need intrinsic +
extrinsic calibration to register to geometry; :mod:`thermotwin.data.thermal_tir` ingests them at
frame level, but they are out of scope for this geometry-coupled rung.) Pipeline:

1. Read the orthophoto + its geo-transform (:mod:`thermotwin.geometry.geotiff`); collapse the
   tone-mapped bands to a single relative-temperature field, with a validity mask (the flight
   covers ~42 % of the raster).
2. Read the CityGML building footprints in **absolute** CRS coords
   (:func:`thermotwin.geometry.citygml.read_citygml_footprints`) and rasterise each onto the
   orthophoto via the geo-transform.
3. Flag heat-loss anomalies with the *same* rule as the ThermoScenes rung
   (:func:`thermotwin.data.thermoscenes.heat_loss_anomaly`) and measure **footprint
   enrichment** — the anomaly rate on the modelled building envelopes vs off them. ``> 1`` means
   measured heat-loss concentrates on the geometry we model; that is the comparable metric.
"""

from __future__ import annotations

import numpy as np

from ..geometry.citygml import read_citygml_footprints
from ..geometry.geotiff import GeoTransform, read_geotiff
from .thermoscenes import heat_loss_anomaly

__all__ = [
    "load_orthophoto",
    "footprint_masks",
    "score",
    "evaluate",
]


def load_orthophoto(tif_path: str) -> tuple[np.ndarray, np.ndarray, GeoTransform]:
    """Read the thermal orthophoto into ``(thermal_rel, valid, transform)``.

    ``thermal_rel`` is the band-averaged relative-temperature field (float); ``valid`` masks the
    flown region (an alpha band if present, else non-zero pixels).
    """
    img, transform = read_geotiff(tif_path)
    img = np.asarray(img)
    if img.ndim == 2:
        bands, valid = img.astype(np.float64), img > 0
    else:
        c = img.shape[2]
        bands = img[..., :3].astype(np.float64).mean(axis=2)
        # Flown-region validity: non-zero thermal, AND alpha if an RGBA channel is present
        # (alpha alone can flag black filler pixels, which would dilute the background rate).
        valid = bands > 0
        if c >= 4:
            valid &= img[..., 3] > 0
    return bands, valid, transform


def footprint_masks(
    footprints: list[tuple[str, np.ndarray]], transform: GeoTransform, shape: tuple[int, int]
) -> list[tuple[str, np.ndarray]]:
    """Rasterise each building footprint polygon onto the raster grid.

    Returns ``(building_id, mask)`` for every footprint that lands at least partly inside the
    raster (off-raster buildings are dropped).
    """
    from PIL import Image, ImageDraw

    h, w = shape
    out: list[tuple[str, np.ndarray]] = []
    for bid, xy in footprints:
        col, row = transform.world_to_pixel(xy[:, 0], xy[:, 1])
        poly = list(zip(col.tolist(), row.tolist(), strict=True))
        canvas = Image.new("1", (w, h), 0)
        ImageDraw.Draw(canvas).polygon(poly, fill=1)
        mask = np.asarray(canvas, dtype=bool)
        if mask.any():
            out.append((bid, mask))
    return out


def score(
    thermal_rel: np.ndarray,
    valid: np.ndarray,
    building_masks: list[tuple[str, np.ndarray]],
    window: int = 25,
    k_sigma: float = 2.0,
    pct: float = 96.0,
) -> dict:
    """Heat-loss-anomaly enrichment on the modelled building footprints.

    The anomaly rule matches the ThermoScenes rung. ``enrichment`` is the anomaly rate inside
    the (union of) building footprints divided by the rate outside them — measured over valid
    (flown) pixels only.
    """
    fill = float(thermal_rel[valid].mean()) if valid.any() else 0.0
    field = np.where(valid, thermal_rel, fill)  # neutralise nodata before local filtering
    anomaly, _ = heat_loss_anomaly(field, window=window, k_sigma=k_sigma, pct=pct)
    anomaly &= valid

    union = np.zeros_like(valid)
    contrasts = []
    bg = valid.copy()
    for _bid, m in building_masks:
        union |= m
    bg &= ~union
    bg_mean = float(thermal_rel[bg].mean()) if bg.any() else 0.0
    for _bid, m in building_masks:
        inside = m & valid
        if inside.any():
            contrasts.append(float(thermal_rel[inside].mean()) - bg_mean)

    inside = union & valid
    outside = bg
    rate_in = float(anomaly[inside].mean()) if inside.any() else 0.0
    rate_out = float(anomaly[outside].mean()) if outside.any() else 0.0
    enrichment = rate_in / rate_out if rate_out > 0 else 0.0
    return {
        "n_buildings": len(building_masks),
        "valid_frac": round(float(valid.mean()), 4),
        "footprint_px": int(inside.sum()),
        "anomaly_rate_on_buildings": round(rate_in, 4),
        "anomaly_rate_off_buildings": round(rate_out, 4),
        "enrichment": round(enrichment, 3),
        "mean_building_thermal_contrast_dn": round(float(np.mean(contrasts)) if contrasts else 0.0, 2),
        "note": (
            "relative (tone-mapped) thermal; enrichment > 1 ⇒ measured heat-loss anomalies "
            "concentrate on the CityGML building envelopes we model"
        ),
    }


def evaluate(tif_path: str, citygml_dir: str) -> dict:
    """End-to-end: orthophoto + CityGML footprints -> measured-thermal localisation summary."""
    thermal, valid, transform = load_orthophoto(tif_path)
    footprints = read_citygml_footprints(citygml_dir)
    masks = footprint_masks(footprints, transform, thermal.shape)
    return score(thermal, valid, masks)
