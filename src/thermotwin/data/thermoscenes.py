"""ThermoScenes loader — calibrated absolute-°C facade thermal for localization validation.

ThermoScenes (Hassan et al., ThermoNeRF) is the real *calibrated* thermal asset: per building
scene it ships a COLMAP reconstruction (geometry + poses), RGB, MSX, and **thermal PNGs whose
grayscale encodes a temperature normalised between** ``temperature_bounds.json``'s
``absolute_min_temperature`` / ``absolute_max_temperature`` (in °C). So a pixel value ``g∈[0,1]``
maps to ``°C = min + g·(max − min)`` — genuine absolute surface temperature, unlike TBBR
(8-bit DN) or the TUM2TWIN TIR sample (uncalibrated counts).

Scope (see `docs/data_inventory.md`): ThermoScenes measures the **exterior surface temperature**
of real facades under real transient/solar conditions, which is a *different* quantity from our
operator's steady through-wall θ. So it validates **heat-loss / thermal-bridge localisation** —
do real warm anomalies stand out exactly where the smooth clear-wall baseline does not explain
the field — not an absolute θ-RMSE. This module provides the calibrated-°C decode and the
anomaly (prior-residual) map; the 3-D COLMAP fusion onto geometry + the operator baseline is the
next layer (`scripts/eval_thermoscenes.py`).
"""

from __future__ import annotations

import json
import zipfile
from io import BytesIO

import numpy as np

__all__ = [
    "BUILDING_SCENES",
    "open_archive",
    "list_scenes",
    "temperature_bounds",
    "thermal_members",
    "decode_celsius",
    "heat_loss_anomaly",
]

# The real-building scenes (the rest of the archive is small objects — cups, kettles, …).
BUILDING_SCENES = (
    "BI-building",
    "buildingA_spring",
    "buildingA_winter",
    "building-sunrise",
    "dorm1",
    "dorm2",
    "exhibition_building",
    "INR-building",
    "MED-building",
)


def open_archive(path: str) -> zipfile.ZipFile:
    """Open the ThermoScenes archive for in-place reading (no extraction)."""
    return zipfile.ZipFile(path)


def list_scenes(zf: zipfile.ZipFile) -> list[str]:
    """All scene folders present in the archive."""
    return sorted({n.split("/")[1] for n in zf.namelist() if n.startswith("ThermoScenes/") and "/" in n[13:]})


def temperature_bounds(zf: zipfile.ZipFile, scene: str) -> tuple[float, float]:
    """``(min_°C, max_°C)`` calibration for a scene's thermal PNGs."""
    d = json.loads(zf.read(f"ThermoScenes/{scene}/temperature_bounds.json"))
    return float(d["absolute_min_temperature"]), float(d["absolute_max_temperature"])


def thermal_members(zf: zipfile.ZipFile, scene: str) -> list[str]:
    """Sorted thermal-PNG member names for a scene."""
    pre = f"ThermoScenes/{scene}/thermal/"
    return sorted(n for n in zf.namelist() if n.startswith(pre) and n.lower().endswith(".png"))


def decode_celsius(zf: zipfile.ZipFile, member: str, bounds: tuple[float, float]) -> np.ndarray:
    """Decode one thermal PNG to an absolute-temperature ``(H, W)`` array in °C."""
    import matplotlib.image as mpimg

    arr = mpimg.imread(BytesIO(zf.read(member)))  # float in [0, 1]
    if arr.ndim == 3:  # grayscale stored as RGB(A) -> luminance of the colour channels
        arr = arr[..., :3].mean(axis=-1)
    lo, hi = bounds
    return (lo + arr.astype(np.float64) * (hi - lo)).astype(np.float32)


def heat_loss_anomaly(celsius: np.ndarray, window: int = 25, k_sigma: float = 2.0, pct: float = 96.0):
    """Localised warm-anomaly (heat-loss) map — the prior-residual on a real facade.

    A clear wall is smooth; thermal bridges / defects are *local* warm anomalies. We flag pixels
    that are both a global warm outlier (above the ``pct`` percentile) AND locally hot (> local
    mean + ``k_sigma``·local std over a ``window`` box) — the same conservative rule as the
    TUM2TWIN TIR pipeline, but on calibrated °C. The ``(measured − local_mean)`` field *is* the
    prior-residual: the smooth local baseline is the "expected clear-wall" stand-in.

    Returns ``(anomaly_mask, residual_celsius)``.
    """
    from scipy.ndimage import uniform_filter

    c = np.asarray(celsius, dtype=np.float64)
    local_mean = uniform_filter(c, size=window)
    local_var = uniform_filter(c * c, size=window) - local_mean**2
    local_std = np.sqrt(np.clip(local_var, 0.0, None))
    residual = c - local_mean
    mask = (c > np.percentile(c, pct)) & (residual > k_sigma * (local_std + 1e-6))
    return mask, residual.astype(np.float32)
