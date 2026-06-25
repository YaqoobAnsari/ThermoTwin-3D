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
import struct
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
    "read_colmap_images",
    "read_colmap_points3D",
    "fuse_thermal_to_geometry",
    "baseline_residual",
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


# --- COLMAP sparse model (binary) — geometry + per-point image observations ----------------
# Minimal readers for the COLMAP ``.bin`` format (pycolmap is absent on the cluster), enough to
# fuse the calibrated thermal onto the reconstructed 3-D facade points via each point's track.


def read_colmap_images(data: bytes) -> dict[int, tuple[str, np.ndarray]]:
    """Parse ``images.bin`` -> ``{image_id: (name, points2D_xy (P,2))}``."""
    out: dict[int, tuple[str, np.ndarray]] = {}
    pos = 0
    (n,) = struct.unpack_from("<Q", data, pos)
    pos += 8
    for _ in range(n):
        image_id = struct.unpack_from("<I", data, pos)[0]
        pos += 4 + 8 * 7 + 4  # id + qvec(4)+tvec(3) doubles + camera_id
        end = data.index(b"\x00", pos)
        name = data[pos:end].decode()
        pos = end + 1
        (npts,) = struct.unpack_from("<Q", data, pos)
        pos += 8
        xy = np.frombuffer(data, dtype=np.float64, count=npts * 3, offset=pos).reshape(npts, 3)[:, :2]
        pos += npts * 24  # x,y double + point3D_id int64
        out[image_id] = (name, np.ascontiguousarray(xy))
    return out


def read_colmap_points3D(data: bytes) -> tuple[np.ndarray, list]:
    """Parse ``points3D.bin`` -> ``(xyz (N,3), tracks)`` where each track is ``[(img_id, p2d_idx)]``."""
    pos = 0
    (n,) = struct.unpack_from("<Q", data, pos)
    pos += 8
    xyz = np.empty((n, 3), dtype=np.float64)
    tracks = []
    for i in range(n):
        pos += 8  # point3D_id (uint64)
        xyz[i] = struct.unpack_from("<3d", data, pos)
        pos += 24 + 3 + 8  # xyz + rgb(3 uint8) + error(double)
        (tlen,) = struct.unpack_from("<Q", data, pos)
        pos += 8
        track = struct.unpack_from("<" + "II" * tlen, data, pos)
        pos += 8 * tlen
        tracks.append([(track[2 * k], track[2 * k + 1]) for k in range(tlen)])
    return xyz, tracks


def fuse_thermal_to_geometry(
    zf: zipfile.ZipFile, scene: str, sparse: str = "0"
) -> tuple[np.ndarray, np.ndarray]:
    """Fuse the calibrated thermal onto the reconstructed 3-D facade points.

    For each COLMAP 3-D point, sample the calibrated °C of every registered (``frame_train``)
    image that observes it — at the point's 2-D track location in that image — and take the
    median across views (occlusion-free by construction: the track is exactly the images that
    *saw* the point). Returns ``(xyz (N,3), celsius (N,))``; points with no thermal observation
    are dropped.
    """
    base = f"ThermoScenes/{scene}/colmap/sparse/{sparse}"
    images = read_colmap_images(zf.read(f"{base}/images.bin"))
    xyz, tracks = read_colmap_points3D(zf.read(f"{base}/points3D.bin"))
    bounds = temperature_bounds(zf, scene)

    therm_cache: dict[str, np.ndarray] = {}

    def _therm(name: str) -> np.ndarray | None:
        if name not in therm_cache:
            member = f"ThermoScenes/{scene}/thermal/{name}"
            try:
                therm_cache[name] = decode_celsius(zf, member, bounds)
            except KeyError:
                therm_cache[name] = None  # no matching thermal frame
        return therm_cache[name]

    keep_xyz, keep_c = [], []
    for p, track in zip(xyz, tracks, strict=True):
        vals = []
        for img_id, p2d_idx in track:
            name, pts2d = images.get(img_id, (None, None))
            if name is None:
                continue
            t = _therm(name)
            if t is None or p2d_idx >= len(pts2d):
                continue
            x, y = pts2d[p2d_idx]
            ix, iy = int(round(x)), int(round(y))
            if 0 <= iy < t.shape[0] and 0 <= ix < t.shape[1]:
                vals.append(float(t[iy, ix]))
        if vals:
            keep_xyz.append(p)
            keep_c.append(np.median(vals))
    return np.asarray(keep_xyz, dtype=np.float32), np.asarray(keep_c, dtype=np.float32)


def baseline_residual(xyz: np.ndarray, celsius: np.ndarray, k: int = 40, k_sigma: float = 2.0):
    """Smooth "expected clear-wall" baseline + the heat-loss residual on a 3-D facade.

    The operator/prior says a clear wall is *smooth*: its surface temperature varies slowly with
    orientation and the bulk ΔT, with localized warm anomalies marking thermal bridges / defects.
    We take the **k-NN local mean** of the fused measured °C as that smooth baseline (the 3-D
    analogue of the 2-D local-mean, and a model-free stand-in for the operator's clear-wall
    field), and the residual ``measured − baseline`` localizes the anomalies — the delta-prior
    decomposition on real, calibrated, 3-D-fused thermal.

    Returns ``(baseline (N,), residual (N,), anomaly_mask (N,))``.
    """
    from scipy.spatial import cKDTree

    xyz = np.asarray(xyz, dtype=np.float64)
    celsius = np.asarray(celsius, dtype=np.float64)
    tree = cKDTree(xyz)
    kk = min(k, len(xyz))
    _, idx = tree.query(xyz, k=kk)
    baseline = celsius[idx].mean(axis=1)
    residual = celsius - baseline
    mask = residual > (residual.mean() + k_sigma * residual.std())
    return baseline.astype(np.float32), residual.astype(np.float32), mask
