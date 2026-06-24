"""Ingest the TUM2TWIN street-level thermal-infrared (TIR) sample.

This is the **real-thermal on-ramp**: a 73-frame street-level TIR sequence recorded
by a Jenoptik IR-TCM 640 microbolometer (FOV 65.2 deg x 51.3 deg) carried on the same
mobile-laser-scanning vehicle that produced the TUM2TWIN point clouds. Each frame is a
16-bit LZW-TIFF of **raw radiometric counts** at 640x480; a sidecar text file gives the
per-frame pose of the *sensor carrier* (the vehicle, not the camera) in a local ENU
frame; the readme carries the 4x4 ENU->ECEF transform.

Honest scope (read before using any number this module returns):

* **No radiometric calibration.** The 16-bit values are uncalibrated microbolometer
  counts, not temperatures. There is no count->Kelvin map, no emissivity correction, no
  reflected-temperature term. Everything here is therefore **qualitative** — relative
  warm/cold structure within a frame, never an absolute temperature or U-value.
* **Carrier pose, not sensor pose.** Columns 2-7 are the *vehicle* pose; the camera's
  intrinsics and lever-arm/boresight extrinsics are not provided, so a pixel cannot be
  back-projected onto a building surface. Pose supports trajectory context only.
* **No thermal ground-truth field.** There is no per-surface temperature or heat-flux
  reference, so this sample can characterise the data and drive qualitative saliency,
  but it cannot validate the operator quantitatively.

What the module does provide, all as plain ``numpy`` arrays so IO stays testable:

* :func:`load_frame` / :func:`load_sequence` — raw uint16 count fields.
* :func:`load_pose_table` — the (N, 7) frame-id + xyz(ENU) + roll/pitch/yaw(deg) table.
* :func:`enu_to_ecef_matrix` — the 4x4 from the readme.
* :func:`tone_map` — percentile-stretch to uint8 for visualisation.
* :func:`heat_loss_saliency` — highlight anomalously *warm* regions (windows, thermal
  bridges) by combining a global high-percentile threshold with local contrast.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image
from scipy.ndimage import uniform_filter

__all__ = [
    "DEFAULT_TIR_DIR",
    "FRAME_HEIGHT",
    "FRAME_WIDTH",
    "ThermalSequence",
    "enu_to_ecef_matrix",
    "heat_loss_saliency",
    "list_frame_paths",
    "load_frame",
    "load_pose_table",
    "load_sequence",
    "tone_map",
    "warm_area_fraction",
]

# Layout of the TUM2TWIN TIR sample as shipped (Fraunhofer IOSB, Oct 2017).
DEFAULT_TIR_DIR = Path("data/raw/tum2twin/thermal_tir_2016")
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
_FRAME_GLOB = "img_*.tif"
_POSE_FILE = "vehicle-track_ir-frames.txt"
_README_FILE = "readme.txt"
_ID_RE = re.compile(r"(\d+)")


@dataclass(frozen=True)
class ThermalSequence:
    """A loaded TIR sequence: raw count frames plus aligned carrier poses.

    Attributes:
        frames: ``(N, H, W)`` uint16 stack of raw radiometric counts.
        frame_ids: ``(N,)`` int frame numbers (parsed from the filenames).
        pose: ``(N, 7)`` float table ``[frame_id, x, y, z, roll, pitch, yaw]`` where
            ``x/y/z`` are sensor-carrier position in local ENU metres and the angles
            are in degrees. Rows are aligned to :attr:`frames` by frame id.
        enu_to_ecef: ``(4, 4)`` float transform mapping homogeneous ENU coordinates to
            ECEF, as given in the readme.
    """

    frames: NDArray[np.uint16]
    frame_ids: NDArray[np.int64]
    pose: NDArray[np.float64]
    enu_to_ecef: NDArray[np.float64]

    @property
    def n_frames(self) -> int:
        """Number of frames in the sequence."""
        return int(self.frames.shape[0])

    def path_length(self) -> float:
        """Total polyline length of the carrier trajectory, in ENU metres."""
        xyz = self.pose[:, 1:4]
        return float(np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum())


def _camera_dir(root: Path | str) -> Path:
    """Resolve the ``camera_ir`` directory under a sample root (or accept it directly)."""
    root = Path(root)
    cam = root / "camera_ir"
    return cam if cam.is_dir() else root


def list_frame_paths(root: Path | str = DEFAULT_TIR_DIR) -> list[Path]:
    """Return the TIR frame paths, sorted by the numeric id in the filename.

    Args:
        root: Sample root (``.../thermal_tir_2016``) or its ``camera_ir`` directory.

    Returns:
        Frame paths ordered by ascending frame id (filename lexical order is *not*
        reliable across all naming, so we sort on the parsed integer).
    """
    cam = _camera_dir(root)
    paths = list(cam.glob(_FRAME_GLOB))
    return sorted(paths, key=lambda p: int(_ID_RE.search(p.stem).group(1)))


def _frame_id(path: Path) -> int:
    return int(_ID_RE.search(path.stem).group(1))


def load_frame(path: Path | str) -> NDArray[np.uint16]:
    """Load a single 16-bit TIR frame as a ``(H, W)`` uint16 array of raw counts.

    Args:
        path: Path to an ``img_*.tif`` frame.

    Returns:
        ``(FRAME_HEIGHT, FRAME_WIDTH)`` uint16 array. Values are uncalibrated
        microbolometer counts (see module docstring) — not temperatures.
    """
    with Image.open(path) as im:
        arr = np.asarray(im)
    if arr.dtype != np.uint16:
        # PIL reads "I;16" as uint16; guard against an unexpected promotion.
        arr = arr.astype(np.uint16)
    return arr


def load_sequence(root: Path | str = DEFAULT_TIR_DIR) -> ThermalSequence:
    """Load the whole TIR sample: frames, poses and the ENU->ECEF transform.

    Frames and pose rows are aligned by frame id; a mismatch raises ``ValueError`` so a
    truncated download fails loudly rather than silently misaligning pose with imagery.

    Args:
        root: Sample root directory (``.../thermal_tir_2016``).

    Returns:
        A populated :class:`ThermalSequence`.

    Raises:
        FileNotFoundError: If no frames are found under ``root``.
        ValueError: If the frame ids and pose-table ids do not match one-to-one.
    """
    root = Path(root)
    paths = list_frame_paths(root)
    if not paths:
        raise FileNotFoundError(f"no TIR frames ({_FRAME_GLOB}) under {root}")

    frame_ids = np.array([_frame_id(p) for p in paths], dtype=np.int64)
    frames = np.stack([load_frame(p) for p in paths], axis=0)

    pose = load_pose_table(root)
    pose_ids = pose[:, 0].astype(np.int64)
    if pose_ids.shape != frame_ids.shape or not np.array_equal(pose_ids, frame_ids):
        raise ValueError(
            f"pose ids ({pose_ids.tolist()[:3]}...) do not match frame ids "
            f"({frame_ids.tolist()[:3]}...); sample may be truncated"
        )

    enu_to_ecef = enu_to_ecef_matrix(root)
    return ThermalSequence(frames, frame_ids, pose, enu_to_ecef)


def load_pose_table(root: Path | str = DEFAULT_TIR_DIR) -> NDArray[np.float64]:
    """Parse ``vehicle-track_ir-frames.txt`` into an ``(N, 7)`` pose table.

    Columns are ``[frame_id, x, y, z, roll, pitch, yaw]``: carrier position in local
    ENU metres and orientation in degrees (per the sample readme).

    Args:
        root: Sample root or its ``camera_ir`` directory.

    Returns:
        ``(N, 7)`` float64 array.

    Raises:
        FileNotFoundError: If the pose file is absent.
        ValueError: If the file does not have 7 columns.
    """
    cam = _camera_dir(root)
    pose_path = cam / _POSE_FILE
    if not pose_path.is_file():
        raise FileNotFoundError(f"pose file not found: {pose_path}")
    table = np.loadtxt(pose_path, dtype=np.float64)
    table = np.atleast_2d(table)
    if table.shape[1] != 7:
        raise ValueError(f"expected 7 columns in {pose_path}, got {table.shape[1]}")
    return table


def enu_to_ecef_matrix(root: Path | str = DEFAULT_TIR_DIR) -> NDArray[np.float64]:
    """Read the 4x4 ENU->ECEF transform from the sample ``readme.txt``.

    The readme lists the matrix as 16 floats (the rows of a 4x4) embedded in prose; we
    extract every float and take the last 16, which are the matrix entries.

    Args:
        root: Sample root directory.

    Returns:
        ``(4, 4)`` float64 transform such that ``ecef = M @ [e, n, u, 1]``.

    Raises:
        FileNotFoundError: If the readme is absent.
        ValueError: If 16 matrix floats cannot be recovered.
    """
    root = Path(root)
    readme = root / _README_FILE
    if not readme.is_file():
        raise FileNotFoundError(f"readme not found: {readme}")
    text = readme.read_text()
    # Matrix entries are the only bare floats with a decimal point / exponent; ids and
    # the FOV degrees in the prose are integers or "65.2"-style — to be robust we take
    # the trailing 16 numeric tokens that look like full-precision matrix entries.
    floats = re.findall(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?", text)
    vals = [float(x) for x in floats]
    if len(vals) < 16:
        raise ValueError(f"could not find 16 matrix floats in {readme} (found {len(vals)})")
    mat = np.array(vals[-16:], dtype=np.float64).reshape(4, 4)
    return mat


def tone_map(
    frame: NDArray[np.number],
    low_pct: float = 2.0,
    high_pct: float = 98.0,
) -> NDArray[np.uint8]:
    """Percentile-stretch a raw count frame to an 8-bit visualisation image.

    Robust linear stretch: clip to the ``[low_pct, high_pct]`` percentiles of the
    frame, then scale that window to ``[0, 255]``. This mirrors the intent of the
    vendor MATLAB tone map (mean-subtract + scale) but is percentile-robust to the
    dead/border pixels present in these frames (counts ~179 against a ~15000 scene).

    Args:
        frame: ``(H, W)`` array of raw counts.
        low_pct: Lower clip percentile.
        high_pct: Upper clip percentile.

    Returns:
        ``(H, W)`` uint8 image with values in ``[0, 255]``.
    """
    f = frame.astype(np.float64)
    lo, hi = np.percentile(f, [low_pct, high_pct])
    if hi <= lo:
        # Degenerate (flat) frame: return mid-grey rather than divide by zero.
        return np.full(f.shape, 128, dtype=np.uint8)
    scaled = (f - lo) / (hi - lo)
    scaled = np.clip(scaled, 0.0, 1.0)
    return np.round(scaled * 255.0).astype(np.uint8)


def heat_loss_saliency(
    frame: NDArray[np.number],
    warm_pct: float = 97.0,
    local_window: int = 25,
    local_sigma: float = 2.0,
) -> NDArray[np.bool_]:
    """Highlight anomalously *warm* regions in a TIR frame (qualitative heat-loss cue).

    Warm pixels in a TIR scene of a building facade are candidate heat-loss features:
    single-glazed windows, uninsulated lintels/balconies and other thermal bridges run
    hotter than the surrounding wall. With no radiometric calibration this can only flag
    *relatively* warm structure, so we combine two complementary cues and AND them:

    * **Global** — pixels above the ``warm_pct`` percentile of the whole frame
      (absolute warm outliers).
    * **Local** — pixels brighter than their local-window mean by more than
      ``local_sigma`` local standard deviations (warm *contrast* against the immediate
      background, which survives vignetting / large-scale gradients).

    Args:
        frame: ``(H, W)`` array of raw counts.
        warm_pct: Global percentile above which a pixel is a warm outlier.
        local_window: Side length (pixels) of the local mean/variance window.
        local_sigma: Local-contrast threshold in local standard deviations.

    Returns:
        ``(H, W)`` boolean mask, ``True`` on salient warm pixels.
    """
    f = frame.astype(np.float64)
    global_thr = np.percentile(f, warm_pct)
    global_mask = f >= global_thr

    local_mean = uniform_filter(f, size=local_window, mode="nearest")
    local_sq = uniform_filter(f * f, size=local_window, mode="nearest")
    local_var = np.clip(local_sq - local_mean * local_mean, 0.0, None)
    local_std = np.sqrt(local_var)
    local_mask = f > (local_mean + local_sigma * local_std)

    return np.logical_and(global_mask, local_mask)


def warm_area_fraction(
    frame: NDArray[np.number],
    warm_pct: float = 97.0,
    local_window: int = 25,
    local_sigma: float = 2.0,
) -> float:
    """Fraction of the frame flagged warm-salient by :func:`heat_loss_saliency`.

    A compact per-frame scalar for sequence-level reporting (how much apparent
    heat-loss structure each frame carries).

    Returns:
        Fraction in ``[0, 1]``.
    """
    mask = heat_loss_saliency(
        frame, warm_pct=warm_pct, local_window=local_window, local_sigma=local_sigma
    )
    return float(mask.mean())
