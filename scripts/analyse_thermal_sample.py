#!/usr/bin/env python
"""Characterise the TUM2TWIN street-level TIR sample (qualitative scope only).

Loads all 73 frames of the Jenoptik IR-TCM 640 sequence
(:mod:`thermotwin.data.thermal_tir`), prints sequence statistics (raw-count range,
per-frame warm-area fraction, vehicle path length), and writes three artefacts under
``results/thermal_sample/``:

* ``contact_sheet.png`` — a few tone-mapped frames spanning the sequence;
* ``saliency_overlay.png`` — a mid-sequence frame with warm-saliency highlighted;
* ``summary.json`` — machine-readable sequence stats + an explicit statement of what
  this sample can and cannot support.

Honest scope: the 16-bit values are **uncalibrated microbolometer counts**, the pose is
the **vehicle carrier** (no camera intrinsics/extrinsics), and there is **no thermal
ground-truth field**. So this is a *characterisation + qualitative saliency* demo —
never an absolute-temperature or U-value validation. See the module docstring and the
``summary.json`` it writes for the full caveat list.

Example
-------
    python scripts/analyse_thermal_sample.py
    python scripts/analyse_thermal_sample.py --root data/raw/tum2twin/thermal_tir_2016 \
        --out results/thermal_sample
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from thermotwin.data.thermal_tir import (  # noqa: E402
    DEFAULT_TIR_DIR,
    heat_loss_saliency,
    load_sequence,
    tone_map,
    warm_area_fraction,
)

_REPO = Path(__file__).resolve().parents[1]

# What this sample does NOT support — written verbatim into summary.json so the caveat
# travels with the numbers.
_CANNOT = [
    "Absolute temperatures: values are uncalibrated 16-bit microbolometer counts; no "
    "count->Kelvin map, emissivity, or reflected-temperature correction is provided.",
    "U-value / heat-flux validation: no thermal ground-truth field accompanies the sample.",
    "Pixel->surface back-projection: pose is the vehicle carrier, and no camera "
    "intrinsics or boresight/lever-arm extrinsics are given.",
]
_CAN = [
    "Qualitative within-frame warm/cold structure and relative heat-loss saliency "
    "(windows / thermal bridges run warmer than surrounding wall).",
    "Sequence + trajectory characterisation (count statistics, vehicle path).",
    "A real-thermal IO + saliency pipeline that TBBR/TBBRv2 calibrated thermography "
    "can later plug into for quantitative work.",
]


def _contact_sheet(seq, n_tiles: int, out_path: Path) -> list[int]:
    """Save a tone-mapped contact sheet of ``n_tiles`` frames spanning the sequence."""
    idx = np.unique(np.linspace(0, seq.n_frames - 1, n_tiles).round().astype(int))
    cols = min(len(idx), 4)
    rows = int(np.ceil(len(idx) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 2.6))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, i in zip(axes, idx, strict=False):
        ax.imshow(tone_map(seq.frames[i]), cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"frame {int(seq.frame_ids[i])}", fontsize=8)
        ax.axis("off")
    fig.suptitle("TUM2TWIN street-level TIR (tone-mapped, uncalibrated counts)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return [int(seq.frame_ids[i]) for i in idx]


def _saliency_overlay(seq, frame_idx: int, out_path: Path) -> int:
    """Save a tone-mapped frame beside the same frame with warm-saliency overlaid."""
    frame = seq.frames[frame_idx]
    base = tone_map(frame)
    mask = heat_loss_saliency(frame)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].imshow(base, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title(f"tone-mapped (frame {int(seq.frame_ids[frame_idx])})", fontsize=9)
    axes[1].imshow(base, cmap="gray", vmin=0, vmax=255)
    overlay = np.zeros((*mask.shape, 4), dtype=float)
    overlay[mask] = (1.0, 0.0, 0.0, 1.0)  # opaque red on salient warm pixels
    axes[1].imshow(overlay)
    axes[1].set_title(
        f"heat-loss saliency (warm outliers, {100 * mask.mean():.2f}% of frame)",
        fontsize=9,
    )
    for ax in axes:
        ax.axis("off")
    fig.suptitle(
        "Qualitative warm-region saliency — candidate windows / thermal bridges "
        "(NOT calibrated temperature)",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return int(seq.frame_ids[frame_idx])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=_REPO / DEFAULT_TIR_DIR)
    ap.add_argument("--out", type=Path, default=_REPO / "results" / "thermal_sample")
    ap.add_argument("--tiles", type=int, default=6, help="frames in the contact sheet")
    args = ap.parse_args()

    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    seq = load_sequence(args.root)

    flat = seq.frames.reshape(seq.n_frames, -1)
    count_min = int(flat.min())
    count_max = int(flat.max())
    per_frame_warm = np.array([warm_area_fraction(f) for f in seq.frames])
    path_len = seq.path_length()

    contact_path = out / "contact_sheet.png"
    overlay_path = out / "saliency_overlay.png"
    mid = seq.n_frames // 2
    sheet_ids = _contact_sheet(seq, args.tiles, contact_path)
    overlay_id = _saliency_overlay(seq, mid, overlay_path)

    summary = {
        "sample": "TUM2TWIN street-level TIR (thermal_tir_2016)",
        "sensor": "Jenoptik IR-TCM 640 uncooled microbolometer, FOV 65.2x51.3 deg",
        "n_frames": seq.n_frames,
        "frame_shape": list(seq.frames.shape[1:]),
        "dtype": str(seq.frames.dtype),
        "frame_id_range": [int(seq.frame_ids[0]), int(seq.frame_ids[-1])],
        "raw_count_range": [count_min, count_max],
        "raw_count_note": (
            "uncalibrated microbolometer counts; the constant low value "
            f"({count_min}) is a fixed border/sentinel, not scene radiance"
        ),
        "warm_area_fraction": {
            "min": float(per_frame_warm.min()),
            "mean": float(per_frame_warm.mean()),
            "max": float(per_frame_warm.max()),
        },
        "vehicle_path_length_m_enu": float(path_len),
        "vehicle_z_range_m_enu": [
            float(seq.pose[:, 3].min()),
            float(seq.pose[:, 3].max()),
        ],
        "enu_to_ecef_4x4": seq.enu_to_ecef.tolist(),
        "artefacts": {
            "contact_sheet": str(contact_path.relative_to(_REPO)),
            "contact_sheet_frame_ids": sheet_ids,
            "saliency_overlay": str(overlay_path.relative_to(_REPO)),
            "saliency_overlay_frame_id": overlay_id,
        },
        "scope": {
            "qualitative_only": True,
            "can_support": _CAN,
            "cannot_support": _CANNOT,
        },
    }
    summary_path = out / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    # Human report.
    print(f"TIR sample: {summary['sample']}")
    print(f"  frames           : {seq.n_frames} x {tuple(seq.frames.shape[1:])} {seq.frames.dtype}")
    print(f"  frame id range   : {summary['frame_id_range'][0]}..{summary['frame_id_range'][1]}")
    print(f"  raw count range  : {count_min}..{count_max}  (uncalibrated)")
    print(
        f"  warm-area frac   : min {per_frame_warm.min():.4f} "
        f"mean {per_frame_warm.mean():.4f} max {per_frame_warm.max():.4f}"
    )
    print(f"  vehicle path len : {path_len:.2f} m (ENU)")
    print(f"  wrote {contact_path.relative_to(_REPO)}")
    print(f"  wrote {overlay_path.relative_to(_REPO)}")
    print(f"  wrote {summary_path.relative_to(_REPO)}")
    print("  SCOPE: qualitative only — no calibration, no thermal GT, carrier pose.")


if __name__ == "__main__":
    main()
