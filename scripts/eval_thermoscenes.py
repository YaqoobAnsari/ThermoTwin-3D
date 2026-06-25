#!/usr/bin/env python
"""ThermoScenes calibrated-thermal localisation — foundation layer.

For each real-building scene: decode a representative thermal frame to **absolute °C**, compute
the heat-loss anomaly (prior-residual) map, write per-scene stats + a figure (calibrated °C +
anomaly overlay) to ``results/thermoscenes/``. This is the calibrated, real-thermal artifact;
the next layer fuses the °C onto the COLMAP geometry and overlays the operator's clear-wall
baseline to localise where the field departs from "expected".

    python scripts/eval_thermoscenes.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from thermotwin.data.thermoscenes import (  # noqa: E402
    BUILDING_SCENES,
    decode_celsius,
    heat_loss_anomaly,
    open_archive,
    temperature_bounds,
    thermal_members,
)
from thermotwin.viz import apply_style, save_figure  # noqa: E402
from thermotwin.viz import style as vstyle  # noqa: E402

ARCHIVE = _REPO / "data" / "raw" / "thermoscenes" / "ThermoScenes_full.zip"


def _figure(scene: str, celsius: np.ndarray, mask: np.ndarray):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    im = axes[0].imshow(celsius, cmap=vstyle.THERMAL)
    axes[0].set_title(f"{scene} — calibrated surface temperature")
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    cb = fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.02)
    cb.set_label("°C", fontsize=8)
    axes[1].imshow(celsius, cmap="gray")
    over = np.zeros((*mask.shape, 4))
    over[mask] = (1.0, 0.0, 0.0, 0.9)  # red where heat-loss anomalies are flagged
    axes[1].imshow(over)
    axes[1].set_title(f"heat-loss anomalies ({100 * mask.mean():.2f}% of pixels)")
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    fig.suptitle("ThermoScenes — calibrated real-thermal heat-loss localisation", fontsize=11)
    return fig


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive", default=str(ARCHIVE))
    p.add_argument("--figures", type=int, default=3, help="how many scenes to render figures for")
    a = p.parse_args()
    apply_style()
    out = _REPO / "results" / "thermoscenes"
    out.mkdir(parents=True, exist_ok=True)

    zf = open_archive(a.archive)
    summary = []
    n_fig = 0
    for scene in BUILDING_SCENES:
        members = thermal_members(zf, scene)
        if not members:
            continue
        bounds = temperature_bounds(zf, scene)
        mid = members[len(members) // 2]
        celsius = decode_celsius(zf, mid, bounds)
        mask, _ = heat_loss_anomaly(celsius)
        summary.append({
            "scene": scene,
            "n_thermal_frames": len(members),
            "temp_min_C": round(bounds[0], 2),
            "temp_max_C": round(bounds[1], 2),
            "frame_min_C": round(float(celsius.min()), 2),
            "frame_max_C": round(float(celsius.max()), 2),
            "anomaly_pixel_frac": round(float(mask.mean()), 5),
        })
        if n_fig < a.figures:
            save_figure(_figure(scene, celsius, mask), out / f"{scene}_thermal")
            n_fig += 1
    (out / "summary.json").write_text(json.dumps({"archive": a.archive, "scenes": summary}, indent=2))
    print(f"wrote {len(summary)} scene summaries + {n_fig} figures to {out}/")
    for s in summary:
        print(f"  {s['scene']:20s} {s['n_thermal_frames']:3d} frames  "
              f"{s['temp_min_C']:6.1f}…{s['temp_max_C']:5.1f} °C  "
              f"anomalies {s['anomaly_pixel_frac'] * 100:.2f}%")


if __name__ == "__main__":
    main()
