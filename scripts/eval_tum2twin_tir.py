#!/usr/bin/env python
"""TUM2TWIN measured-thermal localisation — airborne IR orthophoto vs CityGML geometry.

Registers the TUM2TWIN thermal orthophoto to the LoD2 CityGML footprints and scores whether
measured heat-loss anomalies concentrate on the building envelopes we model (footprint
enrichment). Writes ``results/tum2twin_tir/summary.json`` + an overlay figure. This is the
unified eval's *measured-thermal* cross-task rung (the only one scored on real IR, not simulated
physics).

    python scripts/eval_tum2twin_tir.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from thermotwin.data.thermoscenes import heat_loss_anomaly  # noqa: E402
from thermotwin.data.tum2twin_tir import footprint_masks, load_orthophoto, score  # noqa: E402
from thermotwin.geometry.citygml import read_citygml_footprints  # noqa: E402
from thermotwin.viz import apply_style, save_figure  # noqa: E402

TIF = _REPO / "data/raw/tum2twin-datasets/tif/TUM_flipped_georeferenced.tif"
CITYGML = _REPO / "data/raw/tum2twin-datasets/citygml/lod2-building-datasets"


def _figure(thermal, valid, masks, transform):
    import matplotlib.pyplot as plt

    fill = float(thermal[valid].mean())
    field = np.where(valid, thermal, fill)
    anomaly, _ = heat_loss_anomaly(field)
    anomaly &= valid
    union = np.zeros_like(valid)
    for _bid, m in masks:
        union |= m

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
    disp = np.where(valid, thermal, np.nan)
    ax[0].imshow(disp, cmap="inferno")
    ax[0].set_title("TUM2TWIN thermal orthophoto (relative)")
    ax[1].imshow(disp, cmap="gray")
    fo = np.zeros((*valid.shape, 4))
    fo[union] = (0.1, 0.6, 1.0, 0.35)
    ax[1].imshow(fo)
    ao = np.zeros((*valid.shape, 4))
    ao[anomaly] = (1.0, 0.1, 0.1, 0.95)
    ax[1].imshow(ao)
    ax[1].set_title("CityGML footprints (blue) + heat-loss anomalies (red)")
    for a in ax:
        a.set_xticks([])
        a.set_yticks([])
    fig.suptitle("Measured airborne IR vs modelled building envelopes — TUM campus", fontsize=11)
    return fig


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tif", default=str(TIF))
    p.add_argument("--citygml", default=str(CITYGML))
    a = p.parse_args()
    apply_style()
    out = _REPO / "results" / "tum2twin_tir"
    out.mkdir(parents=True, exist_ok=True)

    thermal, valid, transform = load_orthophoto(a.tif)
    footprints = read_citygml_footprints(a.citygml)
    masks = footprint_masks(footprints, transform, thermal.shape)
    summary = score(thermal, valid, masks)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    save_figure(_figure(thermal, valid, masks, transform), out / "tir_localisation")
    print(
        f"TUM2TWIN-TIR: {summary['n_buildings']} buildings | "
        f"enrichment {summary['enrichment']}× | "
        f"building contrast +{summary['mean_building_thermal_contrast_dn']} DN -> {out}/summary.json"
    )


if __name__ == "__main__":
    main()
