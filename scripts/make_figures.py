#!/usr/bin/env python
"""Generate the ThermoTwin-3D figure gallery — inputs & outputs, paper-ready.

Renders, from the corpora on disk, the views the project needs for reports/papers:
  * Block-1 2-D field heatmaps (conductivity, θ ground truth, 1-D prior, bridge correction);
  * Block-2 / real-CityGML 3-D point clouds (geometry · material · θ heat field · correction);
  * the latent-grid signed-distance field (orthogonal slices + z-montage, zero-contour).

Each figure is written as a vector PDF (manuscript) + 300-dpi PNG (preview) under
``results/figures/``. Optionally overlay a model prediction with ``--pred <npy>`` (an (N,)
or (Nx,Ny) array aligned to the chosen sample) to add prediction + signed-error panels.

    python scripts/make_figures.py
    python scripts/make_figures.py --sample data/processed/block2_realcg_val/sample_00003.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from thermotwin.viz import (  # noqa: E402
    apply_style,
    figure_block1_sample,
    figure_pointcloud_sample,
    figure_sdf,
    save_figure,
)

PROC = _REPO / "data" / "processed"
OUT = _REPO / "results" / "figures"


def _pick(corpus: str, prefer_bridges: bool = True) -> Path | None:
    """A representative sample from a corpus (prefer one with thermal bridges)."""
    mf = PROC / corpus / "manifest.json"
    if not mf.exists():
        return None
    rows = json.loads(mf.read_text())["samples"]
    if prefer_bridges:
        br = [r for r in rows if int(r.get("n_bridges", 0) or 0) > 0]
        rows = br or rows
    return PROC / corpus / rows[0]["file"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=str(OUT), help="output directory for figures")
    p.add_argument("--sample", default=None, help="render this one .npz point-cloud sample only")
    p.add_argument("--pred", default=None, help="optional prediction .npy to overlay (pred + error panels)")
    a = p.parse_args()

    apply_style()
    out = Path(a.out)
    pred = np.load(a.pred) if a.pred else None
    made: list[Path] = []

    if a.sample:  # single-sample mode
        s = Path(a.sample)
        made += save_figure(figure_pointcloud_sample(s, pred=pred), out / f"pointcloud_{s.stem}")
        made += save_figure(figure_sdf(s), out / f"sdf_{s.stem}")
    else:  # full gallery
        b1 = _pick("block1_val")
        if b1:
            made += save_figure(figure_block1_sample(b1), out / "block1_fields")
        for corpus, name in [
            ("block2_realcg_val", "realcg"),  # real CityGML geometry (headline)
            ("block2_irreg_val", "irreg"),
            ("block2_hard_val", "hard"),
            ("block2_val", "box"),
        ]:
            s = _pick(corpus)
            if s:
                made += save_figure(figure_pointcloud_sample(s), out / f"pointcloud_{name}")
                made += save_figure(figure_sdf(s), out / f"sdf_{name}")

    print(f"wrote {len(made)} files to {out}/:")
    for f in made:
        print("  ", f.relative_to(_REPO))


if __name__ == "__main__":
    main()
