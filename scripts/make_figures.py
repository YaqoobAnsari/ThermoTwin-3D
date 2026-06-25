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
    p.add_argument("--sample", default=None, help="render this one .npz sample only (2-D or point-cloud)")
    p.add_argument("--pred", default=None, help="prediction .npy aligned to --sample (pred + error panels)")
    p.add_argument(
        "--preds",
        default=None,
        help="a results/preds/<stem>__<model>.npz dump; the prediction for --sample is matched "
        "by file name and overlaid (pred + signed-error panels).",
    )
    a = p.parse_args()

    apply_style()
    out = Path(a.out)
    made: list[Path] = []

    # Resolve a prediction for the chosen sample, from an explicit .npy or a --preds dump.
    pred = None
    if a.pred:
        pred = np.load(a.pred)
    elif a.preds and a.sample:
        z = np.load(a.preds, allow_pickle=True)
        files = [str(x) for x in z["files"]]
        nm = Path(a.sample).name
        if nm in files:
            pred = z[f"p{files.index(nm)}"]
            print(f"overlaying prediction for {nm} from {Path(a.preds).name}")
        else:
            print(f"warning: {nm} not in {a.preds} — rendering ground truth only")

    if a.sample:  # single-sample mode — dispatch on what the .npz contains
        s = Path(a.sample)
        keys = set(np.load(s, allow_pickle=True).keys())
        if "points" in keys:  # point-cloud sample
            made += save_figure(figure_pointcloud_sample(s, pred=pred), out / f"pointcloud_{s.stem}")
            made += save_figure(figure_sdf(s), out / f"sdf_{s.stem}")
        elif {"k", "temperature"} <= keys:  # Block-1 2-D grid sample
            made += save_figure(figure_block1_sample(s, pred=pred), out / f"block1_{s.stem}")
        else:
            raise SystemExit(f"don't know how to render {s} (keys: {sorted(keys)})")
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
        try:
            print("  ", f.resolve().relative_to(_REPO))
        except ValueError:
            print("  ", f)


if __name__ == "__main__":
    main()
