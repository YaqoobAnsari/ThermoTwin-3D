#!/usr/bin/env python
"""TBBR thermal-bridge localisation eval — heat-loss saliency vs annotated bridges.

Scores the physics-motivated heat-loss saliency against the TBBR annotations over a set of
extracted scenes, aggregates the localisation metrics, and writes
``results/tbbr/summary.json`` + a figure. Comparable metric for the unified eval's TBBR rung.

Prep (one-time): extract scenes from the flight tar, e.g.
    tar --use-compress-program=unzstd -xf data/raw/tbbr/Flug1_100.tar.zst -C <dir> -T <members>

    python scripts/eval_tbbr.py --images-dir <dir>/train/images/Flug1_100
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from thermotwin.data.tbbr import THERMAL_CHANNEL, bridge_masks, saliency, score_image  # noqa: E402
from thermotwin.viz import apply_style, save_figure  # noqa: E402


def _figure(channels, anns):
    import matplotlib.pyplot as plt

    h, w = channels.shape[:2]
    therm = channels[..., THERMAL_CHANNEL]
    gt = np.zeros((h, w), bool)
    for m in bridge_masks(anns, h, w):
        gt |= m
    pred = saliency(therm)
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    ax[0].imshow(therm, cmap="inferno")
    ax[0].set_title("thermal channel")
    ax[1].imshow(therm, cmap="gray")
    ov = np.zeros((h, w, 4))
    ov[gt] = (0, 0.6, 1, 0.45)
    ax[1].imshow(ov)
    ax[1].set_title(f"annotated bridges (n={len(anns)})")
    ax[2].imshow(therm, cmap="gray")
    ov2 = np.zeros((h, w, 4))
    ov2[pred] = (1, 0, 0, 0.9)
    ax[2].imshow(ov2)
    ax[2].set_title("predicted heat-loss saliency")
    for a in ax:
        a.set_xticks([])
        a.set_yticks([])
    fig.suptitle("TBBR — heat-loss saliency vs annotated thermal bridges", fontsize=11)
    return fig


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--images-dir", required=True, help="dir of extracted Flug1_100/*.npy scenes")
    p.add_argument("--coco", default="data/raw/tbbr/Flug1_100-104Media_coco.json")
    a = p.parse_args()
    apply_style()
    out = _REPO / "results" / "tbbr"
    out.mkdir(parents=True, exist_ok=True)

    coco = json.loads(Path(a.coco).read_text())
    by_img = defaultdict(list)
    for ann in coco["annotations"]:
        by_img[ann["image_id"]].append(ann)
    name2id = {im["file_name"].split("/")[-1]: im["id"] for im in coco["images"]}

    per_image, made_fig = [], False
    for npy in sorted(Path(a.images_dir).glob("*.npy")):
        img_id = name2id.get(npy.name)
        if img_id is None or not by_img[img_id]:
            continue
        ch = np.load(npy)
        anns = by_img[img_id]
        per_image.append(score_image(ch, anns))
        if not made_fig:
            save_figure(_figure(ch, anns), out / "bridge_localisation")
            made_fig = True

    def agg(key):
        vals = [s[key] for s in per_image]
        return round(float(np.mean(vals)), 4) if vals else 0.0

    summary = {
        "n_scenes": len(per_image),
        "total_bridges": int(sum(s["n_bridges"] for s in per_image)),
        "precision": agg("precision"),
        "bridge_recall": agg("bridge_recall"),
        "enrichment": agg("enrichment"),
        "note": "physics-motivated heat-loss saliency vs annotated bridges; enrichment > 1 beats random",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"TBBR: {summary['n_scenes']} scenes, {summary['total_bridges']} bridges | "
          f"precision {summary['precision']} · bridge-recall {summary['bridge_recall']} · "
          f"enrichment {summary['enrichment']}× -> {out}/summary.json")


if __name__ == "__main__":
    main()
