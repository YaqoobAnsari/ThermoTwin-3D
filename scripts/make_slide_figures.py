#!/usr/bin/env python
"""Generate the explanatory figures for the AiC presentation into presentations/figs/.

(1) recipe_decomposition.png — one envelope sample shown as input conductivity, the analytic
    1-D prior field, the true field, and the residual (= the thermal bridge the operator learns).
(2) inputs_outputs.png — the same geometry coloured by the model's inputs and its output.
These make "a thermal field" and "a thermal bridge" concrete for a non-expert audience.
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_REPO = Path(__file__).resolve().parents[1]
OUT = _REPO / "presentations" / "figs"
OUT.mkdir(parents=True, exist_ok=True)


def _pick(corpus: str):
    """Pick the validation sample with the strongest, clearest thermal bridge."""
    files = sorted(glob.glob(str(_REPO / f"data/processed/block2_{corpus}_val/sample_*.npz")))
    best, mag = None, -1.0
    for f in files:
        d = np.load(f)
        r = float(np.abs(d["theta"] - d["prior"]).max())
        if r > mag:
            mag, best = r, f
    return np.load(best)


def _scatter(ax, pts, vals, cmap, title, label):
    p = ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=vals, cmap=cmap, s=7, depthshade=False)
    ax.set_title(title, fontsize=10)
    ax.set_axis_off()
    ax.view_init(elev=22, azim=-60)
    cb = ax.figure.colorbar(p, ax=ax, shrink=0.55, pad=0.0)
    cb.set_label(label, fontsize=7)
    cb.ax.tick_params(labelsize=6)


def _highlight_bridge(ax, pts, resid, title, thresh=0.04):
    """Ghost the whole geometry, then light up only the bridge points (large |residual|)."""
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c="0.8", s=2, alpha=0.12, depthshade=False)
    m = np.abs(resid) > thresh
    if m.any():
        p = ax.scatter(pts[m, 0], pts[m, 1], pts[m, 2], c=np.abs(resid[m]),
                       cmap="autumn_r", s=18, depthshade=False, vmin=0)
        cb = ax.figure.colorbar(p, ax=ax, shrink=0.55, pad=0.0)
        cb.set_label(r"$|\Delta\theta|$", fontsize=7)
        cb.ax.tick_params(labelsize=6)
    ax.set_title(title, fontsize=10)
    ax.set_axis_off()
    ax.view_init(elev=22, azim=-60)


def main():
    corpus = sys.argv[1] if len(sys.argv) > 1 else "hard"
    d = _pick(corpus)
    pts = d["points"]
    theta, prior = d["theta"], d["prior"]
    logk = d["feats"][:, 0]
    resid = theta - prior

    # (1) the recipe decomposition — input -> prior -> truth -> residual(bridge)
    fig = plt.figure(figsize=(13.5, 3.6))
    _scatter(fig.add_subplot(1, 4, 1, projection="3d"), pts, logk, "viridis",
             "(a) INPUT: conductivity\n(bridge = high conductivity)", "log-k")
    _scatter(fig.add_subplot(1, 4, 2, projection="3d"), pts, prior, "inferno",
             "(b) 1-D physics PRIOR\n(smooth, bridge-blind)", r"$\theta$")
    _scatter(fig.add_subplot(1, 4, 3, projection="3d"), pts, theta, "inferno",
             "(c) TRUE field (target)", r"$\theta$")
    _highlight_bridge(fig.add_subplot(1, 4, 4, projection="3d"), pts, resid,
                      "(d) RESIDUAL = truth - prior\n(the thermal bridge the operator learns)")
    fig.suptitle(f"The recipe, made visible: the network only learns the residual (the thermal bridge) "
                 f"on top of the analytic prior  [corpus: {corpus}]", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT / "recipe-decomposition.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # (2) inputs / output, plainer
    fig = plt.figure(figsize=(10.5, 3.7))
    _scatter(fig.add_subplot(1, 3, 1, projection="3d"), pts, logk, "viridis",
             "INPUT 1: geometry + conductivity", "log-k")
    _scatter(fig.add_subplot(1, 3, 2, projection="3d"), pts, prior, "inferno",
             "INPUT 2: analytic 1-D prior field", r"$\theta_{\mathrm{prior}}$")
    _scatter(fig.add_subplot(1, 3, 3, projection="3d"), pts, theta, "inferno",
             r"OUTPUT: temperature field $\theta$", r"$\theta$")
    fig.suptitle("What goes in, what comes out (one envelope surface)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT / "inputs-outputs.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    print("wrote", OUT / "recipe_decomposition.png", "and", OUT / "inputs_outputs.png")
    print(f"sample: bridge fraction = {float((np.abs(resid) > 0.02).mean()) * 100:.1f}% of points, "
          f"max |residual| = {float(np.abs(resid).max()):.3f}")


if __name__ == "__main__":
    main()
