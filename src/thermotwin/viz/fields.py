"""2-D field heatmaps — the Block-1 (regular-grid) view: k, θ, prior, residual, error.

The clean "heatmap" view of one wall cross-section: the conductivity field, the true
dimensionless temperature θ, the analytic 1-D clear-wall prior, and what the operator must
learn (θ − prior). If a prediction is supplied, it adds prediction + signed error panels with
a shared, symmetric error scale.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import style


def heatmap(ax, field: np.ndarray, *, cmap: str, vmin=None, vmax=None, title: str = "", cbar_label: str = ""):
    """Render one cell-centred 2-D field. Axis 0 is through-wall (indoor face at top)."""
    im = ax.imshow(field, origin="upper", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax,
                   interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("along-wall")
    ax.set_ylabel("through-wall\n(indoor → outdoor)")
    ax.set_xticks([])
    ax.set_yticks([])
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.ax.tick_params(labelsize=7)
    if cbar_label:
        cb.set_label(cbar_label, fontsize=8)
    return im


def figure_block1_sample(npz_path: str | Path, pred: np.ndarray | None = None):
    """Build the Block-1 panel figure for one corpus sample.

    Panels: log₁₀(k) · θ_true · θ_prior(1-D) · residual (θ_true − prior). With ``pred``:
    also θ_pred and signed error (pred − true) on a symmetric diverging scale.
    """
    import matplotlib.pyplot as plt

    from ..data.dataset import clearwall_theta

    d = np.load(npz_path)
    k = d["k"].astype(float)
    t = d["temperature"].astype(float)
    t_in, t_out = float(d["t_indoor"]), float(d["t_outdoor"])
    r_si, r_se = float(d["r_si"]), float(d["r_se"])
    dx0 = d["dx0"].astype(float)
    theta = (t - t_out) / (t_in - t_out)
    prior = clearwall_theta(k, dx0, r_si, r_se)
    resid = theta - prior

    panels = [
        (np.log10(k), style.MATERIAL, None, None, r"$\log_{10}\,k$  [W/(m·K)]", "conductivity field"),
        (theta, style.THERMAL, 0.0, 1.0, style.THETA_LABEL, "θ  (FV ground truth)"),
        (prior, style.THERMAL, 0.0, 1.0, style.THETA_LABEL, "θ  1-D clear-wall prior"),
    ]
    rmax = float(np.abs(resid).max()) or 1e-6
    panels.append((resid, style.DIVERGING, -rmax, rmax, "θ − prior", "bridge correction (what the operator learns)"))
    if pred is not None:
        pred = np.asarray(pred, float).reshape(theta.shape)
        err = pred - theta
        emax = float(np.abs(err).max()) or 1e-6
        panels += [
            (pred, style.THERMAL, 0.0, 1.0, style.THETA_LABEL, "θ  prediction"),
            (err, style.DIVERGING, -emax, emax, "pred − true", "signed error"),
        ]

    n = len(panels)
    ncol = 4 if n <= 4 else 3
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 3.0 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax, (f, cmap, vmn, vmx, clab, title) in zip(axes, panels, strict=False):
        heatmap(ax, f, cmap=cmap, vmin=vmn, vmax=vmx, title=title, cbar_label=clab)
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle(f"Block-1 wall cross-section — {Path(npz_path).name}", fontsize=11)
    return fig
