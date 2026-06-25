"""3-D point-cloud views — the as-built / Block-2 geometry the operator actually consumes.

Renders one corpus sample (synthetic block, irregular block, or a real CityGML building) as a
panel of feature-coloured point clouds: bare geometry · material (log-conductivity) · the
predicted/true θ heat field · the bridge correction (θ − prior). The same cloud the GINO /
Transolver operators ingest, coloured by what each channel means.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import style

# feats layout = [logk_std, r_si, r_se, theta1d]; de-standardise channel 0 to log10(k).
_LOGK_STD = 1.5
_LOGK_MEAN = 0.0


def scatter_3d(ax, points: np.ndarray, values: np.ndarray, *, cmap: str, vmin=None, vmax=None,
               title: str = "", cbar_label: str = "", s: float = 3.0):
    """Scatter a coloured point cloud on a 3-D axis with equal aspect + clean panes."""
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    p = ax.scatter(x, y, z, c=values, cmap=cmap, vmin=vmin, vmax=vmax, s=s, alpha=0.85,
                   depthshade=False, linewidths=0)
    ax.set_title(title, pad=2)
    # True geometric aspect from the data extent, but zoomed in so an elongated shell fills
    # the panel instead of shrinking to a sliver; tight limits remove matplotlib's auto-margin.
    r = np.ptp(points, axis=0).astype(float)
    r[r == 0] = 1.0
    try:
        ax.set_box_aspect(tuple(r), zoom=1.4)
    except TypeError:  # older matplotlib without the zoom kwarg
        ax.set_box_aspect(tuple(r))
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min(), y.max())
    ax.set_zlim(z.min(), z.max())
    ax.view_init(elev=24, azim=-55)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_alpha(0.04)
        axis.line.set_color((0, 0, 0, 0.25))
    cb = ax.figure.colorbar(p, ax=ax, fraction=0.025, pad=0.0, shrink=0.6)
    cb.ax.tick_params(labelsize=7)
    if cbar_label:
        cb.set_label(cbar_label, fontsize=8)
    return p


def figure_pointcloud_sample(npz_path: str | Path, pred: np.ndarray | None = None, max_points: int = 6000):
    """Panel figure for one point-cloud sample: geometry · material · θ · correction (+pred/err)."""
    import matplotlib.pyplot as plt

    d = np.load(npz_path, allow_pickle=True)
    pts = d["points"].astype(float)
    feats = d["feats"].astype(float)
    theta = d["theta"].astype(float)
    prior = d["prior"].astype(float)
    logk = feats[:, 0] * _LOGK_STD + _LOGK_MEAN  # -> log10(k)

    if len(pts) > max_points:  # thin dense clouds for legible scatter
        idx = np.random.default_rng(0).choice(len(pts), max_points, replace=False)
        pts, feats, theta, prior, logk = pts[idx], feats[idx], theta[idx], prior[idx], logk[idx]
        if pred is not None:
            pred = np.asarray(pred, float).ravel()[idx]

    resid = theta - prior
    rmax = float(np.abs(resid).max()) or 1e-6
    panels = [
        (np.full(len(pts), 0.5), "Greys", 0.0, 1.0, "", "geometry (point cloud)"),
        (logk, style.MATERIAL, None, None, r"$\log_{10}\,k$", "material (conductivity)"),
        (theta, style.THERMAL, 0.0, 1.0, style.THETA_LABEL, "θ  (ground-truth heat field)"),
        (resid, style.DIVERGING, -rmax, rmax, "θ − prior", "bridge correction"),
    ]
    if pred is not None:
        pred = np.asarray(pred, float).ravel()
        err = pred - theta
        emax = float(np.abs(err).max()) or 1e-6
        panels += [
            (pred, style.THERMAL, 0.0, 1.0, style.THETA_LABEL, "θ  prediction"),
            (err, style.DIVERGING, -emax, emax, "pred − true", "signed error"),
        ]

    n = len(panels)
    ncol = 2  # 2-up so each 3-D panel is large and legible (4 panels -> 2x2, 6 -> 3x2)
    nrow = int(np.ceil(n / ncol))
    fig = plt.figure(figsize=(5.2 * ncol, 4.4 * nrow))
    for i, (vals, cmap, vmn, vmx, clab, title) in enumerate(panels):
        ax = fig.add_subplot(nrow, ncol, i + 1, projection="3d")
        scatter_3d(ax, pts, vals, cmap=cmap, vmin=vmn, vmax=vmx, title=title, cbar_label=clab, s=4.0)
    fig.suptitle(f"Point-cloud sample — {Path(npz_path).name}  ({len(d['points'])} pts)", fontsize=11)
    return fig
