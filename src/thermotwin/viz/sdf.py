"""Signed-distance-field views — the latent-grid geometry encoding GINO conditions on.

Shows the ``(G,G,G)`` SDF as three orthogonal central slices with the **zero-contour** (the
solid surface) overlaid, on a symmetric diverging scale (inside < 0 / outside > 0), plus a
montage of slices through the volume so the shape is legible.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import style


def _slice_panel(ax, sl: np.ndarray, vmax: float, title: str):
    im = ax.imshow(sl.T, origin="lower", cmap=style.SDF, vmin=-vmax, vmax=vmax, interpolation="bilinear")
    # zero level set = the surface
    ax.contour(sl.T, levels=[0.0], colors="k", linewidths=1.0)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    return im


def figure_sdf(npz_path_or_array, grid_label: str = ""):
    """Three orthogonal central slices of the SDF + a z-slice montage, with zero-contours."""
    import matplotlib.pyplot as plt

    if isinstance(npz_path_or_array, (str, Path)):
        sdf = np.load(npz_path_or_array, allow_pickle=True)["sdf"].astype(float)
        name = Path(npz_path_or_array).name
    else:
        sdf = np.asarray(npz_path_or_array, float)
        name = grid_label
    g = sdf.shape[0]
    vmax = float(np.abs(sdf).max()) or 1e-6
    c = g // 2

    fig = plt.figure(figsize=(11, 5.4))
    gs = fig.add_gridspec(2, 4)
    # three orthogonal central slices
    im = _slice_panel(fig.add_subplot(gs[0, 0]), sdf[c, :, :], vmax, f"slice  x = {c}/{g}")
    _slice_panel(fig.add_subplot(gs[0, 1]), sdf[:, c, :], vmax, f"slice  y = {c}/{g}")
    _slice_panel(fig.add_subplot(gs[0, 2]), sdf[:, :, c], vmax, f"slice  z = {c}/{g}")
    cb = fig.colorbar(im, ax=fig.axes[:3], fraction=0.04, pad=0.02)
    cb.set_label("signed distance  (inside < 0 < outside)", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    # montage of evenly-spaced z-slices
    zs = np.linspace(1, g - 2, 4).astype(int)
    for j, z in enumerate(zs):
        ax = fig.add_subplot(gs[1, j])
        _slice_panel(ax, sdf[:, :, z], vmax, f"z = {z}")
    fig.suptitle(f"Latent-grid signed-distance field  ({g}³){'  —  ' + name if name else ''}", fontsize=11)
    return fig
