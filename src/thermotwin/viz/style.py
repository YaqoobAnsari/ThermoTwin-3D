"""Publication-grade matplotlib style for ThermoTwin-3D figures.

One place to fix the look of every figure so the gallery is consistent and paper-ready:
a clean sans-serif, sensible sizes, perceptually-uniform colormaps with a fixed semantic
meaning (thermal vs material vs signed error), and a saver that always emits a vector PDF
(for the manuscript) alongside a 300-dpi PNG (for preview / slides).

Colormap semantics (use these, not ad-hoc choices, so a reader learns them once):
  * ``THERMAL`` (inferno)  — the dimensionless temperature θ / any heat field.
  * ``MATERIAL`` (viridis) — (log) conductivity / material identity.
  * ``DIVERGING`` (RdBu_r) — signed residual / error, always centred at 0.
  * ``SDF`` (coolwarm)     — signed distance, centred at 0 (inside<0 / outside>0).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless (HPC / no display)
import matplotlib.pyplot as plt  # noqa: E402

THERMAL = "inferno"
MATERIAL = "viridis"
DIVERGING = "RdBu_r"
SDF = "coolwarm"

# θ label reused across figures.
THETA_LABEL = r"$\theta=(T-T_{\mathrm{out}})/(T_{\mathrm{in}}-T_{\mathrm{out}})$"


def apply_style() -> None:
    """Set the project-wide rcParams. Call once before building figures."""
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.linewidth": 0.8,
            "axes.grid": False,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "legend.fontsize": 8,
            "legend.frameon": False,
            "image.cmap": THERMAL,
            "figure.constrained_layout.use": True,
            "mathtext.default": "regular",
        }
    )


def save_figure(fig, stem: str | Path, formats: tuple[str, ...] = ("png", "pdf")) -> list[Path]:
    """Save ``fig`` to ``<stem>.<ext>`` for each format (vector PDF + raster PNG).

    Returns the written paths. Creates parent dirs. Closes the figure.
    """
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    out = []
    for ext in formats:
        p = stem.with_suffix(f".{ext}")
        fig.savefig(p)
        out.append(p)
    plt.close(fig)
    return out
