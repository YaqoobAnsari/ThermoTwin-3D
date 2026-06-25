"""Publication-grade visualisation for ThermoTwin-3D.

Consistent, paper-ready figures of the model's inputs and outputs: 2-D field heatmaps
(:mod:`fields`), 3-D feature-coloured point clouds (:mod:`pointcloud`), latent-grid SDF slices
(:mod:`sdf`), under one shared style (:mod:`style`). Entry point: ``scripts/make_figures.py``.
"""

from __future__ import annotations

from .fields import figure_block1_sample, heatmap
from .pointcloud import figure_pointcloud_sample, scatter_3d
from .sdf import figure_sdf
from .style import apply_style, save_figure

__all__ = [
    "apply_style",
    "save_figure",
    "heatmap",
    "figure_block1_sample",
    "scatter_3d",
    "figure_pointcloud_sample",
    "figure_sdf",
]
