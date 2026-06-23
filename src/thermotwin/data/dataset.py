"""Torch dataset over the synthetic FEM corpus (``data/processed/<name>/``).

We learn the **dimensionless** temperature field

    θ(x) = (T(x) − T_out) / (T_in − T_out)

which, for linear steady conduction with fixed surface films, depends only on the
conductivity field and the film resistances — not on the absolute boundary
temperatures. So the operator learns the geometry-and-material signal directly, and
absolute temperatures / heat flux are recovered afterwards by rescaling.

Input channels per sample: ``[log10(k) (standardised), r_si, r_se]`` broadcast to
the grid. Target: ``θ`` (one channel). Samples are resampled along the wall (axis 1)
to a common width so they batch; the through-wall axis is already fixed-size.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import zoom
from torch.utils.data import Dataset

__all__ = ["SyntheticFEMDataset", "LOGK_MEAN", "LOGK_STD"]

# Standardisation constants for log10(conductivity); the material library spans
# ~0.035 (insulation) to ~160 (aluminium) W/(m·K).
LOGK_MEAN = 0.0
LOGK_STD = 1.5


class SyntheticFEMDataset(Dataset):
    """Loads ``(input, theta)`` pairs from a generated corpus directory."""

    def __init__(self, root: str | Path, target_width: int = 48):
        self.root = Path(root)
        manifest = json.loads((self.root / "manifest.json").read_text())
        self.samples = manifest["samples"]
        self.target_width = target_width

    def __len__(self) -> int:
        return len(self.samples)

    def _resample(self, arr: np.ndarray, order: int) -> np.ndarray:
        """Resample along axis 1 (along-wall) to ``target_width``."""
        w = arr.shape[1]
        if w == self.target_width:
            return arr
        return zoom(arr, (1.0, self.target_width / w), order=order)

    def __getitem__(self, i: int):
        rec = self.samples[i]
        d = np.load(self.root / rec["file"])
        k = d["k"].astype(np.float32)
        t = d["temperature"].astype(np.float32)
        t_in, t_out = float(d["t_indoor"]), float(d["t_outdoor"])
        r_si, r_se = float(d["r_si"]), float(d["r_se"])

        theta = (t - t_out) / (t_in - t_out)
        # k has sharp bridge edges -> nearest (order 0); theta is smooth -> linear.
        k = self._resample(k, order=0)
        theta = self._resample(theta, order=1)

        logk = (np.log10(k) - LOGK_MEAN) / LOGK_STD
        x = np.stack(
            [
                logk,
                np.full_like(logk, r_si),
                np.full_like(logk, r_se),
            ]
        ).astype(np.float32)
        y = theta[None].astype(np.float32)
        return torch.from_numpy(x), torch.from_numpy(y)
