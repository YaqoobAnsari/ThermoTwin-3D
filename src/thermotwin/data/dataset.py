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

With ``return_physics=True`` each item also yields a ``phys`` dict
``{k, dx0, dy, r_si, r_se}`` of torch tensors on the *same resampled grid* as
``x``/``y`` — the bundle :func:`thermotwin.losses.heat_residual.heat_residual_loss`
needs to evaluate the steady FV residual on the predicted field. ``k`` is the
(order-0) resampled conductivity and ``dy`` is rescaled to keep the wall's physical
width fixed (``dy_resampled = dy_native · Ny_native / target_width``).
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

    def __init__(self, root: str | Path, target_width: int = 48, return_physics: bool = False):
        self.root = Path(root)
        manifest = json.loads((self.root / "manifest.json").read_text())
        self.samples = manifest["samples"]
        self.target_width = target_width
        self.return_physics = return_physics

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
        native_ny = k.shape[1]

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
        x_t, y_t = torch.from_numpy(x), torch.from_numpy(y)
        if not self.return_physics:
            return x_t, y_t

        # Physics bundle on the *same resampled grid* as x/y. Lateral cells were
        # resampled native_ny -> target_width, so the per-cell width scales to keep
        # the wall's physical width fixed; dx0 (through-wall) is untouched.
        dy = float(d["dy"]) * (native_ny / k.shape[1])
        phys = {
            "k": torch.from_numpy(k.astype(np.float32)),
            "dx0": torch.from_numpy(d["dx0"].astype(np.float32)),
            "dy": torch.tensor(dy, dtype=torch.float32),
            "r_si": torch.tensor(r_si, dtype=torch.float32),
            "r_se": torch.tensor(r_se, dtype=torch.float32),
        }
        return x_t, y_t, phys
