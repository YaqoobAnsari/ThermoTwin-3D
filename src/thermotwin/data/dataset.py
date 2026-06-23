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

__all__ = [
    "SyntheticFEMDataset",
    "build_input_channels",
    "clearwall_theta",
    "LOGK_MEAN",
    "LOGK_STD",
    "FEATURE_SETS",
    "ENRICHED_CLEARWALL_CHANNEL",
]

# Standardisation constants for log10(conductivity); the material library spans
# ~0.035 (insulation) to ~160 (aluminium) W/(m·K).
LOGK_MEAN = 0.0
LOGK_STD = 1.5

# Supported input-feature sets and their channel counts.
FEATURE_SETS = {"base": 3, "enriched": 6}
# Channel index of the clear-wall (1-D analytic) θ prior in the 'enriched' stack.
# Layout: [logk_std, r_si, r_se, clearwall_theta, throughwall_coord, lateral_coord].
ENRICHED_CLEARWALL_CHANNEL = 3


def clearwall_theta(k: np.ndarray, dx0: np.ndarray, r_si: float, r_se: float) -> np.ndarray:
    """Analytic per-column 1-D dimensionless temperature for an ``(Nx, Ny)`` k-field.

    Closed-form clear-wall prior (verified against the FV solver to machine
    precision on no-bridge samples):

        θ1d_j = 1 − (r_si + Σ_{m<j} R_m + R_j/2) / R_total
        R_m = dx0_m / k_m,   R_total = r_si + Σ_m R_m + r_se.

    Axis 0 is through-wall; the lo face is indoor (``θ_air = 1``, film ``r_si``) and
    the hi face is outdoor (``θ_air = 0``, film ``r_se``). The ``R_j/2`` term is the
    cell-centre offset, so this reproduces the cell-centred FV field exactly where
    the 1-D assumption holds and leaves only the lateral-spreading correction near
    bridges for the operator to learn.

    Args:
        k: conductivity field [W/(m·K)], shape ``(Nx, Ny)``.
        dx0: through-wall per-cell spacing [m], length ``Nx``.
        r_si: indoor (lo-face) film resistance [m²K/W].
        r_se: outdoor (hi-face) film resistance [m²K/W].

    Returns:
        ``θ1d`` of shape ``(Nx, Ny)``.
    """
    k = np.asarray(k, dtype=np.float64)
    dx0 = np.asarray(dx0, dtype=np.float64)
    r = dx0[:, None] / k  # per-cell resistance, (Nx, Ny)
    cum_before = np.cumsum(r, axis=0) - r  # Σ_{m<j} R_m
    r_total = r_si + r.sum(axis=0, keepdims=True) + r_se  # (1, Ny)
    return 1.0 - (r_si + cum_before + 0.5 * r) / r_total  # (Nx, Ny)


def build_input_channels(
    k: np.ndarray,
    dx0: np.ndarray,
    dy: float,
    r_si: float,
    r_se: float,
    feature_set: str = "base",
) -> np.ndarray:
    """Stack the operator input channels ``(C, Nx, Ny)`` from physical fields.

    Shared by :class:`SyntheticFEMDataset` and the ablation eval so both featurise
    identically. ``dy`` is accepted for signature stability (the per-channel features
    are dimensionless / index-based) and currently unused.

    feature sets:

    * ``'base'`` (3ch): ``[log10(k) standardised, r_si, r_se]`` — current behaviour.
    * ``'enriched'`` (6ch): base **plus**
      ``[clearwall_theta, throughwall_coord, lateral_coord]`` where
      ``clearwall_theta`` is the verified 1-D analytic prior (channel
      :data:`ENRICHED_CLEARWALL_CHANNEL` = 3), ``throughwall_coord`` is the axis-0
      index normalised to ``[0, 1]`` and ``lateral_coord`` the axis-1 index
      normalised to ``[0, 1]``, both broadcast across the grid.

    Args:
        k: conductivity field [W/(m·K)], shape ``(Nx, Ny)``.
        dx0: through-wall per-cell spacing [m], length ``Nx``.
        dy: along-wall cell size [m] (unused; kept for API symmetry).
        r_si: indoor film resistance [m²K/W].
        r_se: outdoor film resistance [m²K/W].
        feature_set: ``'base'`` or ``'enriched'``.

    Returns:
        Input array of shape ``(C, Nx, Ny)`` as ``float32`` (C = 3 or 6).
    """
    del dy  # currently unused; part of the stable physical signature.
    if feature_set not in FEATURE_SETS:
        raise ValueError(
            f"unknown feature_set '{feature_set}'; expected one of {tuple(FEATURE_SETS)}"
        )
    k = np.asarray(k, dtype=np.float32)
    nx, ny = k.shape

    logk = ((np.log10(k.astype(np.float64)) - LOGK_MEAN) / LOGK_STD).astype(np.float32)
    channels = [
        logk,
        np.full((nx, ny), r_si, dtype=np.float32),
        np.full((nx, ny), r_se, dtype=np.float32),
    ]

    if feature_set == "enriched":
        theta1d = clearwall_theta(k, dx0, r_si, r_se).astype(np.float32)
        tw = np.linspace(0.0, 1.0, nx, dtype=np.float32)[:, None]
        lat = np.linspace(0.0, 1.0, ny, dtype=np.float32)[None, :]
        channels += [
            theta1d,
            np.broadcast_to(tw, (nx, ny)).copy(),
            np.broadcast_to(lat, (nx, ny)).copy(),
        ]

    return np.stack(channels).astype(np.float32)


class SyntheticFEMDataset(Dataset):
    """Loads ``(input, theta)`` pairs from a generated corpus directory."""

    def __init__(
        self,
        root: str | Path,
        target_width: int = 48,
        return_physics: bool = False,
        feature_set: str = "base",
    ):
        if feature_set not in FEATURE_SETS:
            raise ValueError(
                f"unknown feature_set '{feature_set}'; expected one of {tuple(FEATURE_SETS)}"
            )
        self.root = Path(root)
        manifest = json.loads((self.root / "manifest.json").read_text())
        self.samples = manifest["samples"]
        self.target_width = target_width
        self.return_physics = return_physics
        self.feature_set = feature_set

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

        dx0 = d["dx0"].astype(np.float32)
        # Lateral cells were resampled native_ny -> target_width, so the per-cell
        # width scales to keep the wall's physical width fixed; dx0 is untouched.
        dy = float(d["dy"]) * (native_ny / k.shape[1])

        x = build_input_channels(k, dx0, dy, r_si, r_se, feature_set=self.feature_set)
        y = theta[None].astype(np.float32)
        x_t, y_t = torch.from_numpy(x), torch.from_numpy(y)
        if not self.return_physics:
            return x_t, y_t

        # Physics bundle on the *same resampled grid* as x/y.
        phys = {
            "k": torch.from_numpy(k.astype(np.float32)),
            "dx0": torch.from_numpy(dx0),
            "dy": torch.tensor(dy, dtype=torch.float32),
            "r_si": torch.tensor(r_si, dtype=torch.float32),
            "r_se": torch.tensor(r_se, dtype=torch.float32),
        }
        return x_t, y_t, phys
