"""Differentiable U-value loss — the building-metric training lever.

The effective transmittance ``U`` of a wall section depends *only* on the predicted
field at the indoor (axis-0 ``lo``) face: it is the boundary heat flux per unit area
and temperature difference,

    U = (1 / A_wall) · Σ_lo_face  g_bnd · (1 − θ_lo),
    g_bnd = dy / ((dx0[0]/2)/k_lo + r_si),   A_wall = dy · Ny.

This is the *same* discrete boundary flux as the numpy reporter
:func:`thermotwin.eval.building.effective_u_from_theta` and the ``g_bnd_lo`` block of
:func:`thermotwin.losses.heat_residual.heat_residual`, written here in differentiable
torch so it can be added to the training loss. Because ``U`` only reads row 0 of
``θ``, the gradient of this loss touches *only* the indoor face — exactly the
near-boundary value that sets U-MAE — making it the most targeted lever for the
U-value metric. It only constrains row 0, so keep the field loss dominant and use a
small ``u_weight``.
"""

from __future__ import annotations

import torch
from torch import Tensor

__all__ = ["u_from_theta", "u_value_loss"]


def _as_field(t: Tensor) -> Tensor:
    """Coerce ``(B,1,Nx,Ny)`` / ``(B,Nx,Ny)`` / ``(Nx,Ny)`` to ``(B,Nx,Ny)``."""
    if t.dim() == 2:
        return t.unsqueeze(0)
    if t.dim() == 3:
        return t
    if t.dim() == 4:
        if t.shape[1] != 1:
            raise ValueError(f"expected a single channel, got shape {tuple(t.shape)}")
        return t[:, 0]
    raise ValueError(f"field must be 2-D, 3-D or 4-D, got {t.dim()}-D")


def u_from_theta(
    theta: Tensor,
    k: Tensor,
    dx0: Tensor,
    dy: Tensor | float,
    r_si: Tensor | float,
) -> Tensor:
    """Effective U-value ``(B,)`` [W/(m²K)] from a dimensionless field ``θ``.

    Differentiable wrt ``theta`` (the gradient is non-zero only on the indoor-face
    row 0). Matches :func:`thermotwin.eval.building.effective_u_from_theta`.

    Args:
        theta: dimensionless temperature, ``(B,1,Nx,Ny)``, ``(B,Nx,Ny)`` or
            ``(Nx,Ny)``. Axis 0 (``Nx``) is through-wall; row 0 is the indoor face.
        k: conductivity [W/(m·K)], same spatial shape as ``theta``.
        dx0: through-wall per-cell sizes [m], ``(Nx,)`` or ``(B,Nx)``.
        dy: along-wall cell size [m], scalar or ``(B,)``.
        r_si: indoor film resistance [m²K/W], scalar or ``(B,)``.

    Returns:
        U-value tensor of shape ``(B,)``.
    """
    theta = _as_field(theta)
    k = _as_field(k)
    if k.shape != theta.shape:
        raise ValueError(f"k shape {tuple(k.shape)} != theta shape {tuple(theta.shape)}")

    batch, nx, ny = theta.shape
    device, dtype = theta.device, theta.dtype

    dx0 = torch.as_tensor(dx0, device=device, dtype=dtype)
    if dx0.dim() == 1:
        dx0 = dx0.unsqueeze(0)
    dx0 = dx0.expand(batch, nx)

    dy = torch.as_tensor(dy, device=device, dtype=dtype).reshape(-1)
    r_si = torch.as_tensor(r_si, device=device, dtype=dtype).reshape(-1)
    if dy.numel() == 1:
        dy = dy.expand(batch)
    if r_si.numel() == 1:
        r_si = r_si.expand(batch)

    k_lo = k[:, 0, :]  # indoor-face row, (B, Ny)
    r_half = (dx0[:, 0:1] / 2.0) / k_lo  # (B, Ny)
    g_bnd = dy.view(batch, 1) / (r_half + r_si.view(batch, 1))  # (B, Ny) [W/K]
    q_over_dt = (g_bnd * (1.0 - theta[:, 0, :])).sum(dim=1)  # (B,) [W/K]
    wall_area = dy * ny  # (B,) [m^2]
    return q_over_dt / wall_area  # (B,) U in W/(m^2 K)


def u_value_loss(
    pred_theta: Tensor,
    target_theta: Tensor,
    k: Tensor,
    dx0: Tensor,
    dy: Tensor | float,
    r_si: Tensor | float,
    smooth_l1: bool = False,
) -> Tensor:
    """Mean U-value discrepancy between predicted and target fields (a scalar loss).

    Computes ``U`` from both fields via the indoor-face boundary flux and penalises
    their difference, batch-averaged. Differentiable wrt ``pred_theta`` (the gradient
    reaches only row 0). By default uses MSE (smoother near zero, better grads); set
    ``smooth_l1=True`` for a smooth-L1 penalty.

    Args:
        pred_theta: predicted dimensionless field (any accepted shape).
        target_theta: ground-truth dimensionless field (same shape contract).
        k: conductivity field matching the spatial shape.
        dx0: through-wall per-cell sizes [m].
        dy: along-wall cell size [m].
        r_si: indoor film resistance [m²K/W].
        smooth_l1: use smooth-L1 instead of squared error.

    Returns:
        Scalar loss tensor.
    """
    u_pred = u_from_theta(pred_theta, k, dx0, dy, r_si)
    u_true = u_from_theta(target_theta, k, dx0, dy, r_si)
    if smooth_l1:
        return torch.nn.functional.smooth_l1_loss(u_pred, u_true)
    return torch.mean((u_pred - u_true) ** 2)
