"""Differentiable steady heat-conduction PDE residual for the PINN loss.

This is the PyTorch, autograd-enabled twin of the finite-volume operator in
:mod:`thermotwin.physics.steady_fv`. Where that module *assembles and solves* the
linear system ``A T = b`` on a cell-centred grid, this module evaluates the same
discrete operator as a *residual* on a predicted field, so it can be added to a
training loss and back-propagated into the network.

We work in the dimensionless temperature

    theta = (T - T_out) / (T_in - T_out),

so the indoor air (the axis-0 *lo* face, with film ``r_si``) is ``theta = 1`` and
the outdoor air (the axis-0 *hi* face, with film ``r_se``) is ``theta = 0``. The
steady equation ``∇·(k ∇T) = 0`` is linear and homogeneous, so it holds for
``theta`` verbatim: the affine map ``T -> theta`` rescales every conductance term
by the same factor ``1/(T_in - T_out)`` and leaves the per-cell balance at zero.

Discretisation (identical to :mod:`steady_fv`)
----------------------------------------------
Axis 0 is through-wall (per-cell sizes ``dx0``); axis 1 is along-wall (uniform
``dy``). For a cell of cross-section ``(dx0_i, dy)`` in this 2-D slab:

* **Internal face** between neighbours ``i`` and ``j`` along an axis has
  conductance ``g = A_face / ((dx_i/2)/k_i + (dx_j/2)/k_j)`` where ``A_face`` is
  the area of the shared face (``= dy`` for an axis-0 face, ``= dx0`` of the cell
  for an axis-1 face). This is the exact series combination of the two cell
  half-resistances (the harmonic mean for equal cells).
* **Boundary film face** on the two axis-0 faces has conductance
  ``g_bnd = A_face / ((dx_face/2)/k_face + r_film)`` with ``A_face = dy``,
  ``r_film = r_si`` at the lo (theta_air = 1) face and ``r_se`` at the hi
  (theta_air = 0) face.
* All **lateral** (axis-1) edges are adiabatic: no term where there is no
  neighbour.

The per-cell residual is the net conductive heat into the cell (which the exact
steady solution drives to zero):

    R_cell = Σ_{existing neighbour faces} g_face * (theta_nb - theta_cell)
             + [cell on lo face] * g_bnd_lo * (1 - theta_cell)
             + [cell on hi face] * g_bnd_hi * (0 - theta_cell).

:func:`heat_residual_loss` returns ``mean(R_cell**2)`` over all cells (and the
batch), differentiable wrt ``theta``.
"""

from __future__ import annotations

import torch
from torch import Tensor

__all__ = ["heat_residual", "heat_residual_loss"]


def _as_field(t: Tensor) -> Tensor:
    """Coerce ``(B,1,Nx,Ny)`` or ``(B,Nx,Ny)`` or ``(Nx,Ny)`` to ``(B,Nx,Ny)``."""
    if t.dim() == 2:  # (Nx, Ny)
        return t.unsqueeze(0)
    if t.dim() == 3:  # (B, Nx, Ny)
        return t
    if t.dim() == 4:  # (B, 1, Nx, Ny)
        if t.shape[1] != 1:
            raise ValueError(f"expected a single channel, got shape {tuple(t.shape)}")
        return t[:, 0]
    raise ValueError(f"field must be 2-D, 3-D or 4-D, got {t.dim()}-D")


def _broadcast_dx0(dx0: Tensor, batch: int, nx: int, device, dtype) -> Tensor:
    """Coerce ``dx0`` to ``(B, Nx)``."""
    dx0 = torch.as_tensor(dx0, device=device, dtype=dtype)
    if dx0.dim() == 1:  # (Nx,)
        dx0 = dx0.unsqueeze(0)
    if dx0.dim() != 2:
        raise ValueError(f"dx0 must be (Nx,) or (B,Nx), got {tuple(dx0.shape)}")
    if dx0.shape[-1] != nx:
        raise ValueError(f"dx0 last dim {dx0.shape[-1]} != Nx {nx}")
    return dx0.expand(batch, nx)


def _broadcast_scalar(v: Tensor, batch: int, device, dtype) -> Tensor:
    """Coerce a scalar / ``(B,)`` per-sample value to ``(B,)``."""
    v = torch.as_tensor(v, device=device, dtype=dtype)
    if v.dim() == 0:
        return v.expand(batch)
    if v.dim() == 1:
        if v.shape[0] == 1:
            return v.expand(batch)
        if v.shape[0] != batch:
            raise ValueError(f"per-sample value has length {v.shape[0]} != batch {batch}")
        return v
    raise ValueError(f"scalar/per-sample value must be 0-D or 1-D, got {v.dim()}-D")


def heat_residual(
    theta: Tensor,
    k: Tensor,
    dx0: Tensor,
    dy: Tensor | float,
    r_si: Tensor | float,
    r_se: Tensor | float,
) -> Tensor:
    """Per-cell steady heat-conduction residual ``R_cell``.

    Evaluates the cell-centred finite-volume operator of
    :mod:`thermotwin.physics.steady_fv` on a predicted dimensionless field
    ``theta`` and returns the net conductive heat flowing into each cell [W].
    The exact discrete steady solution makes every entry ~0.

    Args:
        theta: dimensionless temperature, ``(B,1,Nx,Ny)``, ``(B,Nx,Ny)`` or
            ``(Nx,Ny)``. Axis 0 (``Nx``) is through-wall, axis 1 (``Ny``) lateral.
        k: conductivity [W/(m·K)], same spatial shape as ``theta``.
        dx0: per-cell axis-0 sizes [m], ``(Nx,)`` or ``(B,Nx)``.
        dy: uniform lateral cell size [m], scalar or ``(B,)``.
        r_si: indoor (lo-face, theta=1) film resistance [m²K/W], scalar or ``(B,)``.
        r_se: outdoor (hi-face, theta=0) film resistance [m²K/W], scalar or ``(B,)``.

    Returns:
        Residual tensor ``(B, Nx, Ny)``.
    """
    theta = _as_field(theta)
    k = _as_field(k)
    if k.shape != theta.shape:
        raise ValueError(f"k shape {tuple(k.shape)} != theta shape {tuple(theta.shape)}")

    batch, nx, ny = theta.shape
    device, dtype = theta.device, theta.dtype
    k = k.to(dtype)

    dx0 = _broadcast_dx0(dx0, batch, nx, device, dtype)  # (B, Nx)
    dy = _broadcast_scalar(dy, batch, device, dtype)  # (B,)
    r_si = _broadcast_scalar(r_si, batch, device, dtype)  # (B,)
    r_se = _broadcast_scalar(r_se, batch, device, dtype)  # (B,)

    # Per-cell axis-0 size broadcast over the lateral axis -> (B, Nx, Ny).
    dx0_grid = dx0.unsqueeze(-1).expand(batch, nx, ny)
    dy_grid = dy.view(batch, 1, 1)

    res = torch.zeros_like(theta)

    # --- axis-0 (through-wall) internal faces -------------------------------
    # Face between cells i (0..Nx-2) and i+1. Face area = dy.
    if nx > 1:
        k_lo = k[:, :-1, :]
        k_hi = k[:, 1:, :]
        dx_lo = dx0_grid[:, :-1, :]
        dx_hi = dx0_grid[:, 1:, :]
        r_half_lo = (dx_lo / 2.0) / k_lo
        r_half_hi = (dx_hi / 2.0) / k_hi
        area = dy_grid.expand(batch, nx - 1, ny)
        g0 = area / (r_half_lo + r_half_hi)  # (B, Nx-1, Ny)
        flux = g0 * (theta[:, 1:, :] - theta[:, :-1, :])  # lo cell sees +flux
        res[:, :-1, :] = res[:, :-1, :] + flux
        res[:, 1:, :] = res[:, 1:, :] - flux

    # --- axis-1 (lateral) internal faces ------------------------------------
    # Face between cells j and j+1. Face area = dx0 of that through-wall row.
    if ny > 1:
        k_lo = k[:, :, :-1]
        k_hi = k[:, :, 1:]
        r_half_lo = (dy_grid / 2.0) / k_lo
        r_half_hi = (dy_grid / 2.0) / k_hi
        area = dx0_grid[:, :, :-1]  # shared face area = through-wall cell size
        g1 = area / (r_half_lo + r_half_hi)  # (B, Nx, Ny-1)
        flux = g1 * (theta[:, :, 1:] - theta[:, :, :-1])
        res[:, :, :-1] = res[:, :, :-1] + flux
        res[:, :, 1:] = res[:, :, 1:] - flux

    # --- axis-0 boundary film faces -----------------------------------------
    # Lo face (row 0): air theta = 1, film r_si. Area = dy.
    area_b = dy_grid.expand(batch, 1, ny)[:, 0, :]  # (B, Ny)
    k_lo_face = k[:, 0, :]
    r_half_lo_face = (dx0_grid[:, 0, :] / 2.0) / k_lo_face
    g_bnd_lo = area_b / (r_half_lo_face + r_si.view(batch, 1))
    res[:, 0, :] = res[:, 0, :] + g_bnd_lo * (1.0 - theta[:, 0, :])

    # Hi face (row Nx-1): air theta = 0, film r_se.
    k_hi_face = k[:, -1, :]
    r_half_hi_face = (dx0_grid[:, -1, :] / 2.0) / k_hi_face
    g_bnd_hi = area_b / (r_half_hi_face + r_se.view(batch, 1))
    res[:, -1, :] = res[:, -1, :] + g_bnd_hi * (0.0 - theta[:, -1, :])

    return res


def heat_residual_loss(
    theta: Tensor,
    k: Tensor,
    dx0: Tensor,
    dy: Tensor | float,
    r_si: Tensor | float,
    r_se: Tensor | float,
) -> Tensor:
    """Mean-squared steady heat-conduction PDE residual (a scalar loss).

    Differentiable wrt ``theta``; minimised (to ~0) by the exact discrete steady
    field. See :func:`heat_residual` for the discretisation and argument shapes.

    Returns:
        Scalar tensor ``mean(R_cell**2)`` over all cells and the batch.
    """
    res = heat_residual(theta, k, dx0, dy, r_si, r_se)
    return torch.mean(res**2)
