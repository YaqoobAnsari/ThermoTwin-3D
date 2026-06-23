"""Steady-state heat-conduction solver on a regular grid (finite volume).

This is the geometry-resolved generalisation of :mod:`thermotwin.physics.conduction`:
where that module gives the closed-form 1-D series-resistance answer for a flat
multilayer wall, this one solves

    ∇·(k(x) ∇T(x)) = 0

on an axis-aligned 1-D / 2-D / 3-D grid with a spatially varying conductivity
field ``k(x)`` — so it handles corners, layered assemblies, inclusions and
thermal bridges that the 1-D formula cannot. It is the default lightweight engine
for two jobs:

* **Block-1 ground truth** — generate (geometry, BCs) → temperature/heat-flux
  fields to train and benchmark the operator against.
* **Physics residual reference** — the discrete operator the PINN loss is checked
  against.

The discretisation is cell-centred finite volume. The conductance of the face
between two cells is ``A / (R_half_i + R_half_j)`` where each half-resistance is
``(dx/2)/k`` of the respective cell. For equal cells this is the harmonic mean;
for unequal cells it is the exact series combination. Either way, for
piecewise-constant ``k`` it reproduces the exact series resistance
``R = Σ d/λ`` across a layer interface at *any* resolution, so a layered slab
solved here matches ``steady_state_1d`` to machine precision. That equivalence is
the solver's primary unit test.

Grids may have **non-uniform spacing per axis** (each axis takes either a scalar
cell size or a 1-D array of per-cell sizes), which is what lets a real multilayer
wall keep each layer's exact thickness.

Conventions
-----------
* Axis 0 is the **through-wall** direction; Dirichlet (fixed temperature) /
  film (Robin) conditions are applied on its two faces.
* All other faces are **adiabatic** (zero-flux Neumann).

Units follow :mod:`thermotwin.physics.conduction` (SI: m, W/(m·K), °C, W/m²).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .conduction import Layer

__all__ = [
    "DirichletFilm",
    "ConductionField",
    "layered_k_field",
    "solve_steady_conduction",
]


@dataclass(frozen=True)
class DirichletFilm:
    """Boundary condition on the two through-axis (axis-0) faces.

    Each face fixes an *air* temperature and an optional surface film resistance
    ``r`` [m²K/W]. ``r == 0`` makes it a pure Dirichlet (surface temperature)
    condition; ``r > 0`` is a Robin/convective film with conductance ``h = 1/r``
    per unit area, matching EN ISO 6946 ``R_si`` / ``R_se``.
    """

    t_lo: float  # air temperature at the axis-0 minimum face (e.g. indoor)
    t_hi: float  # air temperature at the axis-0 maximum face (e.g. outdoor)
    r_lo: float = 0.0  # film resistance at the min face (R_si)
    r_hi: float = 0.0  # film resistance at the max face (R_se)


@dataclass(frozen=True)
class ConductionField:
    """Result of a steady conduction solve."""

    temperature: np.ndarray  # T at cell centres, shape == grid shape
    heat_flux: float  # mean through-wall flux density [W/m²], + from lo->hi
    u_value: float  # effective transmittance [W/(m²K)] = q / (t_lo - t_hi)
    r_total: float  # effective resistance [m²K/W] = 1 / u_value

    def summary(self) -> str:
        return (
            f"grid {self.temperature.shape}  "
            f"R_eff = {self.r_total:.4f} m^2K/W   "
            f"U = {self.u_value:.4f} W/m^2K   "
            f"q = {self.heat_flux:.3f} W/m^2"
        )


def _axis_spacings(spacing: Sequence, shape: tuple[int, ...]) -> list[np.ndarray]:
    """Normalise ``spacing`` to one 1-D array of per-cell sizes per axis."""
    if len(spacing) != len(shape):
        raise ValueError(f"spacing has {len(spacing)} axes but grid is {len(shape)}-D")
    out: list[np.ndarray] = []
    for a, s in enumerate(spacing):
        arr = np.atleast_1d(np.asarray(s, dtype=float))
        if arr.size == 1:
            arr = np.full(shape[a], arr.item())
        elif arr.size != shape[a]:
            raise ValueError(f"axis {a}: spacing length {arr.size} != {shape[a]} cells")
        if np.any(arr <= 0):
            raise ValueError(f"axis {a}: spacings must be > 0")
        out.append(arr)
    return out


def layered_k_field(
    layers: Sequence[Layer],
    cells_per_layer: int = 8,
    cross_section: tuple[int, ...] = (),
    transverse_spacing: float = 0.05,
) -> tuple[np.ndarray, list]:
    """Build a conductivity field and (non-uniform) spacing for a multilayer wall.

    The through-axis (axis 0) is discretised layer by layer; each layer keeps its
    *exact* thickness via its own cell size (``thickness / cells_per_layer``), so
    no thickness is distorted regardless of ``cells_per_layer``. ``cross_section``
    adds adiabatic transverse dimensions. Returns ``(k, spacing)`` for
    :func:`solve_steady_conduction`, where ``spacing[0]`` is a per-cell array.
    """
    if not layers:
        raise ValueError("need at least one layer")
    if cells_per_layer < 1:
        raise ValueError("cells_per_layer must be >= 1")

    dx0 = np.concatenate(
        [np.full(cells_per_layer, layer.thickness_m / cells_per_layer) for layer in layers]
    )
    k_axis0 = np.concatenate(
        [np.full(cells_per_layer, layer.conductivity_w_mk) for layer in layers]
    )
    shape = (k_axis0.size, *cross_section)
    k = np.broadcast_to(k_axis0.reshape((-1,) + (1,) * len(cross_section)), shape).copy()
    spacing = [dx0, *([transverse_spacing] * len(cross_section))]
    return k, spacing


def solve_steady_conduction(
    k: np.ndarray,
    spacing: Sequence,
    bc: DirichletFilm,
) -> ConductionField:
    """Solve ``∇·(k ∇T) = 0`` on a regular (optionally non-uniform) grid.

    Args:
        k: conductivity at each cell centre [W/(m·K)]; 1-D / 2-D / 3-D.
        spacing: per-axis cell size — each entry a scalar (uniform) or a 1-D array
            of per-cell sizes [m]. ``len(spacing) == k.ndim``.
        bc: Dirichlet/film conditions on the two axis-0 faces; all other faces
            are adiabatic.

    Returns:
        A :class:`ConductionField` with the temperature field and the effective
        through-wall flux / U-value.
    """
    k = np.asarray(k, dtype=float)
    if np.any(k <= 0):
        raise ValueError("conductivity must be > 0 everywhere")
    dxa = _axis_spacings(spacing, k.shape)
    ndim = k.ndim
    n = k.size
    idx = np.arange(n).reshape(k.shape)

    # Per-cell axis sizes broadcast to the full grid, and transverse face areas.
    dx_grid = [
        dxa[a].reshape([-1 if b == a else 1 for b in range(ndim)]) * np.ones(k.shape)
        for a in range(ndim)
    ]
    vol = np.ones(k.shape)
    for a in range(ndim):
        vol = vol * dx_grid[a]
    # Area of a face normal to axis a == product of the *other* spacings.
    face_area = [vol / dx_grid[a] for a in range(ndim)]

    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    vals: list[np.ndarray] = []
    diag = np.zeros(n)
    rhs = np.zeros(n)

    def sl(start: int, stop: int, axis: int) -> tuple:
        s = [slice(None)] * ndim
        s[axis] = slice(start, stop)
        return tuple(s)

    for axis in range(ndim):
        if k.shape[axis] > 1:
            lo_s = sl(0, k.shape[axis] - 1, axis)
            hi_s = sl(1, k.shape[axis], axis)
            lo = idx[lo_s].reshape(-1)
            hi = idx[hi_s].reshape(-1)
            ki = k[lo_s].reshape(-1)
            kj = k[hi_s].reshape(-1)
            r_half_i = (dx_grid[axis][lo_s].reshape(-1) / 2.0) / ki
            r_half_j = (dx_grid[axis][hi_s].reshape(-1) / 2.0) / kj
            area = face_area[axis][lo_s].reshape(-1)  # shared transverse area
            g = area / (r_half_i + r_half_j)  # face conductance [W/K]
            rows += [lo, hi]
            cols += [hi, lo]
            vals += [-g, -g]
            np.add.at(diag, lo, g)
            np.add.at(diag, hi, g)

        if axis == 0:
            for end, t_air, r_film in (
                (0, bc.t_lo, bc.r_lo),
                (k.shape[0] - 1, bc.t_hi, bc.r_hi),
            ):
                fs = sl(end, end + 1, 0)
                face = idx[fs].reshape(-1)
                kc = k[fs].reshape(-1)
                r_half = (dx_grid[0][fs].reshape(-1) / 2.0) / kc
                g_bnd = face_area[0][fs].reshape(-1) / (r_half + r_film)
                np.add.at(diag, face, g_bnd)
                rhs[face] += g_bnd * t_air

    a_mat = sp.coo_matrix(
        (
            np.concatenate(vals + [diag]),
            (
                np.concatenate(rows + [np.arange(n)]),
                np.concatenate(cols + [np.arange(n)]),
            ),
        ),
        shape=(n, n),
    ).tocsr()

    t = spla.spsolve(a_mat, rhs)
    temperature = t.reshape(k.shape)

    # Effective flux: heat entering the lo face / wall area.
    fs = sl(0, 1, 0)
    face = idx[fs].reshape(-1)
    kc = k[fs].reshape(-1)
    r_half = (dx_grid[0][fs].reshape(-1) / 2.0) / kc
    g_bnd = face_area[0][fs].reshape(-1) / (r_half + bc.r_lo)
    q_total = float(np.sum(g_bnd * (bc.t_lo - t[face])))  # [W]
    wall_area = float(np.sum(face_area[0][fs]))  # [m^2]
    q = q_total / wall_area
    dt = bc.t_lo - bc.t_hi
    u = q / dt if dt != 0 else float("nan")

    return ConductionField(
        temperature=temperature,
        heat_flux=q,
        u_value=float(u),
        r_total=float(1.0 / u) if u not in (0.0,) and np.isfinite(u) else float("nan"),
    )


if __name__ == "__main__":
    demo = [
        Layer("plasterboard", 0.0125, 0.25),
        Layer("mineral wool", 0.120, 0.035),
        Layer("brick", 0.200, 0.77),
    ]
    k, spacing = layered_k_field(demo, cells_per_layer=10, cross_section=(3,))
    res = solve_steady_conduction(k, spacing, DirichletFilm(20.0, 0.0, r_lo=0.13, r_hi=0.04))
    print(res.summary())
