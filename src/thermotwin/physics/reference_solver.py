"""Independent general-boundary steady-conduction reference solver.

Two jobs the production engine (:func:`thermotwin.physics.steady_fv.solve_steady_conduction`)
cannot do, because it hardcodes boundary conditions to the axis-0 faces and reports only the
through-wall flux:

1. **Run the ISO 10211 thermal-bridge reference cases** (corners / junctions need fixed
   temperatures and surface films on *several* faces, and only part of a face at that).
2. **Cross-validate the production solver** as a genuinely independent second implementation.

The interior physics is identical to the production solver on purpose -- cell-centred finite
volume with face conductance ``A / (R_half_i + R_half_j)`` (the exact series / harmonic
combination for piecewise-constant ``k``), so the two must agree to machine precision on any
case the production solver can express. What differs here: boundary conditions are applied via
arbitrary :class:`BoundaryPatch` objects (any axis, any side, optionally only a masked region
of the face; a film resistance of 0 is a pure Dirichlet surface temperature), per-patch heat
flow is extracted, and the surface temperature factor ``f_Rsi`` is available -- the quantities
ISO 10211 is scored on.

Units: SI (m, W/(m.K), degrees C, W/m^2). In 2-D, fluxes are per unit depth (W/m).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

__all__ = ["BoundaryPatch", "ReferenceField", "solve_reference", "temperature_factor"]


@dataclass(frozen=True)
class BoundaryPatch:
    """A boundary condition on (part of) one face of the grid.

    Args:
        axis: face-normal axis.
        side: ``"lo"`` (index 0) or ``"hi"`` (index ``shape[axis]-1``).
        t_air: ambient/air temperature driving this patch.
        r_film: surface film resistance [m^2K/W]; ``0`` => pure Dirichlet (fixed surface temp),
            ``>0`` => Robin/convective film with conductance ``1/r_film`` per unit area.
        mask: optional boolean array over the ``(ndim-1)``-D face (the grid shape with ``axis``
            removed) selecting which face-cells this patch covers; ``None`` => the whole face.
        name: optional label (e.g. ``"interior"``) for reporting / f_Rsi selection.
    """

    axis: int
    side: str
    t_air: float
    r_film: float = 0.0
    mask: np.ndarray | None = None
    name: str = ""


@dataclass(frozen=True)
class ReferenceField:
    """Result of a reference solve."""

    temperature: np.ndarray            # T at cell centres, shape == grid shape
    patch_flux: dict[str, float]       # patch name/key -> heat flow INTO the domain [W or W/m]
    patch_surface_temp: dict[str, float]  # patch key -> min surface-cell temperature
    patch_cells: dict[str, np.ndarray]    # patch key -> flat indices of its boundary cells


def _axis_spacings(spacing: Sequence, shape: tuple[int, ...]) -> list[np.ndarray]:
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


def _face_cells(idx: np.ndarray, axis: int, side: str, mask: np.ndarray | None) -> np.ndarray:
    """Flat indices of the boundary cells on (axis, side), optionally restricted by ``mask``."""
    end = 0 if side == "lo" else idx.shape[axis] - 1
    face = np.take(idx, end, axis=axis)  # (ndim-1)-D array of cell indices on the face
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != face.shape:
            raise ValueError(f"patch mask shape {mask.shape} != face shape {face.shape}")
        return face[mask].reshape(-1)
    return face.reshape(-1)


def solve_reference(
    k: np.ndarray,
    spacing: Sequence,
    patches: Sequence[BoundaryPatch],
) -> ReferenceField:
    """Solve ``div(k grad T) = 0`` with arbitrary per-face(-region) Dirichlet/Robin conditions.

    Faces (or face-regions) not covered by any patch are adiabatic (zero flux). Returns the
    temperature field plus, per patch, the heat flow into the domain and the minimum surface
    temperature (for ``f_Rsi``).
    """
    k = np.asarray(k, dtype=float)
    if np.any(k <= 0):
        raise ValueError("conductivity must be > 0 everywhere")
    dxa = _axis_spacings(spacing, k.shape)
    ndim, n = k.ndim, k.size
    idx = np.arange(n).reshape(k.shape)

    dx_grid = [
        dxa[a].reshape([-1 if b == a else 1 for b in range(ndim)]) * np.ones(k.shape)
        for a in range(ndim)
    ]
    vol = np.ones(k.shape)
    for a in range(ndim):
        vol = vol * dx_grid[a]
    face_area = [vol / dx_grid[a] for a in range(ndim)]  # area normal to axis a

    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    vals: list[np.ndarray] = []
    diag = np.zeros(n)
    rhs = np.zeros(n)

    def sl(start: int, stop: int, axis: int) -> tuple:
        s = [slice(None)] * ndim
        s[axis] = slice(start, stop)
        return tuple(s)

    # Interior face conductances (identical scheme to the production solver).
    for axis in range(ndim):
        if k.shape[axis] <= 1:
            continue
        lo_s, hi_s = sl(0, k.shape[axis] - 1, axis), sl(1, k.shape[axis], axis)
        lo, hi = idx[lo_s].reshape(-1), idx[hi_s].reshape(-1)
        r_half_i = (dx_grid[axis][lo_s].reshape(-1) / 2.0) / k[lo_s].reshape(-1)
        r_half_j = (dx_grid[axis][hi_s].reshape(-1) / 2.0) / k[hi_s].reshape(-1)
        g = face_area[axis][lo_s].reshape(-1) / (r_half_i + r_half_j)
        rows += [lo, hi]
        cols += [hi, lo]
        vals += [-g, -g]
        np.add.at(diag, lo, g)
        np.add.at(diag, hi, g)

    # Boundary patches: Robin/Dirichlet conductance into the named face-cells.
    patch_cells: dict[str, np.ndarray] = {}
    patch_gbnd: dict[str, np.ndarray] = {}
    for p, patch in enumerate(patches):
        key = patch.name or f"patch{p}"
        if key in patch_cells:
            raise ValueError(f"duplicate patch key {key!r}; give patches distinct names")
        cells = _face_cells(idx, patch.axis, patch.side, patch.mask)
        kc = k.reshape(-1)[cells]
        r_half = (dx_grid[patch.axis].reshape(-1)[cells] / 2.0) / kc
        g_bnd = face_area[patch.axis].reshape(-1)[cells] / (r_half + patch.r_film)
        np.add.at(diag, cells, g_bnd)
        np.add.at(rhs, cells, g_bnd * patch.t_air)
        patch_cells[key] = cells
        patch_gbnd[key] = g_bnd

    a_mat = sp.coo_matrix(
        (
            np.concatenate(vals + [diag]) if vals else diag,
            (
                np.concatenate(rows + [np.arange(n)]) if rows else np.arange(n),
                np.concatenate(cols + [np.arange(n)]) if cols else np.arange(n),
            ),
        ),
        shape=(n, n),
    ).tocsr()

    t = spla.spsolve(a_mat, rhs)
    temperature = t.reshape(k.shape)

    flux: dict[str, float] = {}
    surf: dict[str, float] = {}
    for p, patch in enumerate(patches):
        key = patch.name or f"patch{p}"
        cells, g_bnd = patch_cells[key], patch_gbnd[key]
        flux[key] = float(np.sum(g_bnd * (patch.t_air - t[cells])))  # into domain
        surf[key] = float(np.min(t[cells]))
    return ReferenceField(temperature=temperature, patch_flux=flux,
                          patch_surface_temp=surf, patch_cells=patch_cells)


def temperature_factor(field: ReferenceField, interior_key: str, t_i: float, t_e: float) -> float:
    """ISO 10211 surface temperature factor ``f_Rsi = (T_si,min - t_e) / (t_i - t_e)``.

    Uses the *coldest* interior-surface cell (the condensation-risk point ISO reports).
    """
    if t_i == t_e:
        return float("nan")
    return (field.patch_surface_temp[interior_key] - t_e) / (t_i - t_e)
