"""Synthetic 3-D conduction ground truth: layered wall *blocks* with thermal bridges.

Block-2 carries the winning Block-1 recipe — a geometry-conditioned operator that
predicts a **correction on an analytic 1-D clear-wall prior** — into three
dimensions and irregular geometry, on the road to real as-built scans. This module
is the 3-D analogue of :mod:`thermotwin.data.synthetic_fem`: instead of a 2-D
cross-section it generates a 3-D **wall block** layered through axis 0 (the
through-wall direction) and homogeneous across the two in-plane axes (y, z), then
punctures the insulation layer with rectangular-prism thermal bridges that are now
localised in *both* in-plane axes (a stud/nib is finite in y **and** z, not an
infinite 2-D strip). Solved with :func:`thermotwin.physics.steady_fv`
(axis-0 Dirichlet/film, all other faces adiabatic), this gives a genuinely 3-D
temperature field whose departure from the 1-D answer the GINO must learn.

Each solved block is turned into a **GINO sample**: ``N`` points drawn uniformly
inside the block (coordinates normalised to ``[0, 1]^3``), each carrying

    [logk_std, r_si, r_se, theta1d]

where ``theta1d`` is the analytic 1-D clear-wall prior evaluated at the point's
through-wall position from its local k-column (the Block-1 enrichment that proved
decisive OOD), the regression target ``theta`` (the dimensionless temperature,
trilinearly interpolated from the FV field), and a signed-distance field (SDF) on a
regular ``G^3`` latent grid. For an axis-aligned box the SDF is analytic.

Conventions mirror the Block-1 corpus: per-sample ``.npz`` + a ``manifest.json``.
Everything is seeded; each sample carries the parameters needed to reproduce it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..physics.conduction import Layer
from ..physics.steady_fv import (
    ConductionField,
    DirichletFilm,
    layered_k_field,
    solve_steady_conduction,
)
from .synthetic_fem import _BASE_WALLS, _BRIDGE_K

__all__ = [
    "Bridge3D",
    "BlockSample",
    "build_k_field_3d",
    "solve_block",
    "clear_block_u",
    "point_theta1d",
    "box_sdf_grid",
    "sample_block",
    "random_block",
    "random_block_hard",
    "generate_corpus_3d",
    "generate_corpus_hard",
    "FEATURE_LAYOUT",
    "LOGK_MEAN",
    "LOGK_STD",
]

# Standardisation constants for log10(conductivity), shared with the Block-1 dataset
# (material library spans ~0.035 insulation to ~160 aluminium W/(m·K)).
LOGK_MEAN = 0.0
LOGK_STD = 1.5

# Per-point feature layout (channel order in ``feats``).
FEATURE_LAYOUT = ("logk_std", "r_si", "r_se", "theta1d")
FEATURE_DIM = len(FEATURE_LAYOUT)


@dataclass(frozen=True)
class Bridge3D:
    """A rectangular-prism high-conductivity inclusion in the wall block.

    Extents are in metres: ``x`` is through-wall (from the outside face, axis 0),
    ``y`` and ``z`` are the two in-plane axes. ``conductivity`` overrides the base
    material wherever the prism sits. A real bridge (steel stud, concrete nib) is
    finite in both in-plane axes, so it forces 3-D lateral spreading the 2-D
    cross-section could not represent.
    """

    x0: float
    x1: float
    y0: float
    y1: float
    z0: float
    z1: float
    conductivity_w_mk: float


@dataclass(frozen=True)
class BlockSample:
    """Full specification of one synthetic 3-D wall block."""

    layers: tuple[Layer, ...]  # outside -> inside (layered through axis 0)
    width_y_m: float  # in-plane extent along axis 1
    width_z_m: float  # in-plane extent along axis 2
    t_indoor: float
    t_outdoor: float
    r_si: float = 0.13
    r_se: float = 0.04
    bridges: tuple[Bridge3D, ...] = field(default_factory=tuple)
    cells_per_layer: int = 3
    cells_y: int = 16
    cells_z: int = 16

    @property
    def thickness_m(self) -> float:
        return sum(layer.thickness_m for layer in self.layers)


def _cell_centres(spacing_1d: np.ndarray) -> np.ndarray:
    """Cell-centre coordinates from a 1-D per-cell spacing array."""
    edges = np.concatenate([[0.0], np.cumsum(spacing_1d)])
    return 0.5 * (edges[:-1] + edges[1:])


def build_k_field_3d(sample: BlockSample) -> tuple[np.ndarray, list]:
    """Conductivity field ``k`` ``(Nx, Ny, Nz)`` and grid spacing for a block.

    The base field is the layered construction broadcast across the in-plane
    extents; thermal-bridge prisms then overwrite their rectangular footprints in
    ``(x, y, z)``.
    """
    dy = sample.width_y_m / sample.cells_y
    dz = sample.width_z_m / sample.cells_z
    # layered_k_field handles axis 0 (per-layer non-uniform); cross_section adds y, z.
    k, spacing = layered_k_field(
        list(sample.layers),
        cells_per_layer=sample.cells_per_layer,
        cross_section=(sample.cells_y, sample.cells_z),
        transverse_spacing=1.0,  # placeholder; overwritten below to keep y/z distinct
    )
    spacing = [spacing[0], dy, dz]
    if sample.bridges:
        xc = _cell_centres(spacing[0])  # through-wall centres
        yc = (np.arange(sample.cells_y) + 0.5) * dy
        zc = (np.arange(sample.cells_z) + 0.5) * dz
        for b in sample.bridges:
            xm = (xc >= b.x0) & (xc < b.x1)
            ym = (yc >= b.y0) & (yc < b.y1)
            zm = (zc >= b.z0) & (zc < b.z1)
            k[np.ix_(xm, ym, zm)] = b.conductivity_w_mk
    return k, spacing


def solve_block(sample: BlockSample) -> ConductionField:
    """Solve a block to its steady temperature / heat-flux field."""
    k, spacing = build_k_field_3d(sample)
    bc = DirichletFilm(sample.t_indoor, sample.t_outdoor, r_lo=sample.r_si, r_hi=sample.r_se)
    return solve_steady_conduction(k, spacing, bc)


def clear_block_u(sample: BlockSample) -> float:
    """The 1-D clear-wall U-value (no bridges) — the baseline a real bridge beats."""
    no_bridges = BlockSample(
        layers=sample.layers,
        width_y_m=sample.width_y_m,
        width_z_m=sample.width_z_m,
        t_indoor=sample.t_indoor,
        t_outdoor=sample.t_outdoor,
        r_si=sample.r_si,
        r_se=sample.r_se,
        bridges=(),
        cells_per_layer=sample.cells_per_layer,
        cells_y=1,
        cells_z=1,
    )
    return solve_block(no_bridges).u_value


def point_theta1d(
    k_column: np.ndarray,
    dx0: np.ndarray,
    x_frac: np.ndarray,
    r_si: float,
    r_se: float,
) -> np.ndarray:
    """Analytic 1-D clear-wall prior θ at arbitrary through-wall fractions.

    This is the point-sampled analogue of
    :func:`thermotwin.data.dataset.clearwall_theta`: it returns the dimensionless
    temperature of the equivalent layered slab (no lateral spreading) at the
    through-wall positions ``x_frac ∈ [0, 1]`` of a single local k-column.

    Closed form. With cumulative resistance ``R(x) = r_si + ∫_0^x dx'/k(x')`` measured
    from the lo (indoor, θ=1) face and total ``R_tot = r_si + Σ dx/k + r_se``::

        θ1d(x) = 1 − R(x) / R_tot.

    We evaluate ``R(x)`` by piecewise-linear interpolation of the cumulative
    resistance profile sampled at the cell faces, so within a homogeneous layer θ
    is exactly linear and across interfaces it kinks correctly.

    Args:
        k_column: per-cell conductivity along the through-wall axis, length ``Nx``.
        dx0: per-cell through-wall spacing [m], length ``Nx``.
        x_frac: query positions as a fraction of total thickness, in ``[0, 1]``.
        r_si: indoor (lo-face) film resistance [m²K/W].
        r_se: outdoor (hi-face) film resistance [m²K/W].

    Returns:
        ``θ1d`` at each ``x_frac``, same shape as ``x_frac``.
    """
    k_column = np.asarray(k_column, dtype=np.float64)
    dx0 = np.asarray(dx0, dtype=np.float64)
    x_frac = np.asarray(x_frac, dtype=np.float64)

    r_cell = dx0 / k_column  # per-cell resistance
    # Cumulative resistance at cell faces (length Nx+1), starting from r_si at x=0.
    r_face = r_si + np.concatenate([[0.0], np.cumsum(r_cell)])
    r_tot = r_face[-1] + r_se
    thickness = float(dx0.sum())
    x_phys = x_frac * thickness  # physical through-wall coordinate [m]
    x_face = np.concatenate([[0.0], np.cumsum(dx0)])  # physical face positions
    r_at_x = np.interp(x_phys, x_face, r_face)
    return 1.0 - r_at_x / r_tot


def box_sdf_grid(spacing: list, grid: int) -> np.ndarray:
    """Analytic signed-distance field of the block on a ``grid^3`` latent grid.

    The block is the axis-aligned box ``[0, Lx] × [0, Ly] × [0, Lz]`` (with
    ``L_a = Σ spacing[a]``), normalised to the unit cube ``[0, 1]^3`` — the same box
    the sampled points and queries live in, so the SDF and the points share one
    coordinate frame. SDF convention: ``< 0`` inside, ``> 0`` outside, ``≈ 0`` on
    a face. Distance is in the normalised frame.

    For a box, the exact SDF at point ``p`` (in [0,1]^3, box = the whole cube) is::

        q = |p − 0.5| − 0.5
        sdf = ‖max(q, 0)‖ + min(max(q.x, q.y, q.z), 0)

    Since the block *is* the unit cube after normalisation, this is ≤ 0 everywhere
    inside the grid (0 on the boundary). The grid centres sit strictly inside, so
    interior cells are negative and the faces are ~0 — the sign that tells GINO
    "solid here". Points outside the cube would be positive.
    """
    lengths = np.array([float(np.sum(s)) for s in spacing], dtype=np.float64)
    del lengths  # block fills the normalised cube; kept for clarity of intent.
    # Cell-centred coordinates on [0,1]^3.
    c = (np.arange(grid) + 0.5) / grid
    gx, gy, gz = np.meshgrid(c, c, c, indexing="ij")
    p = np.stack([gx, gy, gz], axis=-1)  # (G, G, G, 3)
    q = np.abs(p - 0.5) - 0.5  # box half-extent 0.5 about centre 0.5
    outside = np.linalg.norm(np.maximum(q, 0.0), axis=-1)
    inside = np.minimum(np.max(q, axis=-1), 0.0)
    return (outside + inside).astype(np.float32)


def sample_block(
    sample: BlockSample,
    field_obj: ConductionField,
    spacing: list,
    n_points: int,
    grid: int,
    rng: np.random.Generator,
) -> dict:
    """Turn a solved block into a GINO training sample.

    Draws ``n_points`` uniform random points inside the block, featurises each with
    ``[logk_std, r_si, r_se, theta1d]`` (theta1d = the analytic 1-D prior at the
    point's through-wall fraction, from the local k-column), trilinearly
    interpolates the FV θ field as the target, and builds the analytic box SDF on a
    ``grid^3`` latent grid.

    Returns a dict of arrays ready to serialise to ``.npz``.
    """
    k, _ = build_k_field_3d(sample)
    nx, ny, nz = k.shape
    dx0 = np.asarray(spacing[0], dtype=np.float64)
    dy, dz = float(spacing[1]), float(spacing[2])
    lengths = np.array([dx0.sum(), ny * dy, nz * dz], dtype=np.float64)

    # Dimensionless target field θ = (T − T_out) / (T_in − T_out).
    t_in, t_out = float(sample.t_indoor), float(sample.t_outdoor)
    theta_field = ((field_obj.temperature.astype(np.float64) - t_out) / (t_in - t_out)).astype(
        np.float64
    )

    # Cell-centre coordinates per axis, in physical metres.
    xc = _cell_centres(dx0)
    yc = (np.arange(ny) + 0.5) * dy
    zc = (np.arange(nz) + 0.5) * dz
    centres = [xc, yc, zc]

    # Uniform random points inside the physical box.
    pts_phys = rng.uniform(0.0, 1.0, size=(n_points, 3)) * lengths[None, :]
    # Normalised [0,1]^3 coordinates (the GINO frame).
    pts_norm = (pts_phys / lengths[None, :]).astype(np.float32)

    # Analytic 1-D clear-wall prior as a full grid field, then interpolated the SAME
    # way as θ. At cell centres this field equals the FV θ to machine precision for a
    # clear column (verified), so for a no-bridge block prior and θ agree even through
    # the trilinear sampling; only the FV field's true lateral spreading near bridges
    # makes θ depart from the prior — exactly the residual the operator must learn.
    prior_field = _prior_grid(k, dx0, sample.r_si, sample.r_se)

    # Trilinear interpolation of θ and the prior at each point (shared support).
    theta_pts = _trilinear(theta_field, centres, pts_phys).astype(np.float32)
    theta1d = _trilinear(prior_field, centres, pts_phys).astype(np.float32)

    # Per-point local conductivity (containing cell) for the standardised logk feature.
    iy = np.clip(np.searchsorted(_edges(yc, dy), pts_phys[:, 1]) - 1, 0, ny - 1)
    iz = np.clip(np.searchsorted(_edges(zc, dz), pts_phys[:, 2]) - 1, 0, nz - 1)
    ix = np.clip(
        np.searchsorted(np.concatenate([[0.0], np.cumsum(dx0)]), pts_phys[:, 0]) - 1, 0, nx - 1
    )
    k_at_pt = k[ix, iy, iz].astype(np.float64)
    logk_std = ((np.log10(k_at_pt) - LOGK_MEAN) / LOGK_STD).astype(np.float32)

    feats = np.stack(
        [
            logk_std,
            np.full(n_points, sample.r_si, dtype=np.float32),
            np.full(n_points, sample.r_se, dtype=np.float32),
            theta1d,
        ],
        axis=1,
    ).astype(np.float32)

    sdf = box_sdf_grid(spacing, grid)

    return {
        "points": pts_norm,  # (N, 3) in [0,1]^3
        "feats": feats,  # (N, F)
        "theta": theta_pts,  # (N,)
        "prior": theta1d,  # (N,) the analytic 1-D prior per point
        "sdf": sdf,  # (G, G, G)
        "u_value": np.float32(field_obj.u_value),
    }


def _edges(centres: np.ndarray, spacing: float) -> np.ndarray:
    """Uniform-axis cell edges (length N+1) from cell centres and a scalar spacing."""
    return np.concatenate([centres - spacing / 2.0, [centres[-1] + spacing / 2.0]])


def _prior_grid(k: np.ndarray, dx0: np.ndarray, r_si: float, r_se: float) -> np.ndarray:
    """Analytic 1-D clear-wall prior θ at every cell centre, per through-wall column.

    Vectorised, cell-centred form (identical per-column to
    :func:`thermotwin.data.dataset.clearwall_theta`): for each ``(y, z)`` column with
    its own k-profile,

        θ1d_j = 1 − (r_si + Σ_{m<j} R_m + R_j/2) / R_tot,
        R_m = dx0_m / k_m,   R_tot = r_si + Σ_m R_m + r_se.

    This reproduces the cell-centred FV field exactly where the 1-D assumption holds,
    so a no-bridge block's θ equals this prior; near a bridge the FV field's lateral
    spreading is what departs from it.
    """
    r = dx0[:, None, None] / np.asarray(k, dtype=np.float64)  # (Nx, Ny, Nz)
    cum_before = np.cumsum(r, axis=0) - r  # Σ_{m<j} R_m
    r_tot = r_si + r.sum(axis=0, keepdims=True) + r_se  # (1, Ny, Nz)
    return (1.0 - (r_si + cum_before + 0.5 * r) / r_tot).astype(np.float64)


def _trilinear(field: np.ndarray, centres: list, pts: np.ndarray) -> np.ndarray:
    """Trilinear interpolation of a cell-centred 3-D ``field`` at physical ``pts``.

    Uses the cell-centre grids ``centres = [xc, yc, zc]``; points are clamped to the
    cell-centre support so edge points take the nearest interior value (no
    extrapolation blow-up).
    """
    coords = []
    for a in range(3):
        c = centres[a]
        # Locate the lower bracketing centre for each point; clamp to [0, len-2].
        i0 = np.clip(np.searchsorted(c, pts[:, a]) - 1, 0, len(c) - 2)
        x0 = c[i0]
        x1 = c[i0 + 1]
        w = np.clip((pts[:, a] - x0) / (x1 - x0), 0.0, 1.0)
        coords.append((i0, w))
    (ix, wx), (iy, wy), (iz, wz) = coords
    f = field
    a, b, c = wx, wy, wz
    out = (
        f[ix, iy, iz] * (1 - a) * (1 - b) * (1 - c)
        + f[ix + 1, iy, iz] * a * (1 - b) * (1 - c)
        + f[ix, iy + 1, iz] * (1 - a) * b * (1 - c)
        + f[ix, iy, iz + 1] * (1 - a) * (1 - b) * c
        + f[ix + 1, iy + 1, iz] * a * b * (1 - c)
        + f[ix + 1, iy, iz + 1] * a * (1 - b) * c
        + f[ix, iy + 1, iz + 1] * (1 - a) * b * c
        + f[ix + 1, iy + 1, iz + 1] * a * b * c
    )
    return out


def random_block(rng: np.random.Generator) -> BlockSample:
    """Draw one randomised 3-D wall block (base wall, BCs, 0-3 bridge prisms).

    Mirrors :func:`thermotwin.data.synthetic_fem.random_sample`: bridges puncture
    the **insulation** (lowest-conductivity) layer and are strictly more conductive
    than it (ADR 0006), so every bridge is a genuine thermal bridge that raises U.
    Here each bridge is a finite prism in *both* in-plane axes.
    """
    name = rng.choice(list(_BASE_WALLS))
    layers = _BASE_WALLS[name]
    width_y = float(rng.uniform(0.4, 1.0))
    width_z = float(rng.uniform(0.4, 1.0))
    t_in = float(rng.uniform(18.0, 22.0))
    t_out = float(rng.uniform(-12.0, 8.0))
    r_si = float(rng.choice([0.10, 0.13, 0.17]))

    conductivities = np.array([layer.conductivity_w_mk for layer in layers])
    thicknesses = np.array([layer.thickness_m for layer in layers])
    edges = np.concatenate([[0.0], np.cumsum(thicknesses)])
    insul_idx = int(np.argmin(conductivities))  # the insulation layer
    x_lo, x_hi = float(edges[insul_idx]), float(edges[insul_idx + 1])
    insul_k = float(conductivities[insul_idx])
    bridge_materials = [v for v in _BRIDGE_K.values() if v > insul_k] or list(_BRIDGE_K.values())

    n_bridges = int(rng.integers(0, 4))
    bridges = []
    for _ in range(n_bridges):
        bk = float(rng.choice(bridge_materials))
        by = float(rng.uniform(0.02, 0.10))  # prism extent along y
        bz = float(rng.uniform(0.02, 0.10))  # prism extent along z
        y0 = float(rng.uniform(0.0, max(width_y - by, 0.0)))
        z0 = float(rng.uniform(0.0, max(width_z - bz, 0.0)))
        bridges.append(Bridge3D(x_lo, x_hi, y0, y0 + by, z0, z0 + bz, bk))

    return BlockSample(
        layers=layers,
        width_y_m=width_y,
        width_z_m=width_z,
        t_indoor=t_in,
        t_outdoor=t_out,
        r_si=r_si,
        bridges=tuple(bridges),
        cells_y=int(rng.choice([16, 20, 24])),
        cells_z=int(rng.choice([16, 20, 24])),
    )


def random_block_hard(rng: np.random.Generator, cells_per_layer: int = 6) -> BlockSample:
    """Draw a **high-native-resolution** block with *sub-voxel* thermal fins.

    This is the corpus where a regular voxel grid genuinely *fails* — the diagnosis
    (``docs/block2_redesign.md``) showed the box / rotated-box corpora never escaped a
    16³ grid, so a voxel-FNO matched any point operator. Here the block is solved on a
    **fine** native FV grid (``cells_y = cells_z ∈ {48, 64, 80}``, ``cells_per_layer``
    6 ⇒ ~18 through-wall cells), and each thermal bridge is a **thin fin of 2–4 native
    cells** in each in-plane axis. Because the latent / voxel grid is 16³, a fin of 2–4
    native cells out of 48–80 spans ≤ 1 voxel cell — *sub-voxel*: nearest-cell
    voxelisation drops or smears it (measured ~24 % U-error from aliasing) while the
    point cloud, drawn on the native grid, resolves it exactly. Fins are conductive
    (steel/aluminium/concrete) and puncture the insulation layer (ADR 0006), so each is
    an unambiguous thermal bridge that raises U.

    Axis-aligned (no rotation): the grid-failure signal here is *resolution* (sub-voxel
    features), isolated cleanly from irregular orientation. The indoor face stays at
    world axis-0, so the U-from-indoor-face estimator is exact (no body-frame issue).
    """
    name = rng.choice(list(_BASE_WALLS))
    layers = _BASE_WALLS[name]
    width_y = float(rng.uniform(0.5, 1.0))
    width_z = float(rng.uniform(0.5, 1.0))
    t_in = float(rng.uniform(18.0, 22.0))
    t_out = float(rng.uniform(-12.0, 8.0))
    r_si = float(rng.choice([0.10, 0.13, 0.17]))
    cells_y = int(rng.choice([48, 64, 80]))
    cells_z = int(rng.choice([48, 64, 80]))
    dy, dz = width_y / cells_y, width_z / cells_z

    conductivities = np.array([layer.conductivity_w_mk for layer in layers])
    thicknesses = np.array([layer.thickness_m for layer in layers])
    edges = np.concatenate([[0.0], np.cumsum(thicknesses)])
    insul_idx = int(np.argmin(conductivities))
    x_lo, x_hi = float(edges[insul_idx]), float(edges[insul_idx + 1])
    insul_k = float(conductivities[insul_idx])
    # Conductive bridge materials only (drop the weak timber stud), strictly above host.
    strong = {m: v for m, v in _BRIDGE_K.items() if v > max(insul_k, 1.0)}
    bridge_materials = list(strong.values()) or [v for v in _BRIDGE_K.values() if v > insul_k]

    n_bridges = int(rng.integers(1, 5))  # 1..4 — never a clear block
    bridges = []
    for _ in range(n_bridges):
        bk = float(rng.choice(bridge_materials))
        # Footprint in *native cells*: 3–6 cells. At cells_y≈64 and a 16³ voxel/latent
        # grid, that is ~0.75–1.5 voxel cells — below the grid's ~2-cell Nyquist, so a
        # voxel-FNO (and a 16³-latent GINO) under-resolve / alias the fin, while the FV
        # solver resolves it exactly and a ~4096-point cloud samples it (~10–40 pts/fin)
        # so a gridless operator has real signal to learn the bridge from.
        ny_cells = int(rng.integers(3, 7))
        nz_cells = int(rng.integers(3, 7))
        by, bz = ny_cells * dy, nz_cells * dz
        y0 = float(rng.uniform(0.0, max(width_y - by, 0.0)))
        z0 = float(rng.uniform(0.0, max(width_z - bz, 0.0)))
        bridges.append(Bridge3D(x_lo, x_hi, y0, y0 + by, z0, z0 + bz, bk))

    return BlockSample(
        layers=layers,
        width_y_m=width_y,
        width_z_m=width_z,
        t_indoor=t_in,
        t_outdoor=t_out,
        r_si=r_si,
        bridges=tuple(bridges),
        cells_per_layer=cells_per_layer,
        cells_y=cells_y,
        cells_z=cells_z,
    )


def generate_corpus_hard(
    n: int,
    seed: int = 1337,
    grid: int = 16,
    n_points: int = 4096,
    cells_per_layer: int = 6,
) -> list[dict]:
    """Generate ``n`` fine-native blocks with sub-voxel fins (see :func:`random_block_hard`).

    Same per-sample record schema as :func:`generate_corpus_3d` (axis-aligned, no
    rotation), but solved on a high-resolution native grid with thin sub-voxel thermal
    bridges, and sampled at more points (default 4096) so the cloud resolves the fins a
    16³ voxel grid cannot. This is the corpus that lets a gridless / fine-latent operator
    out-resolve the voxel-FNO baseline.
    """
    rng = np.random.default_rng(seed)
    records: list[dict] = []
    for i in range(n):
        block = random_block_hard(rng, cells_per_layer=cells_per_layer)
        k, spacing = build_k_field_3d(block)
        bc = DirichletFilm(block.t_indoor, block.t_outdoor, r_lo=block.r_si, r_hi=block.r_se)
        res = solve_steady_conduction(k, spacing, bc)
        sample = sample_block(block, res, spacing, n_points=n_points, grid=grid, rng=rng)
        u_clear = clear_block_u(block)
        records.append(
            {
                "id": i,
                **sample,
                "u_clear": np.float32(u_clear),
                "heat_flux": np.float32(res.heat_flux),
                "t_indoor": np.float32(block.t_indoor),
                "t_outdoor": np.float32(block.t_outdoor),
                "r_si": np.float32(block.r_si),
                "r_se": np.float32(block.r_se),
                "grid_shape": np.asarray(k.shape, dtype=np.int32),
                "n_bridges": np.int32(len(block.bridges)),
            }
        )
    return records


def generate_corpus_3d(
    n: int,
    seed: int = 1337,
    grid: int = 16,
    n_points: int = 2048,
    cells_per_layer: int = 3,
) -> list[dict]:
    """Generate ``n`` solved 3-D blocks as GINO samples (plain-dict records).

    Each record holds the sampled points / features / target θ / per-point prior, the
    latent-grid SDF, the effective and clear-wall U-values, and the parameters needed
    to reproduce the block. ``grid`` is the SDF latent resolution ``G`` and
    ``n_points`` the number of sampled points per block.
    """
    rng = np.random.default_rng(seed)
    records: list[dict] = []
    for i in range(n):
        block = random_block(rng)
        # Honour the requested through-wall resolution for the corpus.
        block = BlockSample(
            layers=block.layers,
            width_y_m=block.width_y_m,
            width_z_m=block.width_z_m,
            t_indoor=block.t_indoor,
            t_outdoor=block.t_outdoor,
            r_si=block.r_si,
            r_se=block.r_se,
            bridges=block.bridges,
            cells_per_layer=cells_per_layer,
            cells_y=block.cells_y,
            cells_z=block.cells_z,
        )
        k, spacing = build_k_field_3d(block)
        bc = DirichletFilm(block.t_indoor, block.t_outdoor, r_lo=block.r_si, r_hi=block.r_se)
        res = solve_steady_conduction(k, spacing, bc)
        sample = sample_block(block, res, spacing, n_points=n_points, grid=grid, rng=rng)
        u_clear = clear_block_u(block)
        records.append(
            {
                "id": i,
                **sample,
                "u_clear": np.float32(u_clear),
                "heat_flux": np.float32(res.heat_flux),
                "t_indoor": np.float32(block.t_indoor),
                "t_outdoor": np.float32(block.t_outdoor),
                "r_si": np.float32(block.r_si),
                "r_se": np.float32(block.r_se),
                "grid_shape": np.asarray(k.shape, dtype=np.int32),
                "n_bridges": np.int32(len(block.bridges)),
            }
        )
    return records
