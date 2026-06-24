"""Irregular-geometry 3-D conduction corpus: rotated, off-lattice wall blocks.

This is the Block-2 corpus built **where a regular voxel grid is a poor fit**, so the
geometry-conditioned operator (GINO: point cloud + SDF) can justify itself against a
voxel-FNO baseline. On the axis-aligned box corpus (:mod:`thermotwin.data.synthetic_3d`)
a grid FNO suffices and GINO has no edge — the input points, the FV grid and the
voxel-FNO grid all share one axis-aligned frame, so voxelisation is loss-free. Here we
**break that alignment** two ways:

1. **Random 3-D rotation.** Each block is solved on its own aligned grid (the FV
   physics is untouched), then the whole sample — sampled points *and* the latent SDF
   grid — is rigidly rotated by a random ``SO(3)`` orientation and re-centred in the
   unit cube. The block's faces no longer line up with any axis-aligned voxel grid, so
   the voxel-FNO baseline must voxelise a tilted slab onto an axis-aligned lattice:
   the lattice straddles material interfaces and the domain boundary, smearing the
   sharp through-wall θ profile and the bridge footprints. GINO predicts at the native
   rotated points and pays none of that cost.

2. **Genuinely irregular sampling.** Points are drawn from a low-discrepancy /
   jittered scheme (not a lattice) over the box, so even before rotation the cloud is
   not grid-structured — the operator sees the scattered, non-gridded support a real
   as-built scan produces.

Crucially the FV ground truth is still solved on the block's *own* aligned grid (where
the solver is exact and cheap), and θ / the 1-D prior are interpolated at the point's
**pre-rotation** body coordinates — so the targets are physically identical to the box
corpus. Only the *stored* point coordinates and SDF are rotated. That keeps every
downstream consumer (``pointcloud_dataset``, the GINO / delta_gino / fno_voxel
benchmark) working unchanged: same per-sample keys, same feature layout, same shapes.

Per-sample record mirrors :mod:`thermotwin.data.synthetic_3d` exactly:
``points`` ``(N, 3)`` in ``[0, 1]^3`` (now rotated), ``feats`` ``(N, F)``
``[logk_std, r_si, r_se, theta1d]``, target ``theta`` ``(N,)``, per-point ``prior``
``(N,)``, latent ``sdf`` ``(G, G, G)`` (now the rotated-block SDF), and the scalars
``u_value`` / ``u_clear`` / … Everything is seeded.
"""

from __future__ import annotations

import numpy as np

from ..physics.conduction import Layer
from ..physics.steady_fv import DirichletFilm, solve_steady_conduction
from .synthetic_3d import (
    FEATURE_LAYOUT,
    LOGK_MEAN,
    LOGK_STD,
    BlockSample,
    Bridge3D,
    _cell_centres,
    _edges,
    _prior_grid,
    _trilinear,
    build_k_field_3d,
    clear_block_u,
)
from .synthetic_fem import _BASE_WALLS, _BRIDGE_K

__all__ = [
    "FEATURE_LAYOUT",
    "INSCRIBE_SCALE",
    "random_rotation",
    "halton_unit_cube",
    "rotate_into_unit_cube",
    "rotated_box_sdf_grid",
    "random_block_irregular",
    "sample_block_irregular",
    "generate_corpus_irregular",
]

# A unit cube rotated about its centre has half-diagonal sqrt(3)/2; scaling the body box
# by 1/sqrt(3) about the centre before rotating makes the farthest rotated corner sit at
# distance 1/2 from the centre, so the rotated block *always* fits inside [0, 1]^3. This
# is the single shared affine that keeps the stored points, the output queries and the
# latent-grid coordinates the SDF is built on in ONE [0, 1]^3 frame after rotation — the
# frame GINO's neighbour search and latent grid both assume. The targets (theta, prior,
# u_value) are body-frame and untouched; only the *stored* world coordinates change.
INSCRIBE_SCALE = 1.0 / np.sqrt(3.0)


def random_rotation(rng: np.random.Generator) -> np.ndarray:
    """A uniformly random ``3×3`` rotation matrix (proper, ``det = +1``).

    Uses the QR decomposition of a Gaussian matrix (Mezzadri's method): ``Q`` from
    ``A = QR`` is Haar-distributed on ``O(3)`` once each column is sign-fixed by the
    diagonal of ``R``; flipping one column when ``det = −1`` lands it in ``SO(3)``.
    """
    a = rng.standard_normal((3, 3))
    q, r = np.linalg.qr(a)
    q = q * np.sign(np.diag(r))[None, :]
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q.astype(np.float64)


def _van_der_corput(n: int, base: int) -> np.ndarray:
    """The first ``n`` terms of the van der Corput low-discrepancy sequence."""
    out = np.zeros(n, dtype=np.float64)
    for i in range(n):
        f, r, k = 1.0, 0.0, i + 1  # skip the 0 term (origin) for interior coverage
        while k > 0:
            f /= base
            r += f * (k % base)
            k //= base
        out[i] = r
    return out


def halton_unit_cube(n: int, rng: np.random.Generator) -> np.ndarray:
    """``(n, 3)`` Halton points in ``[0, 1]^3``, Cranley-Patterson randomised.

    A Halton (quasi-Monte-Carlo) cloud is genuinely *not* a lattice — no two points
    share a coordinate plane and the spacing is irregular — yet it covers the domain
    far more evenly than i.i.d. uniform, which matters when ``N`` is modest. A
    per-sample random shift (mod 1) decorrelates successive blocks while preserving
    low discrepancy, so the cloud is reproducible from the seed but distinct per block.
    """
    bases = (2, 3, 5)
    pts = np.stack([_van_der_corput(n, b) for b in bases], axis=1)
    shift = rng.random(3)[None, :]
    return (pts + shift) % 1.0


def rotate_into_unit_cube(body_norm: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    """Map body-frame ``[0, 1]^3`` coordinates into the shared rotated world frame.

    The single shared affine used for *every* stored coordinate (points, output queries,
    and — via :func:`rotated_box_sdf_grid` — the latent grid the SDF lives on):

        ``world = INSCRIBE_SCALE · R · (body − 0.5) + 0.5``.

    Scaling the unit body box by :data:`INSCRIBE_SCALE` ``= 1/√3`` about the centre
    before rotating shrinks its half-diagonal from ``√3/2`` to ``1/2``, so the rotated
    block is inscribed in ``[0, 1]^3`` for *any* ``R`` — every stored coordinate stays in
    range and the cloud shares one frame with the latent grid (which is what makes GINO's
    radius / neighbour search meaningful). Args: ``body_norm`` ``(N, 3)`` in ``[0, 1]^3``,
    ``rotation`` a ``3×3`` proper rotation. Returns ``(N, 3)`` world coordinates.
    """
    body = np.asarray(body_norm, dtype=np.float64)
    world = INSCRIBE_SCALE * ((body - 0.5) @ np.asarray(rotation, dtype=np.float64).T) + 0.5
    return world


def rotated_box_sdf_grid(rotation: np.ndarray, grid: int) -> np.ndarray:
    """SDF of a unit box rotated *and inscribed* into the unit cube, on a ``grid^3`` lattice.

    The block is the axis-aligned box ``[0, 1]^3`` in *body* coordinates; in *world*
    coordinates it is that box mapped by the shared affine of
    :func:`rotate_into_unit_cube` — scaled by :data:`INSCRIBE_SCALE` and rotated by
    ``rotation`` about the cube centre ``(0.5, 0.5, 0.5)``. The SDF is sampled on the
    axis-aligned world lattice, so the solid no longer fills the lattice — corners poke
    into empty cells, faces cut across cells — the very mismatch that makes an
    axis-aligned voxel grid a poor fit, but now the *whole* rotated block stays inside
    ``[0, 1]^3`` (no part is clipped off the lattice). Convention: ``< 0`` inside,
    ``> 0`` outside.

    We evaluate the exact box SDF in body coordinates: invert the shared affine to map
    each world grid centre back to the body frame, apply the analytic unit-box distance
    there, then rescale by :data:`INSCRIBE_SCALE` to undo the shrink (rotation is an
    isometry; the only metric change is the uniform scale, so a body-frame distance ``d``
    is a world-frame distance ``INSCRIBE_SCALE · d``). This keeps the SDF a true signed
    distance in the same world frame the points live in.
    """
    c = (np.arange(grid, dtype=np.float64) + 0.5) / grid
    gx, gy, gz = np.meshgrid(c, c, c, indexing="ij")
    world = np.stack([gx, gy, gz], axis=-1).reshape(-1, 3)  # (G^3, 3)
    # Invert world = scale·R·(body-0.5) + 0.5  ->  body = Rᵀ·(world-0.5)/scale + 0.5.
    body = ((world - 0.5) / INSCRIBE_SCALE) @ rotation + 0.5  # Rᵀ applied via right-mult by R
    q = np.abs(body - 0.5) - 0.5
    outside = np.linalg.norm(np.maximum(q, 0.0), axis=-1)
    inside = np.minimum(np.max(q, axis=-1), 0.0)
    body_dist = (outside + inside).reshape(grid, grid, grid)
    return (INSCRIBE_SCALE * body_dist).astype(np.float32)


def sample_block_irregular(
    sample: BlockSample,
    field_obj,
    spacing: list,
    n_points: int,
    grid: int,
    rng: np.random.Generator,
) -> dict:
    """Turn a solved block into an **off-grid** GINO sample.

    The FV field ``field_obj`` was solved on ``sample``'s own aligned grid. We:

    1. draw ``n_points`` irregular (Halton) points in the *body* unit cube;
    2. featurise + interpolate θ / prior at the body coordinates (physics identical to
       the box corpus — the targets do not move);
    3. apply a random ``SO(3)`` rotation about the cube centre to the **stored** point
       coordinates and to the latent SDF grid, so the cloud and SDF are off-axis.

    The returned dict has the same keys / shapes / dtypes as
    :func:`thermotwin.data.synthetic_3d.sample_block`, plus the rotation matrix for
    reproducibility.
    """
    k, _ = build_k_field_3d(sample)
    nx, ny, nz = k.shape
    dx0 = np.asarray(spacing[0], dtype=np.float64)
    dy, dz = float(spacing[1]), float(spacing[2])
    lengths = np.array([dx0.sum(), ny * dy, nz * dz], dtype=np.float64)

    t_in, t_out = float(sample.t_indoor), float(sample.t_outdoor)
    theta_field = ((field_obj.temperature.astype(np.float64) - t_out) / (t_in - t_out)).astype(
        np.float64
    )

    xc = _cell_centres(dx0)
    yc = (np.arange(ny) + 0.5) * dy
    zc = (np.arange(nz) + 0.5) * dz
    centres = [xc, yc, zc]

    # Irregular body-frame points in [0,1]^3, then mapped to physical metres.
    body_norm = halton_unit_cube(n_points, rng).astype(np.float64)
    pts_phys = body_norm * lengths[None, :]

    prior_field = _prior_grid(k, dx0, sample.r_si, sample.r_se)
    theta_pts = _trilinear(theta_field, centres, pts_phys).astype(np.float32)
    theta1d = _trilinear(prior_field, centres, pts_phys).astype(np.float32)

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

    # Break grid alignment: rotate the stored cloud + SDF about the cube centre, using
    # the single shared affine that inscribes the rotated block in [0, 1]^3. Points, the
    # latent SDF grid (inside rotated_box_sdf_grid) and the output queries (== points)
    # therefore live in ONE [0, 1]^3 frame — what GINO's radius / neighbour search needs.
    rotation = random_rotation(rng)
    pts_rot = rotate_into_unit_cube(body_norm, rotation).astype(np.float32)
    sdf = rotated_box_sdf_grid(rotation, grid)

    # The inscribing scale guarantees in-range storage; assert it so a regression in the
    # frame logic fails loudly at generation time rather than silently breaking GINO.
    assert pts_rot.min() >= -1e-6 and pts_rot.max() <= 1.0 + 1e-6, (
        f"rotated points escaped [0,1]^3: min={pts_rot.min()} max={pts_rot.max()}"
    )

    return {
        "points": pts_rot,  # (N, 3) rotated, off-axis, inscribed in [0,1]^3
        "feats": feats,  # (N, F)
        "theta": theta_pts,  # (N,)
        "prior": theta1d,  # (N,)
        "sdf": sdf,  # (G, G, G) rotated-box SDF, same [0,1]^3 frame as points
        "u_value": np.float32(field_obj.u_value),
        "rotation": rotation.astype(np.float32),  # (3, 3) for reproducibility
    }


def random_block_irregular(rng: np.random.Generator, cells_per_layer: int = 3) -> BlockSample:
    """Draw one randomised 3-D block with **non-trivial** thermal bridges.

    Same base-wall / BC draw and the same ADR-0006 insulation-targeting as
    :func:`thermotwin.data.synthetic_3d.random_block` (bridges puncture the
    lowest-conductivity layer and are strictly more conductive, so every bridge raises
    U), but with deliberately *stronger* bridges so the irregular corpus has something
    for the operator to learn beyond the prior:

    * **More bridges:** 1–4 per block (never zero), vs the box corpus's 0–3 — every
      irregular sample has at least one genuine 3-D bridge.
    * **Wider footprints:** in-plane extents drawn in ``[0.06, 0.18]`` m (vs
      ``[0.02, 0.10]``), so the bridge spans more cells and drives more lateral
      spreading — a larger θ departure from the 1-D prior.
    * **Conductive materials only:** bridge conductivity is restricted to the metals /
      concrete (steel/aluminium/concrete-nib, all ≫ insulation); the weak timber stud is
      dropped so bridges are unambiguous and the prior-alone error is meaningful.

    The body-frame physics is otherwise identical to the box corpus; the strengthening is
    purely in the bridge sampling, so the targets remain the FV solution of a genuine
    insulation-targeted thermal bridge.
    """
    name = rng.choice(list(_BASE_WALLS))
    layers: tuple[Layer, ...] = _BASE_WALLS[name]
    width_y = float(rng.uniform(0.4, 1.0))
    width_z = float(rng.uniform(0.4, 1.0))
    t_in = float(rng.uniform(18.0, 22.0))
    t_out = float(rng.uniform(-12.0, 8.0))
    r_si = float(rng.choice([0.10, 0.13, 0.17]))

    conductivities = np.array([layer.conductivity_w_mk for layer in layers])
    thicknesses = np.array([layer.thickness_m for layer in layers])
    edges = np.concatenate([[0.0], np.cumsum(thicknesses)])
    insul_idx = int(np.argmin(conductivities))  # the insulation layer (ADR-0006)
    x_lo, x_hi = float(edges[insul_idx]), float(edges[insul_idx + 1])
    insul_k = float(conductivities[insul_idx])
    # Restrict to clearly-conductive bridge materials (drop timber, k=0.13) and keep only
    # those strictly above the host insulation, so every bridge is an unambiguous one.
    strong = {m: v for m, v in _BRIDGE_K.items() if v > max(insul_k, 1.0)}
    bridge_materials = list(strong.values()) or [v for v in _BRIDGE_K.values() if v > insul_k]

    n_bridges = int(rng.integers(1, 5))  # 1..4 inclusive — never a clear (no-bridge) block
    bridges = []
    for _ in range(n_bridges):
        bk = float(rng.choice(bridge_materials))
        by = float(rng.uniform(0.06, 0.18))  # wider prism extent along y
        bz = float(rng.uniform(0.06, 0.18))  # wider prism extent along z
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
        cells_y=int(rng.choice([16, 20, 24])),
        cells_z=int(rng.choice([16, 20, 24])),
    )


def generate_corpus_irregular(
    n: int,
    seed: int = 1337,
    grid: int = 16,
    n_points: int = 2048,
    cells_per_layer: int = 3,
) -> list[dict]:
    """Generate ``n`` solved, **rotated / off-grid** 3-D blocks as GINO samples.

    Same block physics and FV solve as the box corpus, but (1) the stored geometry is
    rotated and inscribed into ``[0, 1]^3`` so a regular voxel grid no longer fits the
    domain, and (2) every block carries a *non-trivial* insulation-targeted thermal
    bridge (:func:`random_block_irregular`) so the FV field departs meaningfully from the
    1-D prior — i.e. the operator has a real residual to learn. Records carry the same
    fields as the box corpus plus the per-sample ``rotation``.
    """
    rng = np.random.default_rng(seed)
    records: list[dict] = []
    for i in range(n):
        block = random_block_irregular(rng, cells_per_layer=cells_per_layer)
        k, spacing = build_k_field_3d(block)
        bc = DirichletFilm(block.t_indoor, block.t_outdoor, r_lo=block.r_si, r_hi=block.r_se)
        res = solve_steady_conduction(k, spacing, bc)
        sample = sample_block_irregular(block, res, spacing, n_points=n_points, grid=grid, rng=rng)
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
