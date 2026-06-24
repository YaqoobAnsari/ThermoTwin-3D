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

from ..physics.steady_fv import DirichletFilm, solve_steady_conduction
from .synthetic_3d import (
    FEATURE_LAYOUT,
    LOGK_MEAN,
    LOGK_STD,
    BlockSample,
    _cell_centres,
    _edges,
    _prior_grid,
    _trilinear,
    build_k_field_3d,
    clear_block_u,
    random_block,
)

__all__ = [
    "FEATURE_LAYOUT",
    "random_rotation",
    "halton_unit_cube",
    "rotated_box_sdf_grid",
    "sample_block_irregular",
    "generate_corpus_irregular",
]


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


def rotated_box_sdf_grid(rotation: np.ndarray, grid: int) -> np.ndarray:
    """SDF of a unit box rigidly rotated about its centre, on a ``grid^3`` lattice.

    The block is the axis-aligned box ``[0, 1]^3`` in *body* coordinates; in *world*
    coordinates it is that box rotated by ``rotation`` about the cube centre
    ``(0.5, 0.5, 0.5)``. The SDF is sampled on the axis-aligned world lattice, so the
    solid no longer fills the lattice — corners poke out, faces cut across cells — the
    very mismatch that makes an axis-aligned voxel grid a poor fit. Convention: ``< 0``
    inside, ``> 0`` outside.

    We evaluate the exact box SDF in body coordinates: map each world grid centre back
    through ``Rᵀ`` to the body frame, then apply the analytic box distance there
    (rotation is an isometry, so the body-frame distance is the world-frame distance).
    """
    c = (np.arange(grid, dtype=np.float64) + 0.5) / grid
    gx, gy, gz = np.meshgrid(c, c, c, indexing="ij")
    world = np.stack([gx, gy, gz], axis=-1).reshape(-1, 3)  # (G^3, 3)
    body = (world - 0.5) @ rotation + 0.5  # world -> body via Rᵀ (rotation.T applied on right)
    q = np.abs(body - 0.5) - 0.5
    outside = np.linalg.norm(np.maximum(q, 0.0), axis=-1)
    inside = np.minimum(np.max(q, axis=-1), 0.0)
    return (outside + inside).reshape(grid, grid, grid).astype(np.float32)


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

    # Break grid alignment: rotate the stored cloud + SDF about the cube centre.
    rotation = random_rotation(rng)
    pts_rot = ((body_norm - 0.5) @ rotation.T + 0.5).astype(np.float32)
    sdf = rotated_box_sdf_grid(rotation, grid)

    return {
        "points": pts_rot,  # (N, 3) rotated, off-axis, in (a superset of) [0,1]^3
        "feats": feats,  # (N, F)
        "theta": theta_pts,  # (N,)
        "prior": theta1d,  # (N,)
        "sdf": sdf,  # (G, G, G) rotated-box SDF
        "u_value": np.float32(field_obj.u_value),
        "rotation": rotation.astype(np.float32),  # (3, 3) for reproducibility
    }


def generate_corpus_irregular(
    n: int,
    seed: int = 1337,
    grid: int = 16,
    n_points: int = 2048,
    cells_per_layer: int = 3,
) -> list[dict]:
    """Generate ``n`` solved, **rotated / off-grid** 3-D blocks as GINO samples.

    Identical block physics to :func:`thermotwin.data.synthetic_3d.generate_corpus_3d`
    (same ``random_block`` draw, same FV solve), but the stored geometry is irregular
    and rotated so a regular voxel grid no longer fits the domain. Records carry the
    same fields as the box corpus plus the per-sample ``rotation``.
    """
    rng = np.random.default_rng(seed)
    records: list[dict] = []
    for i in range(n):
        block = random_block(rng)
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
