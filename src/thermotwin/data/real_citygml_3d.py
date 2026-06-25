"""Real-geometry 3-D conduction corpus from TUM2TWIN CityGML buildings (Exp 2.6).

This is the **real-geometry** Block-2 corpus — the test that converts the synthetic
`delta_transolver` win (Exp 2.5) into a real-world one. The diagnosis there sharpened to:
*non-axis-alignment (rotation) is what breaks a voxel grid, and that is where the gridless
operator wins.* A real building envelope is exactly that — many planar surfaces at many real
orientations, assembled into a genuinely non-box shell. So we put the operators on it.

Pipeline per building (real geometry, physics-exact simulated GT — no measured thermal field
exists, see ``docs/data_inventory.md``):

1. Lift the CityGML building into an :class:`~thermotwin.geometry.envelope.Envelope`
   (`geometry/citygml.py`): real surfaces (Wall/Roof/Floor) at their real 3-D orientations,
   each with a material construction from the default library.
2. For each shell surface, solve a **structured FV patch** behind it — the surface's real
   construction layered through-wall, in-plane extent from the surface's real bounding box,
   punctured by conductive thermal-bridge prisms (insulation-targeted, ADR 0006). Reuses the
   synthetic 3-D machinery (`build_k_field_3d`, `solve_steady_conduction`, `sample_block`),
   so the per-surface θ field and the 1-D prior are exactly as validated.
3. **Map** each surface's body-frame sample points onto the real 3-D surface: extrude the
   polygon inward by the wall thickness along the real surface normal. The per-surface clouds
   are concatenated into one **whole-building** point cloud at real orientations.
4. The building cloud is normalised into ``[0, 1]^3`` (one shared affine from the building's
   own bounding box), and the SDF is sampled on a ``G^3`` latent grid in that same frame from
   the real watertight shell mesh. Building geometry (frames, mesh, SDF, normalisation) is
   computed **once** and reused across the ``n_per_building`` augmented samples (which differ
   only in bridge placement + point sampling).

Each sample is one building → the same per-sample record schema as
:mod:`thermotwin.data.synthetic_3d` (points / feats ``[logk_std, r_si, r_se, theta1d]`` /
theta / prior / sdf / u_value / u_clear / …), so the existing `PointCloudDataset` and the
Block-2 benchmark consume it unchanged. Split **by building** (train vs val on disjoint
buildings) so val is genuine generalisation to unseen real geometry.

**Honest scope.** Real as-built geometry (real shells, real orientations, real per-surface
constructions); the conduction physics is per-surface FV (whole-building bridge coupling not
modelled) with synthetic-but-exact bridges — there is no measured 3-D thermal field to use.
The U-value column is **not trustworthy** (multi-normal geometry breaks the indoor-face
estimator) — **field rel-L2 is the metric**, exactly as on the irregular corpus.
"""

from __future__ import annotations

import numpy as np

from ..geometry.citygml import read_citygml_dir
from ..geometry.envelope import Envelope, Surface, surface_films
from ..geometry.sdf import envelope_to_mesh, signed_distance
from ..physics.steady_fv import DirichletFilm, solve_steady_conduction
from .synthetic_3d import (
    FEATURE_LAYOUT,
    LOGK_MEAN,
    LOGK_STD,
    BlockSample,
    Bridge3D,
    build_k_field_3d,
    clear_block_u,
    sample_block,
)
from .synthetic_fem import _BRIDGE_K

__all__ = [
    "FEATURE_LAYOUT",
    "LOGK_MEAN",
    "LOGK_STD",
    "surface_frame",
    "generate_corpus_realcg",
]


def surface_frame(surface: Surface) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Local in-plane frame of a planar surface.

    Returns ``(centroid, t1, t2, n, Lu, Lv)``: the surface centroid, two orthonormal
    in-plane tangent axes, the outward unit normal, and the in-plane extents (the
    surface's bounding box measured along ``t1`` / ``t2``). A body in-plane coordinate
    ``(by, bz) ∈ [0,1]^2`` maps to the 3-D plane point
    ``centroid + (umin + by·Lu)·t1 + (vmin + bz·Lv)·t2`` (umin/vmin folded into the
    returned centroid by shifting it to the bbox corner).
    """
    verts = np.asarray(surface.vertices, dtype=np.float64)
    c = verts.mean(axis=0)
    n = np.asarray(surface.normal, dtype=np.float64)
    nn = np.linalg.norm(n)
    n = n / nn if nn > 0 else np.array([0.0, 0.0, 1.0])
    # First polygon edge, projected into the plane, as t1; t2 completes the frame.
    e = verts[1] - verts[0]
    t1 = e - (e @ n) * n
    t1n = np.linalg.norm(t1)
    if t1n < 1e-9:  # degenerate first edge — pick any in-plane axis
        t1 = np.cross(n, [1.0, 0.0, 0.0])
        if np.linalg.norm(t1) < 1e-9:
            t1 = np.cross(n, [0.0, 1.0, 0.0])
        t1n = np.linalg.norm(t1)
    t1 = t1 / t1n
    t2 = np.cross(n, t1)
    t2 /= np.linalg.norm(t2)
    rel = verts - c
    u, v = rel @ t1, rel @ t2
    umin, umax, vmin, vmax = float(u.min()), float(u.max()), float(v.min()), float(v.max())
    lu, lv = max(umax - umin, 1e-3), max(vmax - vmin, 1e-3)
    # Shift the anchor to the bbox corner so a body (by,bz)=(0,0) sits at the corner.
    corner = c + umin * t1 + vmin * t2
    return corner, t1, t2, n, lu, lv


def _make_surface_bridges(
    layers, lu: float, lv: float, rng: np.random.Generator, n_bridges: int
) -> tuple[Bridge3D, ...]:
    """Insulation-targeted conductive thermal-bridge prisms for a surface patch."""
    conductivities = np.array([layer.conductivity_w_mk for layer in layers])
    thicknesses = np.array([layer.thickness_m for layer in layers])
    edges = np.concatenate([[0.0], np.cumsum(thicknesses)])
    insul_idx = int(np.argmin(conductivities))
    x_lo, x_hi = float(edges[insul_idx]), float(edges[insul_idx + 1])
    insul_k = float(conductivities[insul_idx])
    strong = [v for v in _BRIDGE_K.values() if v > max(insul_k, 1.0)]
    bridge_materials = strong or [v for v in _BRIDGE_K.values() if v > insul_k] or [max(_BRIDGE_K.values())]
    bridges = []
    for _ in range(n_bridges):
        bk = float(rng.choice(bridge_materials))
        # Wider footprints (12–30 % of the surface) so the field departs non-trivially from
        # the 1-D prior even after averaging over the whole-building cloud — otherwise the
        # localized bridge residual is swamped by the clear-wall majority and the benchmark
        # cannot separate the operators (the irregular corpus uses similarly strong bridges).
        by = float(rng.uniform(0.12, 0.30)) * lu
        bz = float(rng.uniform(0.12, 0.30)) * lv
        y0 = float(rng.uniform(0.0, max(lu - by, 0.0)))
        z0 = float(rng.uniform(0.0, max(lv - bz, 0.0)))
        bridges.append(Bridge3D(x_lo, x_hi, y0, y0 + by, z0, z0 + bz, bk))
    return tuple(bridges)


def _building_sample(
    env: Envelope,
    frames: list,
    norm_min: np.ndarray,
    norm_scale: float,
    sdf: np.ndarray,
    n_points: int,
    rng: np.random.Generator,
    cells_per_layer: int,
    cells_in_plane: int,
) -> dict | None:
    """One augmented sample (cloud + θ + prior + feats) for a building; geometry reused."""
    surfaces = env.shell_surfaces()
    areas = np.array([max(s.area, 1e-6) for s in surfaces])
    alloc = rng.multinomial(n_points, areas / areas.sum())

    pts_w, feats, theta, prior, u_clears, areas_used = [], [], [], [], [], []
    t_in, t_out, r_se = 20.0, 0.0, 0.04
    for surface, (corner, t1, t2, n, lu, lv), n_s in zip(surfaces, frames, alloc, strict=True):
        if n_s < 8:  # too few points to be worth a solve
            continue
        constr = env._constr_by_norm[surface.construction_name.strip().casefold()]
        layers = constr.to_conduction_layers()
        if not layers:
            continue
        r_si = float(surface_films(surface.surface_type).r_si)
        n_bridges = int(rng.integers(2, 5))  # 2–4 strong bridges per surface
        bridges = _make_surface_bridges(layers, lu, lv, rng, n_bridges)
        block = BlockSample(
            layers=tuple(layers),
            width_y_m=lu,
            width_z_m=lv,
            t_indoor=t_in,
            t_outdoor=t_out,
            r_si=r_si,
            r_se=r_se,
            bridges=bridges,
            cells_per_layer=cells_per_layer,
            cells_y=cells_in_plane,
            cells_z=cells_in_plane,
        )
        try:
            k, spacing = build_k_field_3d(block)
            res = solve_steady_conduction(
                k, spacing, DirichletFilm(t_in, t_out, r_lo=r_si, r_hi=r_se)
            )
            samp = sample_block(block, res, spacing, n_points=int(n_s), grid=8, rng=rng)
        except Exception:  # a degenerate surface — skip it, keep the rest of the building
            continue
        body = samp["points"].astype(np.float64)  # (n,3): [through-wall, u, v] in [0,1]
        thickness = float(block.thickness_m)
        # Map body -> real 3-D world: in-plane on the polygon, extruded inward by thickness.
        world = (
            corner[None, :]
            + (body[:, 1] * lu)[:, None] * t1[None, :]
            + (body[:, 2] * lv)[:, None] * t2[None, :]
            - (body[:, 0] * thickness)[:, None] * n[None, :]
        )
        pts_w.append(world)
        feats.append(samp["feats"])
        theta.append(samp["theta"])
        prior.append(samp["prior"])
        u_clears.append(clear_block_u(block))
        areas_used.append(float(surface.area))

    if not pts_w:
        return None
    world = np.concatenate(pts_w, axis=0)
    # Clip is a float-safety net only; the slab-corner normalisation already bounds points.
    pts_norm = np.clip((world - norm_min[None, :]) / norm_scale, 0.0, 1.0).astype(np.float32)
    feats = np.concatenate(feats, axis=0).astype(np.float32)
    theta = np.concatenate(theta, axis=0).astype(np.float32)
    prior = np.concatenate(prior, axis=0).astype(np.float32)
    # Area-weighted clear-wall U as the building reference (the U metric is not trusted on
    # multi-normal geometry; stored for completeness only — see module docstring).
    w = np.array(areas_used)
    u_clear = float(np.average(u_clears, weights=w))
    return {
        "points": pts_norm,
        "feats": feats,
        "theta": theta,
        "prior": prior,
        "sdf": sdf.astype(np.float32),
        "u_value": np.float32(u_clear),  # no trusted bridged-U at building scale
        "u_clear": np.float32(u_clear),
        "r_si": np.float32(0.13),
        "r_se": np.float32(r_se),
        "grid_shape": np.asarray(sdf.shape, dtype=np.int32),
        "n_bridges": np.int32(-1),  # per-surface bridges; building count not meaningful
    }


def _building_geometry(env: Envelope, grid: int):
    """Compute the reusable per-building geometry: frames, normalisation, latent SDF.

    Returns ``(frames, norm_min, norm_scale, sdf)`` or ``None`` if the shell cannot be
    meshed / signed-distanced (rare; the building is then skipped by the caller).
    """
    surfaces = env.shell_surfaces()
    if len(surfaces) < 3:
        return None
    frames = [surface_frame(s) for s in surfaces]
    # Normalise by the bbox of every surface *slab corner* (in-plane rectangle extruded
    # inward by the wall thickness): every sample point is a convex combination inside its
    # slab, so this box provably contains all points -> guaranteed in [0,1]^3 (a tilted
    # wall's in-plane rectangle can poke outside the axis-aligned vertex box, which is why
    # the raw vertex bbox is not enough). This keeps GINO's neighbour search / latent grid
    # valid (the out-of-range-points failure mode that confounded the original Exp 2.2).
    corners = []
    for s, (corner, t1, t2, n, lu, lv) in zip(surfaces, frames, strict=True):
        constr = env._constr_by_norm.get(s.construction_name.strip().casefold())
        thickness = sum(layer.thickness_m for layer in constr.to_conduction_layers()) if constr else 0.3
        for by in (0.0, 1.0):
            for bz in (0.0, 1.0):
                for bx in (0.0, 1.0):
                    corners.append(corner + by * lu * t1 + bz * lv * t2 - bx * thickness * n)
    corners = np.asarray(corners)
    norm_min = corners.min(axis=0)
    norm_scale = float(max((corners.max(axis=0) - norm_min).max(), 1e-3))
    try:
        mesh = envelope_to_mesh(env, mode="shell", repair=True)
        c = (np.arange(grid, dtype=np.float64) + 0.5) / grid
        gx, gy, gz = np.meshgrid(c, c, c, indexing="ij")
        grid_norm = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
        grid_world = grid_norm * norm_scale + norm_min[None, :]
        sdf = (signed_distance(mesh, grid_world) / norm_scale).reshape(grid, grid, grid)
    except Exception:
        return None
    return frames, norm_min, norm_scale, sdf


def _corpus_from_envelopes(
    envelopes: list,
    n_per_building: int,
    seed: int,
    grid: int,
    n_points: int,
    cells_per_layer: int,
    cells_in_plane: int,
) -> list[dict]:
    """Core loop: turn a list of real :class:`Envelope` s into corpus records.

    Shared by the TUM2TWIN (CityGML) and 3D BAG (CityJSON) generators — the geometry source
    differs, the per-surface FV + assembly is identical.
    """
    rng = np.random.default_rng(seed)
    records: list[dict] = []
    sample_id = 0
    skipped = 0
    for env in envelopes:
        geom = _building_geometry(env, grid)
        if geom is None:
            skipped += 1
            continue
        frames, norm_min, norm_scale, sdf = geom
        for _ in range(n_per_building):
            rec = _building_sample(
                env, frames, norm_min, norm_scale, sdf, n_points, rng, cells_per_layer, cells_in_plane
            )
            if rec is None:
                continue
            rec["id"] = sample_id
            rec["heat_flux"] = np.float32(0.0)
            rec["t_indoor"] = np.float32(20.0)
            rec["t_outdoor"] = np.float32(0.0)
            records.append(rec)
            sample_id += 1
    if skipped:
        print(f"  (skipped {skipped} buildings that could not be meshed/sampled)")
    return records


def generate_corpus_realcg(
    citygml_dir: str,
    n_per_building: int = 5,
    seed: int = 1337,
    grid: int = 16,
    n_points: int = 4096,
    cells_per_layer: int = 4,
    cells_in_plane: int = 16,
    building_start: int = 0,
    building_end: int | None = None,
) -> list[dict]:
    """Real-geometry corpus from TUM2TWIN ``.gml`` LoD2 buildings (CityGML).

    ``building_start`` / ``building_end`` slice the (deterministically ordered) building list so
    train and val are disjoint real buildings (val geometry unseen). Records match the
    :mod:`synthetic_3d` schema.
    """
    envelopes = read_citygml_dir(citygml_dir)[building_start:building_end]
    return _corpus_from_envelopes(
        envelopes, n_per_building, seed, grid, n_points, cells_per_layer, cells_in_plane
    )


def generate_corpus_bag(
    cityjson_dir: str,
    n_per_building: int = 2,
    seed: int = 1337,
    grid: int = 16,
    n_points: int = 4096,
    cells_per_layer: int = 4,
    cells_in_plane: int = 16,
    building_start: int = 0,
    building_end: int | None = None,
) -> list[dict]:
    """Real-geometry corpus from 3D BAG CityJSON tiles (~thousands of real LoD2.2 shells).

    Same per-surface FV + assembly as :func:`generate_corpus_realcg`; geometry comes from the
    CityJSON reader (:mod:`thermotwin.geometry.cityjson`). ``building_start`` / ``building_end``
    slice the deterministic envelope list (one envelope per BuildingPart, across all tiles) so
    train/val are disjoint real buildings.
    """
    from ..geometry.cityjson import read_cityjson_dir

    envelopes = read_cityjson_dir(cityjson_dir)[building_start:building_end]
    return _corpus_from_envelopes(
        envelopes, n_per_building, seed, grid, n_points, cells_per_layer, cells_in_plane
    )


def generate_corpus_doe(
    doe_dir: str,
    n_per_building: int = 6,
    seed: int = 1337,
    grid: int = 16,
    n_points: int = 4096,
    cells_per_layer: int = 4,
    cells_in_plane: int = 16,
    building_start: int = 0,
    building_end: int | None = None,
) -> list[dict]:
    """Corpus from the DOE Commercial Reference Buildings (EnergyPlus IDFs).

    The 16 DOE buildings carry **real material/construction libraries** on idealised (box-like,
    multi-zone) geometry — the direct-fit dataset that pairs realistic *constructions* with
    clean geometry. Each IDF -> ``Envelope.from_idf`` -> the shared per-surface FV + assembly
    pipeline. ``building_start`` / ``building_end`` slice the (sorted) IDF list for a disjoint
    train/val split.
    """
    from pathlib import Path

    from ..geometry.envelope import Envelope

    idfs = sorted(Path(doe_dir).rglob("*.idf"))[building_start:building_end]
    envelopes = []
    for p in idfs:
        try:
            envelopes.append(Envelope.from_idf(p))
        except Exception as exc:
            print(f"  (skip {p.name}: {exc})")
    return _corpus_from_envelopes(
        envelopes, n_per_building, seed, grid, n_points, cells_per_layer, cells_in_plane
    )
