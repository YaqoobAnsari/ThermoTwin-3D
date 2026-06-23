"""Sample an envelope into a feature-tagged surface point cloud.

This is the **geometry input the operator consumes** (thesis Stage 1): the as-built
envelope as a point cloud where every point carries not just its position and
surface normal but the *thermal* attributes of the surface it sits on — the
construction's U-value and resistance, and the surface type. GINO encodes exactly
this kind of irregular point cloud (+ SDF, see :mod:`thermotwin.geometry.sdf`), so
this module produces the (points, normals, features) triple it ingests.

Sampling is area-weighted across each surface's fan triangulation, so larger
surfaces receive proportionally more points and the density is uniform over the
envelope. A seeded ``numpy`` Generator makes every cloud reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .envelope import Envelope, Surface, _polygon_area_normal

__all__ = ["SurfacePointCloud", "triangulate_polygon", "sample_surface", "envelope_point_cloud"]

# Stable integer ids for surface types (for a categorical feature channel).
_SURFACE_TYPE_ID = {"wall": 0, "roof": 1, "ceiling": 2, "floor": 3}


@dataclass(frozen=True)
class SurfacePointCloud:
    """A sampled point cloud with per-point geometry + thermal features."""

    points: np.ndarray  # (N, 3) coordinates [m]
    normals: np.ndarray  # (N, 3) unit surface normals
    features: np.ndarray  # (N, F) per-point features
    feature_names: tuple[str, ...]

    def __len__(self) -> int:
        return self.points.shape[0]


def triangulate_polygon(vertices: np.ndarray) -> np.ndarray:
    """Fan-triangulate a planar (convex) polygon into triangles (T, 3, 3).

    Building surfaces are simple convex polygons (rectangles, the odd convex
    n-gon), for which a triangle fan from the first vertex is exact.
    """
    v = np.asarray(vertices, dtype=float)
    if len(v) < 3:
        raise ValueError("need >= 3 vertices")
    return np.stack([np.stack([v[0], v[i], v[i + 1]]) for i in range(1, len(v) - 1)])


def _triangle_areas(tris: np.ndarray) -> np.ndarray:
    cross = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    return 0.5 * np.linalg.norm(cross, axis=1)


def sample_surface(
    surface: Surface, n: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Sample ``n`` points uniformly over a surface; returns (points, normals)."""
    tris = triangulate_polygon(surface.vertices)
    areas = _triangle_areas(tris)
    total = areas.sum()
    if total <= 0:
        raise ValueError(f"surface {surface.name!r} has zero area")
    # Distribute n points across triangles proportional to area.
    counts = rng.multinomial(n, areas / total)
    normal = _polygon_area_normal(surface.vertices)[1]

    pts = []
    for tri, c in zip(tris, counts, strict=True):
        if c == 0:
            continue
        # Uniform barycentric sampling within a triangle.
        r1 = np.sqrt(rng.random(c))
        r2 = rng.random(c)
        a = 1.0 - r1
        b = r1 * (1.0 - r2)
        cc = r1 * r2
        pts.append(a[:, None] * tri[0] + b[:, None] * tri[1] + cc[:, None] * tri[2])
    points = np.concatenate(pts) if pts else np.empty((0, 3))
    normals = np.tile(normal, (len(points), 1))
    return points, normals


def envelope_point_cloud(
    envelope: Envelope,
    n_points: int = 4096,
    exterior_only: bool = True,
    seed: int = 1337,
) -> SurfacePointCloud:
    """Sample a whole envelope into a feature-tagged point cloud.

    Per-point features are ``[u_value, resistance, surface_type_id]`` — the thermal
    attributes of the cladding the point sits on — alongside the geometric normals.
    Points are distributed across surfaces proportional to area.
    """
    rng = np.random.default_rng(seed)
    surfaces = (
        envelope.exterior_opaque_surfaces()
        if exterior_only
        else [s for s in envelope.surfaces if _norm_in(s, envelope)]
    )
    if not surfaces:
        raise ValueError("no resolvable surfaces to sample")

    areas = np.array([s.area for s in surfaces])
    alloc = rng.multinomial(n_points, areas / areas.sum())

    all_pts, all_norms, all_feats = [], [], []
    for surface, n in zip(surfaces, alloc, strict=True):
        if n == 0:
            continue
        pts, norms = sample_surface(surface, int(n), rng)
        u = envelope.surface_u_value(surface)
        constr = envelope._constr_by_norm[_norm(surface.construction_name)]
        type_id = _SURFACE_TYPE_ID.get(surface.surface_type.strip().lower(), -1)
        feat = np.tile([u, constr.resistance, type_id], (len(pts), 1))
        all_pts.append(pts)
        all_norms.append(norms)
        all_feats.append(feat)

    return SurfacePointCloud(
        points=np.concatenate(all_pts),
        normals=np.concatenate(all_norms),
        features=np.concatenate(all_feats),
        feature_names=("u_value", "resistance", "surface_type_id"),
    )


def _norm(name: str) -> str:
    return name.strip().casefold()


def _norm_in(surface: Surface, envelope: Envelope) -> bool:
    return _norm(surface.construction_name) in envelope._constr_by_norm
