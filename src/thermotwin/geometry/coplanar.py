"""Recover planar wall/roof faces from a tessellated LoD3 building.

TUM2TWIN LoD3 is photogrammetric: a single building arrives as tens of thousands of
tiny triangles (one ``Surface`` per triangle once read by :mod:`thermotwin.geometry.citygml`).
Our real-geometry corpus solves **one structured FV patch per planar surface** — that abstraction
breaks on a triangle soup (each 0.003 m² sliver would get its own thermal-bridge prism, and the
shell mesh would have 69k facets). This module collapses the soup back into the planar faces a
building actually has: group triangles that share a plane (oriented normal **and** offset), then
represent each group by its in-plane bounding rectangle — exactly the rectangular slab the FV
solver expects.

The result is a *higher-fidelity* counterpart to the LoD2 corpus: LoD3 keeps dormers, facade
breaks, roof superstructures and real orientations that LoD2 smooths into a few big planes, so a
merged-LoD3 building carries more genuine planar faces (and more real normal diversity) than its
LoD2 twin, while staying tractable for the per-surface FV pipeline.

Grouping is two-stage. First by quantised ``(surface_type, oriented unit normal, plane offset
n·c)``: triangles that tessellate one flat face are *exactly* coplanar, so they fall in one bin
regardless of bin edges; mild photogrammetric non-planarity is absorbed by ``normal_tol`` /
``offset_tol``. Plane-grouping alone is **not enough** — two roof pitches (or two facade panels)
can share a normal *and* an offset while sitting on opposite sides of the building, and their
joint bbox would be a giant rectangle spanning the gap. So within each plane we run an in-plane
**connected-components** pass (region-growing on a coarse occupancy grid) and emit one face per
spatially-connected fragment. Each face is that fragment's planar bbox rectangle, wound so
Newell's normal matches the (area-weighted) group normal — i.e. it round-trips through
:func:`thermotwin.data.real_citygml_3d.surface_frame` unchanged. A per-building ``max_faces`` cap
keeps the FV pipeline bounded (smallest faces dropped first).
"""

from __future__ import annotations

from collections import Counter

import numpy as np
from scipy import ndimage

from .envelope import Surface, _polygon_area_normal

__all__ = ["merge_coplanar_surfaces"]


def _in_plane_axes(n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A right-handed ``(t1, t2)`` spanning the plane with normal ``n`` (``t1×t2 = n``)."""
    # Seed t1 from the world axis least aligned with n (avoids a near-zero projection).
    seed = np.eye(3)[int(np.argmin(np.abs(n)))]
    t1 = seed - (seed @ n) * n
    t1 /= np.linalg.norm(t1)
    t2 = np.cross(n, t1)
    t2 /= np.linalg.norm(t2)
    return t1, t2


def _rectangle(verts: np.ndarray, n: np.ndarray) -> tuple[np.ndarray, float]:
    """In-plane bbox rectangle of ``verts`` on the plane with normal ``n``.

    Returns ``(rect4x3, area)`` with corners wound ``corner → +t1 → +t1+t2 → +t2`` so that
    Newell's normal of the ring is ``+n``.
    """
    t1, t2 = _in_plane_axes(n)
    u, w = verts @ t1, verts @ t2
    umin, vmin = float(u.min()), float(w.min())
    lu, lv = float(u.max() - umin), float(w.max() - vmin)
    # Anchor the in-plane corner; keep the plane's normal offset from the centroid.
    centroid = verts.mean(axis=0)
    off = float(centroid @ n)
    corner = umin * t1 + vmin * t2 + off * n
    rect = np.stack([corner, corner + lu * t1, corner + lu * t1 + lv * t2, corner + lv * t2])
    return rect, lu * lv


def merge_coplanar_surfaces(
    surfaces: list[Surface],
    *,
    normal_tol: float = 0.08,
    offset_tol: float = 0.5,
    cluster_cell: float = 1.5,
    min_area: float = 2.0,
    min_fill: float = 0.35,
    max_faces: int = 250,
) -> list[Surface]:
    """Collapse tessellated triangles into planar, spatially-connected bbox-rectangle faces.

    Args:
        surfaces: per-triangle (or per-polygon) surfaces from a LoD3 read.
        normal_tol: normal quantisation step (unit-vector components); ``0.08`` ≈ 4–5° of
            allowed wobble within one face.
        offset_tol: plane-offset quantisation step in metres (separates parallel faces, e.g.
            the two leaves of a wall, that share a normal but sit at different depths).
        cluster_cell: in-plane connected-components grid pitch [m]; fragments closer than this
            merge into one face, fragments separated by a wider gap split apart.
        min_area: drop merged rectangles below this area [m²] (photogrammetric slivers).
        min_fill: drop a rectangle whose triangles fill less than this fraction of its bbox —
            rejects sparse, non-convex spans (e.g. an L-shaped roof's joint bbox) that would
            otherwise inflate area and dominate point allocation.
        max_faces: keep at most this many faces per building (largest by area) so the per-face
            FV stays bounded on the most detailed shells.

    Returns:
        Planar :class:`Surface` faces — vertices are the 4 in-plane bbox corners, wound to
        reproduce the group's outward normal. ``surface_type`` / ``construction_name`` /
        ``zone`` / ``boundary`` are taken by area-weighted majority.
    """
    zone = surfaces[0].zone if surfaces else ""
    # Stage 1 — group triangles by oriented plane (normal direction + offset + type).
    groups: dict[tuple, list[tuple]] = {}
    for s in surfaces:
        v = np.asarray(s.vertices, dtype=np.float64)
        if len(v) < 3:
            continue
        area, n = _polygon_area_normal(v)
        if area <= 0.0:
            continue
        c = v.mean(axis=0)
        key = (
            s.surface_type,
            tuple(np.round(n / normal_tol).astype(int)),
            int(round(float(n @ c) / offset_tol)),
        )
        groups.setdefault(key, []).append((area, n, v, s.boundary, c))

    # Stage 2 — split each plane into spatially-connected fragments, one rectangle per fragment.
    faces: list[tuple[float, Surface]] = []
    fid = 0
    for (surface_type, _, _), items in groups.items():
        areas = np.array([it[0] for it in items])
        n = (areas[:, None] * np.stack([it[1] for it in items])).sum(axis=0)
        nn = np.linalg.norm(n)
        if nn < 1e-9:
            continue
        n /= nn
        t1, t2 = _in_plane_axes(n)
        cents = np.stack([it[4] for it in items])
        cu, cw = cents @ t1, cents @ t2
        iu = np.floor((cu - cu.min()) / cluster_cell).astype(int)
        iw = np.floor((cw - cw.min()) / cluster_cell).astype(int)
        grid = np.zeros((iu.max() + 1, iw.max() + 1), dtype=bool)
        grid[iu, iw] = True
        labels, n_comp = ndimage.label(grid, structure=np.ones((3, 3)))
        tri_label = labels[iu, iw]
        boundary = Counter(it[3] for it in items).most_common(1)[0][0]
        for comp in range(1, n_comp + 1):
            sel = tri_label == comp
            if not sel.any():
                continue
            idx = np.nonzero(sel)[0]
            verts = np.concatenate([items[i][2] for i in idx], axis=0)
            rect, rect_area = _rectangle(verts, n)
            if rect_area < min_area or areas[idx].sum() / rect_area < min_fill:
                continue
            faces.append((
                rect_area,
                Surface(
                    name=f"merged_{surface_type}_{fid}",
                    surface_type=surface_type,
                    construction_name=f"{surface_type}_Construction",
                    zone=zone,
                    boundary=boundary,
                    vertices=rect,
                ),
            ))
            fid += 1

    faces.sort(key=lambda fa: fa[0], reverse=True)
    return [f for _, f in faces[:max_faces]]
