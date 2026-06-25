"""Tests for coplanar-surface merging (LoD3 triangle-soup -> planar faces)."""

from __future__ import annotations

import numpy as np
import pytest

from thermotwin.geometry.coplanar import merge_coplanar_surfaces
from thermotwin.geometry.envelope import Surface


def _tri(verts, stype="Wall", boundary="Outdoors", zone="b"):
    return Surface(
        name="t",
        surface_type=stype,
        construction_name=f"{stype}_Construction",
        zone=zone,
        boundary=boundary,
        vertices=np.asarray(verts, dtype=float),
    )


def _tessellate(origin, t1, t2, lu, lv, nu=4, nv=4, **kw):
    """Split a planar rectangle into a grid of triangles, like an LoD3 face."""
    origin = np.asarray(origin, float)
    tris = []
    for i in range(nu):
        for j in range(nv):
            a = origin + (i / nu * lu) * t1 + (j / nv * lv) * t2
            b = origin + ((i + 1) / nu * lu) * t1 + (j / nv * lv) * t2
            c = origin + (i / nu * lu) * t1 + ((j + 1) / nv * lv) * t2
            d = origin + ((i + 1) / nu * lu) * t1 + ((j + 1) / nv * lv) * t2
            tris.append(_tri([a, b, d], **kw))
            tris.append(_tri([a, d, c], **kw))
    return tris


X, Y, Z = np.eye(3)


def test_single_plane_merges_to_one_face():
    tris = _tessellate([0, 0, 0], X, Y, 4.0, 3.0, nu=4, nv=3)
    merged = merge_coplanar_surfaces(tris, min_area=1.0)
    assert len(merged) == 1
    s = merged[0]
    assert s.area == pytest.approx(12.0, rel=0.05)
    assert abs(abs(s.normal[2]) - 1.0) < 1e-3  # ~ ±z


def test_disjoint_coplanar_split_by_connectivity():
    tris = _tessellate([0, 0, 0], X, Y, 3.0, 3.0) + _tessellate([20, 0, 0], X, Y, 3.0, 3.0)
    merged = merge_coplanar_surfaces(tris, cluster_cell=1.5, min_area=1.0)
    assert len(merged) == 2  # 20 m gap -> two connected components, not one giant bbox


def test_distinct_orientations_split():
    wall = _tessellate([0, 0, 0], X, Z, 3.0, 3.0)  # normal ±y
    roof = _tessellate([0, 0, 0], X, Y, 3.0, 3.0)  # normal ±z
    merged = merge_coplanar_surfaces(wall + roof, min_area=1.0)
    assert len(merged) == 2


def test_normal_sign_preserved_for_frame():
    from thermotwin.data.real_citygml_3d import surface_frame

    tris = _tessellate([0, 0, 0], X, Y, 5.0, 4.0)
    for s in merge_coplanar_surfaces(tris, min_area=1.0):
        _, _, _, n, _, _ = surface_frame(s)
        assert n @ s.normal > 0.99  # frame normal agrees with the face normal


def test_max_faces_cap_keeps_largest():
    tris = []
    for k in range(6):
        tris += _tessellate([10 * k, 0, 0], X, Y, float(k + 1), float(k + 1))
    merged = merge_coplanar_surfaces(tris, cluster_cell=1.0, min_area=0.5, max_faces=3)
    assert len(merged) == 3
    areas = sorted((s.area for s in merged), reverse=True)
    assert areas == sorted(areas, reverse=True)  # returned largest-first
    assert min(areas) >= 9.0  # the three biggest squares (4,5,6)^2 region survive
