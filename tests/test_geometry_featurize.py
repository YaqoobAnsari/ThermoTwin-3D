"""Geometry featurisation: surface point-cloud sampling and SDF."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import trimesh

from thermotwin.geometry import idf
from thermotwin.geometry.envelope import Envelope
from thermotwin.geometry.pointcloud import (
    envelope_point_cloud,
    sample_surface,
    triangulate_polygon,
)
from thermotwin.geometry.sdf import envelope_to_mesh, sdf_grid, signed_distance

# One exterior wall (10 m x 3 m, in the plane y=0) with a simple 2-layer construction.
FIXTURE = """
  Material,
    Concrete, MediumRough, 0.2, 1.5, 2300, 900;
  Material:NoMass,
    Insul, MediumRough, 2.0, 0.9, 0.7, 0.8;
  Construction,
    ExtWall, Concrete, Insul;
  BuildingSurface:Detailed,
    South_Wall, Wall, ExtWall, ZoneA, , Outdoors, , SunExposed, WindExposed, AutoCalculate, 4,
    0.0,0.0,0.0,  10.0,0.0,0.0,  10.0,0.0,3.0,  0.0,0.0,3.0;
"""


@pytest.fixture
def env() -> Envelope:
    return Envelope.from_objects(idf.parse_idf_text(FIXTURE))


def test_triangulate_rectangle_area():
    rect = np.array([[0, 0, 0], [2, 0, 0], [2, 0, 1], [0, 0, 1]], dtype=float)
    tris = triangulate_polygon(rect)
    assert tris.shape == (2, 3, 3)


def test_sample_surface_on_plane(env):
    wall = env.exterior_opaque_surfaces()[0]
    pts, normals = sample_surface(wall, 500, np.random.default_rng(0))
    assert len(pts) == 500
    assert np.allclose(pts[:, 1], 0.0, atol=1e-9)  # all on the y=0 plane
    assert np.allclose(np.abs(normals), [0, 1, 0])  # face normal is +/- y
    # points stay within the wall extent
    assert pts[:, 0].min() >= -1e-9 and pts[:, 0].max() <= 10 + 1e-9
    assert pts[:, 2].min() >= -1e-9 and pts[:, 2].max() <= 3 + 1e-9


def test_envelope_point_cloud_features(env):
    pc = envelope_point_cloud(env, n_points=1000, seed=1337)
    assert len(pc) == 1000
    assert pc.points.shape == (1000, 3)
    assert pc.features.shape == (1000, 3)
    assert pc.feature_names == ("u_value", "resistance", "surface_type_id")
    # the u_value feature must equal the surface's analytic U-value
    wall = env.exterior_opaque_surfaces()[0]
    assert np.allclose(pc.features[:, 0], env.surface_u_value(wall))
    assert np.allclose(pc.features[:, 2], 0)  # wall type id == 0


def test_point_cloud_reproducible(env):
    a = envelope_point_cloud(env, n_points=256, seed=7)
    b = envelope_point_cloud(env, n_points=256, seed=7)
    assert np.array_equal(a.points, b.points)


def test_signed_distance_sign_convention():
    """Standard SDF: negative inside, positive outside, ~0 on the surface."""
    box = trimesh.creation.box(extents=(2.0, 2.0, 2.0))  # [-1, 1]^3
    q = np.array([[0, 0, 0], [0, 0, 2.0], [0, 0, 1.0]])
    sd = signed_distance(box, q)
    assert sd[0] < 0  # centre is inside
    assert sd[1] > 0  # outside
    assert abs(sd[2]) < 1e-6  # on the face


def test_sdf_grid_spans_both_signs():
    box = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
    sdf, (xs, ys, zs) = sdf_grid(box, resolution=8, padding=0.5)
    assert sdf.shape == (8, 8, 8)
    assert sdf.min() < 0 < sdf.max()  # interior negative, padded corners positive


def test_envelope_to_mesh(env):
    mesh = envelope_to_mesh(env, mode="exterior", repair=False)
    assert isinstance(mesh, trimesh.Trimesh)
    assert mesh.faces.shape[0] == 2  # one quad wall -> two triangles
    assert mesh.area == pytest.approx(30.0, rel=1e-6)  # 10 m x 3 m


_DOE = Path(__file__).resolve().parents[1] / "data/raw/doe/idf"
_SMALL_OFFICE = _DOE / "RefBldgSmallOfficeNew2004_Chicago.idf"


@pytest.mark.skipif(not _SMALL_OFFICE.exists(), reason="DOE IDF not downloaded")
def test_shell_mesh_is_watertight_and_sign_correct():
    """The closed shell (outdoors + ground) gives a watertight mesh with a
    reliable inside/outside SDF sign on a real building."""
    env = Envelope.from_idf(_SMALL_OFFICE)
    mesh = envelope_to_mesh(env, mode="shell", repair=True)
    assert mesh.is_watertight
    assert mesh.is_volume
    interior = mesh.bounds.mean(axis=0)  # a point inside the building
    assert signed_distance(mesh, interior[None])[0] < 0
