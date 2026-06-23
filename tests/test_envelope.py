"""IDF → envelope featuriser, and its bridge to the conduction solver.

Two layers of tests:

* **Unit** — on a small inline IDF fixture, so they always run: parsing,
  material/construction resolution, surface geometry, and the key invariant that a
  featurised construction's analytic U-value equals the finite-volume solver's
  effective U over the same layers (Stage-1 geometry ⇆ Stage-2 physics).
* **Integration** — on the real DOE SmallOffice IDF if it has been downloaded,
  else skipped.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from thermotwin.geometry import idf
from thermotwin.geometry.envelope import Envelope, surface_films
from thermotwin.physics.steady_fv import (
    DirichletFilm,
    layered_k_field,
    solve_steady_conduction,
)

# A compact but realistic IDF: a 2-layer exterior wall (mass + no-mass insulation),
# one exterior wall surface (with the optional Space Name field) and one interior
# floor (without it, to exercise version-robust vertex parsing).
FIXTURE = """
  Material,
    Concrete,                !- Name
    MediumRough,             !- Roughness
    0.2000,                  !- Thickness {m}
    1.5000,                  !- Conductivity {W/m-K}
    2300.0,                  !- Density {kg/m3}
    900.0;                   !- Specific Heat {J/kg-K}

  Material:NoMass,
    WallInsulation,          !- Name
    MediumRough,             !- Roughness
    2.0000,                  !- Thermal Resistance {m2-K/W}
    0.9, 0.7, 0.7;           !- absorptances

  Construction,
    ExtWall,                 !- Name
    Concrete,                !- Outside Layer
    WallInsulation;          !- Layer 2

  BuildingSurface:Detailed,
    South_Wall,              !- Name
    Wall,                    !- Surface Type
    ExtWall,                 !- Construction Name
    ZoneA,                   !- Zone Name
    ,                        !- Space Name
    Outdoors,                !- Outside Boundary Condition
    ,                        !- Outside Boundary Condition Object
    SunExposed,              !- Sun Exposure
    WindExposed,             !- Wind Exposure
    AutoCalculate,           !- View Factor to Ground
    4,                       !- Number of Vertices
    0.0, 0.0, 0.0,
    10.0, 0.0, 0.0,
    10.0, 0.0, 3.0,
    0.0, 0.0, 3.0;

  BuildingSurface:Detailed,
    Interior_Floor,          !- Name (no Space Name field — older layout)
    Floor,                   !- Surface Type
    ExtWall,                 !- Construction Name
    ZoneA,                   !- Zone Name
    Surface,                 !- Outside Boundary Condition
    Other_Ceiling,           !- Outside Boundary Condition Object
    NoSun,                   !- Sun Exposure
    NoWind,                  !- Wind Exposure
    AutoCalculate,           !- View Factor to Ground
    4,                       !- Number of Vertices
    0.0, 0.0, 0.0,
    10.0, 0.0, 0.0,
    10.0, 5.0, 0.0,
    0.0, 5.0, 0.0;
"""


@pytest.fixture
def env() -> Envelope:
    return Envelope.from_objects(idf.parse_idf_text(FIXTURE))


def test_materials_parsed(env):
    concrete = env.materials["Concrete"]
    assert concrete.thickness_m == pytest.approx(0.2)
    assert concrete.conductivity_w_mk == pytest.approx(1.5)
    assert concrete.resistance_m2k_w == pytest.approx(0.2 / 1.5)
    insul = env.materials["WallInsulation"]
    assert insul.is_massless
    assert insul.resistance_m2k_w == pytest.approx(2.0)


def test_construction_resistance_and_u(env):
    wall = env.constructions["ExtWall"]
    assert wall.resistance == pytest.approx(0.2 / 1.5 + 2.0)
    film = surface_films("wall")
    expected_u = 1.0 / (film.r_si + wall.resistance + film.r_se)
    assert wall.u_value("wall") == pytest.approx(expected_u)


def test_surface_geometry(env):
    south = next(s for s in env.surfaces if s.name == "South_Wall")
    assert south.is_exterior
    assert south.boundary == "Outdoors"
    assert south.vertices.shape == (4, 3)
    assert south.area == pytest.approx(30.0)  # 10 m x 3 m
    assert np.allclose(np.abs(south.normal), [0.0, 1.0, 0.0])  # faces +/- y


def test_version_robust_vertex_parsing(env):
    """The floor surface omits the Space Name field; parsing must still work."""
    floor = next(s for s in env.surfaces if s.name == "Interior_Floor")
    assert floor.vertices.shape == (4, 3)
    assert floor.area == pytest.approx(50.0)  # 10 m x 5 m
    assert not floor.is_exterior  # boundary == Surface


def test_exterior_opaque_filtering(env):
    ext = env.exterior_opaque_surfaces()
    assert [s.name for s in ext] == ["South_Wall"]


def test_envelope_ua(env):
    wall = env.constructions["ExtWall"]
    expected = wall.u_value("wall") * 30.0
    assert env.envelope_ua() == pytest.approx(expected)


@pytest.mark.parametrize("cells", [4, 12])
def test_featurised_construction_matches_solver(env, cells):
    """The geometry→physics bridge: a featurised construction solved on the grid
    reproduces its analytic U-value (with films)."""
    wall = env.constructions["ExtWall"]
    film = surface_films("wall")
    layers = wall.to_conduction_layers()
    k, spacing = layered_k_field(layers, cells_per_layer=cells)
    res = solve_steady_conduction(
        k, spacing, DirichletFilm(20.0, -2.0, r_lo=film.r_si, r_hi=film.r_se)
    )
    assert res.u_value == pytest.approx(wall.u_value("wall"), rel=1e-6)


# --- Integration against the real DOE building (skipped if not downloaded) ---

_DOE = Path(__file__).resolve().parents[1] / "data/raw/doe/idf"
_SMALL_OFFICE = _DOE / "RefBldgSmallOfficeNew2004_Chicago.idf"


@pytest.mark.skipif(not _SMALL_OFFICE.exists(), reason="DOE IDF not downloaded")
def test_real_doe_small_office():
    env = Envelope.from_idf(_SMALL_OFFICE)
    assert len(env.materials) > 5
    assert len(env.constructions) > 3
    ext = env.exterior_opaque_surfaces()
    assert len(ext) >= 4  # at least the four facade walls
    # Exterior walls should have physically plausible, insulated U-values.
    walls = [s for s in ext if s.surface_type.lower() == "wall"]
    assert walls
    for s in walls:
        u = env.surface_u_value(s)
        assert 0.1 < u < 1.5, f"{s.name}: implausible U={u}"
    assert env.envelope_ua() > 0
