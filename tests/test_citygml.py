"""CityGML -> Envelope reader, and its bridge to the geometry featurisers.

Two layers, mirroring ``test_envelope.py``:

* **Unit** — on a tiny inline CityGML string, so they always run: ``posList``
  parsing (closing-duplicate drop) and a one-surface building lifted into an
  :class:`Envelope` with local-metric geometry.
* **Integration** — on the real TUM2TWIN LoD2 corpus if it has been downloaded,
  else skipped: a representative building parses to Walls + a Roof, vertices are
  local-metric, and the envelope feeds the point-cloud featuriser and SDF mesher
  unchanged.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from thermotwin.geometry.citygml import (
    parse_poslist,
    read_citygml_building,
    read_citygml_dir,
)
from thermotwin.geometry.envelope import Envelope
from thermotwin.geometry.pointcloud import envelope_point_cloud
from thermotwin.geometry.sdf import envelope_to_mesh

# Real TUM2TWIN LoD2 corpus location (integration tests gate on it existing).
_LOD2_DIR = Path("data/raw/tum2twin-datasets/citygml/lod2-building-datasets")
_HAVE_LOD2 = _LOD2_DIR.is_dir() and any(_LOD2_DIR.glob("*.gml"))

# A minimal CityGML 2.0 building: one wall, one roof, one ground surface, each a
# single closed LinearRing. Coordinates carry the EPSG:25832 offset so the test can
# assert that the reader subtracts a per-building origin (local-metric output).
_OX, _OY, _OZ = 691000.0, 5335970.0, 515.0
_INLINE_CITYGML = f"""<?xml version="1.0" encoding="UTF-8"?>
<core:CityModel
    xmlns:gml="http://www.opengis.net/gml"
    xmlns:bldg="http://www.opengis.net/citygml/building/2.0"
    xmlns:core="http://www.opengis.net/citygml/2.0">
  <core:cityObjectMember>
    <bldg:Building gml:id="TEST_BLDG">
      <bldg:measuredHeight uom="urn:adv:uom:m">3.0</bldg:measuredHeight>
      <bldg:boundedBy>
        <bldg:WallSurface gml:id="TEST_WALL">
          <bldg:lod2MultiSurface>
            <gml:MultiSurface>
              <gml:surfaceMember>
                <gml:Polygon gml:id="TEST_WALL_poly">
                  <gml:exterior>
                    <gml:LinearRing>
                      <gml:posList>{_OX} {_OY} {_OZ} {_OX + 4} {_OY} {_OZ} \
{_OX + 4} {_OY} {_OZ + 3} {_OX} {_OY} {_OZ + 3} {_OX} {_OY} {_OZ}</gml:posList>
                    </gml:LinearRing>
                  </gml:exterior>
                </gml:Polygon>
              </gml:surfaceMember>
            </gml:MultiSurface>
          </bldg:lod2MultiSurface>
        </bldg:WallSurface>
      </bldg:boundedBy>
      <bldg:boundedBy>
        <bldg:RoofSurface gml:id="TEST_ROOF">
          <bldg:lod2MultiSurface>
            <gml:MultiSurface>
              <gml:surfaceMember>
                <gml:Polygon gml:id="TEST_ROOF_poly">
                  <gml:exterior>
                    <gml:LinearRing>
                      <gml:posList>{_OX} {_OY} {_OZ + 3} {_OX + 4} {_OY} {_OZ + 3} \
{_OX + 4} {_OY + 4} {_OZ + 3} {_OX} {_OY + 4} {_OZ + 3} {_OX} {_OY} {_OZ + 3}</gml:posList>
                    </gml:LinearRing>
                  </gml:exterior>
                </gml:Polygon>
              </gml:surfaceMember>
            </gml:MultiSurface>
          </bldg:lod2MultiSurface>
        </bldg:RoofSurface>
      </bldg:boundedBy>
      <bldg:boundedBy>
        <bldg:GroundSurface>
          <bldg:lod2MultiSurface>
            <gml:MultiSurface>
              <gml:surfaceMember>
                <gml:Polygon>
                  <gml:exterior>
                    <gml:LinearRing>
                      <gml:posList>{_OX} {_OY} {_OZ} {_OX + 4} {_OY} {_OZ} \
{_OX + 4} {_OY + 4} {_OZ} {_OX} {_OY + 4} {_OZ} {_OX} {_OY} {_OZ}</gml:posList>
                    </gml:LinearRing>
                  </gml:exterior>
                </gml:Polygon>
              </gml:surfaceMember>
            </gml:MultiSurface>
          </bldg:lod2MultiSurface>
        </bldg:GroundSurface>
      </bldg:boundedBy>
    </bldg:Building>
  </core:cityObjectMember>
</core:CityModel>
"""


# --- unit tests (no data dependency) ---------------------------------------


def test_parse_poslist_drops_closing_duplicate() -> None:
    # Five coordinate triples forming a closed square -> four open-ring vertices.
    text = "0 0 0  1 0 0  1 1 0  0 1 0  0 0 0"
    v = parse_poslist(text)
    assert v.shape == (4, 3)
    assert not np.allclose(v[0], v[-1])
    np.testing.assert_allclose(v[0], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(v[2], [1.0, 1.0, 0.0])


def test_parse_poslist_keeps_open_ring() -> None:
    # Already-open ring (no closing dup) is returned untouched.
    v = parse_poslist("0 0 0  1 0 0  1 1 0")
    assert v.shape == (3, 3)


def test_parse_poslist_rejects_ragged() -> None:
    with pytest.raises(ValueError):
        parse_poslist("0 0 0 1 0")  # not a multiple of 3


def test_inline_building_parses(tmp_path: Path) -> None:
    path = tmp_path / "inline.gml"
    path.write_text(_INLINE_CITYGML)
    env = read_citygml_building(path)

    assert isinstance(env, Envelope)
    types = sorted(s.surface_type for s in env.surfaces)
    assert types == ["Floor", "Roof", "Wall"]

    wall = next(s for s in env.surfaces if s.surface_type == "Wall")
    roof = next(s for s in env.surfaces if s.surface_type == "Roof")
    floor = next(s for s in env.surfaces if s.surface_type == "Floor")
    assert wall.boundary == "Outdoors"
    assert roof.boundary == "Outdoors"
    assert floor.boundary == "Ground"

    # Each ring's closing duplicate is dropped -> 4 vertices, (n, 3) shape.
    assert wall.vertices.shape == (4, 3)
    # Origin subtracted -> local-metric (origin corner sits at 0,0,0).
    allv = np.concatenate([s.vertices for s in env.surfaces])
    assert np.abs(allv).max() < 1e3
    np.testing.assert_allclose(allv.min(axis=0), [0.0, 0.0, 0.0], atol=1e-9)

    # Constructions are resolvable -> the envelope reports a finite UA.
    assert env.envelope_ua() > 0.0


def test_truncated_file_raises(tmp_path: Path) -> None:
    bad = tmp_path / "truncated.gml"
    bad.write_text('<?xml version="1.0"?><core:CityModel xmlns:core="x"><bldg:Buil')
    with pytest.raises(ValueError):
        read_citygml_building(bad)


def test_dir_reader_skips_unreadable(tmp_path: Path) -> None:
    (tmp_path / "good.gml").write_text(_INLINE_CITYGML)
    (tmp_path / "bad.gml").write_text("<not-xml")
    envs = read_citygml_dir(tmp_path)
    assert len(envs) == 1  # truncated file skipped, good one kept


# --- integration tests (real TUM2TWIN LoD2 corpus) -------------------------


@pytest.mark.skipif(not _HAVE_LOD2, reason="TUM2TWIN LoD2 CityGML corpus not present")
def test_real_lod2_building_parses_to_envelope() -> None:
    # A representative small building from the corpus.
    path = _LOD2_DIR / "DEBY_LOD2_4906980.gml"
    if not path.exists():
        path = sorted(_LOD2_DIR.glob("*.gml"))[0]
    env = read_citygml_building(path)

    walls = [s for s in env.surfaces if s.surface_type == "Wall"]
    roofs = [s for s in env.surfaces if s.surface_type == "Roof"]
    assert len(walls) > 0
    assert len(roofs) >= 1

    for s in env.surfaces:
        assert s.vertices.ndim == 2 and s.vertices.shape[1] == 3
        assert s.vertices.shape[0] >= 3
    # Local-metric: a single building spans well under a kilometre.
    allv = np.concatenate([s.vertices for s in env.surfaces])
    assert np.abs(allv).max() < 1e3


@pytest.mark.skipif(not _HAVE_LOD2, reason="TUM2TWIN LoD2 CityGML corpus not present")
def test_real_lod2_feeds_featurisers() -> None:
    path = _LOD2_DIR / "DEBY_LOD2_4906980.gml"
    if not path.exists():
        path = sorted(_LOD2_DIR.glob("*.gml"))[0]
    env = read_citygml_building(path)

    # Real geometry flows through the point-cloud featuriser unchanged.
    cloud = envelope_point_cloud(env, n_points=512)
    assert len(cloud) == 512
    assert cloud.points.shape == (512, 3)
    assert cloud.features.shape[0] == 512
    assert np.isfinite(cloud.points).all()

    # ... and through the SDF mesher (watertight shell from the closed building).
    mesh = envelope_to_mesh(env)
    assert len(mesh.faces) > 0
    assert np.isfinite(mesh.vertices).all()


@pytest.mark.skipif(not _HAVE_LOD2, reason="TUM2TWIN LoD2 CityGML corpus not present")
def test_real_lod2_dir_reader() -> None:
    envs = read_citygml_dir(_LOD2_DIR)
    assert len(envs) >= 1
    # Every returned envelope has resolvable geometry.
    for env in envs:
        assert len(env.surfaces) > 0
