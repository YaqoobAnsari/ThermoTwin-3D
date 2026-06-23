"""Read a TUM2TWIN CityGML building into our :class:`~thermotwin.geometry.envelope.Envelope`.

This is the **real-geometry on-ramp** for Block-2 (thesis Stage 1): replace the
synthetic DOE/IDF envelopes with as-built city models so the geometry-conditioned
operator sees the irregular, tessellated shells it must ultimately handle. CityGML
carries geometry *only* — surface polygons tagged Wall / Roof / Ground — so this
module lifts those polygons into our :class:`Surface` objects and pairs each with a
material/construction drawn from a small default library keyed by surface type
(exactly as :mod:`thermotwin.geometry.pointcloud` already assumes). The resulting
:class:`Envelope` flows unchanged into the point-cloud featuriser and the SDF mesher.

The CityGML facts we rely on (TUM2TWIN, CityGML 2.0, EPSG:25832 ETRS89/UTM32N):

* Each ``bldg:boundedBy`` wraps one thematic surface whose *local* tag is
  ``WallSurface`` / ``RoofSurface`` / ``GroundSurface`` (mapped to Wall / Roof /
  Floor). Its geometry is one or more ``gml:Polygon`` s.
* A polygon's outer ring is ``gml:exterior/gml:LinearRing/gml:posList``: a flat
  list of ``x y z`` floats; the last vertex repeats the first (dropped here).
* LoD2 gives one clean polygon per surface; LoD3 tessellates a surface into many
  triangles and adds ``gml:interior`` rings for window/door holes (we keep only the
  exterior rings — holes are treated as part of the opaque wall for v1).
* Coordinates are metric but offset by ~6.9e5 / 5.3e6; we subtract a per-building
  origin (the min corner over all vertices) so vertices are **local-metric**.

Robustness: a truncated file makes :func:`ET.parse` raise ``ParseError``; we wrap
parsing and either re-raise a clear error (single-building API) or skip-and-log
(directory API).
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from .envelope import Construction, Envelope, MaterialLayer, Surface

__all__ = [
    "CITYGML_NS",
    "DEFAULT_MATERIAL_LIBRARY",
    "parse_poslist",
    "read_citygml_building",
    "read_citygml_dir",
]

logger = logging.getLogger(__name__)

# CityGML 2.0 namespaces we resolve by URI (prefixes vary, URIs do not).
CITYGML_NS = {
    "gml": "http://www.opengis.net/gml",
    "bldg": "http://www.opengis.net/citygml/building/2.0",
}

# Local thematic-surface tag -> our surface_type / boundary.
_SURFACE_MAP = {
    "WallSurface": ("Wall", "Outdoors"),
    "RoofSurface": ("Roof", "Outdoors"),
    "GroundSurface": ("Floor", "Ground"),
}


def _default_material_library() -> tuple[dict[str, MaterialLayer], dict[str, Construction]]:
    """A small opaque-construction library keyed by surface type.

    CityGML has no thermal data, so we attach plausible mid-European constructions
    — one per surface type — whose names match the ``<type>_Construction`` keys the
    surfaces reference. Layers are ordered outside -> inside, matching
    :class:`Construction`'s convention and the conduction solver's axis-0 direction.
    """
    # (name, thickness_m, conductivity_w_mk) outside -> inside, per surface type.
    stacks: dict[str, tuple[tuple[str, float, float], ...]] = {
        "Wall": (
            ("stucco", 0.025, 0.72),
            ("brick", 0.24, 0.79),
            ("eps_insulation", 0.10, 0.035),
            ("gypsum", 0.0127, 0.16),
        ),
        "Roof": (
            ("roof_tile", 0.02, 0.84),
            ("mineral_wool", 0.18, 0.035),
            ("osb", 0.018, 0.13),
            ("gypsum", 0.0127, 0.16),
        ),
        "Floor": (
            ("concrete_slab", 0.20, 1.95),
            ("xps_insulation", 0.08, 0.034),
            ("screed", 0.05, 1.40),
        ),
    }
    materials: dict[str, MaterialLayer] = {}
    constructions: dict[str, Construction] = {}
    for stype, stack in stacks.items():
        layers: list[MaterialLayer] = []
        for name, thick, cond in stack:
            mat = MaterialLayer(
                name=name,
                resistance_m2k_w=thick / cond,
                thickness_m=thick,
                conductivity_w_mk=cond,
            )
            materials[name] = mat
            layers.append(mat)
        cname = f"{stype}_Construction"
        constructions[cname] = Construction(cname, tuple(layers))
    return materials, constructions


# Built once: the default library used when the caller supplies none.
DEFAULT_MATERIAL_LIBRARY = _default_material_library()


def parse_poslist(text: str) -> np.ndarray:
    """Parse a ``gml:posList`` body into an ``(n, 3)`` vertex array.

    Drops the closing duplicate vertex (LinearRings repeat the first point) so the
    polygon is an open ring, matching what :func:`triangulate_polygon` expects.

    Args:
        text: whitespace-separated ``x y z x y z ...`` coordinate string.

    Returns:
        ``(n, 3)`` float array of vertices, closing duplicate removed.

    Raises:
        ValueError: if the coordinate count is not a positive multiple of 3.
    """
    vals = [float(v) for v in text.split()]
    if not vals or len(vals) % 3 != 0:
        raise ValueError(f"posList length {len(vals)} is not a positive multiple of 3")
    v = np.asarray(vals, dtype=float).reshape(-1, 3)
    if len(v) >= 2 and np.allclose(v[0], v[-1]):
        v = v[:-1]
    return v


def _local_tag(elem: ET.Element) -> str:
    """The namespace-stripped local tag of an element."""
    return elem.tag.split("}")[-1]


def _polygon_exterior_rings(boundary_child: ET.Element) -> list[np.ndarray]:
    """All exterior-ring vertex arrays under one thematic surface.

    LoD2 yields one ring; LoD3 yields many (tessellated triangles). Interior rings
    (window/door holes) are deliberately ignored — kept as opaque wall for v1.
    Degenerate rings (<3 vertices, or unparseable) are skipped.
    """
    rings: list[np.ndarray] = []
    for poly in boundary_child.iter(f"{{{CITYGML_NS['gml']}}}Polygon"):
        ext = poly.find(f"{{{CITYGML_NS['gml']}}}exterior")
        if ext is None:
            continue
        poslist = ext.find(f"{{{CITYGML_NS['gml']}}}LinearRing/{{{CITYGML_NS['gml']}}}posList")
        if poslist is None or poslist.text is None:
            continue
        try:
            v = parse_poslist(poslist.text)
        except ValueError:
            continue
        if len(v) >= 3:
            rings.append(v)
    return rings


def _building_id(building: ET.Element, fallback: str) -> str:
    """A stable building id from ``gml:id`` (or a fallback)."""
    gml_id = building.get(f"{{{CITYGML_NS['gml']}}}id")
    return gml_id or fallback


def _read_building_element(building: ET.Element, fallback_id: str) -> list[Surface]:
    """Lift every thematic surface of one ``bldg:Building`` into :class:`Surface` s.

    Vertices remain in source CRS coordinates here; the caller subtracts a shared
    per-building origin so all surfaces stay in one local frame.
    """
    surfaces: list[Surface] = []
    bid = _building_id(building, fallback_id)
    counter = 0
    for bb in building.iter(f"{{{CITYGML_NS['bldg']}}}boundedBy"):
        for child in list(bb):
            local = _local_tag(child)
            if local not in _SURFACE_MAP:
                continue
            surface_type, boundary = _SURFACE_MAP[local]
            rings = _polygon_exterior_rings(child)
            if not rings:
                continue
            base_id = child.get(f"{{{CITYGML_NS['gml']}}}id")
            for ring in rings:
                # Synthesise a name when gml:id is absent (common in LoD3).
                if base_id is not None:
                    name = base_id if len(rings) == 1 else f"{base_id}_{counter}"
                else:
                    name = f"{bid}_{surface_type}_{counter}"
                counter += 1
                surfaces.append(
                    Surface(
                        name=name,
                        surface_type=surface_type,
                        construction_name=f"{surface_type}_Construction",
                        zone=bid,
                        boundary=boundary,
                        vertices=ring,
                    )
                )
    return surfaces


def _localise(surfaces: list[Surface], origin: np.ndarray) -> list[Surface]:
    """Return surfaces with ``origin`` subtracted from every vertex."""
    out: list[Surface] = []
    for s in surfaces:
        out.append(
            Surface(
                name=s.name,
                surface_type=s.surface_type,
                construction_name=s.construction_name,
                zone=s.zone,
                boundary=s.boundary,
                vertices=s.vertices - origin,
            )
        )
    return out


def read_citygml_building(
    path: str | Path,
    material_library: tuple[dict[str, MaterialLayer], dict[str, Construction]] | None = None,
) -> Envelope:
    """Read one TUM2TWIN CityGML file into an :class:`Envelope`.

    Wall / Roof / Ground thematic surfaces become :class:`Surface` s; their
    exterior-ring posLists become ``(n, 3)`` local-metric vertex arrays (a shared
    per-building origin — the min corner — is subtracted). A default opaque
    construction library is attached per surface type unless ``material_library`` is
    supplied.

    Args:
        path: path to a ``*.gml`` CityGML 2.0 building file.
        material_library: optional ``(materials, constructions)`` pair overriding the
            default. Surfaces reference ``"<SurfaceType>_Construction"`` by name.

    Returns:
        An :class:`Envelope` with local-metric geometry and a resolvable construction
        library.

    Raises:
        ValueError: if the file cannot be parsed (e.g. truncated) or has no surfaces.
    """
    path = Path(path)
    try:
        tree = ET.parse(path)
    except ET.ParseError as exc:
        raise ValueError(f"could not parse CityGML file {path} (truncated?): {exc}") from exc
    root = tree.getroot()

    surfaces: list[Surface] = []
    for building in root.iter(f"{{{CITYGML_NS['bldg']}}}Building"):
        surfaces.extend(_read_building_element(building, fallback_id=path.stem))
    # Some files nest building parts; fall back to scanning the whole tree if needed.
    if not surfaces:
        surfaces = _read_building_element(root, fallback_id=path.stem)
    if not surfaces:
        raise ValueError(f"no Wall/Roof/Ground surfaces found in {path}")

    origin = np.min(np.concatenate([s.vertices for s in surfaces]), axis=0)
    surfaces = _localise(surfaces, origin)

    if material_library is None:
        materials, constructions = DEFAULT_MATERIAL_LIBRARY
    else:
        materials, constructions = material_library
    return Envelope(
        materials=dict(materials),
        constructions=dict(constructions),
        surfaces=surfaces,
    )


def read_citygml_dir(
    dirpath: str | Path,
    material_library: tuple[dict[str, MaterialLayer], dict[str, Construction]] | None = None,
    pattern: str = "*.gml",
) -> list[Envelope]:
    """Read every CityGML building in a directory, skipping unreadable files.

    Args:
        dirpath: directory holding ``*.gml`` files.
        material_library: optional library forwarded to
            :func:`read_citygml_building`.
        pattern: glob for the files to read.

    Returns:
        One :class:`Envelope` per readable file (truncated / empty files are logged
        and skipped), sorted by filename for determinism.
    """
    dirpath = Path(dirpath)
    envelopes: list[Envelope] = []
    for path in sorted(dirpath.glob(pattern)):
        try:
            envelopes.append(read_citygml_building(path, material_library=material_library))
        except ValueError as exc:
            logger.warning("skipping %s: %s", path.name, exc)
    return envelopes
