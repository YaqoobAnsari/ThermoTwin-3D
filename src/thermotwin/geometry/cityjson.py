"""CityJSON reader (3D BAG) — lift real LoD2.2 building shells into our ``Envelope``.

The TUM2TWIN reader (:mod:`thermotwin.geometry.citygml`) handles CityGML 2.0 (GML/XML); the
Netherlands **3D BAG** ships **CityJSON 2.0** — a compact JSON with quantised integer vertices
(dequantised by ``transform``) and the LoD2.2 geometry on **BuildingPart** objects as a
``Solid`` whose surfaces carry ``semantics`` (Ground / Roof / Wall). This module turns each
BuildingPart's LoD2.2 shell into an :class:`~thermotwin.geometry.envelope.Envelope` — surfaces
tagged Wall/Roof/Floor, default per-type constructions — so it flows into the *same*
point-cloud + per-surface-FV corpus pipeline as the TUM2TWIN envelopes
(:mod:`thermotwin.data.real_citygml_3d`), giving real as-built geometry at scale (~thousands of
buildings per city) for the unified evaluation.

CityJSON carries **geometry only** (like CityGML) — materials are the default-library
placeholders keyed by surface type, exactly as for TUM2TWIN (real material assignment / vintage
archetypes are future work).
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np

from .citygml import _SURFACE_MAP, _default_material_library, _localise
from .envelope import Envelope, Surface

__all__ = ["read_cityjson", "read_cityjson_dir"]


def _open(path: str | Path):
    path = str(path)
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path)


def _pick_lod(geometries: list, prefer: str = "2.2"):
    """The preferred-LoD geometry (``2.2``), else the highest LoD available, else None."""
    if not geometries:
        return None
    for g in geometries:
        if str(g.get("lod")) == prefer:
            return g
    return max(geometries, key=lambda g: str(g.get("lod")))


def _surfaces_from_geometry(geom: dict, verts: np.ndarray, oid: str) -> list[Surface]:
    """Extract semantic Wall/Roof/Floor surfaces from a Solid / MultiSurface geometry."""
    gtype = geom.get("type")
    boundaries = geom.get("boundaries", [])
    sem = geom.get("semantics", {})
    surf_types = sem.get("surfaces", [])
    values = sem.get("values")
    if gtype == "Solid":  # boundaries = [outer_shell, inner_shells...]; semantics.values per shell
        shell = boundaries[0] if boundaries else []
        vals = values[0] if values else [None] * len(shell)
    elif gtype == "MultiSurface":
        shell = boundaries
        vals = values if values else [None] * len(shell)
    else:
        return []

    out: list[Surface] = []
    for i, surface in enumerate(shell):
        if not surface or not surface[0]:
            continue
        ring = np.asarray(surface[0], dtype=np.int64)  # exterior ring of this face
        v = verts[ring]
        if len(v) < 3:
            continue
        stype = None
        vi = vals[i] if i < len(vals) else None
        if vi is not None and vi < len(surf_types):
            stype = surf_types[vi].get("type")
        mapped = _SURFACE_MAP.get(stype)  # WallSurface/RoofSurface/GroundSurface -> (type, bc)
        if mapped is None:  # ClosureSurface / unmapped -> skip
            continue
        surface_type, boundary = mapped
        out.append(
            Surface(
                name=f"{oid}_{surface_type}_{i}",
                surface_type=surface_type,
                construction_name=f"{surface_type}_Construction",
                zone="bag",
                boundary=boundary,
                vertices=v,
            )
        )
    return out


def read_cityjson(path: str | Path) -> list[Envelope]:
    """Read a CityJSON (3D BAG) tile into one :class:`Envelope` per building shell.

    Geometry lives on ``BuildingPart`` objects (3D BAG splits each Pand into parts); each part
    with a usable LoD2.2 (or highest-LoD) Solid and ≥ 3 Wall/Roof/Floor surfaces becomes an
    envelope, with local-metric coordinates (min-corner subtracted) and default per-type
    constructions.
    """
    with _open(path) as fh:
        doc = json.load(fh)
    tr = doc["transform"]
    scale = np.asarray(tr["scale"], dtype=np.float64)
    translate = np.asarray(tr["translate"], dtype=np.float64)
    verts = np.asarray(doc["vertices"], dtype=np.float64) * scale + translate  # (V, 3) metres

    materials, constructions = _default_material_library()
    envelopes: list[Envelope] = []
    for oid, obj in doc.get("CityObjects", {}).items():
        if obj.get("type") not in ("BuildingPart", "Building"):
            continue
        geom = _pick_lod(obj.get("geometry", []))
        if geom is None:
            continue
        surfaces = _surfaces_from_geometry(geom, verts, oid)
        if len(surfaces) < 3:
            continue
        origin = np.concatenate([s.vertices for s in surfaces]).min(axis=0)
        surfaces = _localise(surfaces, origin)
        envelopes.append(Envelope(dict(materials), dict(constructions), surfaces))
    return envelopes


def read_cityjson_dir(directory: str | Path, pattern: str = "*.city.json*") -> list[Envelope]:
    """All building envelopes across every CityJSON tile in ``directory`` (skips unreadable)."""
    out: list[Envelope] = []
    for f in sorted(Path(directory).glob(pattern)):
        try:
            out.extend(read_cityjson(f))
        except Exception as exc:  # a malformed tile shouldn't sink the corpus
            print(f"  (skip {f.name}: {exc})")
    return out
