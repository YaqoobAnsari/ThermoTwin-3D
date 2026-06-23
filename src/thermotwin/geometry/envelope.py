"""Building envelope lifted from an EnergyPlus IDF: surfaces + material layers.

This is Stage-1 of the pipeline (``docs/architecture.md``) for the DOE reference
corpus: turn an IDF into a geometry-tagged, material-resolved envelope that the
physics solver and, later, the neural operator consume. Each exterior surface
carries its polygon geometry **and** its ordered construction layers, so we can:

* compute a per-surface U-value (EN ISO 6946 films by orientation), and
* hand a construction to :mod:`thermotwin.physics.steady_fv` as a conductivity
  field for a geometry-resolved solve.

The bridge to physics is exact: a construction's analytic U-value here equals the
1-D oracle's U-value over the same layers, which in turn equals the finite-volume
solver's effective U (the chain ``conduction`` ⇆ ``steady_fv`` is already gated by
tests). This module adds the IDF → layers link and is tested the same way.

Units are SI throughout (m, W/(m·K), m²K/W, W/(m²K)), matching
:mod:`thermotwin.physics.conduction`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..physics.conduction import Layer, SurfaceFilm, u_value
from . import idf

__all__ = [
    "MaterialLayer",
    "Construction",
    "Surface",
    "Envelope",
    "surface_films",
]

# Known IDF outside-boundary-condition tokens we care about.
_EXTERIOR_BC = {"outdoors"}
# Surfaces forming the closed outer shell: the outside world plus ground contact.
_SHELL_BC = {"outdoors", "ground", "groundfcfactormethod", "groundslabpreprocessoraverage"}
_KNOWN_BC = {
    "outdoors",
    "ground",
    "surface",
    "adiabatic",
    "zone",
    "groundfcfactormethod",
    "othersidecoefficients",
    "othersideconditionsmodel",
}

# EN ISO 6946 surface (film) resistances [m²K/W] by heat-flow direction.
# Internal R_si depends on surface orientation; external R_se ≈ 0.04 for all.
_FILMS = {
    "wall": SurfaceFilm(r_si=0.13, r_se=0.04),  # horizontal flow
    "roof": SurfaceFilm(r_si=0.10, r_se=0.04),  # upward flow
    "ceiling": SurfaceFilm(r_si=0.10, r_se=0.04),
    "floor": SurfaceFilm(r_si=0.17, r_se=0.04),  # downward flow
}


def surface_films(surface_type: str) -> SurfaceFilm:
    """EN ISO 6946 film resistances for a surface type (defaults to wall)."""
    return _FILMS.get(surface_type.strip().lower(), _FILMS["wall"])


def _norm(name: str) -> str:
    """Canonical key for an IDF object name (EnergyPlus names are case-insensitive)."""
    return name.strip().casefold()


@dataclass(frozen=True)
class MaterialLayer:
    """One material layer, resolved from an IDF ``Material`` / ``Material:NoMass``.

    Mass materials carry ``thickness_m`` and ``conductivity_w_mk``; no-mass
    materials carry only a thermal ``resistance_m2k_w`` (no geometric thickness).
    ``density`` / ``specific_heat`` are retained for future transient work.
    """

    name: str
    resistance_m2k_w: float
    thickness_m: float | None = None
    conductivity_w_mk: float | None = None
    density_kg_m3: float | None = None
    specific_heat_j_kgk: float | None = None

    @property
    def is_massless(self) -> bool:
        return self.thickness_m is None

    @classmethod
    def from_mass(cls, fields: idf.IdfObject) -> MaterialLayer:
        # Material: Name, Roughness, Thickness, Conductivity, Density, SpecificHeat, ...
        name, _rough, thick, cond = fields[0], fields[1], float(fields[2]), float(fields[3])
        density = float(fields[4]) if len(fields) > 4 and fields[4] else None
        cp = float(fields[5]) if len(fields) > 5 and fields[5] else None
        return cls(
            name=name,
            resistance_m2k_w=thick / cond,
            thickness_m=thick,
            conductivity_w_mk=cond,
            density_kg_m3=density,
            specific_heat_j_kgk=cp,
        )

    @classmethod
    def from_nomass(cls, fields: idf.IdfObject) -> MaterialLayer:
        # Material:NoMass: Name, Roughness, ThermalResistance, ...
        return cls(name=fields[0], resistance_m2k_w=float(fields[2]))

    def to_conduction_layer(self, nominal_massless_thickness: float = 0.01) -> Layer:
        """Solver-ready layer; a massless layer gets a nominal thickness so it
        occupies grid space, with conductivity chosen to preserve its resistance.
        Only the resistance matters for steady conduction, so the choice is benign.
        """
        if self.is_massless:
            t = nominal_massless_thickness
            return Layer(self.name, t, t / self.resistance_m2k_w)
        return Layer(self.name, self.thickness_m, self.conductivity_w_mk)


@dataclass(frozen=True)
class Construction:
    """An ordered layer stack (IDF order: outside → inside)."""

    name: str
    layers: tuple[MaterialLayer, ...]

    @property
    def resistance(self) -> float:
        """Sum of layer resistances [m²K/W] (excludes surface films)."""
        return sum(layer.resistance_m2k_w for layer in self.layers)

    def u_value(self, surface_type: str = "wall") -> float:
        """Transmittance [W/(m²K)] including EN ISO 6946 films for the type."""
        film = surface_films(surface_type)
        return 1.0 / (film.r_si + self.resistance + film.r_se)

    def to_conduction_layers(self, nominal_massless_thickness: float = 0.01) -> list[Layer]:
        """Solver-ready layers, ordered through-wall (outside → inside)."""
        return [m.to_conduction_layer(nominal_massless_thickness) for m in self.layers]


def _polygon_area_normal(vertices: np.ndarray) -> tuple[float, np.ndarray]:
    """Area [m²] and unit outward normal of a planar polygon (Newell's method)."""
    n = np.zeros(3)
    m = len(vertices)
    for i in range(m):
        a, b = vertices[i], vertices[(i + 1) % m]
        n[0] += (a[1] - b[1]) * (a[2] + b[2])
        n[1] += (a[2] - b[2]) * (a[0] + b[0])
        n[2] += (a[0] - b[0]) * (a[1] + b[1])
    area = 0.5 * np.linalg.norm(n)
    unit = n / np.linalg.norm(n) if area > 0 else n
    return float(area), unit


@dataclass(frozen=True)
class Surface:
    """One building surface: geometry + which construction clads it."""

    name: str
    surface_type: str  # Wall / Roof / Ceiling / Floor
    construction_name: str
    zone: str
    boundary: str  # Outdoors / Ground / Surface / Adiabatic / ...
    vertices: np.ndarray  # (n, 3) in metres

    @property
    def is_exterior(self) -> bool:
        return self.boundary.strip().lower() in _EXTERIOR_BC

    @property
    def area(self) -> float:
        return _polygon_area_normal(self.vertices)[0]

    @property
    def normal(self) -> np.ndarray:
        return _polygon_area_normal(self.vertices)[1]

    @property
    def centroid(self) -> np.ndarray:
        return self.vertices.mean(axis=0)


def _parse_surface(fields: idf.IdfObject) -> Surface | None:
    """Parse a BuildingSurface:Detailed, robust to IDF version field shifts.

    Layout: Name, Type, Construction, Zone, [Space], OutsideBC, BCObject,
    SunExp, WindExp, ViewFactor, NumVertices, then NumVertices XYZ triples.
    The number-of-vertices field is found by anchoring on the trailing triples,
    which sidesteps the optional ``Space Name`` field that varies across versions.
    """
    # Locate NumVertices: the integer field n with exactly 3n fields after it.
    nverts_idx = None
    for i, f in enumerate(fields):
        if f.isdigit() and int(f) >= 3 and len(fields) - (i + 1) == 3 * int(f):
            nverts_idx = i
            break
    if nverts_idx is None:
        return None
    nverts = int(fields[nverts_idx])
    coords = fields[nverts_idx + 1 : nverts_idx + 1 + 3 * nverts]
    vertices = np.array([float(c) for c in coords], dtype=float).reshape(nverts, 3)

    name, stype, constr, zone = fields[0], fields[1], fields[2], fields[3]
    # Boundary condition: first known-BC token in the preamble after the zone.
    boundary = "unknown"
    for f in fields[4:nverts_idx]:
        if f.strip().lower() in _KNOWN_BC:
            boundary = f.strip()
            break
    return Surface(name, stype, constr, zone, boundary, vertices)


@dataclass
class Envelope:
    """An as-built-style envelope: surfaces + the material/construction library."""

    materials: dict[str, MaterialLayer]
    constructions: dict[str, Construction]
    surfaces: list[Surface]

    def __post_init__(self) -> None:
        # Case-insensitive lookup indices (EnergyPlus names are case-insensitive).
        self._constr_by_norm = {_norm(k): v for k, v in self.constructions.items()}

    @classmethod
    def from_idf(cls, path) -> Envelope:
        objs = idf.parse_idf(path)
        return cls.from_objects(objs)

    @classmethod
    def from_objects(cls, objs: dict[str, list[idf.IdfObject]]) -> Envelope:
        materials: dict[str, MaterialLayer] = {}
        for fields in idf.get(objs, "Material"):
            m = MaterialLayer.from_mass(fields)
            materials[m.name] = m
        for fields in idf.get(objs, "Material:NoMass"):
            m = MaterialLayer.from_nomass(fields)
            materials[m.name] = m
        mat_by_norm = {_norm(k): v for k, v in materials.items()}

        constructions: dict[str, Construction] = {}
        for fields in idf.get(objs, "Construction"):
            cname, layer_names = fields[0], [f for f in fields[1:] if f]
            layers = tuple(mat_by_norm[_norm(n)] for n in layer_names if _norm(n) in mat_by_norm)
            # Skip constructions referencing materials we don't model (e.g. glazing).
            if len(layers) == len(layer_names):
                constructions[cname] = Construction(cname, layers)

        surfaces: list[Surface] = []
        for fields in idf.get(objs, "BuildingSurface:Detailed"):
            s = _parse_surface(fields)
            if s is not None:
                surfaces.append(s)

        return cls(materials, constructions, surfaces)

    # --- queries ---------------------------------------------------------

    def exterior_opaque_surfaces(self) -> list[Surface]:
        """Surfaces facing outdoors whose construction we fully resolved."""
        return [
            s
            for s in self.surfaces
            if s.is_exterior and _norm(s.construction_name) in self._constr_by_norm
        ]

    def shell_surfaces(self) -> list[Surface]:
        """Surfaces forming the building's closed outer boundary.

        The watertight shell is everything facing the outside world *or* the ground
        (walls, roof, ground-contact floor) — i.e. boundary in {Outdoors, Ground} —
        with a resolvable construction. Unlike :meth:`exterior_opaque_surfaces`
        (Outdoors only), including ground-contact surfaces closes the bottom of the
        volume, which the SDF needs for a reliable inside/outside sign.
        """
        return [
            s
            for s in self.surfaces
            if s.boundary.strip().lower() in _SHELL_BC
            and _norm(s.construction_name) in self._constr_by_norm
        ]

    def surface_u_value(self, surface: Surface) -> float:
        """U-value [W/(m²K)] of a surface using its construction + orientation films."""
        constr = self._constr_by_norm[_norm(surface.construction_name)]
        return constr.u_value(surface.surface_type)

    def envelope_ua(self) -> float:
        """Whole-envelope transmission UA [W/K] over exterior opaque surfaces."""
        return sum(self.surface_u_value(s) * s.area for s in self.exterior_opaque_surfaces())


def _check_against_oracle(construction: Construction, surface_type: str = "wall") -> bool:
    """Sanity: construction U equals the oracle U over its conduction layers."""
    layers = construction.to_conduction_layers()
    film = surface_films(surface_type)
    return np.isclose(construction.u_value(surface_type), u_value(layers, film), rtol=1e-9)
