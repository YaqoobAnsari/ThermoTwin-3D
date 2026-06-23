"""Synthetic FEM ground-truth corpus: layered walls with thermal bridges.

Block-1 needs a controlled corpus where the temperature field is genuinely
geometry-dependent — otherwise a 1-D U-value would suffice and a geometry-conditioned
operator earns nothing (novelty hook **H1**). We get that by taking a real layered
construction (from the DOE envelope featuriser) and puncturing its insulation with
**thermal bridges**: high-conductivity inclusions (steel studs, concrete nibs,
window reveals) that force lateral heat spreading. The 2-D conduction field then
departs from the clear-wall 1-D answer, and recovering it is a real learning task.

Each sample is a 2-D cross-section: axis 0 is through-wall (layered, non-uniform
spacing), axis 1 is along-wall (uniform). We solve it with
:func:`thermotwin.physics.steady_fv.solve_steady_conduction` and store the
conductivity field, the boundary conditions, and the resulting temperature /
heat-flux field. The generator is fully seeded; every sample carries the parameters
needed to reproduce it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..physics.conduction import Layer
from ..physics.steady_fv import (
    ConductionField,
    DirichletFilm,
    layered_k_field,
    solve_steady_conduction,
)

__all__ = [
    "ThermalBridge",
    "WallSample",
    "build_k_field",
    "solve_sample",
    "clear_wall_u",
    "random_sample",
    "generate_corpus",
]


@dataclass(frozen=True)
class ThermalBridge:
    """A rectangular high-conductivity inclusion in the wall cross-section.

    Extents are in metres: ``x`` is through-wall (from the outside face), ``y`` is
    along-wall. ``conductivity`` overrides the base material wherever the bridge
    sits.
    """

    x0: float
    x1: float
    y0: float
    y1: float
    conductivity_w_mk: float


@dataclass(frozen=True)
class WallSample:
    """Full specification of one synthetic wall cross-section."""

    layers: tuple[Layer, ...]  # outside -> inside
    width_m: float  # along-wall extent
    t_indoor: float
    t_outdoor: float
    r_si: float = 0.13
    r_se: float = 0.04
    bridges: tuple[ThermalBridge, ...] = field(default_factory=tuple)
    cells_per_layer: int = 6
    lateral_cells: int = 48

    @property
    def thickness_m(self) -> float:
        return sum(layer.thickness_m for layer in self.layers)


def _cell_centres(spacing_1d: np.ndarray) -> np.ndarray:
    """Cell-centre coordinates from a 1-D per-cell spacing array."""
    edges = np.concatenate([[0.0], np.cumsum(spacing_1d)])
    return 0.5 * (edges[:-1] + edges[1:])


def build_k_field(sample: WallSample) -> tuple[np.ndarray, list]:
    """Conductivity field ``k`` (Nx, Ny) and grid spacing for a sample.

    The base field is the layered construction broadcast across the wall width;
    thermal bridges then overwrite their rectangular footprints.
    """
    transverse = sample.width_m / sample.lateral_cells
    k, spacing = layered_k_field(
        list(sample.layers),
        cells_per_layer=sample.cells_per_layer,
        cross_section=(sample.lateral_cells,),
        transverse_spacing=transverse,
    )
    if sample.bridges:
        xc = _cell_centres(spacing[0])  # through-wall centres
        yc = (np.arange(sample.lateral_cells) + 0.5) * transverse
        for b in sample.bridges:
            xm = (xc >= b.x0) & (xc < b.x1)
            ym = (yc >= b.y0) & (yc < b.y1)
            k[np.ix_(xm, ym)] = b.conductivity_w_mk
    return k, spacing


def solve_sample(sample: WallSample) -> ConductionField:
    """Solve a sample to its steady temperature / heat-flux field."""
    k, spacing = build_k_field(sample)
    bc = DirichletFilm(sample.t_indoor, sample.t_outdoor, r_lo=sample.r_si, r_hi=sample.r_se)
    return solve_steady_conduction(k, spacing, bc)


def clear_wall_u(sample: WallSample) -> float:
    """The 1-D clear-wall U-value (no bridges) — the baseline the operator must beat."""
    no_bridges = WallSample(
        layers=sample.layers,
        width_m=sample.width_m,
        t_indoor=sample.t_indoor,
        t_outdoor=sample.t_outdoor,
        r_si=sample.r_si,
        r_se=sample.r_se,
        bridges=(),
        cells_per_layer=sample.cells_per_layer,
        lateral_cells=1,
    )
    return solve_sample(no_bridges).u_value


# --- a small library of plausible base constructions and bridge materials ----

_BASE_WALLS: dict[str, tuple[Layer, ...]] = {
    "mass_insulated": (
        Layer("stucco", 0.025, 0.72),
        Layer("concrete", 0.20, 1.95),
        Layer("eps_insulation", 0.10, 0.035),
        Layer("gypsum", 0.0127, 0.16),
    ),
    "light_framed": (
        Layer("cladding", 0.012, 0.20),
        Layer("mineral_wool", 0.14, 0.035),
        Layer("osb", 0.018, 0.13),
        Layer("gypsum", 0.0127, 0.16),
    ),
}
_BRIDGE_K = {"steel_stud": 50.0, "concrete_nib": 1.95, "timber_stud": 0.13, "aluminium": 160.0}


def random_sample(rng: np.random.Generator) -> WallSample:
    """Draw one randomized wall sample (base wall, BCs, 0-3 thermal bridges)."""
    name = rng.choice(list(_BASE_WALLS))
    layers = _BASE_WALLS[name]
    width = float(rng.uniform(0.4, 1.2))
    t_in = float(rng.uniform(18.0, 22.0))
    t_out = float(rng.uniform(-12.0, 8.0))
    r_si = float(rng.choice([0.10, 0.13, 0.17]))

    # Bridges puncture the insulation layer (the 2nd or 3rd layer for these walls).
    thicknesses = np.array([layer.thickness_m for layer in layers])
    edges = np.concatenate([[0.0], np.cumsum(thicknesses)])
    insul_idx = int(np.argmax(thicknesses))  # the thickest layer ~ the insulation
    x_lo, x_hi = edges[insul_idx], edges[insul_idx + 1]

    n_bridges = int(rng.integers(0, 4))
    bridges = []
    for _ in range(n_bridges):
        bk = float(_BRIDGE_K[rng.choice(list(_BRIDGE_K))])
        bw = float(rng.uniform(0.02, 0.08))  # bridge width along wall
        y0 = float(rng.uniform(0.0, max(width - bw, 0.0)))
        bridges.append(ThermalBridge(x_lo, x_hi, y0, y0 + bw, bk))

    return WallSample(
        layers=layers,
        width_m=width,
        t_indoor=t_in,
        t_outdoor=t_out,
        r_si=r_si,
        bridges=tuple(bridges),
        lateral_cells=int(rng.choice([32, 48, 64])),
    )


def generate_corpus(n: int, seed: int = 1337) -> list[dict]:
    """Generate ``n`` solved samples as plain-dict records (arrays + metadata).

    Each record holds the conductivity field, temperature field, per-axis spacing,
    boundary conditions, the effective and clear-wall U-values, and the parameters
    needed to reproduce the sample.
    """
    rng = np.random.default_rng(seed)
    records: list[dict] = []
    for i in range(n):
        sample = random_sample(rng)
        k, spacing = build_k_field(sample)
        bc = DirichletFilm(sample.t_indoor, sample.t_outdoor, r_lo=sample.r_si, r_hi=sample.r_se)
        res = solve_steady_conduction(k, spacing, bc)
        records.append(
            {
                "id": i,
                "k": k.astype(np.float32),
                "temperature": res.temperature.astype(np.float32),
                "dx0": np.asarray(spacing[0], dtype=np.float32),
                "dy": np.float32(spacing[1]),
                "t_indoor": np.float32(sample.t_indoor),
                "t_outdoor": np.float32(sample.t_outdoor),
                "r_si": np.float32(sample.r_si),
                "r_se": np.float32(sample.r_se),
                "u_value": np.float32(res.u_value),
                "u_clear": np.float32(clear_wall_u(sample)),
                "heat_flux": np.float32(res.heat_flux),
                "n_bridges": np.int32(len(sample.bridges)),
            }
        )
    return records
