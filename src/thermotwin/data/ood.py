"""Out-of-distribution (OOD) test corpora for Block-1 generalisation.

The in-distribution corpus (:mod:`thermotwin.data.synthetic_fem`) trains and
validates the operator on two wall assemblies, ``r_si in {0.10, 0.13, 0.17}``,
``r_se = 0.04``, 0-3 thin bridges (0.02-0.08 m) and native resolutions
``cells_per_layer = 6`` / ``lateral_cells in {32, 48, 64}``. A geometry-conditioned
operator only earns its keep if it *generalises* past those draws, so we hold out
four orthogonal distribution shifts as test sets:

* ``ood_walls`` — four **new** base assemblies (brick cavity wall, heavy thick
  concrete + thin insulation, deep timber frame with service void, SIP/OSB-EPS-OSB)
  absent from training; in-distribution films / bridges / width / resolution. Tests
  generalisation to unseen materials and layer geometry.
* ``ood_films`` — in-distribution walls and bridges, but surface films drawn
  *outside* the training set: ``r_si in {0.04, 0.20, 0.25}`` and
  ``r_se in {0.10, 0.13}`` (training used ``r_si in {0.10, 0.13, 0.17}``,
  ``r_se = 0.04``). Tests unseen boundary conditions. Temperatures are *not* varied:
  θ is invariant to the absolute indoor/outdoor temperatures for linear steady
  conduction, so a temperature shift is not a distribution shift in θ.
* ``ood_bridges`` — in-distribution walls, but **harder** bridges: 4-6 of them,
  wider (0.08-0.15 m). Tests bridge-density generalisation — the regime where the
  delta-FNO correction on the 1-D prior is expected to pay off most.
* ``ood_res`` — in-distribution walls / bridges / films, but a **different** native
  resolution: ``cells_per_layer in {10, 11, 12}`` and ``lateral_cells in {96, 128}``.
  Tests cross-resolution / discretisation invariance.

Every generator reuses the in-distribution solve and save logic
(:func:`thermotwin.data.synthetic_fem.build_k_field`,
:func:`~thermotwin.data.synthetic_fem.clear_wall_u`, the steady FV solver), so the
``.npz`` schema and manifest match the existing corpora exactly — only the sampling
distribution changes. All generators are fully seeded.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from ..physics.conduction import Layer
from ..physics.steady_fv import DirichletFilm, solve_steady_conduction
from .synthetic_fem import (
    ThermalBridge,
    WallSample,
    build_k_field,
    clear_wall_u,
)

__all__ = [
    "OOD_WALLS",
    "OOD_FILM_R_SI",
    "OOD_FILM_R_SE",
    "OOD_GENERATORS",
    "random_sample_ood_walls",
    "random_sample_ood_films",
    "random_sample_ood_bridges",
    "random_sample_ood_res",
    "generate_ood_corpus",
]

# --- new base assemblies, held out of training (outside -> inside) -------------
# None of these layer stacks appear in synthetic_fem._BASE_WALLS, so the operator
# has never seen these material/thickness combinations.
OOD_WALLS: dict[str, tuple[Layer, ...]] = {
    # Brick cavity wall: outer brick, ventilated/insulated cavity, inner block, plaster.
    "brick_cavity": (
        Layer("brick_outer", 0.102, 0.77),
        Layer("cavity_insulation", 0.10, 0.040),
        Layer("aac_block", 0.140, 0.15),
        Layer("plaster", 0.013, 0.57),
    ),
    # Heavy thick concrete with only thin external insulation (thermally massive).
    "heavy_concrete_thin_insul": (
        Layer("render", 0.020, 0.87),
        Layer("xps_insulation", 0.050, 0.034),
        Layer("dense_concrete", 0.300, 2.30),
        Layer("gypsum", 0.0127, 0.16),
    ),
    # Deep timber frame with a service void on the warm side.
    "deep_timber_service_void": (
        Layer("fibre_cement", 0.009, 0.35),
        Layer("wood_fibre", 0.060, 0.040),
        Layer("glasswool", 0.200, 0.034),
        Layer("service_void_air", 0.025, 0.18),
        Layer("gypsum", 0.0127, 0.16),
    ),
    # Structural insulated panel: OSB / EPS core / OSB, plus internal lining.
    "sip_osb_eps": (
        Layer("osb_outer", 0.015, 0.13),
        Layer("eps_core", 0.150, 0.036),
        Layer("osb_inner", 0.015, 0.13),
        Layer("gypsum", 0.0127, 0.16),
    ),
}

# Bridge conductivities reused from the in-distribution material library.
_BRIDGE_K = {"steel_stud": 50.0, "concrete_nib": 1.95, "timber_stud": 0.13, "aluminium": 160.0}

# OOD surface films: drawn strictly outside the training support.
OOD_FILM_R_SI = (0.04, 0.20, 0.25)  # training used {0.10, 0.13, 0.17}
OOD_FILM_R_SE = (0.10, 0.13)  # training used 0.04


def _insulation_extent(layers: tuple[Layer, ...]) -> tuple[float, float]:
    """Through-wall ``[x_lo, x_hi)`` of the thickest (insulation) layer, in metres."""
    thicknesses = np.array([layer.thickness_m for layer in layers])
    edges = np.concatenate([[0.0], np.cumsum(thicknesses)])
    insul_idx = int(np.argmax(thicknesses))
    return float(edges[insul_idx]), float(edges[insul_idx + 1])


def _draw_bridges(
    rng: np.random.Generator,
    layers: tuple[Layer, ...],
    width: float,
    n_bridges: int,
    width_lo: float,
    width_hi: float,
) -> tuple[ThermalBridge, ...]:
    """Draw ``n_bridges`` bridges through the insulation layer (in-distribution style)."""
    x_lo, x_hi = _insulation_extent(layers)
    bridges = []
    for _ in range(n_bridges):
        bk = float(_BRIDGE_K[rng.choice(list(_BRIDGE_K))])
        bw = float(rng.uniform(width_lo, width_hi))
        y0 = float(rng.uniform(0.0, max(width - bw, 0.0)))
        bridges.append(ThermalBridge(x_lo, x_hi, y0, y0 + bw, bk))
    return tuple(bridges)


def random_sample_ood_walls(rng: np.random.Generator) -> WallSample:
    """OOD-walls: unseen assemblies, in-distribution films / bridges / width / res."""
    name = rng.choice(list(OOD_WALLS))
    layers = OOD_WALLS[name]
    width = float(rng.uniform(0.4, 1.2))
    t_in = float(rng.uniform(18.0, 22.0))
    t_out = float(rng.uniform(-12.0, 8.0))
    r_si = float(rng.choice([0.10, 0.13, 0.17]))  # in-distribution film
    n_bridges = int(rng.integers(0, 4))  # 0-3, in-distribution
    bridges = _draw_bridges(rng, layers, width, n_bridges, 0.02, 0.08)
    return WallSample(
        layers=layers,
        width_m=width,
        t_indoor=t_in,
        t_outdoor=t_out,
        r_si=r_si,
        r_se=0.04,
        bridges=bridges,
        lateral_cells=int(rng.choice([32, 48, 64])),
    )


def random_sample_ood_films(rng: np.random.Generator) -> WallSample:
    """OOD-films: in-distribution walls/bridges, surface films outside training support."""
    from .synthetic_fem import _BASE_WALLS  # in-distribution assemblies

    name = rng.choice(list(_BASE_WALLS))
    layers = _BASE_WALLS[name]
    width = float(rng.uniform(0.4, 1.2))
    t_in = float(rng.uniform(18.0, 22.0))
    t_out = float(rng.uniform(-12.0, 8.0))
    r_si = float(rng.choice(OOD_FILM_R_SI))  # OOD indoor film
    r_se = float(rng.choice(OOD_FILM_R_SE))  # OOD outdoor film
    n_bridges = int(rng.integers(0, 4))  # in-distribution bridge count
    bridges = _draw_bridges(rng, layers, width, n_bridges, 0.02, 0.08)
    return WallSample(
        layers=layers,
        width_m=width,
        t_indoor=t_in,
        t_outdoor=t_out,
        r_si=r_si,
        r_se=r_se,
        bridges=bridges,
        lateral_cells=int(rng.choice([32, 48, 64])),
    )


def random_sample_ood_bridges(rng: np.random.Generator) -> WallSample:
    """OOD-bridges: in-distribution walls/films, but 4-6 wide (0.08-0.15 m) bridges."""
    from .synthetic_fem import _BASE_WALLS

    name = rng.choice(list(_BASE_WALLS))
    layers = _BASE_WALLS[name]
    width = float(rng.uniform(0.4, 1.2))
    t_in = float(rng.uniform(18.0, 22.0))
    t_out = float(rng.uniform(-12.0, 8.0))
    r_si = float(rng.choice([0.10, 0.13, 0.17]))
    n_bridges = int(rng.integers(4, 7))  # 4-6, harder than training's 0-3
    bridges = _draw_bridges(rng, layers, width, n_bridges, 0.08, 0.15)  # wider
    return WallSample(
        layers=layers,
        width_m=width,
        t_indoor=t_in,
        t_outdoor=t_out,
        r_si=r_si,
        r_se=0.04,
        bridges=bridges,
        lateral_cells=int(rng.choice([32, 48, 64])),
    )


def random_sample_ood_res(rng: np.random.Generator) -> WallSample:
    """OOD-res: in-distribution walls/bridges/films, but a finer native discretisation."""
    from .synthetic_fem import _BASE_WALLS

    name = rng.choice(list(_BASE_WALLS))
    layers = _BASE_WALLS[name]
    width = float(rng.uniform(0.4, 1.2))
    t_in = float(rng.uniform(18.0, 22.0))
    t_out = float(rng.uniform(-12.0, 8.0))
    r_si = float(rng.choice([0.10, 0.13, 0.17]))
    n_bridges = int(rng.integers(0, 4))
    bridges = _draw_bridges(rng, layers, width, n_bridges, 0.02, 0.08)
    return WallSample(
        layers=layers,
        width_m=width,
        t_indoor=t_in,
        t_outdoor=t_out,
        r_si=r_si,
        r_se=0.04,
        bridges=bridges,
        cells_per_layer=int(rng.integers(10, 13)),  # 10-12 vs training's 6
        lateral_cells=int(rng.choice([96, 128])),  # vs training's {32, 48, 64}
    )


# Registry of OOD corpora: name -> per-sample draw function.
OOD_GENERATORS: dict[str, Callable[[np.random.Generator], WallSample]] = {
    "ood_walls": random_sample_ood_walls,
    "ood_films": random_sample_ood_films,
    "ood_bridges": random_sample_ood_bridges,
    "ood_res": random_sample_ood_res,
}


def generate_ood_corpus(name: str, n: int, seed: int) -> list[dict]:
    """Generate ``n`` solved OOD samples for corpus ``name`` as plain-dict records.

    Mirrors :func:`thermotwin.data.synthetic_fem.generate_corpus` record-for-record
    (same keys / dtypes / save schema), differing only in the sampling distribution.

    Args:
        name: one of :data:`OOD_GENERATORS` (``ood_walls`` / ``ood_films`` /
            ``ood_bridges`` / ``ood_res``).
        n: number of samples to draw.
        seed: RNG seed (each corpus should use a distinct seed for independence).

    Returns:
        A list of ``n`` record dicts ready for ``np.savez_compressed`` + manifest.
    """
    if name not in OOD_GENERATORS:
        raise ValueError(f"unknown OOD corpus '{name}'; expected one of {tuple(OOD_GENERATORS)}")
    draw = OOD_GENERATORS[name]
    rng = np.random.default_rng(seed)
    records: list[dict] = []
    for i in range(n):
        sample = draw(rng)
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
