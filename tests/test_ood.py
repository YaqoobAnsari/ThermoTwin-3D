"""Out-of-distribution corpora: validity and provable distribution shift.

Each OOD generator must (a) produce physically valid samples — right shapes,
positive conductivity, plausible U-values, and an effective U that departs from
the clear-wall 1-D answer only through genuine 2-D structure — and (b) shift *at
least one* property provably outside the in-distribution training support (the
whole point of the corpus).

Note on the U / clear-wall relation: bridges puncture the *thickest* layer (this
is inherited verbatim from ``synthetic_fem.random_sample``). For ``light_framed``
that layer is the insulation, so a steel/concrete bridge raises effective U above
clear-wall. For ``mass_insulated`` the thickest layer is structural *concrete*, so
a timber/insulation-grade inclusion is actually *less* conductive than what it
replaces and lowers effective U below clear-wall by up to a couple of percent. So
"U >= clear-wall" is **not** a universal invariant here; we only assert it on the
``light_framed`` family where it provably holds.
"""

from __future__ import annotations

import numpy as np
import pytest

from thermotwin.data.ood import (
    OOD_FILM_R_SE,
    OOD_FILM_R_SI,
    OOD_GENERATORS,
    OOD_WALLS,
    generate_ood_corpus,
)
from thermotwin.data.synthetic_fem import _BASE_WALLS, build_k_field, clear_wall_u, solve_sample

# In-distribution training support (from synthetic_fem.random_sample).
_TRAIN_R_SI = {0.10, 0.13, 0.17}
_TRAIN_R_SE = {0.04}
_TRAIN_LATERAL = {32, 48, 64}
_TRAIN_CELLS_PER_LAYER = 6
_TRAIN_MAX_BRIDGES = 3
_TRAIN_BRIDGE_W = (0.02, 0.08)
_TRAIN_WALL_KEYS = set(_BASE_WALLS)


def test_registry_lists_four_corpora():
    assert set(OOD_GENERATORS) == {"ood_walls", "ood_films", "ood_bridges", "ood_res"}


@pytest.mark.parametrize("name", sorted(OOD_GENERATORS))
def test_corpus_valid_and_deterministic(name: str):
    """Each corpus is well-formed, physically valid, and reproducible from its seed."""
    a = generate_ood_corpus(name, 6, seed=99)
    b = generate_ood_corpus(name, 6, seed=99)
    assert len(a) == 6
    for ra, rb in zip(a, b, strict=True):
        # Reproducible.
        assert np.array_equal(ra["k"], rb["k"])
        assert np.array_equal(ra["temperature"], rb["temperature"])
        # Shapes consistent; through-wall axis fixed by cells_per_layer * n_layers.
        assert ra["k"].ndim == 2
        assert ra["temperature"].shape == ra["k"].shape
        assert ra["dx0"].shape[0] == ra["k"].shape[0]
        # Conductivity strictly positive everywhere.
        assert np.all(ra["k"] > 0.0)
        # Plausible building U-value [W/m^2K]: well-insulated walls to bridged walls.
        assert 0.05 < float(ra["u_value"]) < 6.0
        # Effective U stays in a physical band around the clear-wall value: a bridge
        # perturbs it by lateral spreading, never by more than the clear-wall scale.
        u, uc = float(ra["u_value"]), float(ra["u_clear"])
        assert uc - 0.3 < u  # an inclusion can lower U only mildly (timber-in-concrete)
        assert ra["k"].dtype == np.float32


def test_ood_walls_uses_unseen_assemblies():
    """ood_walls draws only assemblies absent from the training library."""
    assert _TRAIN_WALL_KEYS.isdisjoint(set(OOD_WALLS))
    # The OOD k-fields contain conductivity values not present in any training wall.
    train_k = {
        round(layer.conductivity_w_mk, 6) for layers in _BASE_WALLS.values() for layer in layers
    }
    ood_layer_k = {
        round(layer.conductivity_w_mk, 6) for layers in OOD_WALLS.values() for layer in layers
    }
    assert ood_layer_k - train_k, "ood_walls must introduce unseen layer conductivities"


def test_ood_films_outside_training_support():
    """Every ood_films sample has r_si or r_se strictly outside the training set."""
    assert set(OOD_FILM_R_SI).isdisjoint(_TRAIN_R_SI)
    assert set(OOD_FILM_R_SE).isdisjoint(_TRAIN_R_SE)
    recs = generate_ood_corpus("ood_films", 32, seed=7)
    for r in recs:
        r_si, r_se = float(r["r_si"]), float(r["r_se"])
        assert (r_si not in _TRAIN_R_SI) or (r_se not in _TRAIN_R_SE)
    # At least one sample provably outside on each axis across the corpus.
    assert any(float(r["r_si"]) not in _TRAIN_R_SI for r in recs)
    assert any(float(r["r_se"]) not in _TRAIN_R_SE for r in recs)


def test_ood_bridges_denser_than_training():
    """ood_bridges exceeds the training bridge count, and at least one is wider."""
    recs = generate_ood_corpus("ood_bridges", 32, seed=7)
    n_bridges = [int(r["n_bridges"]) for r in recs]
    assert min(n_bridges) >= _TRAIN_MAX_BRIDGES + 1  # every sample has 4+ bridges
    assert max(n_bridges) <= 6
    # Reconstruct one sample's bridge widths from its k-field to prove they're wider.
    # A bridged column shows high-k cells in the insulation row band; the widest
    # contiguous run along the wall must exceed the training max width in cells.
    widest_seen = 0.0
    for r in recs:
        k = r["k"]
        dy = float(r["dy"])
        # Insulation row = thickest layer's rows; detect via the most common low-k band.
        # Simpler/robust: any along-wall run of cells whose k exceeds the column's min
        # by a large factor is a bridge; measure the widest such run in metres.
        kmin = k.min(axis=0, keepdims=True)
        bridge_mask = (k > 5.0 * kmin).any(axis=0)  # (Ny,) any high-k cell in column
        # widest contiguous True run
        run = best = 0
        for flag in bridge_mask:
            run = run + 1 if flag else 0
            best = max(best, run)
        widest_seen = max(widest_seen, best * dy)
    assert widest_seen > _TRAIN_BRIDGE_W[1], "ood_bridges should contain a wider-than-train bridge"


def test_ood_res_different_native_resolution():
    """ood_res uses lateral and through-wall resolutions absent from training."""
    recs = generate_ood_corpus("ood_res", 24, seed=7)
    n_layers = {name: len(layers) for name, layers in _BASE_WALLS.items()}
    train_nx = {npl * _TRAIN_CELLS_PER_LAYER for npl in n_layers.values()}
    for r in recs:
        nx, ny = r["k"].shape
        assert ny in {96, 128}  # outside training {32, 48, 64}
        assert ny not in _TRAIN_LATERAL
        assert nx not in train_nx  # cells_per_layer 10-12 != 6
    # cells_per_layer recoverable: nx / n_layers in {10, 11, 12}.
    cpl_seen = set()
    for r in recs:
        nx = r["k"].shape[0]
        # match to whichever assembly divides nx exactly
        for npl in set(n_layers.values()):
            if nx % npl == 0:
                cpl_seen.add(nx // npl)
    assert cpl_seen and cpl_seen.issubset({10, 11, 12})


def test_bridge_raises_u_when_more_conductive_than_punctured_layer():
    """A steel/concrete bridge through the insulation layer must raise effective U.

    Uses ``light_framed`` (thickest layer = mineral wool, k=0.035) so any metal or
    concrete inclusion is strictly more conductive than what it replaces.
    """
    from thermotwin.data.synthetic_fem import _BASE_WALLS, ThermalBridge, WallSample

    layers = _BASE_WALLS["light_framed"]
    th = np.array([layer.thickness_m for layer in layers])
    edges = np.concatenate([[0.0], np.cumsum(th)])
    insul = int(np.argmax(th))
    bridge = ThermalBridge(edges[insul], edges[insul + 1], 0.20, 0.32, conductivity_w_mk=50.0)
    s = WallSample(
        layers=layers,
        width_m=0.6,
        t_indoor=20.0,
        t_outdoor=-5.0,
        bridges=(bridge,),
        lateral_cells=48,
    )
    assert solve_sample(s).u_value > clear_wall_u(s) * 1.05


def test_ood_walls_solver_consistency():
    """A no-bridge OOD wall must reduce to its 1-D clear-wall U (solver sanity)."""
    from thermotwin.data.synthetic_fem import WallSample

    layers = OOD_WALLS["brick_cavity"]
    s = WallSample(
        layers=layers,
        width_m=0.6,
        t_indoor=20.0,
        t_outdoor=-5.0,
        bridges=(),
        lateral_cells=48,
    )
    k, _ = build_k_field(s)
    assert np.all(k > 0)
    assert solve_sample(s).u_value == pytest.approx(clear_wall_u(s), rel=1e-6)
