"""Synthetic FEM corpus generator: physics sanity and reproducibility."""

from __future__ import annotations

import numpy as np
import pytest

from thermotwin.data.synthetic_fem import (
    ThermalBridge,
    WallSample,
    build_k_field,
    clear_wall_u,
    generate_corpus,
    solve_sample,
)
from thermotwin.physics.conduction import Layer

WALL = (
    Layer("concrete", 0.20, 1.95),
    Layer("eps_insulation", 0.10, 0.035),
    Layer("gypsum", 0.0127, 0.16),
)


def _sample(bridges=()):
    return WallSample(
        layers=WALL,
        width_m=0.6,
        t_indoor=20.0,
        t_outdoor=-5.0,
        bridges=bridges,
        lateral_cells=48,
    )


def test_field_shapes_and_k_values():
    s = _sample()
    k, spacing = build_k_field(s)
    assert k.ndim == 2
    assert k.shape[1] == 48
    assert len(spacing) == 2
    # Base k only contains the layer conductivities.
    assert set(np.round(np.unique(k), 6)) == {1.95, 0.035, 0.16}


def test_clear_wall_matches_when_no_bridges():
    """With no bridges the 2-D solve equals the clear-wall 1-D U-value."""
    s = _sample()
    assert solve_sample(s).u_value == pytest.approx(clear_wall_u(s), rel=1e-6)


def test_thermal_bridge_increases_u():
    """A steel stud through the insulation must raise effective U above clear-wall."""
    # Insulation spans x in [0.20, 0.30]; place a steel bridge across part of the width.
    bridge = ThermalBridge(0.20, 0.30, 0.25, 0.32, conductivity_w_mk=50.0)
    s = _sample(bridges=(bridge,))
    u_bridged = solve_sample(s).u_value
    u_clear = clear_wall_u(s)
    assert u_bridged > u_clear * 1.05  # at least a 5% penalty from the bridge


def test_bridge_creates_lateral_gradient():
    bridge = ThermalBridge(0.20, 0.30, 0.25, 0.32, conductivity_w_mk=50.0)
    t = solve_sample(_sample(bridges=(bridge,))).temperature
    # The temperature field must vary along the wall (axis 1) near the bridge.
    assert t.std(axis=1).max() > 1e-3


def test_corpus_is_deterministic_and_well_formed():
    a = generate_corpus(4, seed=1337)
    b = generate_corpus(4, seed=1337)
    assert len(a) == 4
    for ra, rb in zip(a, b, strict=True):
        assert np.array_equal(ra["k"], rb["k"])
        assert np.array_equal(ra["temperature"], rb["temperature"])
        assert ra["temperature"].shape == ra["k"].shape
        assert ra["u_value"] >= ra["u_clear"] - 1e-4  # bridges never reduce U
        assert ra["k"].dtype == np.float32
