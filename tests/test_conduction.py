"""Closed-form checks for the 1-D multilayer conduction oracle."""

import math

import pytest

from thermotwin.physics.conduction import (
    Layer,
    SurfaceFilm,
    steady_state_1d,
    total_resistance,
    u_value,
)


def test_single_layer_u_value_closed_form():
    film = SurfaceFilm(r_si=0.13, r_se=0.04)
    layer = Layer("insulation", thickness_m=0.10, conductivity_w_mk=0.04)
    # R = 0.13 + 0.10/0.04 + 0.04 = 2.67 ; U = 1/2.67
    assert math.isclose(total_resistance([layer], film), 2.67, rel_tol=1e-12)
    assert math.isclose(u_value([layer], film), 1.0 / 2.67, rel_tol=1e-12)


def test_heat_flux_and_profile_consistency():
    layer = Layer("insulation", 0.10, 0.04)
    res = steady_state_1d([layer], t_indoor=20.0, t_outdoor=0.0)
    # q = U * dT
    assert math.isclose(res.heat_flux, res.u_value * 20.0, rel_tol=1e-12)
    # profile starts at indoor air and lands on outdoor air within fp tolerance
    assert math.isclose(res.node_temperatures[0], 20.0, abs_tol=1e-9)
    assert math.isclose(res.node_temperatures[-1], 0.0, abs_tol=1e-9)
    # monotonically decreasing when T_in > T_out
    t = res.node_temperatures
    assert all(t[i] >= t[i + 1] for i in range(len(t) - 1))


def test_multilayer_series_resistance_and_nodes():
    layers = [
        Layer("plasterboard", 0.0125, 0.25),
        Layer("mineral wool", 0.120, 0.035),
        Layer("brick", 0.200, 0.77),
    ]
    expected_r = 0.13 + 0.0125 / 0.25 + 0.120 / 0.035 + 0.200 / 0.77 + 0.04
    assert math.isclose(total_resistance(layers), expected_r, rel_tol=1e-12)

    res = steady_state_1d(layers, t_indoor=21.0, t_outdoor=-5.0)
    # total air-to-air drop is conserved
    drop = res.node_temperatures[0] - res.node_temperatures[-1]
    assert math.isclose(drop, 26.0, abs_tol=1e-9)
    # nodes = layers + 3 (indoor air, internal surface, external surface, outdoor air)
    assert len(res.node_temperatures) == len(layers) + 3
    assert len(res.node_names) == len(res.node_temperatures)


def test_validation_guards():
    with pytest.raises(ValueError):
        Layer("bad-thickness", 0.0, 1.0)
    with pytest.raises(ValueError):
        Layer("bad-lambda", 0.1, 0.0)
    with pytest.raises(ValueError):
        total_resistance([])
