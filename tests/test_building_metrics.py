"""Building-metric reconstruction must reproduce the solver's ground-truth U."""

from __future__ import annotations

import numpy as np
import pytest

from thermotwin.data.synthetic_fem import ThermalBridge, WallSample, build_k_field, solve_sample
from thermotwin.eval.building import effective_u_from_theta, u_value_report
from thermotwin.physics.conduction import Layer

WALL = (
    Layer("concrete", 0.20, 1.95),
    Layer("eps_insulation", 0.10, 0.035),
    Layer("gypsum", 0.0127, 0.16),
)


def _theta(field, t_in, t_out):
    return (field.temperature - t_out) / (t_in - t_out)


@pytest.mark.parametrize(
    "bridges",
    [
        (),
        (ThermalBridge(0.20, 0.30, 0.25, 0.32, 50.0),),
        (ThermalBridge(0.20, 0.30, 0.10, 0.16, 160.0), ThermalBridge(0.20, 0.30, 0.40, 0.46, 1.95)),
    ],
)
def test_u_from_theta_matches_solver(bridges):
    """Feeding the ground-truth θ reconstructs the solver's U exactly."""
    s = WallSample(
        WALL, width_m=0.6, t_indoor=21.0, t_outdoor=-4.0, bridges=bridges, lateral_cells=48
    )
    field = solve_sample(s)
    k, spacing = build_k_field(s)
    u = effective_u_from_theta(
        _theta(field, s.t_indoor, s.t_outdoor), k, spacing[0], spacing[1], s.r_si
    )
    assert u == pytest.approx(field.u_value, rel=1e-9)


def test_u_value_report_shapes():
    rep = u_value_report(np.array([0.30, 0.50]), np.array([0.32, 0.45]))
    assert set(rep) == {"u_mae", "u_rmse", "u_mape", "u_bias"}
    assert rep["u_mae"] == pytest.approx(0.035)
    assert rep["u_bias"] == pytest.approx(0.015)
