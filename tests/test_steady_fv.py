"""The grid solver must reproduce the closed-form 1-D oracle.

These tests are the primary correctness gate for :mod:`thermotwin.physics.steady_fv`:
a flat multilayer wall solved on the grid has to match the EN ISO 6946 series
resistance from :mod:`thermotwin.physics.conduction` to (near) machine precision,
with and without surface films, and independent of resolution and of transverse
grid extent.
"""

from __future__ import annotations

import numpy as np
import pytest

from thermotwin.physics.conduction import Layer, SurfaceFilm, steady_state_1d
from thermotwin.physics.steady_fv import (
    DirichletFilm,
    layered_k_field,
    solve_steady_conduction,
)

WALL = [
    Layer("plasterboard", 0.0125, 0.25),
    Layer("mineral wool", 0.120, 0.035),
    Layer("brick", 0.200, 0.77),
]
T_IN, T_OUT = 20.0, -2.0


def test_matches_oracle_without_films():
    """Pure Dirichlet surfaces -> exact series resistance, any resolution."""
    oracle = steady_state_1d(WALL, T_IN, T_OUT, film=SurfaceFilm(0.0, 0.0))
    k, spacing = layered_k_field(WALL, cells_per_layer=6)
    res = solve_steady_conduction(k, spacing, DirichletFilm(T_IN, T_OUT))
    assert res.u_value == pytest.approx(oracle.u_value, rel=1e-9)
    assert res.heat_flux == pytest.approx(oracle.heat_flux, rel=1e-9)


def test_matches_oracle_with_films():
    """EN ISO 6946 films via Robin BCs reproduce the oracle including R_si/R_se."""
    film = SurfaceFilm(r_si=0.13, r_se=0.04)
    oracle = steady_state_1d(WALL, T_IN, T_OUT, film=film)
    k, spacing = layered_k_field(WALL, cells_per_layer=10)
    res = solve_steady_conduction(
        k, spacing, DirichletFilm(T_IN, T_OUT, r_lo=film.r_si, r_hi=film.r_se)
    )
    assert res.u_value == pytest.approx(oracle.u_value, rel=1e-6)
    assert res.r_total == pytest.approx(oracle.r_total, rel=1e-6)
    assert res.heat_flux == pytest.approx(oracle.heat_flux, rel=1e-6)


@pytest.mark.parametrize("cells", [1, 2, 4, 16])
def test_resolution_invariance(cells):
    """Harmonic face conductivity -> U is exact even at 1 cell per layer."""
    oracle = steady_state_1d(WALL, T_IN, T_OUT, film=SurfaceFilm(0.0, 0.0))
    k, spacing = layered_k_field(WALL, cells_per_layer=cells)
    res = solve_steady_conduction(k, spacing, DirichletFilm(T_IN, T_OUT))
    assert res.u_value == pytest.approx(oracle.u_value, rel=1e-9)


@pytest.mark.parametrize("cross_section", [(4,), (3, 5)])
def test_transverse_invariance(cross_section):
    """With adiabatic lateral faces, 2-D/3-D give the same U as 1-D."""
    oracle = steady_state_1d(WALL, T_IN, T_OUT, film=SurfaceFilm(0.0, 0.0))
    k, spacing = layered_k_field(WALL, cells_per_layer=4, cross_section=cross_section)
    res = solve_steady_conduction(k, spacing, DirichletFilm(T_IN, T_OUT))
    assert res.u_value == pytest.approx(oracle.u_value, rel=1e-9)
    # No transverse gradient should develop: every transverse column equals the
    # through-axis profile taken at the first transverse index.
    t = res.temperature
    ref = t[(slice(None),) + (0,) * len(cross_section)]  # shape (Nx,)
    expected = ref.reshape((-1,) + (1,) * len(cross_section))
    assert np.allclose(t, expected)


def test_homogeneous_slab_linear_profile():
    """A single homogeneous layer -> linear temperature profile."""
    slab = [Layer("concrete", 0.2, 1.5)]
    k, spacing = layered_k_field(slab, cells_per_layer=20)
    res = solve_steady_conduction(k, spacing, DirichletFilm(10.0, 0.0))
    profile = res.temperature
    # Cell-centred temperatures should sit on a straight line in x.
    x = (np.arange(profile.size) + 0.5) * spacing[0]
    slope, intercept = np.polyfit(x, profile, 1)
    assert np.allclose(profile, slope * x + intercept, atol=1e-9)


def test_rejects_nonpositive_conductivity():
    k, spacing = layered_k_field([Layer("a", 0.1, 1.0)], cells_per_layer=3)
    k[0] = 0.0
    with pytest.raises(ValueError):
        solve_steady_conduction(k, spacing, DirichletFilm(20.0, 0.0))
