"""Tests for the independent general-BC reference conduction solver."""

from __future__ import annotations

import numpy as np

from thermotwin.physics.conduction import Layer
from thermotwin.physics.reference_solver import (
    BoundaryPatch,
    solve_reference,
    temperature_factor,
)
from thermotwin.physics.steady_fv import (
    DirichletFilm,
    layered_k_field,
    solve_steady_conduction,
)


def _layered_3d(cross=(6, 6), cells=6):
    layers = [Layer("ins", 0.10, 0.04), Layer("brick", 0.20, 0.80)]
    k, spacing = layered_k_field(layers, cells_per_layer=cells, cross_section=cross)
    return k, spacing, layers


# ---- 1-D: matches the analytic series-resistance answer --------------------------------------

def test_1d_layered_matches_analytic_u():
    layers = [Layer("ins", 0.10, 0.04), Layer("brick", 0.20, 0.80)]
    k, spacing = layered_k_field(layers, cells_per_layer=40)  # 1-D
    r_si, r_se, t_i, t_e = 0.13, 0.04, 20.0, 0.0
    patches = [
        BoundaryPatch(0, "lo", t_air=t_i, r_film=r_si, name="in"),
        BoundaryPatch(0, "hi", t_air=t_e, r_film=r_se, name="out"),
    ]
    f = solve_reference(k, spacing, patches)
    r_total = r_si + sum(L.thickness_m / L.conductivity_w_mk for L in layers) + r_se
    u_analytic = 1.0 / r_total
    q_analytic = u_analytic * (t_i - t_e)  # per unit area (1-D area == 1)
    assert np.isclose(f.patch_flux["in"], q_analytic, rtol=1e-6)
    assert np.isclose(f.patch_flux["in"], -f.patch_flux["out"], rtol=1e-6)  # energy balance


# ---- 2-D homogeneous: linear, y-independent, energy-balanced ---------------------------------

def test_2d_homogeneous_dirichlet_is_linear():
    nx, ny = 20, 12
    k = np.ones((nx, ny))
    spacing = (0.01, 0.02)
    patches = [
        BoundaryPatch(0, "lo", t_air=1.0, r_film=0.0, name="hot"),
        BoundaryPatch(0, "hi", t_air=0.0, r_film=0.0, name="cold"),
    ]
    f = solve_reference(k, spacing, patches)
    T = f.temperature
    # independent of y (transverse), monotone decreasing in x, energy-balanced
    assert np.allclose(T - T[:, :1], 0.0, atol=1e-9)
    col = T[:, 0]
    assert np.all(np.diff(col) < 0)
    assert np.isclose(f.patch_flux["hot"], -f.patch_flux["cold"], rtol=1e-9)


# ---- cross-validation: must match the production solver to machine precision -----------------

def test_matches_production_layered_column():
    k, spacing, _ = _layered_3d()
    bc = DirichletFilm(t_lo=20.0, t_hi=0.0, r_lo=0.13, r_hi=0.04)
    prod = solve_steady_conduction(k, spacing, bc)
    ref = solve_reference(
        k, spacing,
        [BoundaryPatch(0, "lo", 20.0, 0.13, name="in"),
         BoundaryPatch(0, "hi", 0.0, 0.04, name="out")],
    )
    assert np.allclose(ref.temperature, prod.temperature, atol=1e-9)  # exact cross-validation
    assert np.isclose(ref.patch_flux["in"], -ref.patch_flux["out"], rtol=1e-6)  # energy balance


def test_matches_production_with_bridge():
    k, spacing, _ = _layered_3d(cross=(8, 8), cells=6)
    k = k.copy()
    k[2:4, 3:6, 3:6] = 5.0  # a high-conductivity inclusion (thermal bridge)
    bc = DirichletFilm(t_lo=20.0, t_hi=0.0, r_lo=0.13, r_hi=0.04)
    prod = solve_steady_conduction(k, spacing, bc)
    ref = solve_reference(
        k, spacing,
        [BoundaryPatch(0, "lo", 20.0, 0.13, name="in"),
         BoundaryPatch(0, "hi", 0.0, 0.04, name="out")],
    )
    assert np.allclose(ref.temperature, prod.temperature, atol=1e-9)


# ---- masked patches + multi-face: a face split into two BC regions ---------------------------

def test_masked_patch_partitions_a_face():
    nx, ny = 16, 16
    k = np.ones((nx, ny))
    spacing = (0.01, 0.01)
    # lo-x face fully hot; hi-x face: lower half cold, upper half adiabatic (masked).
    cold_mask = np.zeros(ny, dtype=bool)
    cold_mask[: ny // 2] = True
    f = solve_reference(
        k, spacing,
        [BoundaryPatch(0, "lo", 1.0, 0.0, name="hot"),
         BoundaryPatch(0, "hi", 0.0, 0.0, mask=cold_mask, name="cold")],
    )
    # the masked (adiabatic) half stays warmer than the cold-driven half at the hi face
    hi_face = f.temperature[-1, :]
    assert hi_face[ny // 2 :].mean() > hi_face[: ny // 2].mean()
    assert np.isclose(f.patch_flux["hot"], -f.patch_flux["cold"], rtol=1e-6)


# ---- f_Rsi on a clear wall matches 1 - U*R_si ------------------------------------------------

def test_clear_wall_temperature_factor():
    layers = [Layer("ins", 0.10, 0.04), Layer("brick", 0.20, 0.80)]
    k, spacing = layered_k_field(layers, cells_per_layer=60)  # 1-D => per-cell face area == 1
    r_si, r_se, t_i, t_e = 0.13, 0.04, 20.0, 0.0
    f = solve_reference(
        k, spacing,
        [BoundaryPatch(0, "lo", t_i, r_si, name="in"),
         BoundaryPatch(0, "hi", t_e, r_se, name="out")],
    )
    r_total = r_si + sum(L.thickness_m / L.conductivity_w_mk for L in layers) + r_se
    u = 1.0 / r_total
    q = f.patch_flux["in"]  # 1-D: already per unit area
    # surface temperature from the film: T_si = t_i - q*R_si
    f_rsi_true = (t_i - q * r_si - t_e) / (t_i - t_e)
    assert np.isclose(f_rsi_true, 1.0 - u * r_si, rtol=1e-6)
    # the cell-centre proxy is close on a fine grid (the first cell sits half a cell deeper)
    assert abs(temperature_factor(f, "in", t_i, t_e) - f_rsi_true) < 0.02
