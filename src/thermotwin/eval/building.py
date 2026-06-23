"""Building-relevant metrics — the venue side of the reporting pair.

``docs/baselines.md`` requires every result to pair an operator-learning metric
(relative L2) with a building metric (U-value / heat-loss error). This module turns
a predicted **dimensionless** temperature field ``θ`` back into the quantity a
building engineer cares about: the effective transmittance ``U`` of the wall
section, thermal bridges and all.

The reconstruction uses the *same* discrete boundary flux as
:func:`thermotwin.physics.steady_fv.solve_steady_conduction`, so feeding the
ground-truth ``θ`` reproduces the solver's stored U to machine precision (the unit
test). Working in ``θ`` makes ``U`` independent of the absolute boundary
temperatures:

    U = (1 / A_wall) · Σ_indoor_face  g_bnd · (1 − θ_face),
    g_bnd = A_face / ((dx/2)/k + r_si)

where the sum is over the cells on the indoor (axis-0 ``lo``) face.
"""

from __future__ import annotations

import numpy as np

__all__ = ["effective_u_from_theta", "u_value_report"]


def effective_u_from_theta(
    theta: np.ndarray,
    k: np.ndarray,
    dx0: np.ndarray,
    dy: float,
    r_si: float,
) -> float:
    """Effective U-value [W/(m²K)] from a 2-D dimensionless field.

    Args:
        theta: dimensionless temperature ``(T−T_out)/(T_in−T_out)``, shape (Nx, Ny).
        k: conductivity field [W/(m·K)], shape (Nx, Ny).
        dx0: through-wall per-cell spacing [m], length Nx.
        dy: along-wall cell size [m] (face area normal to axis 0, per cell).
        r_si: indoor surface film resistance [m²K/W] (the axis-0 ``lo`` face).
    """
    theta = np.asarray(theta, dtype=float)
    k = np.asarray(k, dtype=float)
    dx0 = np.asarray(dx0, dtype=float)

    k_lo = k[0, :]  # indoor-face row
    r_half = (dx0[0] / 2.0) / k_lo
    g_bnd = dy / (r_half + r_si)  # [W/K] per boundary cell
    q_over_dt = np.sum(g_bnd * (1.0 - theta[0, :]))  # [W/K]
    wall_area = dy * theta.shape[1]  # [m^2]
    return float(q_over_dt / wall_area)


def u_value_report(pred_u: np.ndarray, true_u: np.ndarray) -> dict[str, float]:
    """Summary of predicted-vs-true U-value error across a corpus."""
    pred_u = np.asarray(pred_u, dtype=float)
    true_u = np.asarray(true_u, dtype=float)
    err = pred_u - true_u
    return {
        "u_mae": float(np.mean(np.abs(err))),
        "u_rmse": float(np.sqrt(np.mean(err**2))),
        "u_mape": float(np.mean(np.abs(err) / true_u) * 100.0),
        "u_bias": float(np.mean(err)),
    }
