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

__all__ = ["effective_u_from_theta", "u_from_indoor_face_cloud", "u_value_report"]


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


def u_from_indoor_face_cloud(
    theta: np.ndarray,
    prior: np.ndarray,
    points: np.ndarray,
    u_clear: float,
    band: float = 0.08,
    eps: float = 1e-8,
) -> float:
    """Effective U-value from a scattered 3-D θ cloud, via the indoor-face deficit.

    Block-2 predictions are scattered point fields (GINO) or dense voxel fields
    (``fno_voxel``) — both are unstructured w.r.t. the native FV grid, so the 2-D
    cell-indexed :func:`effective_u_from_theta` does not apply. We instead use the
    *ratio* of the indoor-face dimensionless deficit to the analytic clear-wall
    prior's deficit on the same near-face points:

        U ≈ U_clear · mean(1 − θ_face) / mean(1 − θ1d_face),

    over the points within ``band`` of the indoor face (axis-0 ``lo`` plane, the
    normalised through-wall coordinate ``x < band``). The dimensionless deficit
    ``1 − θ`` at the indoor face is the film temperature drop, i.e. proportional to
    the boundary flux, so its ratio to the clear-column flux scales the *known* exact
    clear-wall U-value to the effective one. Two properties make this a fair,
    leakage-free estimator applied identically to ground truth and to every model:

    * it uses only the predicted field, the analytic prior (a known input), and the
      exact ``u_clear`` (a stored physics scalar) — never the target U;
    * it is **exact** on a clear (no-bridge) column, where ``θ ≡ θ1d`` makes the
      ratio 1 and ``U = U_clear``; the residual it must estimate is precisely the
      bridge-driven excess, mirroring the delta-learning recipe.

    Args:
        theta: predicted dimensionless temperature at each point, shape ``(N,)``.
        prior: analytic 1-D clear-wall θ at each point, shape ``(N,)``.
        points: point coordinates in ``[0, 1]^3``, shape ``(N, 3)``; axis 0 is
            through-wall with the indoor (``θ = 1``) face at ``x = 0``.
        u_clear: exact 1-D clear-wall U-value [W/(m²K)] of the same block.
        band: near-indoor-face slab half-width in normalised coords; points with
            ``x < band`` define the face.

    Returns:
        Effective U-value [W/(m²K)].
    """
    theta = np.asarray(theta, dtype=float).ravel()
    prior = np.asarray(prior, dtype=float).ravel()
    x = np.asarray(points, dtype=float)[:, 0]
    mask = x < band
    if not np.any(mask):  # extremely sparse face — fall back to the whole cloud
        mask = np.ones_like(x, dtype=bool)
    num = float(np.mean(1.0 - theta[mask]))
    den = float(np.mean(1.0 - prior[mask]))
    return float(u_clear) * num / (den + eps)


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
