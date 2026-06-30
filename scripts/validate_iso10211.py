#!/usr/bin/env python
"""Validate the independent reference solver against ISO 10211 Case A.1.

Case A.1 is the ISO 10211 *conduction-method* validation case: a 4 m x 8 m homogeneous
(lambda = 1 W/(m.K)) rectangle; the left (x=0) and bottom (y=0) edges are fixed at 0 C, the top
(y=8) edge at +20 C, and the right (x=4) edge is adiabatic. ISO requires the computed
temperatures to match the reference within 0.1 K. The exact reference is the closed-form
separated-variable (Laplace) solution

    T(x,y) = sum_n  c_n sin(lam_n x) sinh(lam_n y) / sinh(lam_n H),
    lam_n = (2n-1) pi / (2 W),   c_n = 40 / (W lam_n),

evaluated at the solver's cell centres (we compute it analytically, so there is no transcription
of a published table). The case has a corner singularity at (0, 8) where the 0 C and 20 C
boundaries meet; ISO scores at 1 m grid-points, and we report both the full-field error and the
error away from that corner.

ISO 10211 Case 2 (the 2-D *material* thermal-bridge benchmark with a metal bridge) is validated
separately in `scripts/validate_iso10211_case2.py` (passes: all nine reference temperatures within
0.039 K, heat flow within 0.009 W/m). Cases 3-4 are 3-D (a wall/floor corner scored by thermal
coupling coefficients; an iron bar through insulation) and remain a documented follow-up.

    python scripts/validate_iso10211.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from thermotwin.physics.reference_solver import BoundaryPatch, solve_reference  # noqa: E402

W, H, T_TOP, TOL = 4.0, 8.0, 20.0, 0.1


def analytic_a1(x: np.ndarray, y: np.ndarray, n_terms: int = 600) -> np.ndarray:
    """Closed-form Laplace solution for Case A.1 at coordinates (x, y). Numerically stable."""
    t = np.zeros(np.broadcast(x, y).shape, dtype=float)
    for m in range(1, n_terms + 1):
        lam = (2 * m - 1) * np.pi / (2 * W)
        c = 2.0 * T_TOP / (W * lam)  # = (2/W) * integral_0^W T_top sin(lam x) dx, cos(lam W)=0
        # sinh(lam y)/sinh(lam H), stable for lam*H large:
        ratio = (np.exp(lam * (y - H)) - np.exp(-lam * (y + H))) / (1.0 - np.exp(-2.0 * lam * H))
        t = t + c * np.sin(lam * x) * ratio
    return t


def main() -> None:
    nx, ny = 160, 320  # dx = dy = 0.025 m
    k = np.ones((nx, ny))
    spacing = (W / nx, H / ny)
    patches = [
        BoundaryPatch(0, "lo", t_air=0.0, r_film=0.0, name="left"),     # x = 0  -> 0 C
        BoundaryPatch(1, "lo", t_air=0.0, r_film=0.0, name="bottom"),   # y = 0  -> 0 C
        BoundaryPatch(1, "hi", t_air=T_TOP, r_film=0.0, name="top"),    # y = 8  -> 20 C
        # x = 4 (axis-0 hi) has no patch => adiabatic, as required.
    ]
    f = solve_reference(k, spacing, patches)

    xc = (np.arange(nx) + 0.5) * (W / nx)
    yc = (np.arange(ny) + 0.5) * (H / ny)
    XX, YY = np.meshgrid(xc, yc, indexing="ij")
    T_ref = analytic_a1(XX, YY)
    err = np.abs(f.temperature - T_ref)

    # distance of each cell centre from the singular corner (0, H)
    dist_corner = np.sqrt(XX**2 + (YY - H) ** 2)
    away = dist_corner > 0.25  # exclude a small neighbourhood of the corner singularity

    out = {
        "case": "ISO 10211 A.1 (conduction method validation)",
        "grid": [nx, ny],
        "tol_K": TOL,
        "max_err_K": float(err.max()),
        "max_err_away_from_corner_K": float(err[away].max()),
        "p99_err_K": float(np.percentile(err, 99)),
        "mean_err_K": float(err.mean()),
        "corner_note": "singularity at (0,8); ISO scores at 1 m grid points and away from it",
        "pass": bool(err[away].max() < TOL),
        "case2": "passed separately (scripts/validate_iso10211_case2.py): 2-D metal thermal bridge",
        "case3_4": "deferred: 3-D corner / iron-bar cases (coupling coefficients)",
    }
    odir = _REPO / "results"
    odir.mkdir(parents=True, exist_ok=True)
    (odir / "iso10211_validation.json").write_text(json.dumps(out, indent=2))

    print(f"ISO 10211 A.1: max |err| = {out['max_err_K']:.4f} K (full field), "
          f"{out['max_err_away_from_corner_K']:.4f} K away from the (0,8) corner; "
          f"p99 = {out['p99_err_K']:.4f} K")
    print(f"  PASS (away-from-corner < {TOL} K): {out['pass']}")
    print(f"wrote {odir / 'iso10211_validation.json'}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(11, 4))
        for a, (data, title, cmap) in zip(ax, [
            (f.temperature.T, "reference solver T [C]", "inferno"),
            (T_ref.T, "analytic Laplace T [C]", "inferno"),
            (err.T, "|error| [K]", "viridis"),
        ], strict=False):
            im = a.imshow(data, origin="lower", aspect="auto", cmap=cmap,
                          extent=[0, W, 0, H])
            a.set_title(title, fontsize=10)
            fig.colorbar(im, ax=a, fraction=0.046)
        fig.suptitle("ISO 10211 Case A.1 - reference solver vs analytic Laplace solution", fontsize=11)
        fig.tight_layout()
        (odir / "figures").mkdir(parents=True, exist_ok=True)
        fig.savefig(odir / "figures" / "iso10211_a1.png", dpi=130, bbox_inches="tight")
        print(f"wrote {odir / 'figures' / 'iso10211_a1.png'}")
    except Exception as exc:  # pragma: no cover - figure is optional
        print(f"(figure skipped: {exc})")


if __name__ == "__main__":
    main()
