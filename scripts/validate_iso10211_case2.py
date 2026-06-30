#!/usr/bin/env python
"""Validate the independent reference solver against ISO 10211:2007 Case 2 (material thermal bridge).

Case 2 is the standard's two-dimensional *material* benchmark: a 0.5 m x 0.0475 m section in which
four materials of widely differing conductivity (slab 1.15, insulation 0.12, air 0.029, metal 230
W/(m.K)) form a metal thermal bridge, driven by an interior film (20 C, Rsi 0.11) on the bottom and
an exterior film (0 C, Rse 0.06) on the top, with adiabatic sides. ISO 10211 publishes nine
reference temperatures (points A..I) to be matched within **0.1 K** and a reference heat flow of
**9.5 W/m** to within **0.1 W/m**.

Geometry, materials, boundary conditions and the nine ISO reference temperatures are taken from the
Blocon HEAT2/HEAT3 validation report (buildingphysics.com/download/iso/ISO_10211_HEAT2_HEAT3.pdf),
which reproduces the ISO 10211:2007 reference values. The metal bridge is a bracket: a 1.5 mm full
width strip on the interior face, a 1.5 mm riser up the left (symmetry) edge, and a 1.5 mm shelf
under the insulation block (the near-isothermal left edge, H=16.8 -> F=16.4 C, fixes the riser; the
8.5 K drop F=16.4 -> C=7.9 C fixes the 5 mm insulation block on top of it).

    python scripts/validate_iso10211_case2.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from thermotwin.physics.reference_solver import BoundaryPatch, solve_reference  # noqa: E402

W, HY = 0.5, 0.0475          # domain: width (x) x height (y), metres
T_INT, R_SI = 20.0, 0.11     # interior film on the bottom edge (y = 0)
T_EXT, R_SE = 0.0, 0.06      # exterior film on the top edge (y = HY)
K_SLAB, K_INS, K_AIR, K_METAL = 1.15, 0.12, 0.029, 230.0
TOL_T, TOL_Q, Q_REF = 0.1, 0.1, 9.5

# ISO 10211 Case 2 reference temperatures (point: (x, y) in m -> reference T in C, surface? ).
REF_POINTS = {
    "A": ((0.000, 0.0475), 7.1, "top"),     # top surface, left
    "B": ((0.500, 0.0475), 0.8, "top"),     # top surface, right
    "C": ((0.000, 0.0415), 7.9, "bulk"),    # under the slab, left
    "D": ((0.015, 0.0415), 6.3, "bulk"),
    "E": ((0.500, 0.0415), 0.8, "bulk"),
    "F": ((0.000, 0.0365), 16.4, "bulk"),   # top of the metal riser, left
    "G": ((0.015, 0.0365), 16.3, "bulk"),
    "H": ((0.000, 0.0000), 16.8, "bottom"), # bottom surface, left
    "I": ((0.500, 0.0000), 18.3, "bottom"), # bottom surface, right
}


def build_case2(dx: float):
    """Conductivity field k(x, y), spacing, and cell-centre coordinate axes for the chosen dx."""
    nx, ny = int(round(W / dx)), int(round(HY / dx))
    xc = (np.arange(nx) + 0.5) * (W / nx)
    yc = (np.arange(ny) + 0.5) * (HY / ny)
    X, Y = np.meshgrid(xc, yc, indexing="ij")

    k = np.full((nx, ny), K_AIR)                                   # region 3: air
    k[Y >= 0.0415] = K_SLAB                                        # region 1: slab (top 6 mm)
    k[(X < 0.015) & (Y >= 0.0365) & (Y < 0.0415)] = K_INS         # region 2: insulation block
    # region 4: the metal bracket (assign last so it wins on overlaps)
    k[Y < 0.0015] = K_METAL                                        #   interior-face strip (full width)
    k[(X < 0.0015) & (Y < 0.0365)] = K_METAL                      #   riser up the left edge
    k[(X < 0.015) & (Y >= 0.035) & (Y < 0.0365)] = K_METAL        #   shelf under the insulation
    return k, (W / nx, HY / ny), xc, yc


def _surface_temp(t_cell, k_cell, half, t_air, r_film):
    """Reconstruct the surface temperature from the film + half-cell series at a Robin boundary."""
    g_film, g_cell = 1.0 / r_film, k_cell / half
    return (t_air * g_film + t_cell * g_cell) / (g_film + g_cell)


def _yface_temp(field, k, i, py, dy):
    """Conductivity-weighted temperature of the horizontal material interface nearest ``py``.

    The bulk reference points sit on y-interfaces (which align with cell faces here). The physical
    interface temperature follows from flux continuity across the face: with equal half-cells,
    ``T_face = (k_below T_below + k_above T_above) / (k_below + k_above)`` -- so a high-conductivity
    layer (e.g. the metal, 2000x the insulation) pins the interface to its own temperature, exactly
    as the standard reports.
    """
    jb = int(np.clip(round(py / dy) - 1, 0, field.shape[1] - 2))
    ja = jb + 1
    kb, ka = k[i, jb], k[i, ja]
    return float((kb * field[i, jb] + ka * field[i, ja]) / (kb + ka))


def evaluate(field, k, xc, yc, dy):
    """Computed temperature at each ISO reference point (surface points reconstructed via the film)."""
    out = {}
    for name, ((px, py), _ref, kind) in REF_POINTS.items():
        i = int(np.argmin(np.abs(xc - px)))
        if kind == "top":
            out[name] = _surface_temp(field[i, -1], k[i, -1], dy / 2, T_EXT, R_SE)
        elif kind == "bottom":
            out[name] = _surface_temp(field[i, 0], k[i, 0], dy / 2, T_INT, R_SI)
        else:
            out[name] = _yface_temp(field, k, i, py, dy)
    return out


def main() -> None:
    dx = 0.00025  # 0.25 mm uniform grid: resolves the 1.5 mm metal features (~6 cells each)
    k, spacing, xc, yc = build_case2(dx)
    patches = [
        BoundaryPatch(1, "lo", t_air=T_INT, r_film=R_SI, name="interior"),  # bottom edge y = 0
        BoundaryPatch(1, "hi", t_air=T_EXT, r_film=R_SE, name="exterior"),  # top edge y = HY
        # x = 0 and x = W are unpatched => adiabatic symmetry planes, as ISO requires.
    ]
    field = solve_reference(k, spacing, patches)
    comp = evaluate(field.temperature, k, xc, yc, spacing[1])

    rows, max_dt = [], 0.0
    for name, ((px, py), ref, kind) in REF_POINTS.items():
        c = comp[name]
        d = abs(c - ref)
        max_dt = max(max_dt, d)
        rows.append({"point": name, "xy": [px, py], "kind": kind,
                     "ref_C": ref, "computed_C": round(c, 3), "abs_err_K": round(d, 3)})

    q_in = field.patch_flux["interior"]      # into the domain from the warm interior (W/m)
    q_out = -field.patch_flux["exterior"]    # out through the cold exterior (W/m)
    q_err = max(abs(q_in - Q_REF), abs(q_out - Q_REF))

    out = {
        "case": "ISO 10211:2007 Case 2 (2-D material thermal bridge)",
        "source": "Blocon HEAT2/HEAT3 validation report (reproduces ISO 10211 reference values)",
        "grid": [int(round(W / dx)), int(round(HY / dx))],
        "tol_K": TOL_T, "tol_Q_W_per_m": TOL_Q,
        "max_temp_err_K": round(float(max_dt), 3),
        "heat_flow_in_W_per_m": round(float(q_in), 4),
        "heat_flow_out_W_per_m": round(float(q_out), 4),
        "heat_flow_ref_W_per_m": Q_REF,
        "heat_flow_err_W_per_m": round(float(q_err), 4),
        "energy_balance_W_per_m": round(float(q_in - q_out), 5),
        "points": rows,
        "pass": bool(max_dt < TOL_T and q_err < TOL_Q),
    }
    odir = _REPO / "results"
    odir.mkdir(parents=True, exist_ok=True)
    (odir / "iso10211_case2_validation.json").write_text(json.dumps(out, indent=2))

    print(f"ISO 10211 Case 2: max |T_err| = {out['max_temp_err_K']:.3f} K (tol {TOL_T}), "
          f"Q in/out = {out['heat_flow_in_W_per_m']:.3f}/{out['heat_flow_out_W_per_m']:.3f} W/m "
          f"(ref {Q_REF}, err {out['heat_flow_err_W_per_m']:.3f})")
    for r in rows:
        flag = "" if r["abs_err_K"] < TOL_T else "  <-- exceeds tol"
        print(f"  {r['point']}: ref {r['ref_C']:5.1f}  computed {r['computed_C']:7.3f}  "
              f"|err| {r['abs_err_K']:.3f} K{flag}")
    print(f"  PASS: {out['pass']}")
    print(f"wrote {odir / 'iso10211_case2_validation.json'}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 3.2))
        im = ax.imshow(field.temperature.T, origin="lower", aspect="auto", cmap="inferno",
                       extent=[0, W, 0, HY])
        for name, ((px, py), _r, _k) in REF_POINTS.items():
            ax.plot(px, py, "o", ms=5, mfc="cyan", mec="k")
            ax.annotate(name, (px, py), textcoords="offset points", xytext=(4, 4),
                        fontsize=8, color="white")
        ax.set_title("ISO 10211 Case 2 - reference-solver T [C] with the nine reference points")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="C")
        fig.tight_layout()
        (odir / "figures").mkdir(parents=True, exist_ok=True)
        fig.savefig(odir / "figures" / "iso10211_case2.png", dpi=130, bbox_inches="tight")
        print(f"wrote {odir / 'figures' / 'iso10211_case2.png'}")
    except Exception as exc:  # pragma: no cover - figure is optional
        print(f"(figure skipped: {exc})")


if __name__ == "__main__":
    main()
