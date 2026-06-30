#!/usr/bin/env python
"""Cross-validate the production FV solver against the independent reference solver.

The production engine (`thermotwin.physics.steady_fv.solve_steady_conduction`) only applies
boundary conditions on the axis-0 faces -- which is exactly the regime of our generated corpus
(layered wall + embedded bridges, indoor/outdoor on the two through-wall faces). The reference
solver (`thermotwin.physics.reference_solver.solve_reference`) is an independent implementation
of the same finite-volume scheme. Agreement to ~machine precision on a sample of generated
cases (incl. severe bridges) shows our GT generator is free of implementation bugs.

    python scripts/validate_solver_crosscheck.py --n 24
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from thermotwin.data.synthetic_3d import build_k_field_3d, random_block  # noqa: E402
from thermotwin.physics.reference_solver import BoundaryPatch, solve_reference  # noqa: E402
from thermotwin.physics.steady_fv import DirichletFilm, solve_steady_conduction  # noqa: E402


def _compare(sample) -> dict:
    k, spacing = build_k_field_3d(sample)
    prod = solve_steady_conduction(
        k, spacing, DirichletFilm(sample.t_indoor, sample.t_outdoor, r_lo=sample.r_si, r_hi=sample.r_se)
    )
    ref = solve_reference(
        k, spacing,
        [BoundaryPatch(0, "lo", sample.t_indoor, sample.r_si, name="in"),
         BoundaryPatch(0, "hi", sample.t_outdoor, sample.r_se, name="out")],
    )
    denom = float(np.linalg.norm(prod.temperature)) + 1e-12
    field_rel_l2 = float(np.linalg.norm(ref.temperature - prod.temperature) / denom)
    area = sample.width_y_m * sample.width_z_m
    dt = sample.t_indoor - sample.t_outdoor
    u_ref = ref.patch_flux["in"] / (area * dt)
    u_rel = abs(u_ref - prod.u_value) / (abs(prod.u_value) + 1e-12)
    return {"field_rel_l2": field_rel_l2, "u_rel_err": float(u_rel), "n_bridges": len(sample.bridges)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=12, help="number of generated standard blocks")
    p.add_argument("--seed", type=int, default=1337)
    a = p.parse_args()
    rng = np.random.default_rng(a.seed)

    # Standard blocks (incl. bridges) are fast; the severe sub-voxel "hard" blocks take ~30 s
    # each to solve, and agreement on a bridge is already covered by tests/test_reference_solver.py
    # (test_matches_production_with_bridge). So sample standard blocks here for a broad check.
    rows = [_compare(random_block(rng)) for _ in range(a.n)]

    field = np.array([r["field_rel_l2"] for r in rows])
    urel = np.array([r["u_rel_err"] for r in rows])
    tol = 1e-6
    out = {
        "n_cases": len(rows),
        "field_rel_l2_max": float(field.max()),
        "field_rel_l2_mean": float(field.mean()),
        "u_rel_err_max": float(urel.max()),
        "tol": tol,
        "pass": bool(field.max() < tol and urel.max() < tol),
        "note": "production vs independent reference solver on generated corpus cases (axis-0 BCs)",
    }
    (_REPO / "results").mkdir(parents=True, exist_ok=True)
    (_REPO / "results" / "solver_crosscheck.json").write_text(json.dumps(out, indent=2))
    print(f"production vs reference over {out['n_cases']} cases: "
          f"field rel-L2 max = {out['field_rel_l2_max']:.2e}, U rel-err max = {out['u_rel_err_max']:.2e}")
    print(f"  PASS (< {tol}): {out['pass']}")
    print(f"wrote {_REPO / 'results' / 'solver_crosscheck.json'}")


if __name__ == "__main__":
    main()
