#!/usr/bin/env python
"""Generate the Block-2 synthetic 3-D conduction ground-truth corpus.

Block-2 carries the Block-1 recipe (geometry-conditioned operator predicting a
correction on an analytic 1-D clear-wall prior) into 3-D, irregular-geometry
operators (GINO). This writes solved 3-D wall **blocks** — layered constructions
punctured by finite thermal-bridge prisms — into ``data/processed/<name>/`` as
per-sample ``.npz`` files plus a ``manifest.json`` that makes the corpus
reproducible (generator, seed, per-sample scalars).

Each ``.npz`` stores a GINO sample: ``points`` (N,3) in [0,1]^3, ``feats`` (N,F)
``[logk_std, r_si, r_se, theta1d]``, the target ``theta`` (N,), the per-point 1-D
``prior`` (N,), the latent-grid ``sdf`` (G,G,G), plus ``u_value`` and metadata.

With ``--irregular`` the corpus is generated on **rotated, off-lattice** geometry
(see :mod:`thermotwin.data.synthetic_3d_irreg`) — the variant where a regular voxel
grid is a poor fit, so GINO can justify itself over the voxel-FNO baseline. The stored
per-sample fields are identical (the irregular records also carry a ``rotation``
matrix), so the existing dataset loader / benchmark consume them unchanged.

Examples
--------
    python scripts/generate_3d_gt.py --n 96 --seed 1337 --name block2_train
    python scripts/generate_3d_gt.py --n 32 --seed 99 --name block2_val
    python scripts/generate_3d_gt.py --n 8 --grid 12 --npts 1024 --name smoke
    python scripts/generate_3d_gt.py --irregular --n 96 --seed 1337 --name block2_irreg_train
    python scripts/generate_3d_gt.py --irregular --n 32 --seed 99 --name block2_irreg_val
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from thermotwin.data.real_citygml_3d import (  # noqa: E402
    generate_corpus_realcg,
)
from thermotwin.data.synthetic_3d import (  # noqa: E402
    FEATURE_LAYOUT,
    generate_corpus_3d,
    generate_corpus_hard,
)
from thermotwin.data.synthetic_3d_irreg import (  # noqa: E402
    generate_corpus_irregular,
)

PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"

# Scalars stored per-sample in the .npz and summarised in the manifest.
_SCALARS = ("u_value", "u_clear", "heat_flux", "t_indoor", "t_outdoor", "r_si", "r_se", "n_bridges")
# Arrays stored per-sample in the .npz.
_ARRAYS = ("points", "feats", "theta", "prior", "sdf", "grid_shape")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--n", type=int, default=96, help="number of blocks")
    p.add_argument("--seed", type=int, default=1337, help="RNG seed")
    p.add_argument("--name", default="block2_train", help="corpus subdirectory name")
    p.add_argument("--grid", type=int, default=16, help="SDF latent grid resolution G")
    p.add_argument("--npts", type=int, default=2048, help="sampled points per block")
    p.add_argument(
        "--cells-per-layer", type=int, default=3, help="through-wall cells per material layer"
    )
    p.add_argument(
        "--irregular",
        action="store_true",
        help="generate rotated, off-lattice geometry (the voxel-grid-is-a-poor-fit variant)",
    )
    p.add_argument(
        "--hard",
        action="store_true",
        help=(
            "generate fine-native blocks with sub-voxel thermal fins — the corpus where a "
            "16³ voxel grid genuinely aliases the bridge (the diagnosis-driven variant). "
            "Axis-aligned; pair with --npts 4096 and --cells-per-layer 6."
        ),
    )
    p.add_argument(
        "--real-citygml",
        default=None,
        metavar="DIR",
        help=(
            "generate the REAL-geometry corpus from a directory of TUM2TWIN CityGML .gml "
            "buildings (Exp 2.6) — each building -> a whole-shell point cloud at real "
            "orientations with per-surface FV + bridge physics. Overrides --n; use "
            "--n-per-building / --building-start / --building-end."
        ),
    )
    p.add_argument("--n-per-building", type=int, default=6, help="augmented samples per building")
    p.add_argument("--building-start", type=int, default=0, help="building index slice start")
    p.add_argument("--building-end", type=int, default=None, help="building index slice end (exclusive)")
    a = p.parse_args()

    if sum(bool(x) for x in (a.irregular, a.hard, a.real_citygml)) > 1:
        p.error("--irregular / --hard / --real-citygml are mutually exclusive")

    out = PROCESSED / a.name
    out.mkdir(parents=True, exist_ok=True)

    if a.real_citygml:
        print(
            f"generating REAL-geometry corpus from {a.real_citygml} "
            f"(buildings [{a.building_start}:{a.building_end}], {a.n_per_building}/building, "
            f"grid {a.grid}, npts {a.npts}) -> {out}"
        )
        records = generate_corpus_realcg(
            a.real_citygml,
            n_per_building=a.n_per_building,
            seed=a.seed,
            grid=a.grid,
            n_points=a.npts,
            cells_per_layer=a.cells_per_layer,
            building_start=a.building_start,
            building_end=a.building_end,
        )
    else:
        kind = (
            "irregular (rotated, off-grid)"
            if a.irregular
            else "hard (fine-native, sub-voxel fins)"
            if a.hard
            else "axis-aligned box"
        )
        print(
            f"generating {a.n} {kind} 3-D blocks (seed {a.seed}, grid {a.grid}, npts {a.npts}) -> {out}"
        )
        gen = (
            generate_corpus_irregular
            if a.irregular
            else generate_corpus_hard
            if a.hard
            else generate_corpus_3d
        )
        records = gen(
            a.n, seed=a.seed, grid=a.grid, n_points=a.npts, cells_per_layer=a.cells_per_layer
        )
    # The irregular corpus also persists the per-sample rotation matrix.
    arrays = (*_ARRAYS, "rotation") if a.irregular else _ARRAYS
    manifest_rows = []
    for r in records:
        fname = f"sample_{r['id']:05d}.npz"
        np.savez_compressed(
            out / fname,
            **{k: r[k] for k in arrays},
            **{s: r[s] for s in _SCALARS},
        )
        manifest_rows.append(
            {
                "file": fname,
                "n_points": int(r["points"].shape[0]),
                "feat_dim": int(r["feats"].shape[1]),
                "grid": int(r["sdf"].shape[0]),
                "grid_shape": [int(v) for v in r["grid_shape"]],
                **{s: float(r[s]) for s in _SCALARS},
            }
        )

    penalties = [row["u_value"] / row["u_clear"] - 1.0 for row in manifest_rows]
    manifest = {
        "generator": (
            "thermotwin.data.real_citygml_3d.generate_corpus_realcg"
            if a.real_citygml
            else "thermotwin.data.synthetic_3d_irreg.generate_corpus_irregular"
            if a.irregular
            else "thermotwin.data.synthetic_3d.generate_corpus_hard"
            if a.hard
            else "thermotwin.data.synthetic_3d.generate_corpus_3d"
        ),
        "irregular": bool(a.irregular),
        "hard": bool(a.hard),
        "real_citygml": a.real_citygml or False,
        "seed": a.seed,
        "n_samples": len(manifest_rows),
        "grid": a.grid,
        "n_points": a.npts,
        "cells_per_layer": a.cells_per_layer,
        "feature_layout": list(FEATURE_LAYOUT),
        "fields": {
            "arrays": list(_ARRAYS),
            "scalars": list(_SCALARS),
            "target": "theta",
        },
        "u_bridge_penalty": {
            "mean": float(np.mean(penalties)),
            "max": float(np.max(penalties)),
            "frac_with_bridges": float(np.mean([row["n_bridges"] > 0 for row in manifest_rows])),
        },
        "samples": manifest_rows,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    pen = manifest["u_bridge_penalty"]
    print(f"done: {len(records)} samples in {out}")
    print(
        f"  thermal-bridge U-penalty: mean {pen['mean'] * 100:.1f}%  "
        f"max {pen['max'] * 100:.1f}%  "
        f"({pen['frac_with_bridges'] * 100:.0f}% of blocks have bridges)"
    )


if __name__ == "__main__":
    main()
