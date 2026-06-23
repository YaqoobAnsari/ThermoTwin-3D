#!/usr/bin/env python
"""Generate the Block-1 synthetic FEM heat-conduction ground-truth corpus.

No off-the-shelf "envelope heat-conduction field on building geometry" dataset
exists, so we generate our own (a release asset). This writes solved 2-D wall
cross-sections — layered constructions punctured by thermal bridges — into
``data/processed/<name>/`` as per-sample ``.npz`` files plus a ``manifest.json``
that makes the corpus reproducible (generator, seed, per-sample scalars).

Examples
--------
    python scripts/generate_fem_groundtruth.py                 # 256 samples, seed 1337
    python scripts/generate_fem_groundtruth.py --n 1024 --name block1_train
    python scripts/generate_fem_groundtruth.py --n 128 --seed 7 --name block1_val
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from thermotwin.data.synthetic_fem import generate_corpus  # noqa: E402

PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"

# Scalar fields stored both per-sample (in the .npz) and in the manifest summary.
_SCALARS = ("t_indoor", "t_outdoor", "r_si", "r_se", "u_value", "u_clear", "heat_flux", "n_bridges")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--n", type=int, default=256, help="number of samples")
    p.add_argument("--seed", type=int, default=1337, help="RNG seed")
    p.add_argument("--name", default="block1_synthetic", help="corpus subdirectory name")
    a = p.parse_args()

    out = PROCESSED / a.name
    out.mkdir(parents=True, exist_ok=True)
    print(f"generating {a.n} samples (seed {a.seed}) -> {out}")

    records = generate_corpus(a.n, seed=a.seed)
    manifest_rows = []
    for r in records:
        fname = f"sample_{r['id']:05d}.npz"
        np.savez_compressed(
            out / fname,
            k=r["k"],
            temperature=r["temperature"],
            dx0=r["dx0"],
            dy=r["dy"],
            **{s: r[s] for s in _SCALARS},
        )
        manifest_rows.append(
            {"file": fname, "shape": list(r["k"].shape), **{s: float(r[s]) for s in _SCALARS}}
        )

    # A bridged sample's effective U exceeds the clear-wall U; report the spread.
    penalties = [row["u_value"] / row["u_clear"] - 1.0 for row in manifest_rows]
    manifest = {
        "generator": "thermotwin.data.synthetic_fem.generate_corpus",
        "seed": a.seed,
        "n_samples": a.n,
        "fields": {
            "inputs": ["k", "dx0", "dy", "t_indoor", "t_outdoor", "r_si", "r_se"],
            "target": "temperature",
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
    print(f"done: {a.n} samples in {out}")
    print(
        f"  thermal-bridge U-penalty: mean {pen['mean'] * 100:.1f}%  "
        f"max {pen['max'] * 100:.1f}%  "
        f"({pen['frac_with_bridges'] * 100:.0f}% of samples have bridges)"
    )


if __name__ == "__main__":
    main()
