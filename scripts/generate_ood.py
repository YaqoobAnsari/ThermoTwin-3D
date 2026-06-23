#!/usr/bin/env python
"""Generate the Block-1 out-of-distribution (OOD) test corpora.

These are held-out generalisation sets: the same solved-2-D-wall ``.npz`` schema and
``manifest.json`` as the in-distribution corpora (``block1_train`` / ``block1_val``),
but drawn from deliberately shifted distributions — unseen wall assemblies, unseen
surface films, harder bridge configurations, and a different native resolution. See
:mod:`thermotwin.data.ood` for the precise shifts and the rationale.

Examples
--------
    python scripts/generate_ood.py                       # all 4 corpora, 64 each
    python scripts/generate_ood.py --n 64                # explicit size
    python scripts/generate_ood.py --only ood_walls      # one corpus
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from thermotwin.data.ood import OOD_GENERATORS, generate_ood_corpus  # noqa: E402

PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"

# Scalar fields stored both per-sample (in the .npz) and in the manifest summary —
# identical to scripts/generate_fem_groundtruth.py so loaders read OOD sets unchanged.
_SCALARS = ("t_indoor", "t_outdoor", "r_si", "r_se", "u_value", "u_clear", "heat_flux", "n_bridges")

# Distinct per-corpus seeds (each independent of the others and of the train/val seeds).
_SEEDS = {"ood_walls": 4001, "ood_films": 4002, "ood_bridges": 4003, "ood_res": 4004}


def _write_corpus(name: str, n: int, seed: int) -> dict:
    out = PROCESSED / name
    out.mkdir(parents=True, exist_ok=True)
    print(f"generating {n} samples (seed {seed}) -> {out}")

    records = generate_ood_corpus(name, n, seed=seed)
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

    penalties = [row["u_value"] / row["u_clear"] - 1.0 for row in manifest_rows]
    manifest = {
        "generator": "thermotwin.data.ood.generate_ood_corpus",
        "corpus": name,
        "ood": True,
        "seed": seed,
        "n_samples": n,
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
    print(
        f"  done: {n} samples in {out}\n"
        f"  thermal-bridge U-penalty: mean {pen['mean'] * 100:.1f}%  "
        f"max {pen['max'] * 100:.1f}%  "
        f"({pen['frac_with_bridges'] * 100:.0f}% have bridges)"
    )
    return manifest


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--n", type=int, default=64, help="samples per corpus")
    p.add_argument(
        "--only",
        choices=sorted(OOD_GENERATORS),
        help="generate only this corpus (default: all four)",
    )
    a = p.parse_args()

    names = [a.only] if a.only else list(OOD_GENERATORS)
    for name in names:
        _write_corpus(name, a.n, _SEEDS[name])


if __name__ == "__main__":
    main()
