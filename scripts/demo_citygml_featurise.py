#!/usr/bin/env python
"""Real-geometry featurisation demo: a TUM2TWIN LoD2 building -> point cloud + SDF.

Proves the **real-geometry path is benchmark-ready**: it loads one as-built TUM2TWIN
CityGML building (:func:`thermotwin.geometry.citygml.read_citygml_building`), lifts it
into an :class:`~thermotwin.geometry.envelope.Envelope`, then runs the exact two
featurisers the Block-2 operator consumes —

* the surface point cloud (:func:`thermotwin.geometry.pointcloud.envelope_point_cloud`):
  area-weighted points carrying per-point ``[u_value, resistance, surface_type_id]``;
* the latent-grid SDF (:func:`thermotwin.geometry.sdf.sdf_grid` over the watertight
  shell mesh) — the same "where is the solid" encoding GINO conditions on.

and reports stats (n points, bounding box, per-surface-type U-values, SDF sign split).
Measured thermal ground truth is gated (IR calibration is a later block), so this is a
*featurisation* demo, not a training corpus — but it shows a real building flows through
the identical input pipeline as the synthetic Block-2 corpus.

Example
-------
    python scripts/demo_citygml_featurise.py \
        --gml data/raw/tum2twin-datasets/citygml/lod2-building-datasets/DEBY_LOD2_4906965.gml \
        --npoints 4096 --grid 24
If ``--gml`` is omitted, the first ``*.gml`` in the default TUM2TWIN LoD2 directory is
used. ``--json`` prints a machine-readable summary instead of the human table.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from thermotwin.geometry.citygml import read_citygml_building  # noqa: E402
from thermotwin.geometry.pointcloud import envelope_point_cloud  # noqa: E402
from thermotwin.geometry.sdf import envelope_to_mesh, sdf_grid  # noqa: E402

_DEFAULT_DIR = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "raw"
    / "tum2twin-datasets"
    / "citygml"
    / "lod2-building-datasets"
)


def featurise(gml: Path, npoints: int, grid: int, seed: int) -> dict:
    """Load + featurise one CityGML building; return a summary dict."""
    env = read_citygml_building(gml)

    # Per-surface-type U-values (W/m^2K) and total area.
    by_type_u: dict[str, float] = {}
    area_by_type: dict[str, float] = defaultdict(float)
    for s in env.exterior_opaque_surfaces():
        by_type_u[s.surface_type] = env.surface_u_value(s)
        area_by_type[s.surface_type] += float(s.area)

    cloud = envelope_point_cloud(env, n_points=npoints, exterior_only=True, seed=seed)
    bbox_lo = cloud.points.min(axis=0)
    bbox_hi = cloud.points.max(axis=0)

    # Watertight shell SDF on a regular latent grid (the GINO geometry encoding).
    mesh = envelope_to_mesh(env, mode="shell", repair=True)
    sdf, axes = sdf_grid(mesh, resolution=grid, padding=0.1)
    frac_inside = float(np.mean(sdf < 0.0))

    return {
        "file": gml.name,
        "n_surfaces_total": len(env.surfaces),
        "n_surfaces_exterior_opaque": len(env.exterior_opaque_surfaces()),
        "n_points": int(len(cloud)),
        "feature_names": list(cloud.feature_names),
        "bbox_lo_m": [round(float(v), 3) for v in bbox_lo],
        "bbox_hi_m": [round(float(v), 3) for v in bbox_hi],
        "extent_m": [round(float(h - lo), 3) for lo, h in zip(bbox_lo, bbox_hi, strict=True)],
        "u_values_w_m2k": {k: round(float(v), 4) for k, v in sorted(by_type_u.items())},
        "area_m2_by_type": {k: round(float(v), 1) for k, v in sorted(area_by_type.items())},
        "envelope_ua_w_k": round(float(env.envelope_ua()), 1),
        "sdf_grid": int(grid),
        "sdf_frac_inside": round(frac_inside, 4),
        "watertight": bool(mesh.is_watertight),
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--gml", type=Path, default=None, help="path to a CityGML building (.gml)")
    p.add_argument("--npoints", type=int, default=4096, help="point-cloud sample size")
    p.add_argument("--grid", type=int, default=24, help="SDF latent grid resolution")
    p.add_argument("--seed", type=int, default=1337, help="point-cloud RNG seed")
    p.add_argument("--json", action="store_true", help="print a JSON summary")
    a = p.parse_args()

    gml = a.gml
    if gml is None:
        candidates = sorted(_DEFAULT_DIR.glob("*.gml"))
        if not candidates:
            raise SystemExit(f"no CityGML files in {_DEFAULT_DIR}; pass --gml")
        gml = candidates[0]
    if not gml.exists():
        raise SystemExit(f"file not found: {gml}")

    summary = featurise(gml, a.npoints, a.grid, a.seed)

    if a.json:
        print(json.dumps(summary, indent=2))
        return

    print(f"CityGML featurisation demo: {summary['file']}")
    print(
        f"  surfaces: {summary['n_surfaces_total']} total, "
        f"{summary['n_surfaces_exterior_opaque']} exterior-opaque (sampled)"
    )
    print(f"  point cloud: {summary['n_points']} points, feats {summary['feature_names']}")
    print(
        f"  bbox (local-metric): lo {summary['bbox_lo_m']}  hi {summary['bbox_hi_m']}  "
        f"extent {summary['extent_m']} m"
    )
    print("  per-surface-type U [W/m^2K] (area m^2):")
    for st, u in summary["u_values_w_m2k"].items():
        print(f"    {st:6s}: U = {u:.4f}   area = {summary['area_m2_by_type'].get(st, 0.0)}")
    print(f"  envelope UA = {summary['envelope_ua_w_k']} W/K")
    print(
        f"  SDF: {summary['sdf_grid']}^3 grid, "
        f"{summary['sdf_frac_inside'] * 100:.1f}% cells inside the shell "
        f"(watertight={summary['watertight']})"
    )


if __name__ == "__main__":
    main()
