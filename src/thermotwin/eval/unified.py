"""Unified cross-dataset evaluation — one matrix from the per-corpus benchmarks.

The operators are scored per corpus by ``scripts/benchmark_block2.py`` (all models, the full
metric suite). This module assembles those per-corpus results into ONE consolidated
**model × dataset × metric** matrix, and carries an honest **coverage map** of the datasets
that still need bespoke adapters (the real-thermal rungs live in different formats / measure
different quantities and are wired as deliberate next steps, not faked into this matrix).

Design: a unified harness has (1) a model registry — already the Block-2 roster; (2) a dataset
registry with adapters — geometry/field-prediction datasets are native (point cloud → θ field),
the thermal datasets each get their own adapter + metric; (3) a metric suite; (4) one report.
This file covers (1)+(3)+(4) for the native datasets and *declares* (2) for the rest.
"""

from __future__ import annotations

import json
from pathlib import Path

_RESULTS = Path(__file__).resolve().parents[3] / "results"

__all__ = ["DATASETS", "MODELS", "METRICS", "CROSS_TASK", "load_results", "load_cross_task", "cell"]

# Geometry / field-prediction datasets we already have full results for.
# (json stem, display name, family, note)
DATASETS = [
    ("block2_benchmark", "synthetic-box", "synthetic geometry", "axis-aligned box (legacy 4-model run)"),
    ("block2_irreg_ops_benchmark", "synthetic-irregular", "synthetic geometry", "rotated / off-lattice"),
    ("block2_hard_benchmark", "synthetic-hard", "synthetic geometry", "sub-voxel thermal fins"),
    ("block2_realcg_benchmark", "real-CityGML", "real geometry", "TUM2TWIN LoD2 shells, sim. physics"),
    ("block2_bag_benchmark", "real-3DBAG", "real geometry", "3D BAG Amsterdam LoD2.2 shells, sim. physics"),
    ("block2_doe_benchmark", "DOE-refbldg", "real constructions", "DOE Reference Buildings (real materials, idealised geometry)"),
]

MODELS = ["delta_transolver", "transolver", "delta_gino", "gino", "fno_voxel", "prior_only"]

# (metric key, label, lower-is-better)
METRICS = [
    ("field_rel_l2", "field rel-L2 ↓", True),
    ("u_mae", "U-MAE [W/m²K] ↓", True),
    ("bridge_correction_rel_l2", "correction rel-L2 (vs prior) ↓", True),
    ("bridge_bridge_corr_rel_l2_t002", "bridge corr-relL2 (τ=0.02) ↓", True),
    ("bridge_correction_r2", "correction R² ↑", False),
    ("infer_ms_per_sample", "infer ms ↓", True),
]

# Non-direct datasets, each made comparable via a bespoke adapter with its OWN metric (they
# validate different quantities in different formats, so they can't share the θ-field matrix).
# (name, family, summary.json, what it validates, metric keys to surface)
CROSS_TASK = [
    ("Twin Houses", "real measured U", "results/twin_houses/summary.json",
     "per-element U vs documented (real assemblies)", ["u_mae", "u_max_error", "n_elements"]),
    ("ThermoScenes", "real calibrated thermal", "results/thermoscenes/summary.json",
     "calibrated-°C heat-loss localisation (3-D fused)", ["fused_3d"]),
    ("TBBR", "real bridge detection", "results/tbbr/summary.json",
     "heat-loss saliency vs annotated bridges", ["precision", "bridge_recall", "enrichment"]),
]


def load_cross_task() -> list[dict]:
    """Load each cross-task adapter's summary.json (skipping any not yet run)."""
    out = []
    for name, fam, path, what, keys in CROSS_TASK:
        p = _RESULTS.parent / path
        row = {"name": name, "family": fam, "what": what, "metrics": None}
        if p.exists():
            row["metrics"] = json.loads(p.read_text())
            row["keys"] = keys
        out.append(row)
    return out


def load_results() -> dict:
    """Load the per-corpus benchmark JSONs into ``{display_name: {...}}``."""
    out: dict[str, dict] = {}
    for stem, name, fam, note in DATASETS:
        p = _RESULTS / f"{stem}.json"
        if not p.exists():
            continue
        r = json.loads(p.read_text())
        out[name] = {
            "family": fam,
            "note": note,
            "seeds": r["results"][0].get("seeds"),
            "models": {m["model"]: m for m in r["results"]},
        }
    return out


def cell(by_model: dict, model: str, metric: str) -> tuple[float, float] | None:
    """``(mean, std)`` for one model/metric, or ``None`` if absent."""
    m = by_model.get(model)
    if not m:
        return None
    mean = m.get(f"{metric}_mean")
    if mean is None:
        return None
    return float(mean), float(m.get(f"{metric}_std", 0.0))
