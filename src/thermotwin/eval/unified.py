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

__all__ = ["DATASETS", "MODELS", "METRICS", "PLANNED_ADAPTERS", "load_results", "cell"]

# Geometry / field-prediction datasets we already have full results for.
# (json stem, display name, family, note)
DATASETS = [
    ("block2_benchmark", "synthetic-box", "synthetic geometry", "axis-aligned box (legacy 4-model run)"),
    ("block2_irreg_ops_benchmark", "synthetic-irregular", "synthetic geometry", "rotated / off-lattice"),
    ("block2_hard_benchmark", "synthetic-hard", "synthetic geometry", "sub-voxel thermal fins"),
    ("block2_realcg_benchmark", "real-CityGML", "real geometry", "TUM2TWIN LoD2 shells, sim. physics"),
    ("block2_bag_benchmark", "real-3DBAG", "real geometry", "3D BAG Amsterdam LoD2.2 shells, sim. physics"),
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

# Real-thermal / extra datasets that need a bespoke adapter (the honest coverage map). Each
# validates a different quantity in a different format — they are NOT in this matrix yet.
# (name, family, what it is, adapter + metric it will contribute)
PLANNED_ADAPTERS = [
    ("ThermoScenes", "real CALIBRATED thermal", "8 facades, absolute °C + COLMAP geometry",
     "multi-view thermal-fusion adapter → surface-°C RMSE / pattern correlation"),
    ("Twin Houses", "real measured U / heat flux", "2 houses, point sensors + drawings",
     "geometry-encode + XLSX adapter → per-element U-value / heat-flux error"),
    ("TBBR", "real bridge localization", "926 UAV scenes, 6 927 annotations",
     "2D saliency/detection adapter → precision / recall / AP vs annotated bridges"),
]


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
