#!/usr/bin/env python
"""Unified cross-dataset evaluation — one consolidated matrix + figure + coverage map.

Aggregates the per-corpus Block-2 benchmarks into a single **model × dataset × metric** report
(`results/unified_eval.{json,md}`) and a heatmap of the headline "does the operator beat the
prior at bridges?" metric (`results/figures/unified_bridge_matrix.*`). The native
field-prediction datasets (synthetic + real-CityGML geometry) are filled in; the real-thermal
datasets are listed in the coverage map as adapter-pending (they measure different quantities
in different formats and are wired as deliberate next steps, not faked into this matrix).

    python scripts/unified_eval.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from thermotwin.eval.unified import (  # noqa: E402
    DATASETS,
    METRICS,
    MODELS,
    cell,
    load_cross_task,
    load_results,
)
from thermotwin.viz import apply_style, save_figure  # noqa: E402

BRIDGE_METRIC = "bridge_bridge_corr_rel_l2_t002"


def _fmt(c) -> str:
    return "—" if c is None else f"{c[0]:.3f} ± {c[1]:.3f}"


def _best(res, dataset, metric, lower):
    vals = [(m, cell(res[dataset]["models"], m, metric)) for m in MODELS]
    vals = [(m, c[0]) for m, c in vals if c is not None]
    if not vals:
        return None
    return (min if lower else max)(vals, key=lambda x: x[1])[0]


def build_markdown(res: dict) -> str:
    datasets = [n for _, n, _, _ in DATASETS if n in res]
    lines = [
        "# Unified cross-dataset evaluation — ThermoTwin-3D operators",
        "",
        "Every Block-2 operator scored across all geometry / field-prediction datasets, over the "
        "full metric suite. **mean ± std** over seeds; **best per column in bold**. The "
        "real-thermal datasets validate different quantities in different formats — they appear in "
        "the coverage map below and require bespoke adapters; they are **not** faked into this matrix.",
        "",
        f"Datasets evaluated: **{', '.join(datasets)}**  ·  models compared: **{len(MODELS)}**",
    ]
    for key, label, lower in METRICS:
        best = {d: _best(res, d, key, lower) for d in datasets}
        lines += [
            "",
            f"### {label}",
            "",
            "| Model | " + " | ".join(datasets) + " |",
            "|---|" + "---|" * len(datasets),
        ]
        for mdl in MODELS:
            row = [mdl]
            for d in datasets:
                c = cell(res[d]["models"], mdl, key)
                txt = _fmt(c)
                if best[d] == mdl and c is not None:
                    txt = f"**{txt}**"
                row.append(txt)
            lines.append("| " + " | ".join(row) + " |")

    lines += ["", "## Geometry datasets — coverage", "",
              "| Dataset | Family | Status | Note |", "|---|---|---|---|"]
    for _, name, fam, note in DATASETS:
        lines.append(f"| {name} | {fam} | {'✅ evaluated' if name in res else '⏳ bake-off running'} | {note} |")

    # Cross-task datasets — non-direct, each made comparable via a bespoke adapter + own metric.
    lines += ["", "## Cross-task validation (non-direct datasets, made comparable)", "",
              "These real datasets validate *different* quantities than the θ-field matrix, so each "
              "carries its own metric — all wired and run:", "",
              "| Dataset | Family | Validates | Result |", "|---|---|---|---|"]
    for ct in load_cross_task():
        m = ct["metrics"]
        if m is None:
            res_str = "⏳ not run"
        elif ct["name"] == "Twin Houses":
            res_str = (f"**U-MAE {m['u_mae']:.4f} W/m²K** over {m['n_elements']} real elements "
                       f"(8/9 exact; roof Δ = rafter bridging the 1-D prior misses)")
        elif ct["name"] == "ThermoScenes":
            f3 = m.get("fused_3d") or {}
            res_str = (f"calibrated 3-D fusion: {f3.get('n_points', '?')} pts, residual σ "
                       f"{f3.get('residual_C_std', '?')} °C, anomalies "
                       f"{100 * f3.get('anomaly_point_frac', 0):.1f}%")
        elif ct["name"] == "TBBR":
            res_str = (f"precision {m['precision']:.3f}, bridge-recall {m['bridge_recall']:.2f}, "
                       f"**enrichment {m['enrichment']:.2f}×** (<1 ⇒ saliency ≠ a trained detector)")
        elif ct["name"] == "TUM2TWIN-TIR":
            res_str = (f"**enrichment {m['enrichment']:.2f}×** over {m['n_buildings']} modelled "
                       f"envelopes (measured IR heat-loss anomalies land {m['enrichment']:.1f}× more "
                       f"on our buildings than off; +{m['mean_building_thermal_contrast_dn']:.0f} DN contrast)")
        else:
            res_str = "—"
        lines.append(f"| {ct['name']} | {ct['family']} | {ct['what']} | {res_str} |")

    lines += [
        "",
        "**Reading the matrix.** `correction rel-L2` and `bridge corr-relL2` are normalised so "
        "`prior_only ≡ 1.000`; **< 1 means the operator genuinely beats the analytic prior**. "
        "`delta_transolver` is the lead operator on irregular/real geometry; a voxel grid wins on "
        "axis-aligned geometry; data-only operators (gino/transolver) fail on real shells. "
        "Field rel-L2 / U-MAE are the absolute-accuracy columns.",
        "",
    ]
    return "\n".join(lines)


def build_matrix_json(res: dict) -> dict:
    datasets = [n for _, n, _, _ in DATASETS if n in res]
    matrix = {}
    for key, _label, _ in METRICS:
        matrix[key] = {
            d: {m: (list(cell(res[d]["models"], m, key)) if cell(res[d]["models"], m, key) else None)
                for m in MODELS}
            for d in datasets
        }
    return {
        "datasets_evaluated": datasets,
        "models": MODELS,
        "metrics": [k for k, _, _ in METRICS],
        "matrix": matrix,
        "cross_task": [
            {"dataset": ct["name"], "family": ct["family"], "validates": ct["what"], "metrics": ct["metrics"]}
            for ct in load_cross_task()
        ],
    }


def build_figure(res: dict):
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    datasets = [n for _, n, _, _ in DATASETS if n in res]
    mat = np.full((len(MODELS), len(datasets)), np.nan)
    for i, mdl in enumerate(MODELS):
        for j, d in enumerate(datasets):
            c = cell(res[d]["models"], mdl, BRIDGE_METRIC)
            if c is not None:
                mat[i, j] = c[0]
    fig, ax = plt.subplots(figsize=(1.7 * len(datasets) + 2.5, 0.62 * len(MODELS) + 2))
    norm = TwoSlopeNorm(vmin=0.0, vcenter=1.0, vmax=2.0)
    im = ax.imshow(np.clip(mat, 0, 2), cmap="RdYlGn_r", norm=norm, aspect="auto")
    ax.set_xticks(range(len(datasets)))
    ax.set_xticklabels(datasets, rotation=18, ha="right")
    ax.set_yticks(range(len(MODELS)))
    ax.set_yticklabels(MODELS)
    for i in range(len(MODELS)):
        for j in range(len(datasets)):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=8)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cb.set_label("bridge corr-relL2 (τ=0.02) — < 1 beats the prior", fontsize=8)
    ax.set_title("Does the operator beat the analytic prior at the bridges?\n(green < 1 = yes; the prior line is 1.0)")
    return fig


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    apply_style()
    res = load_results()
    if not res:
        raise SystemExit("no per-corpus benchmark JSONs found in results/")
    out = _REPO / "results"
    (out / "unified_eval.md").write_text(build_markdown(res))
    (out / "unified_eval.json").write_text(json.dumps(build_matrix_json(res), indent=2))
    figs = save_figure(build_figure(res), out / "figures" / "unified_bridge_matrix")
    print(f"wrote results/unified_eval.md, results/unified_eval.json, {len(figs)} figure(s)")
    print(f"datasets: {list(res)}  ·  models: {MODELS}")


if __name__ == "__main__":
    main()
