#!/usr/bin/env python
"""Twin Houses U-value validation — our conduction physics vs real documented U-values.

Parses the IEA Annex-71 Twin Houses construction build-ups, computes each element's U with our
physics (``1/(Rsi+Σ d/λ+Rse)``), compares to the building's documented U, and writes
``results/twin_houses/summary.json`` + a predicted-vs-documented figure. The comparable metric
for the unified eval's Twin Houses rung is **per-element U-value MAE**.

    python scripts/eval_twin_houses.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from thermotwin.data.twin_houses import evaluate  # noqa: E402
from thermotwin.viz import apply_style, save_figure  # noqa: E402

XLSX = _REPO / "data" / "raw" / "twin_houses" / "01_Constructions_TwinHouses.xlsx"


def _figure(res: dict):
    import matplotlib.pyplot as plt

    rows = res["elements"]
    ours = [r["u_computed"] for r in rows]
    doc = [r["u_documented"] for r in rows]
    names = [r["element"] for r in rows]
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    lim = max(max(ours), max(doc)) * 1.08
    ax.plot([0, lim], [0, lim], "k--", lw=0.8, label="y = x (exact)")
    ax.scatter(doc, ours, s=45, c="#b2182b", zorder=3)
    for n, x, y in zip(names, doc, ours, strict=True):
        ax.annotate(n, (x, y), fontsize=7, xytext=(4, 2), textcoords="offset points")
    ax.set_xlabel("documented U  [W/(m²K)]")
    ax.set_ylabel("our physics U  [W/(m²K)]")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_title(f"Twin Houses — U-value validation\nU-MAE {res['u_mae']:.4f} W/(m²K) over {res['n_elements']} real elements")
    ax.legend()
    return fig


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--xlsx", default=str(XLSX))
    a = p.parse_args()
    apply_style()
    out = _REPO / "results" / "twin_houses"
    out.mkdir(parents=True, exist_ok=True)
    res = evaluate(a.xlsx)
    (out / "summary.json").write_text(json.dumps(res, indent=2))
    save_figure(_figure(res), out / "u_value_validation")
    print(f"U-MAE {res['u_mae']} W/m²K · max {res['u_max_error']} · {res['n_elements']} elements "
          f"-> {out}/summary.json + figure")


if __name__ == "__main__":
    main()
