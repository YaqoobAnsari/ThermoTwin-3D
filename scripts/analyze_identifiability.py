#!/usr/bin/env python
"""Identifiability analysis of the inverse twin (Exp 3).

Surface IR under-determines the interior conductivity field. We can *assert* that integrals
(U, the bridge conductance Psi) are recoverable while the full point-level field is not, or we
can *measure* it. `benchmark_inverse.py --identifiability` stores, per validation block, the
recovered conductivity field (ensemble mean), its per-point ensemble spread, the true field,
the through-wall coordinate, and the distance to the nearest true bridge, plus an
observation-masking sweep and a multi-band U decomposition. This script turns those into the
quantified, practitioner-facing result:

  1. recovery error vs through-wall depth  -> the field is identifiable near the observed
     surface, not in the interior;
  2. recovery error vs distance-to-bridge   -> error concentrates at the bridge cores;
  3. predicted spread vs actual error        -> the UQ tracks the non-uniqueness;
  4. U recovered across face-slab widths      -> the integral is stable and accurate even though
     the point field is not (the integration-scale decomposition);
  5/6. observation masking (surface / interior / full) -> what seeing the interior buys for
     field recovery and bridge localisation, while U stays recoverable from the surface alone.

    python scripts/analyze_identifiability.py --corpus hard
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))


def _binned(x: np.ndarray, y: np.ndarray, edges: np.ndarray):
    """Mean and standard error of ``y`` in bins of ``x`` defined by ``edges``."""
    centres = 0.5 * (edges[:-1] + edges[1:])
    mean = np.full(centres.shape, np.nan)
    sem = np.full(centres.shape, np.nan)
    for i in range(len(centres)):
        m = (x >= edges[i]) & (x < edges[i + 1])
        if m.sum() >= 5:
            yi = y[m]
            mean[i] = float(yi.mean())
            sem[i] = float(yi.std() / np.sqrt(yi.size))
    return centres, mean, sem


def _per_sample_field_rel_l2(sample_id, logk_hat, logk_true):
    """Field relative-L2 per block, then averaged (the point-level recovery the integrals beat)."""
    vals = []
    for s in np.unique(sample_id):
        m = sample_id == s
        num = float(np.linalg.norm(logk_hat[m] - logk_true[m]))
        den = float(np.linalg.norm(logk_true[m])) + 1e-9
        vals.append(num / den)
    return float(np.mean(vals)), float(np.std(vals))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", default="hard")
    a = p.parse_args()

    npz_path = _REPO / "results" / f"inverse_fields_{a.corpus}.npz"
    json_path = _REPO / "results" / f"inverse_{a.corpus}.json"
    if not npz_path.exists():
        raise SystemExit(f"missing {npz_path} -- run benchmark_inverse.py --corpus {a.corpus} first")
    d = np.load(npz_path)
    sid = d["sample_id"]
    logk_true, logk_hat = d["logk_true"], d["logk_hat"]
    std = d["logk_std"]
    x = d["x_throughwall"]
    dist = d["dist_to_bridge"]
    err = np.abs(logk_hat - logk_true)

    bench = json.loads(json_path.read_text()) if json_path.exists() else {}
    ident = bench.get("identifiability", {})

    # (1) recovery error vs through-wall depth
    xe = np.linspace(0.0, 1.0, 13)
    xc, x_err, x_sem = _binned(x, err, xe)

    # (2) recovery error vs distance to the nearest true bridge (drop blocks with no bridge)
    finite = np.isfinite(dist)
    if finite.any():
        de = np.linspace(0.0, float(np.nanpercentile(dist[finite], 95)), 11)
        dc, d_err, d_sem = _binned(dist[finite], err[finite], de)
    else:
        dc = d_err = d_sem = np.array([])

    # (3) predicted spread vs actual error
    s = std.clip(min=1e-8)
    pear = float(np.corrcoef(s, err)[0, 1])
    # Spearman = Pearson on the rank-transformed variables.
    rank_s = np.argsort(np.argsort(s)) / s.size
    rank_e = np.argsort(np.argsort(err)) / err.size
    spear = float(np.corrcoef(rank_s, rank_e)[0, 1])
    se = np.linspace(float(s.min()), float(np.percentile(s, 98)), 11)
    sc, s_err, _ = _binned(s, err, se)

    # (4) point field error vs integrated U recovery
    field_rel_l2, field_rel_l2_std = _per_sample_field_rel_l2(sid, logk_hat, logk_true)
    ubands = ident.get("u_bands", {})

    summary = {
        "corpus": a.corpus,
        "n_points": int(err.size),
        "n_blocks": int(np.unique(sid).size),
        "point_field_rel_l2_mean": field_rel_l2,
        "point_field_rel_l2_std": field_rel_l2_std,
        "err_vs_depth": {"x": xc.tolist(), "err": np.nan_to_num(x_err, nan=0.0).tolist()},
        "err_surface_mean": float(err[x < 0.08].mean()) if (x < 0.08).any() else float("nan"),
        "err_interior_mean": float(err[(x > 0.2) & (x < 0.8)].mean()) if ((x > 0.2) & (x < 0.8)).any() else float("nan"),
        "err_vs_dist_to_bridge": {"d": dc.tolist(), "err": np.nan_to_num(d_err, nan=0.0).tolist()},
        "uq_err_std_pearson": pear,
        "uq_err_std_spearman": spear,
        "u_bands": ubands,
        "observation": ident.get("observation", {}),
        "bridge_focused": ident.get("bridge_focused", {}),
    }
    out_json = _REPO / "results" / f"identifiability_{a.corpus}.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"point-level field rel-L2 = {field_rel_l2:.3f} +/- {field_rel_l2_std:.3f}")
    print(f"err surface vs interior  = {summary['err_surface_mean']:.3f} vs {summary['err_interior_mean']:.3f}")
    print(f"UQ |err|-sigma corr      = pearson {pear:.2f}, spearman {spear:.2f}")
    if ubands:
        print("U-MAE by face-band       = " + "  ".join(f"{k.split('_b')[-1]}:{v:.4f}"
              for k, v in ubands.items() if k.startswith("u_mae")))
    obs = summary["observation"]
    for mt in ("full", "surface", "interior"):
        if obs.get(mt):
            o = obs[mt]
            print(f"  obs={mt:8s} logk_relL2={o.get('logk_rel_l2_mean', float('nan')):.3f} "
                  f"bridgeIoU={o.get('bridge_iou_mean', float('nan')):.2f} U-MAE={o.get('u_mae', float('nan')):.4f}")
    print(f"wrote {out_json}")

    # ---- figure ---------------------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(2, 3, figsize=(13.5, 7.5))

        # 1. error vs depth
        ok = np.isfinite(x_err)
        ax[0, 0].plot(xc[ok], x_err[ok], "o-", color="C3")
        ax[0, 0].fill_between(xc[ok], (x_err - x_sem)[ok], (x_err + x_sem)[ok], color="C3", alpha=0.2)
        ax[0, 0].axvspan(0, 0.08, color="C0", alpha=0.15, label="observed surface")
        ax[0, 0].set_xlabel("through-wall position x  (0 = indoor face)")
        ax[0, 0].set_ylabel("recovery error |delta log k|")
        ax[0, 0].set_title("(1) field identifiable near surface, not interior", fontsize=10)
        ax[0, 0].legend(fontsize=8)

        # 2. error vs distance to bridge
        if dc.size:
            ok = np.isfinite(d_err)
            ax[0, 1].plot(dc[ok], d_err[ok], "o-", color="C4")
            ax[0, 1].fill_between(dc[ok], (d_err - d_sem)[ok], (d_err + d_sem)[ok], color="C4", alpha=0.2)
        ax[0, 1].set_xlabel("distance to nearest true bridge")
        ax[0, 1].set_ylabel("recovery error |delta log k|")
        ax[0, 1].set_title("(2) recovery error vs distance to bridge", fontsize=10)

        # 3. UQ: spread vs error (binned + scatter subsample)
        ss = slice(None, None, max(1, s.size // 4000))
        ax[0, 2].scatter(s[ss], err[ss], s=3, alpha=0.15, color="gray")
        ok = np.isfinite(s_err)
        ax[0, 2].plot(sc[ok], s_err[ok], "o-", color="C2", label="binned mean")
        ax[0, 2].set_xlabel("predicted spread (ensemble sigma)")
        ax[0, 2].set_ylabel("actual error |delta log k|")
        ax[0, 2].set_title(f"(3) UQ tracks error  (spearman {spear:.2f})", fontsize=10)
        ax[0, 2].legend(fontsize=8)

        # 4. integration-scale: U-MAE by band + point-field-error annotation
        bands = [k for k in ubands if k.startswith("u_mae")]
        if bands:
            vals = [ubands[k] for k in bands]
            labs = [k.split("_b")[-1] for k in bands]
            ax[1, 0].bar(labs, vals, color="C0")
            ax[1, 0].set_xlabel("indoor-face slab width (integration scale)")
            ax[1, 0].set_ylabel("U-MAE  [W/(m^2 K)]")
        ax[1, 0].set_title(f"(4) integral recovers (U-MAE small)\npoint field does not (rel-L2 {field_rel_l2:.2f})",
                           fontsize=10)

        # 5/6. observation masking
        masks = [m for m in ("surface", "interior", "full") if obs.get(m)]
        if masks:
            rel = [obs[m].get("logk_rel_l2_mean", np.nan) for m in masks]
            iou = [obs[m].get("bridge_iou_mean", np.nan) for m in masks]
            umae = [obs[m].get("u_mae", np.nan) for m in masks]
            ax[1, 1].bar(masks, rel, color="C3")
            ax[1, 1].set_ylabel("field rel-L2")
            ax[1, 1].set_title("(5) field recovery vs what is observed", fontsize=10)
            xpos = np.arange(len(masks))
            w = 0.38
            ax[1, 2].bar(xpos - w / 2, iou, w, label="bridge IoU", color="C1")
            ax[1, 2].bar(xpos + w / 2, umae, w, label="U-MAE", color="C0")
            ax[1, 2].set_xticks(xpos)
            ax[1, 2].set_xticklabels(masks)
            ax[1, 2].set_title("(6) bridge IoU + U-MAE vs observation", fontsize=10)
            ax[1, 2].legend(fontsize=8)

        fig.suptitle(f"Inverse-twin identifiability ({a.corpus}): integrals recover, full field does not",
                     fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        fig_dir = _REPO / "results" / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(fig_dir / f"identifiability_{a.corpus}.png", dpi=130, bbox_inches="tight")
        print(f"wrote {fig_dir / f'identifiability_{a.corpus}.png'}")
    except Exception as exc:  # pragma: no cover - figure is optional
        print(f"(figure skipped: {exc})")


if __name__ == "__main__":
    main()
