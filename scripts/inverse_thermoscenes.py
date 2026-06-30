#!/usr/bin/env python
"""Measured-field inverse on ThermoScenes (Exp 2): the inverse twin, run end-to-end on a real
3-D thermal field instead of a simulated one.

The limitation this closes: our differentiable inverse (optimize_inverse + sparsity + kNN total
variation + ensemble UQ) had only ever been run on *simulated* theta fields. ThermoScenes gives
a calibrated absolute-degC surface temperature fused onto reconstructed COLMAP facade geometry.
It ships no conductivity, no materials, no air temperatures and is transient/solar, so an
*absolute* U/property inverse is impossible. What is well posed, and exactly the inverse twin's
job, is a **relative source localisation**:

    model the measured heat-loss residual r (= measured - geometry-smooth baseline) as the
    graph-diffused footprint of a *sparse, spatially coherent* source field s on the facade,
        r  ~=  P^m s          (P = row-normalised kNN diffusion on the 3-D points),
    and recover s by the same regularised differentiable inverse we use on simulated data.

We then validate **convergently**: the recovered source map and an *independent* local-statistics
heat-loss detector (`baseline_residual`, a different operator) agree on where the real facade
loses heat (precision / recall / IoU at a matched anomaly fraction). The ensemble spread is a
confidence map; absolute UQ calibration needs labels the dataset does not provide, so we report
ensemble *stability* instead and say so. This is a relative, convergent validation, not absolute
U and not held-out labels.

    python scripts/inverse_thermoscenes.py --scenes BI-building exhibition_building INR-building

A sim-trained-operator forward (transfer of a delta_pointnet2 from synthetic theta) is a noted,
deferred exploratory check: the unit mismatch (degC vs dimensionless theta, no T_in/T_out) and the
absence of a saved forward checkpoint make it out of scope here.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from thermotwin.calibration.inverse import (  # noqa: E402
    ensemble_inverse,
    knn_edges,
    optimize_inverse,
)
from thermotwin.data.thermoscenes import (  # noqa: E402
    BUILDING_SCENES,
    baseline_residual,
    fuse_thermal_to_geometry,
    open_archive,
)
from thermotwin.eval.bridge_metrics import bridge_focused_metrics  # noqa: E402

ARCHIVE = _REPO / "data" / "raw" / "thermoscenes" / "ThermoScenes_full.zip"


def _diffusion_op(coords: torch.Tensor, k: int, m: int):
    """Row-normalised kNN diffusion ``P`` and its ``m``-step low-pass ``G(s) = P^m s``.

    Encodes the physics that a heat-loss source warms a *neighbourhood* on the facade, so the
    measured residual is a smoothed footprint of sharper underlying sources. Recovering ``s``
    from the residual is therefore a geometry-aware deconvolution, which the sparsity + TV prior
    regularises. Also returns the symmetric kNN edge list for the TV regulariser.
    """
    i, j = knn_edges(coords, k=k)
    n = coords.shape[0]
    self_idx = torch.arange(n, device=coords.device)
    ii = torch.cat([i, j, self_idx])
    jj = torch.cat([j, i, self_idx])
    w = torch.ones(ii.shape[0], dtype=coords.dtype, device=coords.device)
    a = torch.sparse_coo_tensor(torch.stack([ii, jj]), w, (n, n)).coalesce()
    deg = torch.sparse.sum(a, dim=1).to_dense().clamp_min(1.0)
    vals = a.values() / deg[a.indices()[0]]
    p = torch.sparse_coo_tensor(a.indices(), vals, (n, n)).coalesce()

    def g(s: torch.Tensor) -> torch.Tensor:
        x = s.reshape(-1, 1)
        for _ in range(m):
            x = torch.sparse.mm(p, x)
        return x.reshape(-1)

    return g, (i, j)


def _mask_prf(pred: np.ndarray, true: np.ndarray) -> dict:
    """Precision / recall / IoU of a predicted boolean mask vs a reference boolean mask."""
    tp = float(np.sum(pred & true))
    fp = float(np.sum(pred & ~true))
    fn = float(np.sum(~pred & true))
    union = tp + fp + fn
    return {
        "precision": tp / (tp + fp) if (tp + fp) else float("nan"),
        "recall": tp / (tp + fn) if (tp + fn) else float("nan"),
        "iou": tp / union if union else float("nan"),
    }


def _topfrac_mask(score: np.ndarray, frac: float) -> np.ndarray:
    """Boolean mask of the top-``frac`` highest-scoring points (heat-loss = warm = high score)."""
    frac = float(np.clip(frac, 1e-4, 0.999))
    thr = np.quantile(score, 1.0 - frac)
    return score >= thr


def _invert_scene(xyz, cel, args):
    """Run the measured-field inverse on one fused scene. Returns (result dict, arrays for fig)."""
    base, resid, det_mask = baseline_residual(xyz, cel, k=args.baseline_k)
    sigma = float(resid.std()) + 1e-9
    r = resid / sigma  # standardise the residual so the absolute reg weights transfer from sim

    coords = torch.as_tensor(xyz, dtype=torch.float32)
    r_t = torch.as_tensor(r, dtype=torch.float32)
    g, edges = _diffusion_op(coords, k=args.knn, m=args.diffuse)

    reg = dict(l1=args.l1, l2=args.l2, tv=args.tv)
    init = r_t.clone()  # warm start from the residual itself

    def member(idx, _init=init, _r=r_t, _g=g, _edges=edges, _reg=reg):
        torch.manual_seed(1000 + idx)
        return optimize_inverse(_g, _r, _init + 0.1 * torch.randn_like(_init), 0.0,
                                n_steps=args.steps, lr=args.lr, tv_edges=_edges, **_reg)

    mean, std = ensemble_inverse(member, n=args.ensemble)
    s_hat = mean.cpu().numpy()
    s_std = std.cpu().numpy()
    recon_r = g(mean).cpu().numpy()           # P^m s, the explained (standardised) residual
    data_fit = float(np.linalg.norm(recon_r - r) / (np.linalg.norm(r) + 1e-9))

    # convergent validation vs the independent detector, at a matched anomaly fraction.
    frac = float(det_mask.mean())
    pred_mask = _topfrac_mask(s_hat, frac) if frac > 0 else np.zeros_like(s_hat, dtype=bool)
    prf = _mask_prf(pred_mask, det_mask)
    iou_random = frac  # expected IoU of a random mask of the same fraction (the null)

    # ensemble stability (UQ proxy without ground-truth labels): how reproducible is the mask?
    member_masks = []
    for idx in range(args.ensemble):
        sm = member(idx).cpu().numpy()
        member_masks.append(_topfrac_mask(sm, frac))
    pair_iou = [
        _mask_prf(member_masks[a], member_masks[b])["iou"]
        for a in range(len(member_masks)) for b in range(a + 1, len(member_masks))
    ]
    stability = float(np.nanmean(pair_iou)) if pair_iou else float("nan")

    # reconstruction skill on the bridge region (theta-space correction metric, reused from sim).
    recon_C = base + recon_r * sigma
    bf = bridge_focused_metrics(recon_C, cel, base)

    res = {
        "n_points": int(len(xyz)),
        "measured_std_C": round(sigma, 3),
        "detector_anomaly_frac": round(frac, 4),
        "data_fit_rel_l2": round(data_fit, 3),
        "overlap_precision": round(prf["precision"], 3),
        "overlap_recall": round(prf["recall"], 3),
        "overlap_iou": round(prf["iou"], 3),
        "overlap_iou_random_null": round(iou_random, 4),
        "overlap_iou_lift_over_null": round(prf["iou"] / (iou_random + 1e-9), 1)
        if prf["iou"] == prf["iou"] else float("nan"),
        "ensemble_mask_stability_iou": round(stability, 3),
        "recon_correction_rel_l2": round(float(bf["correction_rel_l2"]), 3),
        "recon_correction_corr": round(float(bf["correction_corr"]), 3),
    }
    fig_arrays = dict(xyz=xyz, cel=cel, base=base, resid=resid, det_mask=det_mask,
                      s_hat=s_hat, s_std=s_std, pred_mask=pred_mask)
    return res, fig_arrays


def _figure(scene, A, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from thermotwin.viz import apply_style, save_figure
    from thermotwin.viz import style as vstyle
    from thermotwin.viz.pointcloud import scatter_3d

    apply_style()
    xyz = A["xyz"]
    rmax = float(np.abs(A["resid"]).max()) or 1e-6
    smax = float(np.abs(A["s_hat"]).max()) or 1e-6
    # agreement categories: TP (both), FN (detector only), FP (inverse only)
    tp = A["pred_mask"] & A["det_mask"]
    fn = ~A["pred_mask"] & A["det_mask"]
    fp = A["pred_mask"] & ~A["det_mask"]
    agree = np.zeros(len(xyz))            # 0 = neither (grey)
    agree[fp] = 1.0                        # inverse only
    agree[fn] = 2.0                        # detector only
    agree[tp] = 3.0                        # both (convergent hit)

    fig = plt.figure(figsize=(15.5, 8.6))
    panels = [
        (A["cel"], vstyle.THERMAL, None, None, "degC", "measured surface temperature"),
        (A["base"], vstyle.THERMAL, float(A["cel"].min()), float(A["cel"].max()), "degC",
         "geometry-smooth baseline"),
        (A["resid"], vstyle.DIVERGING, -rmax, rmax, "degC", "measured residual (observation)"),
        (A["s_hat"], vstyle.DIVERGING, -smax, smax, "", "recovered sources (inverse twin)"),
        (A["s_std"], "viridis", None, None, "", "ensemble spread (confidence)"),
        (agree, "turbo", 0.0, 3.0, "", "convergence: TP=3 detectorFN=2 inverseFP=1"),
    ]
    for k, (vals, cmap, vmn, vmx, clab, title) in enumerate(panels):
        ax = fig.add_subplot(2, 3, k + 1, projection="3d")
        scatter_3d(ax, np.asarray(xyz), vals, cmap=cmap, vmin=vmn, vmax=vmx,
                   title=title, cbar_label=clab, s=6)
    fig.suptitle(f"ThermoScenes {scene}: measured-field inverse twin, {len(xyz)} pts "
                 f"(relative, convergent validation)", fontsize=12)
    save_figure(fig, out_path)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive", default=str(ARCHIVE))
    p.add_argument("--scenes", nargs="*", default=["BI-building", "exhibition_building", "INR-building"],
                   help=f"scene ids to invert (choices: {', '.join(BUILDING_SCENES)})")
    p.add_argument("--knn", type=int, default=8, help="kNN degree for the diffusion / TV graph")
    p.add_argument("--diffuse", type=int, default=3, help="diffusion steps m in the forward P^m")
    p.add_argument("--baseline-k", type=int, default=40, help="kNN for the smooth baseline / detector")
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--lr", type=float, default=5e-2)
    p.add_argument("--l1", type=float, default=5e-3)
    p.add_argument("--l2", type=float, default=1e-3)
    p.add_argument("--tv", type=float, default=5e-3)
    p.add_argument("--ensemble", type=int, default=5)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--no-figures", action="store_true")
    a = p.parse_args()
    torch.manual_seed(a.seed)

    out = _REPO / "results" / "inverse_thermoscenes"
    out.mkdir(parents=True, exist_ok=True)
    zf = open_archive(a.archive)

    scenes = {}
    for scene in a.scenes:
        try:
            xyz, cel = fuse_thermal_to_geometry(zf, scene)
        except Exception as exc:
            print(f"[{scene}] fusion failed: {exc}", flush=True)
            continue
        if len(xyz) < 50:
            print(f"[{scene}] too few fused points ({len(xyz)}), skipping", flush=True)
            continue
        res, arrays = _invert_scene(xyz, cel, a)
        scenes[scene] = res
        print(f"[{scene}] n={res['n_points']:5d}  data-fit={res['data_fit_rel_l2']:.3f}  "
              f"overlap P/R/IoU={res['overlap_precision']:.2f}/{res['overlap_recall']:.2f}/"
              f"{res['overlap_iou']:.2f}  (IoU lift x{res['overlap_iou_lift_over_null']})  "
              f"stability={res['ensemble_mask_stability_iou']:.2f}", flush=True)
        if not a.no_figures:
            _figure(scene, arrays, out / f"{scene}_inverse")

    summary = {"archive": a.archive, "config": vars(a), "scenes": scenes}
    if scenes:
        ious = [s["overlap_iou"] for s in scenes.values() if s["overlap_iou"] == s["overlap_iou"]]
        lifts = [s["overlap_iou_lift_over_null"] for s in scenes.values()
                 if s["overlap_iou_lift_over_null"] == s["overlap_iou_lift_over_null"]]
        summary["aggregate"] = {
            "n_scenes": len(scenes),
            "mean_overlap_iou": round(float(np.mean(ious)), 3) if ious else float("nan"),
            "mean_iou_lift_over_null": round(float(np.mean(lifts)), 1) if lifts else float("nan"),
            "mean_data_fit_rel_l2": round(float(np.mean([s["data_fit_rel_l2"] for s in scenes.values()])), 3),
            "mean_stability_iou": round(float(np.mean(
                [s["ensemble_mask_stability_iou"] for s in scenes.values()
                 if s["ensemble_mask_stability_iou"] == s["ensemble_mask_stability_iou"]])), 3),
        }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out / 'summary.json'}  ({len(scenes)} scenes)")
    if "aggregate" in summary:
        ag = summary["aggregate"]
        print(f"AGGREGATE: mean overlap IoU {ag['mean_overlap_iou']} "
              f"(x{ag['mean_iou_lift_over_null']} over random null), "
              f"mean data-fit {ag['mean_data_fit_rel_l2']}, mean stability {ag['mean_stability_iou']}")


if __name__ == "__main__":
    main()
