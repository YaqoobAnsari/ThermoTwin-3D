#!/usr/bin/env python
"""Boundary-layer A/B benchmark — does the SDF-keyed window + uncertainty head earn its keep?

The pre-registered ablation for the AiC system paper's method section (the mechanism is NOT a
standalone novelty — every leg is 2025-26 prior art; this measures whether the boundary-layer
structure improves results over plain additive residual learning, and whether the uncertainty
head is calibrated). Per corpus, for the lead backbone:

  prior_only : zero-parameter analytic prior (the baseline to beat)
  delta      : additive residual            θ̂ = prior + Δ            (the now-standard recipe)
  bl         : boundary-layer window        θ̂ = prior + w(d;ε)·Δ      (window ON, no UQ)
  bl_nowin   : window OFF control           θ̂ = prior + Δ (with the d feature)  — isolates the
               window from the extra interface-distance input channel
  blu        : boundary-layer + uncertainty (heteroscedastic head, calibrated reliability map)

Metrics: field rel-L2, mean-removed fluct rel-L2, bridge correction-rel-L2 (<1 beats prior),
U-MAE, and (blu) UQ calibration — coverage at 1σ/2σ and err-σ correlation.

    python scripts/benchmark_bl.py --corpus hard --epochs 200 --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from thermotwin.calibration.inverse import uq_calibration  # noqa: E402
from thermotwin.data.pointcloud_dataset import PointCloudDataset, collate_pointcloud  # noqa: E402
from thermotwin.eval.bridge_metrics import bridge_focused_metrics  # noqa: E402
from thermotwin.eval.building import u_from_indoor_face_cloud, u_value_report  # noqa: E402
from thermotwin.eval.metrics import relative_l2  # noqa: E402
from thermotwin.models.boundary_layer import (  # noqa: E402
    build_bl_pointnet2,
    build_blu_pointnet2,
    heteroscedastic_loss,
    interface_distance,
)
from thermotwin.models.pointnet2 import build_delta_pointnet2  # noqa: E402
from thermotwin.utils.seed import seed_everything  # noqa: E402

CORPORA = {
    "box": ("data/processed/block2_train", "data/processed/block2_val"),
    "irreg": ("data/processed/block2_irreg_train", "data/processed/block2_irreg_val"),
    "hard": ("data/processed/block2_hard_train", "data/processed/block2_hard_val"),
    "realcg": ("data/processed/block2_realcg_train", "data/processed/block2_realcg_val"),
    "realcg_lod3": ("data/processed/block2_realcg_lod3_train", "data/processed/block2_realcg_lod3_val"),
    "bag": ("data/processed/block2_bag_train", "data/processed/block2_bag_val"),
    "doe": ("data/processed/block2_doe_train", "data/processed/block2_doe_val"),
}
LATENT_GRID = 16
U_FACE_BAND = 0.08
KINDS = ("prior_only", "delta", "delta_input", "bl", "blu")


def _build(kind):
    if kind == "prior_only":
        return None
    if kind == "delta":
        return build_delta_pointnet2(in_channels=4, k=16, width=128)
    if kind == "delta_input":  # window OFF: additive, but WITH the interface-distance feature
        return build_bl_pointnet2(feat_channels=4, width=128, window=False)
    if kind == "bl":
        return build_bl_pointnet2(feat_channels=4, width=128, window=True)
    if kind == "blu":
        return build_blu_pointnet2(feat_channels=4, width=128, window=True)
    raise KeyError(kind)


def _d_cache(ds, device):
    """Interface distance per sample (the SDF-stretched coordinate), computed once."""
    cache = {}
    for i in range(len(ds)):
        it = ds[i]
        cache[i] = interface_distance(it["points"].to(device), it["feats"][:, 0].to(device))
    return cache


def _forward(model, kind, b, d, device):
    ig = b["input_geom"].to(device)
    feats = b["feats"].to(device)
    lat, sdf = b["latent_queries"].to(device), b["sdf"].to(device)
    prior = b["prior"].to(device)
    if kind == "delta":
        return model(ig, feats, lat, sdf, ig, prior), None
    out = model(ig, feats, lat, sdf, ig, prior, d)  # bl / blu / delta_input
    if kind == "blu":
        return out[0], out[1]
    return out, None


def _train(model, kind, ds, dcache, args, device):
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_pointcloud)
    samples = [(b, dcache[i]) for i, b in enumerate(loader)]
    t0 = time.perf_counter()
    for _ in range(args.epochs):
        model.train()
        order = torch.randperm(len(samples))
        for idx in order:
            b, d = samples[int(idx)]
            opt.zero_grad()
            pred, log_var = _forward(model, kind, b, d, device)
            target = b["theta"].to(device).unsqueeze(-1)
            loss = heteroscedastic_loss(pred, log_var, target) if log_var is not None else relative_l2(pred, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
    return time.perf_counter() - t0


def _eval(model, kind, ds, dcache, device):
    rel, fluct, P, T, Pr, uq = [], [], [], [], [], []
    u_pred, u_true, u_clear = [], [], []
    if model is not None:
        model.eval()
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_pointcloud)
    with torch.no_grad():
        for i, b in enumerate(loader):
            theta = b["theta"][0].numpy()
            prior = b["prior"][0].numpy()
            uc = float(b["u_clear"][0])
            if kind == "prior_only":
                pred = prior.copy()
                log_var = None
            else:
                p, lv = _forward(model, kind, b, dcache[i], device)
                pred = p[0, :, 0].cpu().numpy()
                log_var = lv[0, :, 0].cpu().numpy() if lv is not None else None
            rel.append(float(np.linalg.norm(pred - theta) / (np.linalg.norm(theta) + 1e-9)))
            fl = float(np.linalg.norm(theta - theta.mean())) + 1e-9
            fluct.append(float(np.linalg.norm(pred - theta) / fl))
            u_pred.append(u_from_indoor_face_cloud(pred, prior, b["points"][0].numpy(), uc, band=U_FACE_BAND))
            u_true.append(float(b["u_value"][0]))
            u_clear.append(uc)
            P.append(pred)
            T.append(theta)
            Pr.append(prior)
            if log_var is not None:
                std = np.exp(0.5 * np.clip(log_var, -10, 10))
                uq.append(uq_calibration(torch.from_numpy(pred), torch.from_numpy(std), torch.from_numpy(theta)))
    bridge = bridge_focused_metrics(np.concatenate(P), np.concatenate(T), np.concatenate(Pr))
    out = {
        "field_rel_l2": float(np.mean(rel)),
        "field_rel_l2_fluct": float(np.mean(fluct)),
        "u_mae": u_value_report(np.array(u_pred), np.array(u_true))["u_mae"],
        "n_params": int(sum(p.numel() for p in model.parameters())) if model is not None else 0,
        **{f"bridge_{k}": v for k, v in bridge.items()},
    }
    if uq:
        out.update({k: float(np.mean([u[k] for u in uq])) for k in uq[0]})
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", choices=list(CORPORA), default="hard")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seeds", type=int, nargs="+", default=[1337, 1])
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    device = a.device if (a.device == "cpu" or torch.cuda.is_available()) else "cpu"
    tr, va = CORPORA[a.corpus]
    train_ds = PointCloudDataset(_REPO / tr, latent_grid=LATENT_GRID, cache_in_memory=True)
    val_ds = PointCloudDataset(_REPO / va, latent_grid=LATENT_GRID, cache_in_memory=True)
    dtr, dva = _d_cache(train_ds, device), _d_cache(val_ds, device)

    results = []
    for kind in KINDS:
        per_seed = []
        for seed in a.seeds:
            seed_everything(seed)
            model = None if kind == "prior_only" else _build(kind).to(device)
            tt = 0.0 if model is None else _train(model, kind, train_ds, dtr, a, device)
            m = _eval(model, kind, val_ds, dva, device)
            m["seed"] = seed
            per_seed.append(m)
            print(f"[{a.corpus}/{kind} seed={seed}] relL2={m['field_rel_l2']:.4f} "
                  f"fluct={m['field_rel_l2_fluct']:.4f} corr={m['bridge_correction_rel_l2']:.3f} "
                  f"u_mae={m['u_mae']:.4f}"
                  + (f" cov1σ={m.get('uq_cov_1sigma', float('nan')):.2f}" if "uq_cov_1sigma" in m else "")
                  + f" train={tt:.0f}s", flush=True)
        keys = [k for k in per_seed[0] if isinstance(per_seed[0][k], (int, float)) and k != "seed"]
        agg = {f"{k}_mean": float(np.mean([s[k] for s in per_seed])) for k in keys}
        agg |= {f"{k}_std": float(np.std([s[k] for s in per_seed])) for k in keys}
        results.append({"kind": kind, "seeds": a.seeds, **agg, "per_seed": per_seed})

    out = _REPO / "results" / f"bl_{a.corpus}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"config": vars(a), "results": results}, indent=2, default=str))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
