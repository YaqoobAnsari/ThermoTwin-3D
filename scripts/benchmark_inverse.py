#!/usr/bin/env python
"""Inverse-twin benchmark — recover the conductivity / thermal-bridge field from observed θ.

Trains a frozen differentiable forward surrogate on a Block-2 corpus, then inverts it three
ways (optimization-based, amortized, hybrid) with regularisation + uncertainty variants, and
scores the recovered conductivity field on field accuracy, **bridge localisation**, U-MAE, and
UQ calibration. The honest test of the inverse twin: how well can we read per-surface
properties + thermal bridges back out of a temperature field, and where does identifiability
break (severe-bridge `hard` should recover well; clear-wall `realcg` is the hard, ill-posed case).

    python scripts/benchmark_inverse.py --corpus hard --fwd-epochs 150 --device cuda
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

from thermotwin.calibration.inverse import (  # noqa: E402
    bridge_localization,
    ensemble_inverse,
    knn_edges,
    optimize_inverse,
    uq_calibration,
)
from thermotwin.data.pointcloud_dataset import PointCloudDataset, collate_pointcloud  # noqa: E402
from thermotwin.eval.bridge_metrics import bridge_focused_metrics  # noqa: E402
from thermotwin.eval.building import u_from_indoor_face_cloud, u_value_report  # noqa: E402
from thermotwin.eval.metrics import relative_l2  # noqa: E402
from thermotwin.models.pointnet2 import build_delta_pointnet2, build_pointnet2  # noqa: E402
from thermotwin.utils.seed import seed_everything  # noqa: E402

CORPORA = {
    "hard": ("data/processed/block2_hard_train", "data/processed/block2_hard_val"),
    "realcg": ("data/processed/block2_realcg_train", "data/processed/block2_realcg_val"),
    "realcg_lod3": ("data/processed/block2_realcg_lod3_train", "data/processed/block2_realcg_lod3_val"),
    "box": ("data/processed/block2_train", "data/processed/block2_val"),
}
LATENT_GRID = 16
U_FACE_BAND = 0.08
# Identifiability study (Exp 3): is the *full field* recoverable, or only the integrals (U, Ψ)?
U_BANDS = (0.04, 0.08, 0.16)   # indoor-face slab widths -> integration-scale decomposition of U
BRIDGE_MARGIN = 0.25           # logk departure that defines a "bridge" point (matches bridge_localization)
OBS_MASKS = ("full", "surface", "interior")  # observation lever: what does seeing the interior buy?
INTERIOR_X = 0.2               # "interior" observation = points deeper than this through-wall coord
# Optimization-inverse regularisation variants (l1=sparsity, l2=stay-near-clear, tv=smoothness).
REG_VARIANTS = {
    "opt_none": dict(l1=0.0, l2=1e-3, tv=0.0),
    "opt_sparse": dict(l1=5e-3, l2=1e-3, tv=0.0),
    "opt_smooth": dict(l1=0.0, l2=1e-3, tv=5e-3),
    "opt_sparse_smooth": dict(l1=5e-3, l2=1e-3, tv=5e-3),
}


def _train_forward(model, train_ds, device, epochs, lr=1e-3):
    """Train the delta forward surrogate (property field -> θ). Returns wall-clock s."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loader = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=collate_pointcloud)
    t0 = time.perf_counter()
    for _ in range(epochs):
        model.train()
        for b in loader:
            opt.zero_grad()
            ig = b["input_geom"].to(device)
            feats = b["feats"].to(device)
            pred = model(ig, feats, b["latent_queries"].to(device), b["sdf"].to(device), ig, b["prior"].to(device))
            loss = relative_l2(pred, b["theta"].to(device).unsqueeze(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
    return time.perf_counter() - t0


def _train_amortized(model, train_ds, device, epochs, lr=1e-3):
    """Train an amortized inverse: θ (1 channel) -> logk field. MSE on the true conductivity."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loader = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=collate_pointcloud)
    for _ in range(epochs):
        model.train()
        for b in loader:
            opt.zero_grad()
            ig = b["input_geom"].to(device)
            theta_in = b["theta"].to(device).unsqueeze(-1)  # (1,N,1)
            logk_true = b["feats"][..., :1].to(device)       # (1,N,1) channel-0 conductivity
            pred = model(ig, theta_in, b["latent_queries"].to(device), b["sdf"].to(device), ig)
            loss = torch.mean((pred - logk_true) ** 2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()


def _forward_fn(model, b, device):
    """Closure logk(N,) -> θ_pred(1,N,1), with the nominal construction (chans 1:) + prior fixed."""
    ig = b["input_geom"].to(device)
    feats0 = b["feats"].to(device)                       # (1,N,4)
    prior = b["prior"].to(device)
    latent = b["latent_queries"].to(device)
    sdf = b["sdf"].to(device)
    oq = b["output_queries"].to(device)

    def fwd(logk):
        logk_col = logk.reshape(1, -1, 1)
        feats = torch.cat([logk_col, feats0[..., 1:]], dim=-1)  # differentiable in logk
        return model(ig, feats, latent, sdf, oq, prior)

    return fwd


def _metrics(logk_hat, logk_true, clear_ref, theta_obs, fwd, u_true, prior_np, points_np, u_clear):
    """Field recovery + bridge localisation + U-MAE + data fit for one recovered field."""
    with torch.no_grad():
        theta_rec = fwd(logk_hat).reshape(-1).cpu().numpy()
        data_fit = float(np.linalg.norm(theta_rec - theta_obs) / (np.linalg.norm(theta_obs) + 1e-9))
        u_rec = u_from_indoor_face_cloud(theta_rec, prior_np, points_np, u_clear, band=U_FACE_BAND)
    lk_rel = float(torch.linalg.norm(logk_hat - logk_true) / (torch.linalg.norm(logk_true) + 1e-9))
    m = {"logk_rel_l2": lk_rel, "data_fit": data_fit, "u_rec": u_rec, "u_true": u_true}
    m.update(bridge_localization(logk_hat, logk_true, clear_ref))
    return m


def _obs_index(points_x, mask_type, band, device):
    """Indices of the *observed* points for an observation-masking variant.

    The identifiability lever: solve the same inverse but only let the data-fit see a subset of
    the field. ``surface`` = the indoor-face slab (the realistic IR view), ``interior`` = the
    deep points, ``full`` = everything. The recovered field is still scored over the *whole*
    domain, so this measures what observing the interior actually buys.
    """
    if mask_type == "surface":
        m = points_x < band
    elif mask_type == "interior":
        m = points_x > INTERIOR_X
    else:  # full
        m = np.ones_like(points_x, dtype=bool)
    return torch.as_tensor(np.nonzero(m)[0], device=device, dtype=torch.long)


def _masked_forward(fwd, idx):
    """Wrap a forward closure so its output (and thus the data-fit) is restricted to ``idx``."""
    def f(logk):
        return fwd(logk).reshape(-1)[idx]
    return f


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", choices=list(CORPORA), default="hard")
    p.add_argument("--fwd-epochs", type=int, default=150)
    p.add_argument("--amort-epochs", type=int, default=150)
    p.add_argument("--inv-steps", type=int, default=300)
    p.add_argument("--inv-lr", type=float, default=5e-2)
    p.add_argument("--ensemble", type=int, default=4)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-identifiability", dest="identifiability", action="store_false",
                   help="skip the Exp-3 identifiability study (field storage + obs-masking + U-bands)")
    a = p.parse_args()
    device = a.device if (a.device == "cpu" or torch.cuda.is_available()) else "cpu"
    tr_root, va_root = CORPORA[a.corpus]
    train_ds = PointCloudDataset(_REPO / tr_root, latent_grid=LATENT_GRID, cache_in_memory=True)
    val_ds = PointCloudDataset(_REPO / va_root, latent_grid=LATENT_GRID, cache_in_memory=True)

    seed_everything(a.seed)
    print(f"[{a.corpus}] training forward surrogate (delta_pointnet2) ...", flush=True)
    fwd_model = build_delta_pointnet2(in_channels=4, k=16, width=128).to(device)
    t_fwd = _train_forward(fwd_model, train_ds, device, a.fwd_epochs)
    fwd_model.eval()
    for q in fwd_model.parameters():
        q.requires_grad_(False)
    print(f"  forward trained in {t_fwd:.0f}s", flush=True)

    print(f"[{a.corpus}] training amortized inverse (θ -> logk) ...", flush=True)
    amort = build_pointnet2(in_channels=1, k=16, width=128).to(device)
    _train_amortized(amort, train_ds, device, a.amort_epochs)
    amort.eval()
    for q in amort.parameters():
        q.requires_grad_(False)

    # Per-variant accumulators.
    variants = list(REG_VARIANTS) + ["amortized", "hybrid", "opt_sparse_smooth_ens"]
    acc = {v: [] for v in variants}
    uq_acc = []
    # Exp-3 identifiability accumulators.
    fields_rows: list[dict] = []          # per-point storage (concatenated at the end)
    mask_acc = {mt: [] for mt in OBS_MASKS}  # observation-masking recovery
    uband_rows: list[dict] = []           # multi-band U from the recovered field
    bf_acc: list[dict] = []               # bridge-focused correction metrics on theta_rec
    loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate_pointcloud)
    for b in loader:
        fwd = _forward_fn(fwd_model, b, device)
        theta_obs_t = b["theta"].to(device).reshape(-1)
        theta_obs_np = b["theta"][0].numpy()
        logk_true = b["feats"][0, :, 0].to(device)
        clear_ref = float(logk_true.median())  # known nominal clear-wall conductivity (proxy)
        logk_clear = torch.full_like(logk_true, clear_ref)
        coords = b["input_geom"][0].to(device)
        edges = knn_edges(coords, k=8)
        u_true = float(b["u_value"][0])
        u_clear = float(b["u_clear"][0])
        prior_np = b["prior"][0].numpy()
        points_np = b["points"][0].numpy()
        common = dict(u_true=u_true, prior_np=prior_np, points_np=points_np, u_clear=u_clear)

        # 1) optimization-based inverse, each regularisation variant (init = clear wall).
        for name, reg in REG_VARIANTS.items():
            lk = optimize_inverse(fwd, theta_obs_t, logk_clear, clear_ref,
                                  n_steps=a.inv_steps, lr=a.inv_lr, tv_edges=edges, **reg)
            acc[name].append(_metrics(lk, logk_true, clear_ref, theta_obs_np, fwd, **common))

        # 2) amortized inverse (θ -> logk in one shot).
        with torch.no_grad():
            lk_amort = amort(coords.unsqueeze(0), theta_obs_t.reshape(1, -1, 1),
                             b["latent_queries"].to(device), b["sdf"].to(device),
                             coords.unsqueeze(0)).reshape(-1)
        acc["amortized"].append(_metrics(lk_amort, logk_true, clear_ref, theta_obs_np, fwd, **common))

        # 3) hybrid: amortized init -> optimization refine (sparse+smooth).
        reg = REG_VARIANTS["opt_sparse_smooth"]
        lk_hy = optimize_inverse(fwd, theta_obs_t, lk_amort, clear_ref,
                                 n_steps=a.inv_steps, lr=a.inv_lr, tv_edges=edges, **reg)
        acc["hybrid"].append(_metrics(lk_hy, logk_true, clear_ref, theta_obs_np, fwd, **common))

        # 4) ensemble UQ on the sparse+smooth optimization (init = clear wall + noise/member).
        def _member(m, _reg=reg, _clear=logk_clear, _cr=clear_ref, _edges=edges, _fwd=fwd, _obs=theta_obs_t):
            torch.manual_seed(1000 + m)
            init = _clear + 0.1 * torch.randn_like(_clear)
            return optimize_inverse(_fwd, _obs, init, _cr, n_steps=a.inv_steps, lr=a.inv_lr,
                                    tv_edges=_edges, **_reg)
        mean, std = ensemble_inverse(_member, n=a.ensemble)
        acc["opt_sparse_smooth_ens"].append(_metrics(mean, logk_true, clear_ref, theta_obs_np, fwd, **common))
        uq_acc.append(uq_calibration(mean, std, logk_true))

        if not a.identifiability:
            continue
        # ---- Exp 3: identifiability study --------------------------------------------------
        # Canonical recovered field = the ensemble mean (full obs); std = ensemble spread.
        lk_hat = mean.reshape(-1)
        with torch.no_grad():
            theta_rec = fwd(lk_hat).reshape(-1).cpu().numpy()
        # (a) integration-scale: U recovered at several indoor-face slab widths.
        urow = {"u_true": u_true}
        for bnd in U_BANDS:
            urow[f"u_rec_b{bnd}"] = u_from_indoor_face_cloud(theta_rec, prior_np, points_np, u_clear, band=bnd)
        uband_rows.append(urow)
        # correction skill of the recovered field on the bridge region (theta space).
        bf_acc.append(bridge_focused_metrics(theta_rec, theta_obs_np, prior_np))
        # (c) per-point storage for the UQ-vs-error correlation + recovery-vs-position curves.
        x_tw = points_np[:, 0]
        true_b = (logk_true - clear_ref) > BRIDGE_MARGIN
        if bool(true_b.any()):
            dist = torch.cdist(coords, coords[true_b]).min(dim=1).values.cpu().numpy()
        else:
            dist = np.full(coords.shape[0], np.nan, dtype=np.float32)
        sid = len(fields_rows)
        fields_rows.append({
            "sample_id": np.full(x_tw.shape, sid, dtype=np.int32),
            "logk_true": logk_true.cpu().numpy().astype(np.float32),
            "logk_hat": lk_hat.cpu().numpy().astype(np.float32),
            "logk_std": std.reshape(-1).cpu().numpy().astype(np.float32),
            "x_throughwall": x_tw.astype(np.float32),
            "dist_to_bridge": dist.astype(np.float32),
            "clear_ref": np.full(x_tw.shape, clear_ref, dtype=np.float32),
        })
        # (b) observation masking: same inverse config, observation restricted to a subset.
        reg_sm = REG_VARIANTS["opt_sparse_smooth"]
        for mt in OBS_MASKS:
            oidx = _obs_index(x_tw, mt, U_FACE_BAND, device)
            if oidx.numel() == 0:
                continue  # e.g. realcg has no indoor-face points -> no "surface" view
            lk_m = optimize_inverse(_masked_forward(fwd, oidx), theta_obs_t[oidx], logk_clear,
                                    clear_ref, n_steps=a.inv_steps, lr=a.inv_lr, tv_edges=edges, **reg_sm)
            with torch.no_grad():
                th_m = fwd(lk_m).reshape(-1).cpu().numpy()
            row = {
                "logk_rel_l2": float(torch.linalg.norm(lk_m - logk_true) / (torch.linalg.norm(logk_true) + 1e-9)),
                "n_obs": int(oidx.numel()),
                "u_rec": u_from_indoor_face_cloud(th_m, prior_np, points_np, u_clear, band=U_FACE_BAND),
                "u_true": u_true,
            }
            row.update(bridge_localization(lk_m, logk_true, clear_ref))
            mask_acc[mt].append(row)

    # Aggregate.
    def agg(rows):
        keys = [k for k in rows[0] if isinstance(rows[0][k], (int, float))]
        out = {}
        for k in keys:
            vals = [r[k] for r in rows if r[k] == r[k]]  # drop NaN
            if vals:
                out[f"{k}_mean"] = float(np.mean(vals))
                out[f"{k}_std"] = float(np.std(vals))
        return out

    results = {}
    for v in variants:
        results[v] = agg(acc[v])
        r = results[v]
        # U-MAE from recovered vs true U over the val set.
        urec = [row["u_rec"] for row in acc[v]]
        utru = [row["u_true"] for row in acc[v]]
        results[v]["u_mae"] = u_value_report(np.array(urec), np.array(utru))["u_mae"]
        print(f"[{a.corpus}/{v}] logk_relL2={r.get('logk_rel_l2_mean', float('nan')):.3f} "
              f"bridge P/R/IoU={r.get('bridge_precision_mean', float('nan')):.2f}/"
              f"{r.get('bridge_recall_mean', float('nan')):.2f}/{r.get('bridge_iou_mean', float('nan')):.2f} "
              f"U-MAE={results[v]['u_mae']:.4f} datafit={r.get('data_fit_mean', float('nan')):.3f}", flush=True)
    uq = agg(uq_acc)
    print(f"[{a.corpus}/UQ] cov1σ={uq.get('uq_cov_1sigma_mean', float('nan')):.2f} "
          f"cov2σ={uq.get('uq_cov_2sigma_mean', float('nan')):.2f} "
          f"err-σ corr={uq.get('uq_err_std_corr_mean', float('nan')):.2f}", flush=True)

    out_dict = {"config": vars(a), "forward_train_s": t_fwd, "variants": results, "uq": uq}

    # ---- Exp 3: identifiability summary + per-point field dump -----------------------------
    if a.identifiability and fields_rows:
        uband = {}
        utru = np.array([r["u_true"] for r in uband_rows])
        for bnd in U_BANDS:
            rec = np.array([r[f"u_rec_b{bnd}"] for r in uband_rows])
            rep = u_value_report(rec, utru)
            uband[f"u_mae_b{bnd}"] = rep["u_mae"]
            uband[f"u_mape_b{bnd}"] = rep["u_mape"]
        obs = {}
        for mt, rows in mask_acc.items():
            if not rows:
                continue
            obs[mt] = agg(rows)
            obs[mt]["u_mae"] = u_value_report(
                np.array([r["u_rec"] for r in rows]), np.array([r["u_true"] for r in rows])
            )["u_mae"]
            obs[mt]["n_obs_mean"] = float(np.mean([r["n_obs"] for r in rows]))
        out_dict["identifiability"] = {
            "u_bands": uband, "bridge_focused": agg(bf_acc), "observation": obs,
        }
        cat = {k: np.concatenate([r[k] for r in fields_rows]) for k in fields_rows[0]}
        npz = _REPO / "results" / f"inverse_fields_{a.corpus}.npz"
        npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(npz, **cat)
        print(f"[{a.corpus}/IDENT] U-MAE by face-band  " +
              "  ".join(f"{b}:{uband[f'u_mae_b{b}']:.4f}" for b in U_BANDS), flush=True)
        for mt in OBS_MASKS:
            if obs.get(mt):
                o = obs[mt]
                print(f"  obs={mt:8s} n~{int(o['n_obs_mean']):5d}  "
                      f"logk_relL2={o.get('logk_rel_l2_mean', float('nan')):.3f}  "
                      f"bridgeIoU={o.get('bridge_iou_mean', float('nan')):.2f}  "
                      f"U-MAE={o['u_mae']:.4f}", flush=True)
        print(f"wrote {npz}", flush=True)

    out = _REPO / "results" / f"inverse_{a.corpus}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(out_dict, indent=2, default=str))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
