#!/usr/bin/env python
"""Phase-0 deconfound gate on the WHNO conduction-with-inclusions benchmark (cross-PDE test).

A thermal bridge is mathematically a high-conductivity inclusion in a matrix. WHNO
(Cavallazzi 2025, arXiv 2511.07347) is exactly that: a 64x64 domain, background conductivity
k=1, four k=10 inclusions, with a fixed central-hot-square (T=1) / cold-border (T=0) BC. The
**homogenised prior** — the uniform-conductivity solution (no inclusions), one fixed field —
plays the role of our analytic 1-D clear-wall prior; the inclusions are the bridges.

We run the SAME deconfound gate as ``scripts/benchmark_block2.py``, flattening the grid to a
point cloud so our gridless operators consume it:

  prior_only        : the uniform-medium field (zero parameters)
  predict_mean      : constant floor (train-mean temperature)
  <bb>              : data-only (logk feature only), predict the full field
  cond_<bb>         : logk + prior as INPUT, predict the full field      (prior-as-input)
  delta_<bb>        : logk + prior, predict the correction (prior added back)   (the recipe)
  delta_const_<bb>  : delta with a constant (non-physics) prior

across gridless backbones (pointnet2, transolver, deeponet). It decides whether the recipe is
(a) necessary and (b) beats the homogenised prior on a credible PUBLISHED PDE with genuinely
sharp inclusions — the cross-PDE evidence the ICLR pitch needs.

    python scripts/benchmark_whno.py --n-train 1000 --n-val 200 --epochs 150 --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))
_WHNO = _REPO / "data" / "raw" / "external" / "WHNO"
sys.path.insert(0, str(_WHNO))

import heat_utils  # noqa: E402  (from the cloned WHNO repo)

from thermotwin.eval.bridge_metrics import bridge_focused_metrics  # noqa: E402
from thermotwin.eval.metrics import relative_l2  # noqa: E402
from thermotwin.models.deeponet import build_deeponet, build_delta_deeponet  # noqa: E402
from thermotwin.models.pointnet2 import build_delta_pointnet2, build_pointnet2  # noqa: E402
from thermotwin.models.transolver import build_delta_transolver, build_transolver  # noqa: E402
from thermotwin.utils.seed import seed_everything  # noqa: E402

GRID = 64
N_POINTS = 1500  # subsample of the 4096 grid points (matches our building cloud density)
# WHNO data config (config_heat_whno.yaml), the central-hot-square / cold-border problem.
DATA_CFG = {
    "num_blocks": 4, "min_block_size": 12, "max_block_size": 18,
    "k_background": 1.0, "k_inclusion": 10.0, "solver_iterations": 500,
    "T_hot": 1.0, "T_cold": 0.0, "geometry": "rect",
}

# Backbone builders: (data-only/cond builder, delta builder). cond reuses the data-only
# architecture with an extra input channel; delta uses the DeltaX wrapper.
BACKBONES = {
    "pointnet2": (
        lambda c: build_pointnet2(in_channels=c, k=16, width=128),
        lambda c: build_delta_pointnet2(in_channels=c, k=16, width=128),
    ),
    "transolver": (
        lambda c: build_transolver(in_channels=c, n_hidden=128, n_layers=8, n_head=8, slice_num=64, mlp_ratio=2),
        lambda c: build_delta_transolver(in_channels=c, n_hidden=128, n_layers=8, n_head=8, slice_num=64, mlp_ratio=2),
    ),
    "deeponet": (
        lambda c: build_deeponet(in_channels=c, p=128, hidden=256, depth=4),
        lambda c: build_delta_deeponet(in_channels=c, p=128, hidden=256, depth=4),
    ),
}


def _solve(k_field: torch.Tensor) -> torch.Tensor:
    return heat_utils.solve_heat_equation(
        k_field, iterations=DATA_CFG["solver_iterations"],
        T_hot=DATA_CFG["T_hot"], T_cold=DATA_CFG["T_cold"],
    )


def gen_or_load(n: int, split: str, device: str, seed: int):
    """Conductivity + temperature grids, cached to data/raw/external/WHNO_data/."""
    cache = _WHNO.parent / "WHNO_data"
    cache.mkdir(parents=True, exist_ok=True)
    fk, ft = cache / f"k_{split}_{n}.pt", cache / f"T_{split}_{n}.pt"
    if fk.exists() and ft.exists():
        return torch.load(fk), torch.load(ft)
    np.random.seed(seed)  # noqa: NPY002 — WHNO generators use global np.random
    torch.manual_seed(seed)
    ks, ts = [], []
    for i in range(n):
        k = heat_utils.generate_conductivity_by_geometry("rect", (GRID, GRID), DATA_CFG, device="cpu")
        ks.append(k)
        ts.append(_solve(k.clone()))
        if (i + 1) % 200 == 0:
            print(f"  [{split}] generated {i + 1}/{n}", flush=True)
    K, T = torch.stack(ks), torch.stack(ts)
    torch.save(K, fk)
    torch.save(T, ft)
    return K, T


def _grid_points(idx: np.ndarray) -> torch.Tensor:
    ys, xs = np.meshgrid(np.arange(GRID), np.arange(GRID), indexing="ij")
    coords = np.stack([xs.ravel(), ys.ravel(), np.zeros(GRID * GRID)], axis=-1).astype(np.float32)
    coords[:, 0] /= GRID - 1
    coords[:, 1] /= GRID - 1
    return torch.from_numpy(coords[idx])  # (N_POINTS, 3), z=0 planar


class WHNODataset(Dataset):
    """Grid-as-points: per sample -> (coords, logk_std, prior, theta) on a fixed subsample."""

    def __init__(self, K, T, prior_grid, idx, logk_mean, logk_std):
        self.K, self.T = K, T
        self.prior = torch.from_numpy(prior_grid.reshape(-1)[idx].astype(np.float32))
        self.coords = _grid_points(idx)
        self.idx = idx
        self.lm, self.ls = logk_mean, logk_std

    def __len__(self):
        return self.K.shape[0]

    def __getitem__(self, i):
        k = self.K[i].reshape(-1).numpy()[self.idx]
        theta = self.T[i].reshape(-1).numpy()[self.idx].astype(np.float32)
        logk_std = ((np.log(k) - self.lm) / self.ls).astype(np.float32)
        return {
            "coords": self.coords,
            "logk": torch.from_numpy(logk_std).unsqueeze(-1),  # (N,1)
            "prior": self.prior,                               # (N,)
            "theta": torch.from_numpy(theta),                  # (N,)
        }


def _forward(model, kind, batch, device, const):
    ig = batch["coords"].to(device)        # (1,N,3) — DataLoader already added the batch dim
    logk = batch["logk"].to(device)        # (1,N,1)
    prior = batch["prior"].to(device)      # (1,N)
    if kind == "data":
        return model(ig, logk, None, None, ig)
    feats = torch.cat([logk, prior.unsqueeze(-1)], dim=-1)  # (1,N,2)
    if kind == "cond":
        return model(ig, feats, None, None, ig)
    if kind == "delta_const":
        feats = feats.clone()
        feats[..., 1] = const
        prior = torch.full_like(prior, const)
    return model(ig, feats, None, None, ig, prior)  # delta / delta_const


def _train(model, kind, ds, args, device):
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loader = DataLoader(ds, batch_size=1, shuffle=True)
    t0 = time.perf_counter()
    for _ in range(args.epochs):
        model.train()
        for b in loader:
            opt.zero_grad()
            pred = _forward(model, kind, b, device, args._const)  # (1,N,1)
            target = b["theta"].to(device).unsqueeze(-1)
            loss = relative_l2(pred, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
    return time.perf_counter() - t0


def _eval(model, kind, ds, device, const):
    rel, fluct, P, Tt, Pr = [], [], [], [], []
    if model is not None:
        model.eval()
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    with torch.no_grad():
        for b in loader:
            theta = b["theta"][0].numpy()
            prior = b["prior"][0].numpy()
            if kind == "prior_only":
                pred = prior.copy()
            elif kind == "predict_mean":
                pred = np.full_like(theta, const)
            else:
                pred = _forward(model, kind, b, device, const)[0, :, 0].cpu().numpy()
            rel.append(float(np.linalg.norm(pred - theta) / (np.linalg.norm(theta) + 1e-12)))
            fl = float(np.linalg.norm(theta - theta.mean())) + 1e-9
            fluct.append(float(np.linalg.norm(pred - theta) / fl))
            P.append(pred)
            Tt.append(theta)
            Pr.append(prior)
    bridge = bridge_focused_metrics(np.concatenate(P), np.concatenate(Tt), np.concatenate(Pr))
    n_params = int(sum(p.numel() for p in model.parameters())) if model is not None else 0
    return {
        "field_rel_l2": float(np.mean(rel)),
        "field_rel_l2_fluct": float(np.mean(fluct)),
        "n_params": n_params,
        **{f"bridge_{k}": v for k, v in bridge.items()},
    }


def _build(name, kind, device):
    if kind in ("prior_only", "predict_mean"):
        return None
    base, delta = BACKBONES[name]
    if kind in ("data",):
        return base(1).to(device)
    if kind == "cond":
        return base(2).to(device)
    return delta(2).to(device)  # delta / delta_const


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-train", type=int, default=1000)
    p.add_argument("--n-val", type=int, default=200)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seeds", type=int, nargs="+", default=[1337, 1])
    p.add_argument("--backbones", nargs="+", default=list(BACKBONES))
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    device = a.device if (a.device == "cpu" or torch.cuda.is_available()) else "cpu"

    print("generating / loading WHNO data ...", flush=True)
    Ktr, Ttr = gen_or_load(a.n_train, "train", device, seed=1337)
    Kva, Tva = gen_or_load(a.n_val, "val", device, seed=99)
    prior_grid = _solve(torch.full((GRID, GRID), DATA_CFG["k_background"])).numpy()
    rng = np.random.default_rng(0)
    idx = rng.choice(GRID * GRID, size=N_POINTS, replace=False)
    logk_all = np.log(Ktr.reshape(Ktr.shape[0], -1).numpy())
    lm, ls = float(logk_all.mean()), float(logk_all.std() + 1e-9)
    a._const = float(Ttr.reshape(Ttr.shape[0], -1).mean())
    print(f"prior(uniform-k) rel-L2 vs samples is the baseline; const={a._const:.4f}", flush=True)

    train_ds = WHNODataset(Ktr, Ttr, prior_grid, idx, lm, ls)
    val_ds = WHNODataset(Kva, Tva, prior_grid, idx, lm, ls)

    # Roster: controls + per-backbone {data, cond, delta, delta_const}.
    roster = [("prior_only", "prior_only", "prior_only"), ("predict_mean", "predict_mean", "predict_mean")]
    for bb in a.backbones:
        roster += [(f"{bb}", bb, "data"), (f"cond_{bb}", bb, "cond"),
                   (f"delta_{bb}", bb, "delta"), (f"delta_const_{bb}", bb, "delta_const")]

    results = []
    for label, name, kind in roster:
        per_seed = []
        for seed in a.seeds:
            seed_everything(seed)
            model = _build(name, kind, device)
            tr = 0.0 if model is None else _train(model, kind, train_ds, a, device)
            m = _eval(model, kind, val_ds, device, a._const)
            m["seed"] = seed
            per_seed.append(m)
            print(f"[{label} seed={seed}] field_relL2={m['field_rel_l2']:.4f} "
                  f"fluct={m['field_rel_l2_fluct']:.4f} corr={m['bridge_correction_rel_l2']:.3f} "
                  f"bridge_t002={m.get('bridge_bridge_corr_rel_l2_t002', float('nan')):.3f} "
                  f"(<1 beats prior) train={tr:.0f}s", flush=True)
        keys = [k for k, v in per_seed[0].items() if isinstance(v, (int, float)) and k != "seed"]
        agg = {f"{k}_mean": float(np.mean([s[k] for s in per_seed if s.get(k) is not None])) for k in keys}
        agg |= {f"{k}_std": float(np.std([s[k] for s in per_seed if s.get(k) is not None])) for k in keys}
        results.append({"model": label, "backbone": name, "kind": kind, "seeds": a.seeds, **agg, "per_seed": per_seed})

    out = _REPO / "results" / "whno_phase0.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"config": vars(a), "results": results}, indent=2, default=str))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
