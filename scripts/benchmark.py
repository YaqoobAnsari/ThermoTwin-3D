#!/usr/bin/env python
"""Block-1 benchmark: train every model in the roster and tabulate the numbers.

Establishes the leaderboard we then try to beat. For each model it reports the
paired metrics the venue expects — field **relative L2** and **U-value error** —
at each sample's native resolution (also a discretisation-invariance check), plus
the inference **speedup vs the finite-volume solver** (the operator-learning
headline) and the gap to the geometry-blind **1-D clear-wall baseline** (H1).

Writes ``results/block1_benchmark.json`` and a markdown leaderboard.

    python scripts/benchmark.py --models fno cnn --epochs 200 --device cuda
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from thermotwin.data.dataset import LOGK_MEAN, LOGK_STD, SyntheticFEMDataset
from thermotwin.eval.building import effective_u_from_theta, u_value_report
from thermotwin.eval.metrics import relative_l2
from thermotwin.models.registry import build_model
from thermotwin.utils.seed import seed_everything

_REPO = Path(__file__).resolve().parents[1]


def _native_eval(model, val_root: Path, device) -> dict:
    """Per-sample native-resolution metrics + inference timing."""
    manifest = json.loads((val_root / "manifest.json").read_text())
    rel_l2s, u_pred, u_true, u_clear, times = [], [], [], [], []
    model.eval()
    with torch.no_grad():
        for row in manifest["samples"]:
            d = np.load(val_root / row["file"])
            k = d["k"].astype(np.float32)
            t_in, t_out = float(d["t_indoor"]), float(d["t_outdoor"])
            r_si = float(d["r_si"])
            theta_gt = (d["temperature"].astype(np.float32) - t_out) / (t_in - t_out)
            logk = (np.log10(k) - LOGK_MEAN) / LOGK_STD
            x = np.stack([logk, np.full_like(logk, r_si), np.full_like(logk, float(d["r_se"]))])
            xt = torch.from_numpy(x[None].astype(np.float32)).to(device)
            t0 = time.perf_counter()
            pred = model(xt)[0, 0].cpu().numpy()
            times.append(time.perf_counter() - t0)
            rel_l2s.append(
                relative_l2(
                    torch.from_numpy(pred)[None, None], torch.from_numpy(theta_gt)[None, None]
                ).item()
            )
            u_pred.append(effective_u_from_theta(pred, k, d["dx0"], float(d["dy"]), r_si))
            u_true.append(float(d["u_value"]))
            u_clear.append(float(d["u_clear"]))
    op = u_value_report(np.array(u_pred), np.array(u_true))
    base = u_value_report(np.array(u_clear), np.array(u_true))
    return {
        "field_rel_l2": float(np.mean(rel_l2s)),
        "u_mae": op["u_mae"],
        "u_mape": op["u_mape"],
        "u_mae_clear_baseline": base["u_mae"],
        "u_improvement_x": base["u_mae"] / op["u_mae"] if op["u_mae"] else None,
        "infer_ms_per_sample": float(np.mean(times) * 1e3),
    }


def _fv_solver_ms(val_root: Path, n: int = 16) -> float:
    """Mean wall-clock of the finite-volume GT solver per sample (speedup ref)."""
    from thermotwin.physics.steady_fv import DirichletFilm, solve_steady_conduction

    manifest = json.loads((val_root / "manifest.json").read_text())
    ts = []
    for row in manifest["samples"][:n]:
        d = np.load(val_root / row["file"])
        k, dx0, dy = d["k"], d["dx0"], float(d["dy"])
        bc = DirichletFilm(float(d["t_indoor"]), float(d["t_outdoor"]),
                           r_lo=float(d["r_si"]), r_hi=float(d["r_se"]))
        t0 = time.perf_counter()
        solve_steady_conduction(k, [dx0, dy], bc)
        ts.append(time.perf_counter() - t0)
    return float(np.mean(ts) * 1e3)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", default=["fno", "cnn"])
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--device", default="cuda")
    p.add_argument("--train_root", default="data/processed/block1_train")
    p.add_argument("--val_root", default="data/processed/block1_val")
    p.add_argument("--target_width", type=int, default=48)
    a = p.parse_args()

    device = a.device if (a.device == "cpu" or torch.cuda.is_available()) else "cpu"
    train_root, val_root = _REPO / a.train_root, _REPO / a.val_root
    train_ds = SyntheticFEMDataset(train_root, a.target_width)

    fv_ms = _fv_solver_ms(val_root)
    rows = []
    for name in a.models:
        seed_everything(a.seed)  # same init seed -> fair comparison
        model_cfg = OmegaConf.load(_REPO / "configs" / "model" / f"{name}.yaml")
        model = build_model(model_cfg).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
        loader = DataLoader(train_ds, batch_size=a.batch_size, shuffle=True)

        t0 = time.perf_counter()
        for _ in range(a.epochs):
            model.train()
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                opt.zero_grad()
                loss = relative_l2(model(x), y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            sched.step()
        train_s = time.perf_counter() - t0

        m = _native_eval(model, val_root, device)
        m["speedup_vs_fv"] = fv_ms / m["infer_ms_per_sample"] if m["infer_ms_per_sample"] else None
        m.update({"model": name, "params": int(n_params), "train_time_s": round(train_s, 1)})
        rows.append(m)
        print(
            f"[{name}] relL2={m['field_rel_l2']:.4f} U-MAE={m['u_mae']:.4f} "
            f"(clear {m['u_mae_clear_baseline']:.4f}, {m['u_improvement_x']:.2f}x) "
            f"speedup={m['speedup_vs_fv']:.0f}x params={n_params} train={train_s:.0f}s"
        )

    out = _REPO / "results"
    out.mkdir(exist_ok=True)
    report = {
        "config": vars(a) | {"device": device, "fv_solver_ms_per_sample": fv_ms},
        "results": rows,
    }
    (out / "block1_benchmark.json").write_text(json.dumps(report, indent=2))
    _write_markdown(out / "block1_benchmark.md", report)
    print(f"\nwrote {out/'block1_benchmark.json'} and .md")


def _write_markdown(path: Path, report: dict) -> None:
    cfg = report["config"]
    lines = [
        "# Block-1 Benchmark — synthetic FEM (layered walls + thermal bridges)",
        "",
        f"- device: `{cfg['device']}` · epochs: {cfg['epochs']} · batch: {cfg['batch_size']} "
        f"· seed: {cfg['seed']}",
        f"- FV ground-truth solver: {cfg['fv_solver_ms_per_sample']:.2f} ms/sample",
        "",
        "| Model | Field rel-L2 ↓ | U-value MAE ↓ (W/m²K) | U-MAPE ↓ | vs 1-D clear ↑ | "
        "Speedup vs FV ↑ | Params | Train (s) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in sorted(report["results"], key=lambda x: x["field_rel_l2"]):
        lines.append(
            f"| {r['model']} | {r['field_rel_l2']:.4f} | {r['u_mae']:.4f} | "
            f"{r['u_mape']:.1f}% | {r['u_improvement_x']:.2f}× | "
            f"{r['speedup_vs_fv']:.0f}× | {r['params']:,} | {r['train_time_s']:.0f} |"
        )
    lines += [
        "",
        f"Geometry-blind 1-D clear-wall baseline U-MAE: "
        f"{report['results'][0]['u_mae_clear_baseline']:.4f} W/m²K — the number any "
        "geometry-aware model must beat (H1).",
        "",
    ]
    path.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
