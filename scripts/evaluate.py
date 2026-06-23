#!/usr/bin/env python
"""Evaluate a trained operator: field accuracy + building U-value error.

Runs the model at each sample's **native resolution** (not the training grid), so
this doubles as a discretisation-invariance check. Reports the paired metrics the
venue expects — relative L2 (field) and U-value error (building) — and contrasts
the operator's U-error against the **1-D clear-wall baseline** (assume no thermal
bridge), which is the direct "does geometry resolution earn its keep" number (H1).

    python scripts/evaluate.py experiment=block1_synthetic_fem ckpt=outputs/.../best.pt
    python scripts/evaluate.py experiment=block1_synthetic_fem ckpt=best.pt data.val_root=data/processed/block1_val
"""

from __future__ import annotations

import json
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

from thermotwin.data.dataset import build_input_channels
from thermotwin.eval.building import effective_u_from_theta, u_value_report
from thermotwin.eval.metrics import relative_l2
from thermotwin.models.registry import build_model
from thermotwin.utils.seed import seed_everything

_REPO = Path(__file__).resolve().parents[1]


def _native_input(d, feature_set: str = "base") -> tuple[torch.Tensor, np.ndarray, dict]:
    """Build the model input at a sample's native resolution + carry GT scalars.

    Uses the same featuriser as training (``build_input_channels``), so models that
    need the enriched channels (e.g. ``delta_fno``) are evaluated correctly.
    """
    k = d["k"].astype(np.float32)
    t_in, t_out = float(d["t_indoor"]), float(d["t_outdoor"])
    r_si, r_se = float(d["r_si"]), float(d["r_se"])
    theta_gt = (d["temperature"].astype(np.float32) - t_out) / (t_in - t_out)
    x = build_input_channels(k, d["dx0"], float(d["dy"]), r_si, r_se, feature_set)[None]
    meta = {
        "k": k,
        "dx0": d["dx0"],
        "dy": float(d["dy"]),
        "r_si": r_si,
        "u_value": float(d["u_value"]),
        "u_clear": float(d["u_clear"]),
    }
    return torch.from_numpy(x.astype(np.float32)), theta_gt, meta


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed)
    if cfg.get("ckpt") is None:
        raise SystemExit("provide a checkpoint: ckpt=<path to best.pt>")
    device = cfg.device if (cfg.device == "cpu" or torch.cuda.is_available()) else "cpu"

    model = build_model(cfg.model).to(device)
    ckpt = Path(cfg.ckpt)
    if not ckpt.is_absolute():
        ckpt = _REPO / ckpt
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    val_root = _REPO / cfg.data.val_root
    manifest = json.loads((val_root / "manifest.json").read_text())
    feature_set = cfg.data.get("feature_set", "base")

    rel_l2s, u_pred, u_true, u_clear = [], [], [], []
    with torch.no_grad():
        for row in manifest["samples"]:
            d = np.load(val_root / row["file"])
            x, theta_gt, m = _native_input(d, feature_set)
            pred = model(x.to(device))[0, 0].cpu().numpy()
            rel_l2s.append(
                relative_l2(
                    torch.from_numpy(pred)[None, None], torch.from_numpy(theta_gt)[None, None]
                ).item()
            )
            u_pred.append(effective_u_from_theta(pred, m["k"], m["dx0"], m["dy"], m["r_si"]))
            u_true.append(m["u_value"])
            u_clear.append(m["u_clear"])

    operator = u_value_report(np.array(u_pred), np.array(u_true))
    baseline = u_value_report(np.array(u_clear), np.array(u_true))  # 1-D clear-wall guess
    report = {
        "n_val": len(u_true),
        "field_rel_l2_mean": float(np.mean(rel_l2s)),
        "u_operator": operator,
        "u_clear_baseline": baseline,
        "u_mae_improvement_x": baseline["u_mae"] / operator["u_mae"] if operator["u_mae"] else None,
    }
    Path("eval_metrics.json").write_text(json.dumps(report, indent=2))

    print(f"val samples: {report['n_val']}  (native-resolution eval)")
    print(f"field relative L2 : {report['field_rel_l2_mean']:.4f}")
    print(
        f"U-value MAE  operator : {operator['u_mae']:.4f} W/m2K  (MAPE {operator['u_mape']:.1f}%)"
    )
    print(
        f"U-value MAE  1-D clear: {baseline['u_mae']:.4f} W/m2K  (MAPE {baseline['u_mape']:.1f}%)"
    )
    if report["u_mae_improvement_x"]:
        print(
            f"=> operator cuts U-value error {report['u_mae_improvement_x']:.1f}x vs ignoring bridges (H1)"
        )


if __name__ == "__main__":
    main()
