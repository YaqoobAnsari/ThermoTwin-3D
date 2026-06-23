#!/usr/bin/env python
"""Train a ThermoTwin-3D neural operator on the synthetic FEM corpus.

Block-1 entry point. Composes a Hydra config (data + model + train) and trains the
operator to predict the dimensionless temperature field, reporting the relative-L2
metric on a held-out split. GPU if available, else CPU (e.g. a login-node smoke run).

    python scripts/train.py experiment=block1_synthetic_fem
    python scripts/train.py experiment=block1_synthetic_fem train.epochs=15 device=cpu
"""

from __future__ import annotations

import json
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from thermotwin.data.dataset import SyntheticFEMDataset
from thermotwin.eval.metrics import relative_l2, rmse
from thermotwin.losses.heat_residual import heat_residual_loss
from thermotwin.models.registry import build_model
from thermotwin.utils.seed import seed_everything

_REPO = Path(__file__).resolve().parents[1]


@torch.no_grad()
def evaluate(model, loader, device) -> dict[str, float]:
    model.eval()
    rl2, rms, n = 0.0, 0.0, 0
    for batch in loader:
        x, y = batch[0], batch[1]
        x, y = x.to(device), y.to(device)
        pred = model(x)
        bs = x.shape[0]
        rl2 += relative_l2(pred, y).item() * bs
        rms += rmse(pred, y).item() * bs
        n += bs
    return {"rel_l2": rl2 / n, "rmse": rms / n}


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed)
    device = cfg.device if (cfg.device == "cpu" or torch.cuda.is_available()) else "cpu"
    print(f"device={device}  model={cfg.model.name}")

    physics_weight = float(cfg.train.get("physics_weight", 0.0))
    use_physics = physics_weight > 0.0
    if use_physics:
        print(f"physics-informed: physics_weight={physics_weight}")

    # Data roots are repo-relative; resolve against the repo, not Hydra's job dir.
    feature_set = cfg.data.get("feature_set", "base")
    train_ds = SyntheticFEMDataset(
        _REPO / cfg.data.train_root,
        cfg.data.target_width,
        return_physics=use_physics,
        feature_set=feature_set,
    )
    val_ds = SyntheticFEMDataset(
        _REPO / cfg.data.val_root, cfg.data.target_width, feature_set=feature_set
    )
    train_loader = DataLoader(
        train_ds, batch_size=cfg.train.batch_size, shuffle=True, num_workers=cfg.data.num_workers
    )
    val_loader = DataLoader(val_ds, batch_size=cfg.train.batch_size)
    print(f"train={len(train_ds)}  val={len(val_ds)}")

    model = build_model(cfg.model).to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.train.epochs)

    best = float("inf")
    history = []
    for epoch in range(cfg.train.epochs):
        model.train()
        running = 0.0
        for batch in train_loader:
            x, y = batch[0].to(device), batch[1].to(device)
            opt.zero_grad()
            pred = model(x)
            loss = relative_l2(pred, y)
            if use_physics:
                phys = {key: val.to(device) for key, val in batch[2].items()}
                loss = loss + physics_weight * heat_residual_loss(pred, **phys)
            loss.backward()
            if cfg.train.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            opt.step()
            running += loss.item() * x.shape[0]
        sched.step()

        train_l2 = running / len(train_ds)
        val = evaluate(model, val_loader, device)
        history.append({"epoch": epoch, "train_rel_l2": train_l2, **val})
        if epoch % max(1, cfg.train.log_every // 10) == 0 or epoch == cfg.train.epochs - 1:
            print(
                f"epoch {epoch:3d}  train_relL2 {train_l2:.4f}  "
                f"val_relL2 {val['rel_l2']:.4f}  val_rmse {val['rmse']:.4f}"
            )
        if val["rel_l2"] < best:
            best = val["rel_l2"]
            torch.save(model.state_dict(), "best.pt")

    Path("metrics.json").write_text(
        json.dumps(
            {
                "best_val_rel_l2": best,
                "config": OmegaConf.to_container(cfg, resolve=True),
                "history": history,
            },
            indent=2,
        )
    )
    print(f"done. best val rel_L2 = {best:.4f}  (best.pt, metrics.json in {Path.cwd()})")


if __name__ == "__main__":
    main()
