#!/usr/bin/env python
"""Block-2 benchmark: geometry-conditioned operators on 3-D irregular wall blocks.

Block-1 proved the winning recipe is *delta learning on an analytic 1-D clear-wall
prior* (``delta_fno`` generalised 5–10× better OOD than every alternative). Block-2
carries that recipe off the regular grid and onto the irregular point clouds of real
as-built scans, with **GINO** as the geometry-conditioned backbone. This runner trains
and tabulates the Block-2 roster on the synthetic 3-D corpus
(``data/processed/block2_train`` / ``block2_val``):

* ``gino`` — the data-only geometry operator. Features ``[logk_std, r_si, r_se]``
  (the prior channel is **dropped**): it must learn θ from scratch on the cloud.
* ``delta_gino`` — the prior-based operator. Same network, but it predicts only the
  *correction* on the analytic 1-D prior, which is added back per output query (the
  Block-1 recipe, now on irregular geometry).
* ``fno_voxel`` — the regular-grid reference. A 3-D FNO over the dense voxel field,
  reconstructed once from the scattered cloud. This is the grid baseline the
  point-cloud operators must match or beat without ever seeing a grid.
* ``prior_only`` — the zero-parameter control. Its per-point prediction *is* the
  analytic 1-D clear-wall prior channel; no network, no training. Scored with the same
  field rel-L2 and the same U-from-indoor-face estimator as every learned model, it is
  the row that separates "the operator helps" from "the prior is already good" — without
  it a low rel-L2 cannot be attributed to the operator. (Integrity control: the operator
  only earns the geometry result if it beats this baseline.)

For each model and seed (``SEEDS``) it reports the paired metrics the venue expects —
field **relative L2** and **U-value error** (U derived from the predicted field at the
indoor face via :func:`thermotwin.eval.building.u_from_indoor_face_cloud`, the same
estimator on every model and on the ground truth). All three are scored on the *same*
support — the original sampled points — so the voxel baseline is resampled back to the
cloud before scoring, a fair head-to-head. Results (mean ± std over seeds) are written
to ``results/block2_benchmark.{json,md}``.

    python scripts/benchmark_block2.py --epochs 150 --device cuda
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from thermotwin.data.pointcloud_dataset import (  # noqa: E402
    PointCloudDataset,
    collate_pointcloud,
    latent_grid_coords,
)
from thermotwin.eval.bridge_metrics import bridge_focused_metrics  # noqa: E402
from thermotwin.eval.building import u_from_indoor_face_cloud, u_value_report  # noqa: E402
from thermotwin.eval.metrics import relative_l2  # noqa: E402
from thermotwin.models.fno import build_fno  # noqa: E402
from thermotwin.models.gino import build_delta_gino, build_gino  # noqa: E402
from thermotwin.models.transolver import build_delta_transolver, build_transolver  # noqa: E402
from thermotwin.utils.seed import seed_everything  # noqa: E402

# Roster: name -> kind. The GINO / Transolver pairs differ only in whether the analytic
# 1-D prior is fed + added back (delta_*) or hidden (data-only); fno_voxel is the
# regular-grid reference; prior_only is the zero-parameter control whose prediction is the
# prior channel itself. Transolver (gridless physics-attention) is the diagnosis-driven
# contender added to break the equal-grid null — it predicts at the native points with no
# latent grid, so on the sub-voxel-fin `hard` corpus it can resolve features the 16³ voxel
# grid (and a 16³-latent GINO) alias. Data-only kinds drop the prior channel; delta kinds
# feed it and add it back.
ROSTER: dict[str, str] = {
    "gino": "gino",
    "delta_gino": "delta_gino",
    "transolver": "transolver",
    "delta_transolver": "delta_transolver",
    "fno_voxel": "fno_voxel",
    "prior_only": "prior_only",
}
# Data-only point-cloud kinds (prior channel dropped from the features they see).
_DATA_ONLY_CLOUD = ("gino", "transolver")
# Delta point-cloud kinds (full features incl. theta1d + the prior added back per query).
_DELTA_CLOUD = ("delta_gino", "delta_transolver")
SEEDS = [1337, 1]

# Latent / voxel grid resolution (the SDF is stored at G=16); FNO modes stay < G//2.
LATENT_GRID = 16
VOXEL_GRID = 16
FNO_MODES = (6, 6, 6)
# Generous GNO radii for the ~2k-point clouds (sparser than a dense scan).
GNO_RADIUS = 0.12
U_FACE_BAND = 0.08


def _build(kind: str, device: str, accel: bool = True) -> torch.nn.Module | None:
    """Construct a roster model. The two GINO entries share the architecture.

    Returns ``None`` for the zero-parameter ``prior_only`` control (no network).

    ``accel`` turns on the GINO GPU-throughput accelerators (per-sample neighbour-graph
    cache + on-GPU torch_cluster radius search) — a correctness-preserving change (the
    cached / torch_cluster CRS is set-identical to the native one); see
    :mod:`thermotwin.models.gino_accel`.
    """
    if kind == "prior_only":
        # Zero-parameter control: no network to build. The forward is the prior channel
        # itself, handled directly in _eval_model; nothing trains.
        return None
    gino_accel = dict(cache_neighbours=accel, neighbour_search_backend="auto")
    if kind == "gino":
        model = build_gino(
            in_channels=3,  # [logk_std, r_si, r_se] — prior dropped (data-only)
            fno_in_channels=32,
            fno_n_modes=FNO_MODES,
            fno_hidden_channels=64,
            fno_n_layers=4,
            in_gno_radius=GNO_RADIUS,
            out_gno_radius=GNO_RADIUS,
            latent_grid=LATENT_GRID,
            **gino_accel,
        )
    elif kind == "delta_gino":
        model = build_delta_gino(
            in_channels=4,  # [logk_std, r_si, r_se, theta1d] — prior fed + added back
            fno_in_channels=32,
            fno_n_modes=FNO_MODES,
            fno_hidden_channels=64,
            fno_n_layers=4,
            in_gno_radius=GNO_RADIUS,
            out_gno_radius=GNO_RADIUS,
            latent_grid=LATENT_GRID,
            **gino_accel,
        )
    elif kind == "transolver":
        # Gridless physics-attention; data-only (3-ch, prior dropped). No latent grid, so
        # it resolves sub-voxel fins the 16³ voxel / 16³-latent GINO both alias.
        model = build_transolver(
            in_channels=3, n_hidden=128, n_layers=8, n_head=8, slice_num=64, mlp_ratio=2
        )
    elif kind == "delta_transolver":
        # Same, but predicts the correction on the analytic 1-D prior (4-ch in, prior added).
        model = build_delta_transolver(
            in_channels=4, n_hidden=128, n_layers=8, n_head=8, slice_num=64, mlp_ratio=2
        )
    elif kind == "fno_voxel":
        model = build_fno(
            in_channels=4,  # voxelised [logk_std, r_si, r_se, theta1d]
            out_channels=1,
            n_modes=FNO_MODES,
            hidden_channels=64,
            n_layers=4,
        )
    else:  # pragma: no cover - guarded by ROSTER
        raise KeyError(kind)
    return model.to(device)


def _accelerate(model) -> contextlib.AbstractContextManager:
    """Activate the GINO neighbour-cache/search accelerators if the model exposes them.

    Returns a no-op context for the grid baselines (plain FNO), so callers can wrap any
    roster model uniformly.
    """
    accel = getattr(model, "accelerate", None)
    return accel() if callable(accel) else contextlib.nullcontext()


def _set_sample_key(model, batch: dict) -> None:
    """Point the GINO neighbour cache at this batch's sample (no-op for grid models)."""
    setter = getattr(model, "set_sample_key", None)
    if callable(setter):
        setter(int(batch["sample_index"][0]))


def _forward_cloud(model, kind: str, batch: dict, device: str) -> torch.Tensor:
    """One forward of a point-cloud model (GINO or Transolver) -> ``(B, n_out, 1)``.

    Both families share the ``(input_geom, feats, latent_queries, sdf, output_queries)``
    signature (Transolver ignores the latent grid / SDF / queries); the delta variants
    take an extra per-query ``prior``. Data-only kinds see the prior-dropped features.
    """
    input_geom = batch["input_geom"].to(device)
    sdf = batch["sdf"].to(device)
    latent_queries = batch["latent_queries"].to(device)
    output_queries = batch["output_queries"].to(device)
    if kind in _DATA_ONLY_CLOUD:
        feats = batch["gino_feats"].to(device)  # prior channel dropped
        return model(input_geom, feats, latent_queries, sdf, output_queries)
    feats = batch["feats"].to(device)  # delta_*: full 4-ch incl theta1d
    prior = batch["prior"].to(device)
    return model(input_geom, feats, latent_queries, sdf, output_queries, prior)


def _trilinear_sample(field: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """Sample a ``(G, G, G)`` cell-centred field at ``points`` in ``[0, 1]^3``.

    Uses ``grid_sample`` so the voxel baseline can be scored on the *same* scattered
    points as the GINO models. Returns ``(N,)``.
    """
    grid = field.shape[-1]
    # grid_sample wants coords in [-1, 1]; cell centres sit at (i + 0.5)/G, so map
    # the unit cube through that cell-centred convention and flip axis order to x,y,z.
    norm = points * 2.0 - 1.0  # [0,1] -> [-1,1]
    coords = norm.flip(-1).view(1, -1, 1, 1, 3)  # (1, N, 1, 1, 3), order (z, y, x)
    vol = field.view(1, 1, grid, grid, grid)
    out = torch.nn.functional.grid_sample(
        vol, coords, mode="bilinear", align_corners=False, padding_mode="border"
    )
    return out.view(-1)


def _eval_model(model, kind: str, val_ds, device: str) -> dict:
    """Per-sample field rel-L2 + U-MAE on the validation corpus (scored on points)."""
    rel_l2s, u_pred, u_true, u_clear, times = [], [], [], [], []
    pred_all, true_all, prior_all = [], [], []  # concatenated for the bridge-focused metric
    if model is not None:
        model.eval()
    loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate_pointcloud)
    with torch.no_grad(), _accelerate(model):
        for batch in loader:
            theta_gt = batch["theta"][0].numpy()  # (n,)
            points = batch["points"][0].numpy()
            prior = batch["prior"][0].numpy()
            uc = float(batch["u_clear"][0])
            t0 = time.perf_counter()
            if kind == "prior_only":
                # Zero-parameter control: the prediction *is* the per-point prior channel,
                # scored on the same support with the same field rel-L2 and U estimator.
                pred = prior.copy()
            elif kind == "fno_voxel":
                vx = batch["voxel_feats"].to(device)  # (1, F, G, G, G)
                vox_pred = model(vx)[0, 0]  # (G, G, G)
                pred = _trilinear_sample(vox_pred, batch["points"][0].to(device)).cpu().numpy()
            else:
                _set_sample_key(model, batch)
                pred = _forward_cloud(model, kind, batch, device)[0, :, 0].cpu().numpy()
            times.append(time.perf_counter() - t0)
            rel_l2s.append(
                relative_l2(torch.from_numpy(pred)[None], torch.from_numpy(theta_gt)[None]).item()
            )
            u_pred.append(u_from_indoor_face_cloud(pred, prior, points, uc, band=U_FACE_BAND))
            u_true.append(float(batch["u_value"][0]))
            u_clear.append(uc)
            pred_all.append(pred)
            true_all.append(theta_gt)
            prior_all.append(prior)
    op = u_value_report(np.array(u_pred), np.array(u_true))
    base = u_value_report(np.array(u_clear), np.array(u_true))
    # Bridge-focused metrics over the whole val cloud (the localized correction the global
    # field rel-L2 washes out). correction_rel_l2 < 1 means the model beats the prior;
    # prior_only is exactly 1.0 by construction. See eval/bridge_metrics.py.
    bridge = bridge_focused_metrics(
        np.concatenate(pred_all), np.concatenate(true_all), np.concatenate(prior_all)
    )
    return {
        "field_rel_l2": float(np.mean(rel_l2s)),
        "u_mae": op["u_mae"],
        "u_mape": op["u_mape"],
        "u_mae_clear_baseline": base["u_mae"],
        "u_improvement_x": (base["u_mae"] / op["u_mae"]) if op["u_mae"] else None,
        "infer_ms_per_sample": float(np.mean(times) * 1e3),
        **{f"bridge_{k}": v for k, v in bridge.items()},
        # Per-sample val predictions (popped by main for --save-preds; never JSON-serialised).
        "_sample_preds": pred_all,
        "_sample_files": [Path(f).name for f in getattr(val_ds, "files", [])],
    }


def _train_one(model, kind: str, train_ds, args, device: str) -> float:
    """Train one model (AdamW + cosine). Returns wall-clock seconds."""
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_pointcloud,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
    )
    t0 = time.perf_counter()
    with _accelerate(model):  # activates the GINO neighbour cache for the whole run
        for _ in range(args.epochs):
            model.train()
            for batch in loader:
                opt.zero_grad()
                if kind == "fno_voxel":
                    vx = batch["voxel_feats"].to(device)  # (1, F, G, G, G)
                    pred = model(vx)  # (1, 1, G, G, G)
                    target = batch["voxel_theta"].to(device).unsqueeze(1)  # (1, 1, G, G, G)
                else:
                    _set_sample_key(model, batch)
                    pred = _forward_cloud(model, kind, batch, device)  # (1, n, 1)
                    target = batch["theta"].to(device).unsqueeze(-1)  # (1, n, 1)
                loss = relative_l2(pred, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            sched.step()
    return time.perf_counter() - t0


def _aggregate(per_seed: list[dict]) -> dict:
    """Mean ± std over seeds for every numeric metric (incl. the bridge-focused keys).

    Generic over keys so new metrics aggregate automatically; ``None`` and ``NaN`` are
    dropped per-key (e.g. a bridge threshold with no points in some seeds).
    """
    skip = {"seed", "train_time_s"}
    keys = [
        k
        for k in per_seed[0]
        if k not in skip and isinstance(per_seed[0][k], (int, float))
    ]
    agg: dict[str, float] = {}
    for k in keys:
        vals = [
            s[k]
            for s in per_seed
            if s.get(k) is not None and not (isinstance(s[k], float) and np.isnan(s[k]))
        ]
        if vals:
            agg[f"{k}_mean"] = float(np.mean(vals))
            agg[f"{k}_std"] = float(np.std(vals))
    return agg


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--models", nargs="+", default=list(ROSTER))
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    p.add_argument("--device", default="cuda")
    p.add_argument("--train_root", default=None)
    p.add_argument("--val_root", default=None)
    p.add_argument(
        "--corpus",
        choices=["box", "irreg", "hard", "realcg"],
        default="box",
        help=(
            "Which corpus to benchmark. 'box' -> block2_train/val + results/block2_benchmark.*; "
            "'irreg' -> block2_irreg_train/val + results/block2_irreg_benchmark.*; "
            "'hard' -> block2_hard_train/val + results/block2_hard_benchmark.* (the fine-native "
            "sub-voxel-fin corpus where a 16³ voxel grid genuinely aliases the bridge). "
            "Sets train_root/val_root and output stem unless they are overridden explicitly."
        ),
    )
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument(
        "--save-preds",
        dest="save_preds",
        action="store_true",
        help="dump per-sample val predictions (first seed) to results/preds/<stem>__<model>.npz "
        "for the viz prediction-overlay (scripts/make_figures.py --preds).",
    )
    p.add_argument(
        "--out_stem",
        default=None,
        help=(
            "Override the results file stem (results/<stem>.{json,md}). Defaults to the "
            "corpus stem; set it to avoid clobbering committed artefacts when re-running an "
            "existing corpus with a different roster (e.g. block2_irreg_ops_benchmark)."
        ),
    )
    p.add_argument(
        "--no-accel",
        dest="accel",
        action="store_false",
        help="disable the GINO neighbour-cache/GPU-search accelerators (default: on)",
    )
    p.set_defaults(accel=True)
    a = p.parse_args()

    # Resolve the corpus -> default roots + output stem. Explicit --train_root/--val_root
    # win over the corpus default so the runner stays fully overridable.
    corpus_defaults = {
        "box": ("data/processed/block2_train", "data/processed/block2_val", "block2_benchmark"),
        "irreg": (
            "data/processed/block2_irreg_train",
            "data/processed/block2_irreg_val",
            "block2_irreg_benchmark",
        ),
        "hard": (
            "data/processed/block2_hard_train",
            "data/processed/block2_hard_val",
            "block2_hard_benchmark",
        ),
        "realcg": (
            "data/processed/block2_realcg_train",
            "data/processed/block2_realcg_val",
            "block2_realcg_benchmark",
        ),
    }
    default_train, default_val, out_stem = corpus_defaults[a.corpus]
    if a.train_root is None:
        a.train_root = default_train
    if a.val_root is None:
        a.val_root = default_val
    if a.out_stem:  # avoid clobbering committed artefacts when re-running with a new roster
        out_stem = a.out_stem

    device = a.device if (a.device == "cpu" or torch.cuda.is_available()) else "cpu"
    train_root, val_root = _REPO / a.train_root, _REPO / a.val_root

    # The two GINO entries train on the cloud; fno_voxel needs the dense voxel field.
    need_voxel = "fno_voxel" in a.models
    # The corpus is a few MB: cache it in RAM so np.load is paid once, not per step.
    train_ds = PointCloudDataset(
        train_root,
        latent_grid=LATENT_GRID,
        voxelise=need_voxel,
        voxel_grid=VOXEL_GRID,
        cache_in_memory=True,
    )
    val_ds = PointCloudDataset(
        val_root,
        latent_grid=LATENT_GRID,
        voxelise=need_voxel,
        voxel_grid=VOXEL_GRID,
        cache_in_memory=True,
    )
    # Latent queries are shared across the corpus (one fixed latent grid).
    _ = latent_grid_coords(LATENT_GRID)

    results = []
    for name in a.models:
        kind = ROSTER[name]
        per_seed = []
        params = None
        for seed in a.seeds:
            seed_everything(seed)
            model = _build(kind, device, accel=a.accel)
            # prior_only is a zero-parameter control: nothing to count, nothing to train.
            params = 0 if model is None else sum(q.numel() for q in model.parameters())
            train_s = 0.0 if model is None else _train_one(model, kind, train_ds, a, device)
            m = _eval_model(model, kind, val_ds, device)
            # Per-sample predictions are popped here so they never bloat the JSON; saved (first
            # seed only) when --save-preds, for the viz prediction-overlay (scripts/make_figures).
            sample_preds = m.pop("_sample_preds", None)
            sample_files = m.pop("_sample_files", None)
            if a.save_preds and seed == a.seeds[0] and sample_preds is not None:
                pdir = _REPO / "results" / "preds"
                pdir.mkdir(parents=True, exist_ok=True)
                np.savez(
                    pdir / f"{out_stem}__{name}.npz",
                    files=np.array(sample_files, dtype=object),
                    **{f"p{i}": v for i, v in enumerate(sample_preds)},
                )
            m["seed"] = seed
            m["train_time_s"] = round(train_s, 1)
            per_seed.append(m)
            print(
                f"[{name} seed={seed}] relL2={m['field_rel_l2']:.4f} "
                f"corr_relL2={m['bridge_correction_rel_l2']:.3f} "
                f"bridge_corr={m.get('bridge_bridge_corr_rel_l2_t002', float('nan')):.3f} "
                f"(<1 beats prior) train={train_s:.0f}s"
            )
        agg = _aggregate(per_seed)
        results.append(
            {
                "model": name,
                "kind": kind,
                "params": int(params),
                "seeds": a.seeds,
                **agg,
                "per_seed": per_seed,
            }
        )

    out = _REPO / "results"
    (out / "logs").mkdir(parents=True, exist_ok=True)
    report = {
        "config": vars(a)
        | {
            "device": device,
            "latent_grid": LATENT_GRID,
            "voxel_grid": VOXEL_GRID,
            "u_face_band": U_FACE_BAND,
            "gino_accel": a.accel,
        },
        "results": results,
    }
    (out / f"{out_stem}.json").write_text(json.dumps(report, indent=2))
    _write_markdown(out / f"{out_stem}.md", report)
    print(f"\nwrote {out / f'{out_stem}.json'} and .md")


def _write_markdown(path: Path, report: dict) -> None:
    cfg = report["config"]
    corpus = cfg.get("corpus", "box")
    corpus_label = {
        "box": "axis-aligned box corpus",
        "irreg": "rotated / off-lattice irregular corpus",
    }.get(corpus, corpus)
    lines = [
        f"# Block-2 Benchmark — 3-D wall blocks · {corpus_label} (GINO vs grid FNO)",
        "",
        f"- corpus: `{corpus}` · train: `{cfg['train_root']}` · val: `{cfg['val_root']}`",
        f"- device: `{cfg['device']}` · epochs: {cfg['epochs']} · batch: {cfg['batch_size']} "
        f"· seeds: {cfg['seeds']} · latent/voxel grid: {cfg['latent_grid']}",
        "- U-value derived from the predicted field at the indoor face "
        f"(near-face band {cfg['u_face_band']} in normalised coords); the same "
        "estimator is applied to every model and to the ground truth.",
        "- field rel-L2 and U-MAE are scored on the **sampled points** for all models "
        "(the voxel FNO is resampled back to the cloud), a fair head-to-head.",
        "",
        "| Model | Kind | Field rel-L2 ↓ | U-value MAE ↓ (W/m²K) | U-MAPE ↓ | "
        "vs 1-D clear ↑ | Infer (ms) | Params |",
        "|---|---|---|---|---|---|---|---|",
    ]

    def fmt(r, key):
        m, s = r.get(f"{key}_mean"), r.get(f"{key}_std")
        return f"{m:.4f} ± {s:.4f}" if m is not None else "—"

    for r in sorted(report["results"], key=lambda x: x.get("field_rel_l2_mean", 1e9)):
        imp = r.get("u_improvement_x_mean")
        imp_cell = f"{imp:.2f}×" if imp is not None else "—"
        lines.append(
            f"| {r['model']} | {r['kind']} | {fmt(r, 'field_rel_l2')} | "
            f"{fmt(r, 'u_mae')} | {r.get('u_mape_mean', float('nan')):.1f}% | "
            f"{imp_cell} | {r.get('infer_ms_per_sample_mean', float('nan')):.2f} | "
            f"{r['params']:,} |"
        )
    base = report["results"][0].get("u_mae_clear_baseline_mean")
    lines += [
        "",
        f"Geometry-blind 1-D clear-wall baseline U-MAE: "
        f"{base:.4f} W/m²K — the number any geometry-aware operator must beat (H1)."
        if base is not None
        else "",
        "",
    ]

    # --- Bridge-focused table: does the operator's correction beat the prior? --------------
    # correction_rel_l2 = ‖pred−true‖ / ‖prior−true‖ — prior_only ≡ 1.0, so < 1 beats the
    # prior; the bridge columns focus that on the points the bridge actually perturbs
    # (|true−prior| > τ in θ). This is the metric the global field rel-L2 washes out.
    bf02 = report["results"][0].get("bridge_bridge_frac_t002_mean")
    lines += [
        "## Bridge-focused — does the learned correction beat the prior?",
        "",
        "`correction rel-L2` = ‖pred−true‖ / ‖prior−true‖ — **`prior_only` ≡ 1.000 by "
        "construction, so < 1.000 means the operator genuinely beats the analytic prior**; "
        "the *bridge* columns restrict this to points the bridge perturbs (|true−prior| > τ "
        "in θ). `clear rel-L2` guards that clear walls are not corrupted.",
        ""
        + (
            f"Bridge region (τ=0.02) covers ~{bf02 * 100:.1f}% of points.\n"
            if bf02 is not None
            else ""
        ),
        "| Model | correction rel-L2 ↓ | bridge corr-relL2 (τ=0.02) ↓ | "
        "bridge corr-relL2 (τ=0.05) ↓ | correction R² ↑ | clear rel-L2 ↓ |",
        "|---|---|---|---|---|---|",
    ]
    for r in sorted(report["results"], key=lambda x: x.get("bridge_correction_rel_l2_mean", 1e9)):
        lines.append(
            f"| {r['model']} | {fmt(r, 'bridge_correction_rel_l2')} | "
            f"{fmt(r, 'bridge_bridge_corr_rel_l2_t002')} | "
            f"{fmt(r, 'bridge_bridge_corr_rel_l2_t005')} | "
            f"{fmt(r, 'bridge_correction_r2')} | {fmt(r, 'bridge_clear_field_rel_l2')} |"
        )
    lines.append("")
    path.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
