#!/usr/bin/env python
"""Block-1 ablation runner — does any variant *robustly* beat the data-only FNO?

The number to beat is the data-only FNO (field rel-L2 0.0144, U-MAE 0.0205). U-MAE
is set by the through-wall theta gradient at the indoor (axis-0 lo) face, and the
plain FNO loses there twice: spectral bias smears the sharp bridge gradient, and the
FFT's periodic wraparound contaminates our non-periodic Dirichlet/film faces. This
script sweeps a fixed roster of countermeasures — domain padding, the enriched
clear-wall prior, the delta (residual-on-prior) head, the U-FNO local path, and the
U-value / physics-residual loss levers — each trained over several seeds, and asks
whether the mean U-MAE improvement clears the pooled seed noise (a *robust* win).

Self-contained: it does **not** import :mod:`scripts.benchmark`. The roster lives
in-code (:data:`ROSTER`); everything else is reusable library calls into
``thermotwin``. Evaluation is at each val sample's **native** resolution (the val
corpus mixes widths 32/48/64), featurised with the variant's own feature set via
:func:`thermotwin.data.dataset.build_input_channels`, so the discretisation-invariance
of the operator is exercised and the U-value boundary flux is read on the true grid.

    python scripts/ablate.py --epochs 300 --device cuda
    python scripts/ablate.py --variants fno delta_fno --seeds 1337 --epochs 50

Writes ``results/block1_ablations.json`` and ``results/block1_ablations.md`` (sorted
by overall U-MAE ascending, with the data-only ``fno`` row marked as the reference).
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from thermotwin.data.dataset import SyntheticFEMDataset, build_input_channels
from thermotwin.eval.building import effective_u_from_theta
from thermotwin.eval.metrics import relative_l2
from thermotwin.losses.building_loss import u_value_loss
from thermotwin.losses.heat_residual import heat_residual_loss
from thermotwin.models.registry import build_model
from thermotwin.utils.seed import seed_everything

_REPO = Path(__file__).resolve().parents[1]

# Reference variant whose mean U-MAE every other variant is measured against.
REFERENCE = "fno"


@dataclass(frozen=True)
class Variant:
    """One ablation cell: a model build config, its featuriser and loss weights.

    Args:
        name: roster key / report label.
        model_cfg: mapping handed to :func:`thermotwin.models.registry.build_model`
            (must carry ``name`` and ``in_channels``); per-axis ``domain_padding``
            and ``clearwall_index`` are read straight from here.
        feature_set: ``'base'`` (3ch) or ``'enriched'`` (6ch) — drives both the
            dataset featuriser and the native-resolution eval featuriser.
        loss_weights: weights for ``{data, u_value, physics}``; ``data`` is the
            field relative-L2 term, ``u_value`` adds the indoor-face U discrepancy,
            ``physics`` adds the steady-FV PDE residual.
    """

    name: str
    model_cfg: dict
    feature_set: str = "base"
    loss_weights: dict[str, float] = field(default_factory=lambda: {"data": 1.0})

    @property
    def u_weight(self) -> float:
        return float(self.loss_weights.get("u_value", 0.0))

    @property
    def physics_weight(self) -> float:
        return float(self.loss_weights.get("physics", 0.0))

    @property
    def needs_physics(self) -> bool:
        """Whether the per-cell physics bundle is needed during training."""
        return self.u_weight > 0.0 or self.physics_weight > 0.0


# Shared FNO/operator hyper-parameters (match configs/model/*.yaml).
_N_MODES = [8, 16]
_HIDDEN = 32
_N_LAYERS = 4

# The roster. Each cell pins the architecture, its input featuriser and the loss
# levers; everything else (optimiser, schedule, seeds) is fixed across the sweep so
# the only moving part is the countermeasure under test.
ROSTER: tuple[Variant, ...] = (
    Variant(
        name="fno",  # reference / the number to beat (data-only FNO)
        model_cfg={
            "name": "fno",
            "in_channels": 3,
            "out_channels": 1,
            "n_modes": _N_MODES,
            "hidden_channels": _HIDDEN,
            "n_layers": _N_LAYERS,
        },
        feature_set="base",
        loss_weights={"data": 1.0},
    ),
    Variant(
        name="fno_padded",  # pad only the non-periodic through-wall axis
        model_cfg={
            "name": "fno",
            "in_channels": 3,
            "out_channels": 1,
            "n_modes": _N_MODES,
            "hidden_channels": _HIDDEN,
            "n_layers": _N_LAYERS,
            "domain_padding": [0.25, 0.0],
        },
        feature_set="base",
        loss_weights={"data": 1.0},
    ),
    Variant(
        name="fno_enriched",  # base FNO fed the analytic clear-wall prior + coords
        model_cfg={
            "name": "fno",
            "in_channels": 6,
            "out_channels": 1,
            "n_modes": _N_MODES,
            "hidden_channels": _HIDDEN,
            "n_layers": _N_LAYERS,
        },
        feature_set="enriched",
        loss_weights={"data": 1.0},
    ),
    Variant(
        name="delta_fno",  # learn theta - theta_prior on the enriched stack
        model_cfg={
            "name": "delta_fno",
            "in_channels": 6,
            "out_channels": 1,
            "n_modes": _N_MODES,
            "hidden_channels": _HIDDEN,
            "n_layers": _N_LAYERS,
            "clearwall_index": 3,
            "domain_padding": [0.25, 0.0],
        },
        feature_set="enriched",
        loss_weights={"data": 1.0},
    ),
    Variant(
        name="ufno",  # parallel local-conv path per spectral block
        model_cfg={
            "name": "ufno",
            "in_channels": 3,
            "out_channels": 1,
            "n_modes": _N_MODES,
            "hidden_channels": _HIDDEN,
            "n_layers": _N_LAYERS,
        },
        feature_set="base",
        loss_weights={"data": 1.0},
    ),
    Variant(
        name="fno_uloss",  # data-only FNO + targeted indoor-face U-value loss
        model_cfg={
            "name": "fno",
            "in_channels": 3,
            "out_channels": 1,
            "n_modes": _N_MODES,
            "hidden_channels": _HIDDEN,
            "n_layers": _N_LAYERS,
        },
        feature_set="base",
        loss_weights={"data": 1.0, "u_value": 0.1},
    ),
    Variant(
        name="fno_physics",  # data-only FNO + steady-FV PDE residual loss
        model_cfg={
            "name": "fno",
            "in_channels": 3,
            "out_channels": 1,
            "n_modes": _N_MODES,
            "hidden_channels": _HIDDEN,
            "n_layers": _N_LAYERS,
        },
        feature_set="base",
        loss_weights={"data": 1.0, "physics": 0.05},
    ),
    Variant(
        name="delta_fno_uloss",  # delta head + U-value loss (both U-MAE levers)
        model_cfg={
            "name": "delta_fno",
            "in_channels": 6,
            "out_channels": 1,
            "n_modes": _N_MODES,
            "hidden_channels": _HIDDEN,
            "n_layers": _N_LAYERS,
            "clearwall_index": 3,
            "domain_padding": [0.25, 0.0],
        },
        feature_set="enriched",
        loss_weights={"data": 1.0, "u_value": 0.1},
    ),
)

DEFAULT_SEEDS = (1337, 1, 2)


def get_variant(name: str) -> Variant:
    """Look up a roster :class:`Variant` by name."""
    for v in ROSTER:
        if v.name == name:
            return v
    raise KeyError(f"unknown variant '{name}'. Roster: {[v.name for v in ROSTER]}.")


def _phys_to_device(phys: dict, device) -> dict:
    """Move a physics bundle of tensors onto ``device``."""
    return {key: val.to(device) for key, val in phys.items()}


def train_one(
    variant: Variant,
    seed: int,
    train_root: Path,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    target_width: int,
    device: str,
) -> tuple[torch.nn.Module, int, float]:
    """Train a single (variant, seed) run; return (model, n_params, train_seconds).

    The dataset is featurised with the variant's own feature set and returns the
    per-cell physics bundle whenever a U-value or physics loss is active, so the
    target U-value (for the U-value loss) and the PDE residual can both be evaluated
    on the same resampled grid the field lives on.
    """
    seed_everything(seed)
    train_ds = SyntheticFEMDataset(
        train_root,
        target_width=target_width,
        return_physics=variant.needs_physics,
        feature_set=variant.feature_set,
    )
    model = build_model(variant.model_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    data_w = float(variant.loss_weights.get("data", 1.0))
    u_w, phys_w = variant.u_weight, variant.physics_weight

    t0 = time.perf_counter()
    for _ in range(epochs):
        model.train()
        for batch in loader:
            x, y = batch[0].to(device), batch[1].to(device)
            opt.zero_grad()
            pred = model(x)
            loss = data_w * relative_l2(pred, y)
            if variant.needs_physics:
                phys = _phys_to_device(batch[2], device)
                if u_w > 0.0:
                    # Derive the *target* U from the GT field y on this same grid, so
                    # the U-value loss penalises pred-vs-true indoor-face flux.
                    loss = loss + u_w * u_value_loss(
                        pred, y, phys["k"], phys["dx0"], phys["dy"], phys["r_si"]
                    )
                if phys_w > 0.0:
                    loss = loss + phys_w * heat_residual_loss(pred, **phys)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
    train_s = time.perf_counter() - t0
    return model, int(n_params), float(train_s)


def _strat_metrics(rel_l2s, u_pred, u_true, mask) -> dict:
    """field rel-L2 mean and U-MAE over the subset selected by boolean ``mask``."""
    rel_l2s, u_pred, u_true = map(np.asarray, (rel_l2s, u_pred, u_true))
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return {"field_rel_l2": float("nan"), "u_mae": float("nan"), "n": 0}
    return {
        "field_rel_l2": float(np.mean(rel_l2s[mask])),
        "u_mae": float(np.mean(np.abs(u_pred[mask] - u_true[mask]))),
        "n": int(mask.sum()),
    }


def evaluate_native(variant: Variant, model: torch.nn.Module, val_root: Path, device: str) -> dict:
    """Per-sample native-resolution metrics, stratified by bridge presence.

    Featurises each val sample at its own resolution with the variant's feature set,
    forwards once, and reports field relative-L2 (mean) and U-MAE both overall and
    split into the no-bridge (``n_bridges == 0``) and bridged (``>= 1``) strata.
    """
    manifest = json.loads((val_root / "manifest.json").read_text())
    rel_l2s, u_pred, u_true, n_bridges = [], [], [], []
    model.eval()
    with torch.no_grad():
        for row in manifest["samples"]:
            d = np.load(val_root / row["file"])
            k = d["k"].astype(np.float32)
            dx0 = d["dx0"].astype(np.float32)
            dy = float(d["dy"])
            r_si, r_se = float(d["r_si"]), float(d["r_se"])
            t_in, t_out = float(d["t_indoor"]), float(d["t_outdoor"])
            theta_gt = (d["temperature"].astype(np.float32) - t_out) / (t_in - t_out)

            x = build_input_channels(k, dx0, dy, r_si, r_se, feature_set=variant.feature_set)
            xt = torch.from_numpy(x[None]).to(device)
            pred = model(xt)[0, 0].cpu().numpy()

            rel_l2s.append(
                relative_l2(
                    torch.from_numpy(pred)[None, None],
                    torch.from_numpy(theta_gt)[None, None],
                ).item()
            )
            u_pred.append(effective_u_from_theta(pred, k, dx0, dy, r_si))
            u_true.append(float(d["u_value"]))
            n_bridges.append(int(d["n_bridges"]))

    rel_l2s, u_pred, u_true = map(np.asarray, (rel_l2s, u_pred, u_true))
    n_bridges = np.asarray(n_bridges)
    overall = _strat_metrics(rel_l2s, u_pred, u_true, np.ones_like(n_bridges, dtype=bool))
    overall.pop("n")
    return {
        "field_rel_l2": overall["field_rel_l2"],
        "u_mae": overall["u_mae"],
        "no_bridge": _strat_metrics(rel_l2s, u_pred, u_true, n_bridges == 0),
        "bridge": _strat_metrics(rel_l2s, u_pred, u_true, n_bridges >= 1),
    }


def _mean_std(values) -> dict:
    """mean / population std (ddof=0) / raw list for a per-seed metric."""
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "values": [float(v) for v in arr],
    }


def aggregate(name: str, per_seed: list[dict], params: int, train_times: list[float]) -> dict:
    """Collapse per-seed eval dicts into mean±std summaries for one variant."""

    def col(*path):
        out = []
        for s in per_seed:
            cur = s
            for key in path:
                cur = cur[key]
            out.append(cur)
        return out

    return {
        "variant": name,
        "n_seeds": len(per_seed),
        "params": int(params),
        "train_time_s": _mean_std(train_times),
        "field_rel_l2": _mean_std(col("field_rel_l2")),
        "u_mae": _mean_std(col("u_mae")),
        "no_bridge": {
            "field_rel_l2": _mean_std(col("no_bridge", "field_rel_l2")),
            "u_mae": _mean_std(col("no_bridge", "u_mae")),
            "n": per_seed[0]["no_bridge"]["n"],
        },
        "bridge": {
            "field_rel_l2": _mean_std(col("bridge", "field_rel_l2")),
            "u_mae": _mean_std(col("bridge", "u_mae")),
            "n": per_seed[0]["bridge"]["n"],
        },
    }


def add_verdicts(rows: list[dict]) -> None:
    """Annotate each row with a robust-win verdict against the reference U-MAE.

    A variant *robustly* beats the reference when its mean U-MAE is lower by more
    than the pooled (root-sum-square) seed std of the two — i.e. the improvement is
    real, not seed jitter. The reference row is labelled ``reference`` and never a
    win against itself.
    """
    ref = next((r for r in rows if r["variant"] == REFERENCE), None)
    ref_mean = ref["u_mae"]["mean"] if ref else None
    ref_std = ref["u_mae"]["std"] if ref else 0.0
    for r in rows:
        if r["variant"] == REFERENCE:
            r["beats_reference"] = None
            r["robust_win"] = None
            r["verdict"] = f"reference (U-MAE {r['u_mae']['mean']:.4f})"
            continue
        delta = ref_mean - r["u_mae"]["mean"]  # positive => variant is better
        pooled = float(np.hypot(ref_std, r["u_mae"]["std"]))
        robust = delta > pooled
        r["beats_reference"] = bool(delta > 0.0)
        r["robust_win"] = bool(robust)
        r["pooled_std"] = pooled
        if robust:
            r["verdict"] = (
                f"yes — robustly beats {REFERENCE} (ΔU-MAE {delta:+.4f} > pooled σ {pooled:.4f})"
            )
        elif delta > 0.0:
            r["verdict"] = (
                f"no — better mean but within noise (ΔU-MAE {delta:+.4f} ≤ pooled σ {pooled:.4f})"
            )
        else:
            r["verdict"] = f"no — does not beat {REFERENCE} (ΔU-MAE {delta:+.4f})"


def _fmt(ms: dict) -> str:
    return f"{ms['mean']:.4f}±{ms['std']:.4f}"


def write_markdown(path: Path, report: dict) -> None:
    """Write the leaderboard markdown, sorted by overall U-MAE ascending."""
    cfg = report["config"]
    rows = sorted(report["results"], key=lambda r: r["u_mae"]["mean"])
    lines = [
        "# Block-1 Ablations — beating the data-only FNO on U-MAE",
        "",
        f"- device: `{cfg['device']}` · epochs: {cfg['epochs']} · batch: {cfg['batch_size']} "
        f"· lr: {cfg['lr']} · seeds: {cfg['seeds']}",
        f"- eval: native resolution on `{cfg['val_root']}` "
        f"({report['n_val']} samples: {report['n_no_bridge']} clear / "
        f"{report['n_bridge']} bridged)",
        f"- reference: **{REFERENCE}** (data-only FNO) — mean U-MAE "
        f"{report['reference_u_mae']:.4f} W/m²K, the number to beat",
        "- *robust win* = mean U-MAE lower than the reference by more than the pooled seed σ.",
        "",
        "| Variant | Field rel-L2 ↓ | U-MAE ↓ (W/m²K) | U-MAE clear | U-MAE bridge | "
        "rel-L2 clear | rel-L2 bridge | Params | Train (s) | Robust win |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        ref_tag = " *(ref)*" if r["variant"] == REFERENCE else ""
        win = "—" if r["robust_win"] is None else ("**yes**" if r["robust_win"] else "no")
        lines.append(
            f"| {r['variant']}{ref_tag} | {_fmt(r['field_rel_l2'])} | {_fmt(r['u_mae'])} | "
            f"{_fmt(r['no_bridge']['u_mae'])} | {_fmt(r['bridge']['u_mae'])} | "
            f"{_fmt(r['no_bridge']['field_rel_l2'])} | {_fmt(r['bridge']['field_rel_l2'])} | "
            f"{r['params']:,} | {r['train_time_s']['mean']:.0f} | {win} |"
        )
    lines += ["", "## Verdict per variant", ""]
    for r in rows:
        lines.append(f"- **{r['variant']}**: {r['verdict']}")
    lines += [
        "",
        "All metrics are mean±std over the seeds. U-MAE columns split the val set by "
        "thermal-bridge presence (`n_bridges == 0` *clear* vs `>= 1` *bridge*); U-MAE is "
        "driven by the near-boundary gradient, so the bridge stratum is where geometry "
        "awareness has to pay off.",
        "",
    ]
    path.write_text("\n".join(lines))


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--variants",
        nargs="+",
        default=[v.name for v in ROSTER],
        help="subset of roster variants to run (default: all).",
    )
    p.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default="cuda")
    p.add_argument("--train_root", default="data/processed/block1_train")
    p.add_argument("--val_root", default="data/processed/block1_val")
    p.add_argument("--target_width", type=int, default=48)
    a = p.parse_args()

    device = a.device if (a.device == "cpu" or torch.cuda.is_available()) else "cpu"
    train_root, val_root = _REPO / a.train_root, _REPO / a.val_root
    variants = [get_variant(name) for name in a.variants]

    # Bridge-stratum sizes (for the report header) — read once from the val manifest.
    val_manifest = json.loads((val_root / "manifest.json").read_text())
    n_bridges = np.array(
        [int(np.load(val_root / r["file"])["n_bridges"]) for r in val_manifest["samples"]]
    )

    rows = []
    for variant in variants:
        per_seed, params, train_times = [], None, []
        for seed in a.seeds:
            model, n_params, train_s = train_one(
                variant,
                seed,
                train_root,
                epochs=a.epochs,
                batch_size=a.batch_size,
                lr=a.lr,
                target_width=a.target_width,
                device=device,
            )
            params = n_params
            train_times.append(train_s)
            m = evaluate_native(variant, model, val_root, device)
            per_seed.append(m)
            print(
                f"[{variant.name} seed={seed}] relL2={m['field_rel_l2']:.4f} "
                f"U-MAE={m['u_mae']:.4f} "
                f"(clear {m['no_bridge']['u_mae']:.4f}, bridge {m['bridge']['u_mae']:.4f}) "
                f"train={train_s:.0f}s"
            )
        agg = aggregate(variant.name, per_seed, params, train_times)
        rows.append(agg)
        print(
            f"== {variant.name}: U-MAE {agg['u_mae']['mean']:.4f}±{agg['u_mae']['std']:.4f} "
            f"relL2 {agg['field_rel_l2']['mean']:.4f}±{agg['field_rel_l2']['std']:.4f} =="
        )

    add_verdicts(rows)
    ref = next((r for r in rows if r["variant"] == REFERENCE), None)

    out = _REPO / "results"
    out.mkdir(exist_ok=True)
    report = {
        "config": vars(a) | {"device": device},
        "n_val": len(n_bridges),
        "n_no_bridge": int((n_bridges == 0).sum()),
        "n_bridge": int((n_bridges >= 1).sum()),
        "reference": REFERENCE,
        "reference_u_mae": ref["u_mae"]["mean"] if ref else None,
        "results": rows,
    }
    (out / "block1_ablations.json").write_text(json.dumps(report, indent=2))
    write_markdown(out / "block1_ablations.md", report)
    print(f"\nwrote {out / 'block1_ablations.json'} and .md")


if __name__ == "__main__":
    main()
