#!/usr/bin/env python
"""Block-1 OOD generalisation runner — does the in-distribution winner *travel*?

The in-distribution ablation (:mod:`scripts.ablate`) crowned ``delta_fno`` (the
residual-on-the-analytic-prior head) on U-MAE. That sweep, however, trains and
evaluates inside one distribution: the same wall library, the same film band, the
same bridge regime, the same discretisation. The question this runner answers is the
one a reviewer at *Automation in Construction* will actually ask — **does the model
hold up off-distribution?** Concretely, on four held-out shifts that each move a
*physically meaningful* axis while keeping θ well-posed:

* ``ood_walls``  — four wall assemblies never seen in training (new materials/layers).
* ``ood_films``  — surface-film resistances (r_si / r_se) outside the training band.
* ``ood_bridges``— a denser/wider thermal-bridge regime than training ever shows.
* ``ood_res``    — finer through-wall *and* lateral discretisation (cross-resolution).

(θ is invariant to the absolute indoor/outdoor temperatures under linear steady
conduction, so none of the shifts touch temperatures — only films, walls, bridges
and resolution, the axes that actually change the field.)

For every contender we train **once per (regime, seed)** and then score that single
model on **every** test set — the in-distribution val plus all four OOD corpora — at
each sample's native resolution, featurised with the variant's own feature set via
:func:`thermotwin.data.dataset.build_input_channels`. Two training regimes probe how
much the generalisation depends on data volume:

* ``full``     — the whole 256-sample ``block1_train``.
* ``lowdata``  — a fixed, seeded 64-sample :class:`torch.utils.data.Subset` of it.

This script is a thin orchestration layer: the model build, featuriser, losses and
the per-sample native-resolution evaluator are all reused verbatim from
:mod:`scripts.ablate` (it does not redefine training or eval), so the OOD numbers are
directly comparable to the in-distribution leaderboard.

    python scripts/ood_ablate.py --epochs 300 --device cuda
    python scripts/ood_ablate.py --variants delta_fno fno --regimes full --epochs 50

Writes ``results/block1_ood.json`` and ``results/block1_ood.md`` (one section per
test set, variants as rows and {full, lowdata} columns, sorted by U-MAE, with a
generalisation-gap column = OOD U-MAE − in-distribution U-MAE per variant).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

# Reuse the in-distribution runner's roster type, device helper and native-resolution
# evaluator verbatim. ``scripts/`` is not an importable package, so make ``ablate``
# resolvable regardless of how this file is launched (``python scripts/ood_ablate.py``
# puts this dir on sys.path[0]; ``python -m`` / pytest may not).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ablate import (  # noqa: E402
    Variant,
    _phys_to_device,
    evaluate_native,
)
from thermotwin.data.dataset import SyntheticFEMDataset  # noqa: E402
from thermotwin.eval.metrics import relative_l2  # noqa: E402
from thermotwin.losses.building_loss import u_value_loss  # noqa: E402
from thermotwin.losses.heat_residual import heat_residual_loss  # noqa: E402
from thermotwin.models.registry import build_model  # noqa: E402
from thermotwin.utils.seed import seed_everything  # noqa: E402

_REPO = Path(__file__).resolve().parents[1]

# Shared FNO/operator hyper-parameters (match configs/model/*.yaml and scripts.ablate).
_N_MODES = [8, 16]
_HIDDEN = 32
_N_LAYERS = 4

# The OOD contenders — the meaningful Block-1 models (drop the padding/enriched
# diagnostics from the in-distribution sweep; keep the residual head, the data-only
# reference, the two loss levers and the local-path U-FNO). delta_fno rides the
# enriched (6ch) stack so it sees the analytic clear-wall prior; the rest are base.
ROSTER: tuple[Variant, ...] = (
    Variant(
        name="delta_fno",  # in-distribution winner: residual on the analytic 1-D prior
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
        name="fno",  # data-only FNO reference
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
)

DEFAULT_SEEDS = (1337, 1, 2)
DEFAULT_REGIMES = ("full", "lowdata")
# Test sets scored for every trained model. The first is the in-distribution
# reference (used for the generalisation-gap baseline); the rest are the OOD shifts.
IN_DIST = "in_dist"
TEST_SETS: dict[str, str] = {
    IN_DIST: "data/processed/block1_val",
    "ood_walls": "data/processed/ood_walls",
    "ood_films": "data/processed/ood_films",
    "ood_bridges": "data/processed/ood_bridges",
    "ood_res": "data/processed/ood_res",
}
OOD_SETS: tuple[str, ...] = tuple(name for name in TEST_SETS if name != IN_DIST)

# Fixed size of the low-data Subset of block1_train.
LOWDATA_N = 64
# Seed for the (deterministic) low-data subset draw — independent of the model seed
# so every variant/seed trains on the *same* 64 samples in the lowdata regime.
LOWDATA_SUBSET_SEED = 1337


def get_variant(name: str) -> Variant:
    """Look up a roster :class:`Variant` by name."""
    for v in ROSTER:
        if v.name == name:
            return v
    raise KeyError(f"unknown variant '{name}'. Roster: {[v.name for v in ROSTER]}.")


def _make_train_dataset(variant: Variant, train_root: Path, target_width: int):
    """Featurised full training dataset for ``variant``."""
    return SyntheticFEMDataset(
        train_root,
        target_width=target_width,
        return_physics=variant.needs_physics,
        feature_set=variant.feature_set,
    )


def _lowdata_indices(n_total: int) -> list[int]:
    """Deterministic LOWDATA_N indices into the training corpus (seed-independent)."""
    rng = np.random.default_rng(LOWDATA_SUBSET_SEED)
    return sorted(int(i) for i in rng.choice(n_total, size=LOWDATA_N, replace=False))


def train_regime(
    variant: Variant,
    seed: int,
    regime: str,
    train_root: Path,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    target_width: int,
    device: str,
) -> tuple[torch.nn.Module, int, float, int]:
    """Train one (variant, regime, seed); return (model, n_params, seconds, n_train).

    Mirrors :func:`scripts.ablate.train_one` but takes a ``regime`` that selects the
    training pool: ``'full'`` uses the whole corpus, ``'lowdata'`` a fixed seeded
    64-sample :class:`~torch.utils.data.Subset`. The subset draw is independent of the
    model seed, so a given variant trains on the *same* 64 samples across all seeds —
    only the optimisation noise varies. Loss wiring (data / U-value / PDE residual) is
    identical to the in-distribution runner.
    """
    if regime not in DEFAULT_REGIMES:
        raise ValueError(f"unknown regime '{regime}'; expected one of {DEFAULT_REGIMES}.")

    seed_everything(seed)
    full_ds = _make_train_dataset(variant, train_root, target_width)
    if regime == "lowdata":
        train_ds: SyntheticFEMDataset | Subset = Subset(full_ds, _lowdata_indices(len(full_ds)))
    else:
        train_ds = full_ds
    n_train = len(train_ds)

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
    return model, int(n_params), float(train_s), int(n_train)


def _mean_std(values) -> dict:
    """mean / population std (ddof=0) / raw list for a per-seed metric."""
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "values": [float(v) for v in arr],
    }


def aggregate_cell(per_seed: list[dict]) -> dict:
    """Collapse a list of per-seed eval dicts (one test set) into mean±std summaries."""
    return {
        "n_seeds": len(per_seed),
        "field_rel_l2": _mean_std([s["field_rel_l2"] for s in per_seed]),
        "u_mae": _mean_std([s["u_mae"] for s in per_seed]),
    }


def _fmt(ms: dict) -> str:
    return f"{ms['mean']:.4f}±{ms['std']:.4f}"


def _gen_gap(cell: dict, in_dist_cell: dict | None) -> float | None:
    """Generalisation gap = OOD U-MAE mean − in-distribution U-MAE mean."""
    if in_dist_cell is None:
        return None
    return float(cell["u_mae"]["mean"] - in_dist_cell["u_mae"]["mean"])


def write_markdown(path: Path, report: dict) -> None:
    """One section per test set; variants as rows, {full, lowdata} as columns.

    Within a section, rows are sorted by the ``full``-regime U-MAE mean ascending. A
    *Gen. gap* column reports OOD U-MAE − in-distribution U-MAE for the same variant
    and regime (omitted on the in-distribution section, where it is identically zero).
    """
    cfg = report["config"]
    res = report["results"]  # {variant: {regime: {test_set: cell}}}
    regimes = cfg["regimes"]
    variants = cfg["variants"]

    lines = [
        "# Block-1 OOD generalisation — does the in-distribution winner travel?",
        "",
        f"- device: `{cfg['device']}` · epochs: {cfg['epochs']} · batch: {cfg['batch_size']} "
        f"· lr: {cfg['lr']} · seeds: {cfg['seeds']}",
        f"- regimes: full (train on all {report['n_full']} of `{cfg['train_root']}`) "
        f"vs lowdata (fixed seeded {report['n_lowdata']}-sample subset)",
        "- eval: native resolution per sample, featurised with each variant's own "
        "feature set; metrics are mean±std over seeds.",
        "- test sets: " + ", ".join(f"`{k}` ({report['n_test'][k]})" for k in TEST_SETS) + ".",
        "- **U-MAE** (W/m²K) is primary; *Gen. gap* = OOD U-MAE − in-distribution U-MAE "
        "(same variant & regime; lower/negative is better).",
        "",
    ]

    for test_set in TEST_SETS:
        is_in_dist = test_set == IN_DIST
        title = "in-distribution (`block1_val`)" if is_in_dist else f"OOD · `{test_set}`"
        lines += [f"## {title}", ""]

        # Sort rows by the 'full' regime U-MAE mean (fallback to first available regime).
        sort_regime = "full" if "full" in regimes else regimes[0]

        def _key(v, sr=sort_regime, ts=test_set):
            cell = res[v].get(sr, {}).get(ts)
            return cell["u_mae"]["mean"] if cell else float("inf")

        ordered = sorted(variants, key=_key)

        header = ["Variant"]
        for reg in regimes:
            header += [f"U-MAE [{reg}]", f"rel-L2 [{reg}]"]
            if not is_in_dist:
                header.append(f"Gen. gap [{reg}]")
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "---|" * len(header))

        for v in ordered:
            cells = [str(v)]
            for reg in regimes:
                cell = res[v].get(reg, {}).get(test_set)
                if cell is None:
                    cells += ["—", "—"] + ([] if is_in_dist else ["—"])
                    continue
                cells += [_fmt(cell["u_mae"]), _fmt(cell["field_rel_l2"])]
                if not is_in_dist:
                    gap = res[v][reg][test_set].get("gen_gap")
                    cells.append("—" if gap is None else f"{gap:+.4f}")
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

    lines += [
        "Each cell is mean±std over seeds for a single model trained once on the named "
        "regime and scored on the named test set at native resolution. The OOD shifts "
        "move one physically meaningful axis each — unseen walls, films outside the "
        "training band, a denser/wider bridge regime, and finer discretisation — while "
        "keeping θ well-posed (temperatures are untouched, as θ is invariant to them).",
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
    p.add_argument(
        "--regimes",
        nargs="+",
        default=list(DEFAULT_REGIMES),
        choices=list(DEFAULT_REGIMES),
        help="training data regimes to run (default: full lowdata).",
    )
    p.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default="cuda")
    p.add_argument("--train_root", default="data/processed/block1_train")
    p.add_argument("--target_width", type=int, default=48)
    a = p.parse_args()

    device = a.device if (a.device == "cpu" or torch.cuda.is_available()) else "cpu"
    train_root = _REPO / a.train_root
    variants = [get_variant(name) for name in a.variants]
    test_roots = {name: _REPO / rel for name, rel in TEST_SETS.items()}

    # Test-set sizes for the report header.
    n_test = {
        name: len(json.loads((root / "manifest.json").read_text())["samples"])
        for name, root in test_roots.items()
    }
    n_full = len(_make_train_dataset(variants[0], train_root, a.target_width))

    # results[variant][regime][test_set] = aggregated cell.
    results: dict[str, dict[str, dict[str, dict]]] = {v.name: {} for v in variants}
    n_runs = 0
    params_by_variant: dict[str, int] = {}

    for variant in variants:
        for regime in a.regimes:
            # Per-seed eval dicts, one list per test set.
            per_seed: dict[str, list[dict]] = {name: [] for name in test_roots}
            train_times: list[float] = []
            for seed in a.seeds:
                model, n_params, train_s, n_train = train_regime(
                    variant,
                    seed,
                    regime,
                    train_root,
                    epochs=a.epochs,
                    batch_size=a.batch_size,
                    lr=a.lr,
                    target_width=a.target_width,
                    device=device,
                )
                params_by_variant[variant.name] = n_params
                train_times.append(train_s)
                n_runs += 1
                for name, root in test_roots.items():
                    m = evaluate_native(variant, model, root, device)
                    per_seed[name].append(m)
                in_d = per_seed[IN_DIST][-1]
                ood_str = " ".join(f"{name}={per_seed[name][-1]['u_mae']:.4f}" for name in OOD_SETS)
                print(
                    f"[{variant.name} {regime} seed={seed}] n_train={n_train} "
                    f"in_dist U-MAE={in_d['u_mae']:.4f} | {ood_str} train={train_s:.0f}s"
                )

            # Aggregate each test set over seeds; attach the generalisation gap.
            cell_by_set = {name: aggregate_cell(per_seed[name]) for name in test_roots}
            in_dist_cell = cell_by_set[IN_DIST]
            for name, cell in cell_by_set.items():
                cell["gen_gap"] = _gen_gap(cell, None if name == IN_DIST else in_dist_cell)
            cell_by_set["_train_time_s"] = _mean_std(train_times)
            cell_by_set["_n_train"] = n_train
            results[variant.name][regime] = cell_by_set
            print(
                f"== {variant.name} [{regime}]: in_dist U-MAE "
                f"{in_dist_cell['u_mae']['mean']:.4f}±{in_dist_cell['u_mae']['std']:.4f} =="
            )

    out = _REPO / "results"
    out.mkdir(exist_ok=True)
    report = {
        "config": vars(a) | {"device": device},
        "test_sets": TEST_SETS,
        "ood_sets": list(OOD_SETS),
        "n_full": n_full,
        "n_lowdata": LOWDATA_N,
        "n_test": n_test,
        "params": params_by_variant,
        "results": results,
        "n_training_runs": n_runs,
    }
    (out / "block1_ood.json").write_text(json.dumps(report, indent=2))
    write_markdown(out / "block1_ood.md", report)
    print(f"\nwrote {out / 'block1_ood.json'} and .md  ({n_runs} training runs)")


if __name__ == "__main__":
    main()
