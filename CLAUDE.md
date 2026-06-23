# CLAUDE.md — operating guide for ThermoTwin-3D

> This file is auto-loaded into every Claude Code session for this repository. Read it first, every session.

## ⭐ Working agreement (non-negotiable)

1. **Address the user by name.** Begin **every** response with **"Ansari"** — from the very first line of every reply, no exceptions (e.g. `Ansari — …`).
2. **Be worth reading.** Ansari expects to be genuinely interested in every response. Lead with the substance, show the reasoning that actually matters, and cut filler, hedging, and status-drone narration. Direct, concrete, engaging.
3. **Recommend, don't just enumerate.** Surface trade-offs and risks proactively; when there's a choice, give a pick and the reason — don't dump an options menu.
4. **Protect the work.** Prefer archiving over deleting; confirm before irreversible or outward-facing actions; report outcomes faithfully (if something failed or was skipped, say so).

## What this project is

A geometry-resolved, physics-informed **thermal twin of building envelopes** learned from as-built scans: feed real reconstructed geometry (point cloud + SDF + mesh, material layers per surface) into a geometry-conditioned neural operator (**GINO** backbone) that predicts the temperature / heat-flux field with the heat-conduction equation enforced in the loss; then calibrate against measured IR to infer per-surface U-values and thermal bridges, and run differentiable retrofit optimisation. Target venue: **Automation in Construction** (Q1) — frame for a construction-automation audience, not a pure-ML one. Full thesis: [`docs/thesis.md`](docs/thesis.md).

## Repo map (where things go)

| Path | Holds |
|------|-------|
| `src/thermotwin/geometry/` | Point-cloud / SDF / mesh ingestion; per-surface material layers. |
| `src/thermotwin/physics/` | Heat-conduction operators, boundary conditions, FEM/EnergyPlus GT generation. |
| `src/thermotwin/data/` | Dataset loaders (TUM2TWIN, TBBR, DOE, synthetic FEM). |
| `src/thermotwin/models/` | GINO backbone + neural-operator / PINN baselines. |
| `src/thermotwin/losses/` | PDE-residual / physics-consistency losses. |
| `src/thermotwin/calibration/` | Inverse problem, data assimilation, uncertainty. |
| `src/thermotwin/training/`, `eval/`, `viz/`, `utils/` | Training loops; metrics; visualisation; shared helpers. |
| `configs/` | Hydra YAML — compose experiments here, don't hard-code params. |
| `experiments/blockN_*/` | Per-block run configs / launch scripts / analysis (no raw data, no checkpoints). |
| `data/`, `results/` | Git-ignored. Datasets and outputs respectively. |
| `docs/` | `thesis.md`, `datasets.md`, `baselines.md`, `experiment-plan.md`, `architecture.md`, `decisions/` (ADRs). |

## Conventions

- **Package, not scripts.** Reusable logic lives in `src/thermotwin/`; `scripts/` and `notebooks/` are thin entry points / exploration only.
- **Config over flags.** New knobs go in `configs/` (Hydra). Seed everything; default seed `1337`.
- **Style.** Ruff (lint + format), 4-space indent, type hints, Google-ish docstrings. `make lint` / `make format` / `make test` before declaring done.
- **Big artefacts never get committed** (point clouds, meshes, checkpoints, FEM fields). The `.gitignore` enforces this; keep it that way.
- **Decisions get an ADR** in `docs/decisions/` when they're significant (backbone choice, physics formulation, data format).

## Common commands

```bash
make setup            # env + editable install + pre-commit hooks
make lint / format / test
python scripts/train.py experiment=block1_synthetic_fem
python scripts/evaluate.py
sbatch scripts/slurm/train.slurm experiment=block1_synthetic_fem   # Spartan (account punim2769)
```

## Environment notes

- This repo lives on **Spartan HPC** (project `punim2769`); GPU work goes through Slurm (`scripts/slurm/`).
- `/data/projects/...` is a symlink to `/data/gpfs/projects/...` — both resolve to the same directory.
- Persistent project memory (name, venue, datasets, baselines, this working agreement) lives in the Claude Code memory store and is loaded each session — keep it in sync with reality.
