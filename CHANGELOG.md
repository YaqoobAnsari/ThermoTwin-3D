# Changelog

All notable changes to ThermoTwin-3D are recorded here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
the project aims to follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Professional repository skeleton: `src/thermotwin/` package with one subpackage per
  pipeline stage (`geometry`, `physics`, `data`, `models`, `losses`, `calibration`,
  `training`, `eval`, `viz`, `utils`).
- Hydra config tree under `configs/` (`data`, `model`, `physics`, `train`, `experiment`).
- Documentation set: `README.md`, `CLAUDE.md` (operating guide incl. working agreement),
  and `docs/` — `thesis.md`, `datasets.md`, `baselines.md`, `experiment-plan.md`,
  `architecture.md`, and an ADR log under `docs/decisions/`.
- Tooling: `pyproject.toml` (setuptools + ruff/mypy/pytest config), `Makefile`,
  `.pre-commit-config.yaml`, `.editorconfig`, `.gitattributes`, `.gitignore`,
  conda `env/environment.yml` + `requirements.txt`.
- Entry-point stubs (`scripts/train.py`, `evaluate.py`, `download_data.py`,
  `generate_fem_groundtruth.py`) and Spartan Slurm templates (`scripts/slurm/`).
- Smoke test and `data/` / `results/` / `experiments/` / `notebooks/` scaffolding.
- Initialised git repository (branch `main`).
- First physics slice — `physics/conduction.py`: closed-form 1-D multilayer
  steady-state conduction (U-value, heat flux, full temperature profile, EN ISO 6946
  surface films) with closed-form unit tests. The analytic oracle and first
  ground-truth source for `eval/`.

### Changed
- **Renamed the project `BuildTrust-3D` → `ThermoTwin-3D`** to match the thermal-twin thesis
  and the target venue (Automation in Construction).

### Deprecated / Archived
- Retired the earlier *robustness / corruption-benchmark* framing. Its planning docs
  (`experiments.md`, `models-and-code.md`) are preserved under
  `docs/_archive/legacy-robustness-benchmark/` for provenance only.

---

_Dates use ISO-8601. Most recent changes first. Initial scaffolding: 2026-06-23._
