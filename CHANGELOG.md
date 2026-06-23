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
- Geometry-resolved solver — `physics/steady_fv.py`: cell-centred finite-volume
  steady conduction `∇·(k∇T)=0` on 1-D/2-D/3-D non-uniform grids with Dirichlet/film
  BCs. Exact-series face conductivities reproduce the 1-D oracle to machine precision
  (gated by `tests/test_steady_fv.py`). Default Block-1 ground-truth + PINN-residual
  engine; see ADR `0002`.
- Geometry featuriser (Stage 1) — `geometry/idf.py` (dependency-free EnergyPlus IDF
  reader) + `geometry/envelope.py`: lifts Materials, Constructions and
  BuildingSurface:Detailed from a DOE IDF into a material-tagged envelope with
  per-surface U-values (EN ISO 6946 films by orientation), polygon area/normal
  (Newell), case-insensitive name resolution, and version-robust vertex parsing.
  Bridges into `steady_fv` (a featurised construction's analytic U equals the
  solver's effective U); gated by `tests/test_envelope.py` incl. a real-DOE
  integration test.
- Data acquisition — `data/sources.py` registry + real `scripts/download_data.py`
  CLI (resume, provenance SOURCE.md, gated-dataset stubs). Fetched the open
  critical-path corpora: 16 DOE Commercial Reference Buildings (EnergyPlus IDFs +
  Chicago TMY3) and the TBBRv2 thermal-bridge set (CC-BY-4.0). Placeholders written
  for the EULA-gated corpora (TUM2TWIN, ScanNet++, Matterport3D, Structured3D,
  Building3D).
- Baseline code vendored via `scripts/fetch_baselines.sh` into `vendored/`
  (git-ignored, pinned commits in `MANIFEST.txt`): neuraloperator (GINO/FNO),
  Transolver, GNOT, DeepXDE (DeepONet), MeshGraphNet, NVIDIA Modulus, pytorch-3dunet,
  PointNet++, and Gokhale's building-PINN. Di Natale PCNN and Yang GNN have no public
  code — flagged for reimplementation.
- Block-1 learning pipeline, end to end and runnable:
  - `data/synthetic_fem.py` — parametric layered walls punctured by thermal bridges,
    solved by `steady_fv`; `scripts/generate_fem_groundtruth.py` writes a seeded
    corpus + manifest. Bridges shift effective U 40–50% (max 4×) off the 1-D
    clear-wall value — the geometry signal motivating the operator (H1).
  - `data/dataset.py` — torch dataset predicting the dimensionless field
    θ=(T−T_out)/(T_in−T_out) (geometry/material-determined, BC-scale-free).
  - `models/fno.py` — Fourier Neural Operator backbone (GINO follows on Block-2
    point clouds); `eval/metrics.py` — relative L2 + RMSE; `utils/seed.py`.
  - `scripts/train.py` — Hydra training loop (AdamW + cosine, val metric, isolated
    run dirs). Smoke run: **val relative L2 0.65 → 0.033 in 15 CPU epochs.**
  - Configs wired: `model/fno.yaml`, `data/synthetic_fem.yaml` → corpus,
    `experiment/block1_synthetic_fem` composes FNO.
  - `eval/building.py` — recovers the effective U-value from a predicted θ field
    (reproduces the solver's U exactly); `scripts/evaluate.py` reports the venue's
    paired metrics at **native resolution** (discretisation-invariance check) and
    contrasts the operator's U-error against the 1-D clear-wall baseline (H1).
    First run: operator cuts U-value MAE ~1.6× vs ignoring bridges (undertrained
    checkpoint; widens with the full GPU run).
  - `scripts/slurm/train.slurm` rewritten for the real entry point (env's Python
    directly, `PYTHONNOUSERSITE=1`, no `conda activate`).
  - 35 tests total (added `tests/test_building_metrics.py`); all green.
- Baseline comparison wiring — `models/registry.py` (`build_model` behind one
  `(B,C,H,W)→(B,1,H,W)` contract) + `models/cnn.py` (size-agnostic conv baseline,
  the "no neural-operator" control). `train.py`/`evaluate.py` now build via the
  registry, so `model=fno` / `model=cnn` are config-selectable. The vendored
  point-cloud operators (GINO/GNOT/Transolver/MeshGraphNet/DeepONet/PointNet++) are
  registered as explicit "deferred" (Block-2 competitors on irregular geometry) and
  raise loudly rather than silently. Gated by `tests/test_registry.py` (43 tests total).
- Conda env on project disk (`/data/gpfs/projects/punim2769/envs/thermotwin`):
  Python 3.10 + PyTorch 2.5.1/CUDA 12.1 + neuraloperator 2.0 + geometry/IO stack,
  via `scripts/setup_env.sh`.

### Changed
- **Renamed the project `BuildTrust-3D` → `ThermoTwin-3D`** to match the thermal-twin thesis
  and the target venue (Automation in Construction).

### Deprecated / Archived
- Retired the earlier *robustness / corruption-benchmark* framing. Its planning docs
  (`experiments.md`, `models-and-code.md`) are preserved under
  `docs/_archive/legacy-robustness-benchmark/` for provenance only.

---

_Dates use ISO-8601. Most recent changes first. Initial scaffolding: 2026-06-23._
