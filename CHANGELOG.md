# Changelog

All notable changes to ThermoTwin-3D are recorded here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
the project aims to follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **`docs/experiments.md`** — the running, paper-grade record of every experiment:
  setup, **all** variants (winners and losers), numbers, and conclusions, with raw
  artefacts in `results/*.json`/`*.md`. Consolidates Exp 1.1 (first benchmark),
  Exp 1.2 (physics loss + UNet), and Exp 1.3 (the 8-variant model ablation).
- **Adopted `delta_fno` as the Block-1 experimental default** (ADR 0004 winner):
  `configs/experiment/block1_synthetic_fem.yaml` now composes `model=delta_fno` with
  the `enriched` feature set. `feature_set` threaded through `configs/data`,
  `scripts/train.py`, and `scripts/evaluate.py` (the latter now featurises via the
  shared `build_input_channels`, so any model's channels are built correctly at
  native-resolution eval). The other seven ablation variants stay registered and
  config-selectable as competitive alternatives.
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
- Watertight mesh repair — `Envelope.shell_surfaces()` (outdoors + ground = the
  closed outer boundary) + `envelope_to_mesh(mode="shell", repair=...)`. The real
  DOE SmallOffice now meshes watertight, so the SDF inside/outside **sign is
  reliable** (interior SDF −3 m). Gated by a new real-building integration test.
- Block-1 benchmark harness — `scripts/benchmark.py` + `scripts/slurm/benchmark.slurm`:
  trains every model in the roster and tabulates field relative L2, U-value error,
  the gap to the 1-D clear-wall baseline (H1), inference speedup vs the FV solver,
  params and train time → `results/block1_benchmark.{json,md}` leaderboard.
- Geometry featurisation for the operator (Stage-1 input) — `geometry/pointcloud.py`
  (area-weighted surface sampling into a feature-tagged point cloud: position,
  normal, U-value, resistance, surface type) + `geometry/sdf.py` (envelope→mesh,
  signed distance, regular SDF grid). The point cloud + SDF are GINO's native input.
  Validated on the real DOE SmallOffice (8k points, 24³ SDF). Gated by
  `tests/test_geometry_featurize.py`. Adds `rtree` (trimesh spatial index).
  Caveat: real multi-zone envelopes aren't watertight as assembled — SDF distances
  exact, signs heuristic until mesh repair lands.
- Physics-informed residual loss — `losses/heat_residual.py`: the autograd-enabled
  twin of `physics/steady_fv`. It evaluates the same cell-centred finite-volume
  steady operator as a *residual* on a predicted dimensionless field θ
  (indoor air θ=1 / `r_si` lo face, outdoor air θ=0 / `r_se` hi face; adiabatic
  lateral edges) and returns `mean(R_cell²)`, differentiable wrt θ and addable to
  the training loss. Gated by `tests/test_heat_residual.py` (residual ~0 on GT,
  gradient checks).
- Second Block-1 baseline — `models/unet.py`: a 2-D multiscale encoder/decoder
  (`configs/model/unet.yaml`, depth-2, 32 base channels) registered in
  `models/registry.py` (`WIRED_MODELS` now `fno`/`cnn`/`unet`), so `model=unet`
  is config-selectable under the one `(B,C,H,W)→(B,1,H,W)` contract. Gated by
  `tests/test_unet.py`.
- Dataset physics bundle — `data/dataset.py` gains `return_physics=True`: each
  item additionally yields a `phys` dict `{k, dx0, dy, r_si, r_se}` of tensors on
  the *same resampled grid* as `x`/`y`, the bundle `heat_residual_loss` needs to
  evaluate the FV residual on the prediction. Off by default (data-only path
  unchanged).
- Physics integration in the training paths — `configs/train/default.yaml` adds
  `physics_weight` (default `0.0`; `>0` adds `physics_weight · heat_residual_loss`).
  `scripts/train.py` switches the dataset into physics mode and augments the loss
  when the weight is positive; `scripts/benchmark.py` adds an `fno_physics` roster
  entry (the FNO architecture trained with `physics_weight=0.1`) alongside the new
  `unet`, sharing one physics-mode dataset across the roster.
- Re-ran the Block-1 GPU benchmark (`results/block1_benchmark.{json,md}`, 300
  epochs · batch 64 · seed 1337, A100). New leaderboard (field rel-L2 / U-MAE
  [W/m²K] / U-MAPE / vs 1-D clear):
  - `fno_physics` — 0.0143 / 0.0218 / 5.1% / 5.36×
  - `fno`         — 0.0144 / 0.0205 / 4.9% / 5.70×
  - `unet`        — 0.0167 / 0.0343 / 9.1% / 3.41×
  - `cnn`         — 0.0170 / 0.0254 / 5.5% / 4.59×
  Every geometry-aware model clears the geometry-blind 1-D clear-wall baseline
  (U-MAE 0.1168 W/m²K) by 3.4–5.7× (H1). The PDE-residual term shaves field
  rel-L2 marginally (0.0144→0.0143) but does not improve U-MAE at this weight; see
  ADR `0003`.
- Conda env on project disk (`/data/gpfs/projects/punim2769/envs/thermotwin`):
  Python 3.10 + PyTorch 2.5.1/CUDA 12.1 + neuraloperator 2.0 + geometry/IO stack,
  via `scripts/setup_env.sh`.
- **Block-1 model ablation — beating the data-only FNO on U-MAE.** A targeted
  countermeasure sweep, motivated by the diagnosis that U-MAE is governed by the
  through-wall θ-gradient at the indoor face, where the plain FNO loses twice:
  spectral bias smears the sharp high-k bridge gradient, and the FFT's periodic
  wraparound contaminates the non-periodic Dirichlet/film faces. New machinery:
  - `data/dataset.py` — pluggable `feature_set`/`build_input_channels`, adding the
    `enriched` set that carries the analytic closed-form 1-D clear-wall θ as a
    dedicated input channel (the geometry/physics prior).
  - `models/delta_fno.py` (`configs/model/delta_fno.yaml`) — predicts
    `θ = θ_prior + fno(x)`, i.e. learns only the residual lateral-spreading
    correction near bridges instead of regressing the whole sharp field; the prior
    nails bridge-free columns by construction.
  - `models/ufno.py` (`configs/model/ufno.yaml`) — U-FNO (Wen et al., 2022): a
    parallel **local** Conv2d path summed with the spectral path inside every
    block, restoring the boundary-aware high-frequency capacity the FFT lacks.
  - `models/fno.py` — `domain_padding` exposed (`fno_padded`: pad only the
    non-periodic through-wall axis to break the FFT wraparound at the boundary).
  - `losses/building_loss.py` (`u_value_loss`) — a differentiable indoor-face
    U-value loss whose gradient touches only row 0 of θ, the most targeted lever
    for U-MAE (`fno_uloss`, `delta_fno_uloss`).
  - `scripts/ablate.py` + `scripts/slurm/ablate.slurm` — self-contained runner
    over the 8-variant roster × seeds `{1337, 1, 2}`, evaluated at each val
    sample's native resolution, stratified by thermal-bridge presence, with a
    *robust win* gate (mean U-MAE below reference by more than the pooled seed σ).
    Writes `results/block1_ablations.{json,md}`.
  - Gated by `tests/test_variants.py`, `tests/test_input_channels.py`,
    `tests/test_building_loss.py`; `models/registry.py` wires the new variants.
  - **Roster:** `fno` (reference), `fno_padded` (domain padding), `fno_enriched`
    (prior as an input channel), `delta_fno` (prior as an additive residual),
    `ufno` (local-conv path), `fno_uloss` (U-value loss), `delta_fno_uloss`
    (delta head + U-value loss), `fno_physics` (PDE-residual loss, weight 0.1).
  - **Outcome (mean ± std over 3 seeds, 300 epochs · A100, native-resolution eval
    on 64 val samples = 15 clear / 49 bridged; reference `fno` U-MAE
    0.0242 ± 0.0034, field rel-L2 0.0147):** five of eight variants robustly beat
    the reference. **Winner `delta_fno`: U-MAE 0.0105 ± 0.0009 W/m²K** (−56% vs
    reference, 3.9× pooled σ) at the **best field rel-L2 in the sweep, 0.0131**
    — it improves the primary metric without trading off the secondary.
    `delta_fno_uloss` is statistically indistinguishable (0.0111 ± 0.0005); once
    the prior pins the boundary flux the U-loss has nothing left to correct, so the
    simpler `delta_fno` is the pick. `fno_enriched` 0.0162 ± 0.0017 (prior as a mere
    input channel gets ~half the gain), `ufno` 0.0196 ± 0.0010, `fno_uloss`
    0.0200 ± 0.0009. **The two pure boundary-treatment levers lose:** `fno_physics`
    0.0248 ± 0.0035 and `fno_padded` 0.0256 ± 0.0064 (also worst rel-L2, highest
    seed variance) sit within noise of the reference. The win is **stratified**:
    `delta_fno` collapses clear U-MAE 0.0076 → 0.0017 (−78%) and bridge
    0.0293 → 0.0133 (−55%), so the largest absolute gain lands on the hard bridged
    cases. **Decision: adopt `delta_fno` as the Block-1 backbone**; the geometry/
    physics prior that hands the operator the boundary structure wins, while
    loss-only and architecture-only boundary tweaks without that prior do not.
    See ADR `0004`; leaderboard in `results/block1_ablations.md`.
- **Block-1 out-of-distribution generalization study — the in-distribution winner
  travels.** ADR [`0005`](docs/decisions/0005-ood-generalization.md); Exp 1.4 in
  `docs/experiments.md`; artefacts `results/block1_ood.{json,md}` (git-tracked). Five
  variants (`delta_fno`, `fno`, `fno_uloss`, `ufno`, `fno_physics`) × 3 seeds ×
  2 data regimes (full 256 / lowdata 64) × 5 native-resolution test sets (30 training
  runs, A100, 300 ep). Four OOD axes each move **one** physically meaningful covariate
  — unseen wall assemblies, surface films outside the training band, a denser/wider
  bridge regime, finer discretisation — while **holding boundary temperatures fixed**,
  because θ=(T−T_out)/(T_in−T_out) is invariant to them under linear steady conduction
  (shifting temperatures is a no-op OOD test).
  - **`delta_fno` does not lose anywhere:** lowest U-MAE *and* smallest generalization
    gap on **all 8 OOD cells** (4 axes × 2 regimes). Full-regime U-MAE — `ood_walls`
    **0.0680±0.0089** (next ufno 0.3026, **4.4×** lead), `ood_bridges`
    **0.0491±0.0081** (next 0.1669, **3.4×**), `ood_films` **0.0617±0.0054**,
    `ood_res` **0.0143±0.0008** (next 0.0713, **5.0×**) — with near-flat gaps
    (`ood_res` +0.0037 vs ufno +0.0517). The hard analytic 1-D θ prior is a property
    of the physics, not the training distribution, so it carries across shift.
  - **Unseen wall assemblies are the binding risk** (hardest axis for everyone, mean
    gap +0.2419 full): walls ≫ bridges > films ≫ res. The in-distribution auxiliary
    winners overfit — `ufno` posts the largest gap in the study (lowdata `ood_walls`
    **+0.3791**), `fno_uloss` +0.2852 on walls.
  - **Soft PDE-residual loss earns a marginal low-data keep only:** a wash in full
    data, a consistent (but within-σ, directional) win on all 5 cells in lowdata,
    dominated ~8× by the hard prior. Prior strength: hard θ-channel ≫ U-supervision >
    soft residual.
  - **Decision:** `delta_fno` confirmed as the Block-1 default and the paper's lead
    contribution (report per-axis OOD U-MAE with assemblies as the headline stressor);
    the PDE-residual loss stays a low-data consistency rail, **not** re-enabled by
    default. Mandate for Block-2 GINO: make the **assembly/material-layer encoding**
    the design and OOD-evaluation focus.

### Changed
- **Renamed the project `BuildTrust-3D` → `ThermoTwin-3D`** to match the thermal-twin thesis
  and the target venue (Automation in Construction).

### Deprecated / Archived
- Retired the earlier *robustness / corruption-benchmark* framing. Its planning docs
  (`experiments.md`, `models-and-code.md`) are preserved under
  `docs/_archive/legacy-robustness-benchmark/` for provenance only.

---

_Dates use ISO-8601. Most recent changes first. Initial scaffolding: 2026-06-23._
