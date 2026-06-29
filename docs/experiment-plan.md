# Experiment Plan — ThermoTwin-3D

Four experiment blocks. Each maps to a sub-directory under `experiments/` and composes from
`configs/experiment/`.

> **Current status (2026-06-30) — pivoted to the SYSTEM; target Automation in Construction.**
> The forward gate (ADR 0010) confirmed the analytic prior is near-optimal on real clear-wall
> envelopes — the forward operator is the *engine*, not the contribution. A three-agent adversarial
> novelty re-scan confirmed there is **no defensible residual-mechanism novelty** (every leg is
> 2025–26 prior art; ~5/10 for ML), but the **end-to-end system is unoccupied (~8/10 for AiC)**: the
> scan→thermal-field and inverse-thermography rails have never been joined on as-built envelopes.
> **The contribution is the geometry-resolved inverse twin** — see
> **[ADR 0011](decisions/0011-aic-system-contribution.md)** (supersedes ADR 0010's open fork).
> First inverse-twin results are in (`results/inverse_*.json`); a boundary-layer A/B ablation
> (`results/bl_*.json`) is the method-section evidence.

## Block 1 — Controlled synthetic FEM benchmark  (`experiments/block1_synthetic_fem/`)
**Goal:** establish field accuracy and the speedup headline on a clean, controlled corpus.
- Train on FEM heat-conduction fields generated over the geometry corpus (`scripts/generate_fem_groundtruth.py`).
- Report temperature RMSE/MAE, heat-flux error, relative L2; inference time and **speedup vs FEM**.
- Generalisation sweeps: unseen geometries, unseen BCs, cross-resolution (discretisation-invariance).

## Block 2 — Real-building validation  (`experiments/block2_real_validation/`)
**Goal:** show the twin holds on real measured thermal fields.
- Validate predictions against **TUM2TWIN** thermal point clouds and **TBBR** thermal-bridge maps.
- Building-relevant metrics: per-surface U-value error, whole-building transmission heat-loss
  error, thermal-bridge localisation + quantification.

## Block 3 — Ablations  (`experiments/block3_ablations/`)
**Goal:** prove each component earns its keep.
- physics loss **on/off**
- geometry encoding **on/off** (vs lumped/zone input)
- **PINN vs neural-operator** formulation
- thermal-data assimilation **with/without**

## Block 4 — Retrofit what-if  (`experiments/block4_retrofit/`)
**Goal:** the payoff — a *differentiable* twin optimising an intervention on real geometry.
- Gradient-based retrofit optimisation (e.g. where to add insulation / fix a bridge) on the
  real envelope; report predicted heat-loss reduction and the optimisation trajectory.

## Cross-cutting design
- **Seed everything** (default `1337`); persist a per-run manifest so results reproduce exactly.
- **Pair metrics:** every result reports one operator-learning metric (relative L2) and one
  building metric (U-value or heat-loss error).
- **Sanity gates:** before trusting a learned number, confirm the FEM/EnergyPlus reference
  itself is converged and that the model reproduces a held-out FEM field within tolerance.
- Configs live in `configs/experiment/blockN_*.yaml`; launch via
  `python scripts/train.py experiment=blockN_*` (or `sbatch scripts/slurm/train.slurm ...`).
