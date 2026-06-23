# Experiment Plan — ThermoTwin-3D

Four experiment blocks. Each maps to a sub-directory under `experiments/` and composes from
`configs/experiment/`.

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
