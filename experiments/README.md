# experiments/

One sub-directory per experiment block (see `../docs/experiment-plan.md`):

1. `block1_synthetic_fem/`    — controlled synthetic FEM benchmark.
2. `block2_real_validation/`  — TUM2TWIN / TBBR real-building validation.
3. `block3_ablations/`        — physics-loss, geometry-encoding, PINN-vs-operator, assimilation.
4. `block4_retrofit/`         — differentiable retrofit what-if optimisation.

Each block should contain its run configs, launch scripts, and analysis — not raw data or checkpoints.
