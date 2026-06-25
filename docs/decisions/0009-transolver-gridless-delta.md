# 9. Gridless physics-attention (Transolver) + the delta prior wins on irregular 3-D geometry

Date: 2026-06-25

## Status

Accepted

## Context

Block-2 (Exp 2.1–2.2) produced a **null**: no geometry-conditioned operator beat the
voxel-FNO grid baseline on 3-D wall blocks. A six-spike research + diagnosis campaign
(see [`block2_redesign.md`](../block2_redesign.md)) found the null was a property of the
*benchmark*, not the operators:

1. **Equal-resolution framing.** GINO's latent grid was 16³ with FNO modes (6,6,6) — the
   *same* as the voxel baseline — so GINO inherited the grid's resolution ceiling and could
   only add GNO encode/decode error on top.
2. **The "irregular" corpus was a shrunk tilted box** (the 1/√3 inscribe left the solid in
   19 % of cells) that a 16³ grid voxelises near-losslessly.
3. **The irregular U-MAE was a frame-bug artefact** (indoor-face band in world axis-0 selects
   ~1/2048 points on rotated blocks).

The literature scout's top pick to break the grid bottleneck was **Transolver** (Wu et al.,
*Transolver: A Fast Transformer Solver for PDEs on General Geometries*, ICML 2024,
arXiv:2402.02366) — a **gridless** operator that soft-assigns points to learnable
physics-attention "slices" and attends over the slices, with documented wins over GINO on
irregular 3-D (Shape-Net Car 0.0207 vs GINO 0.0386). It runs in our env (pure torch + einops).

## Decision

1. **Adopt `delta_transolver` as the Block-2 lead operator** — a self-contained vendored
   Transolver (`models/transolver.py`, `TransolverOperator`) wrapped with the proven hard 1-D
   clear-wall delta prior (`DeltaTransolver`, mirroring `DeltaGino`): predict the correction at
   the native points, add the analytic prior back. Wired into `models/registry.py` and
   `scripts/benchmark_block2.py` (full roster: gino, delta_gino, transolver, delta_transolver,
   fno_voxel, prior_only).
2. **Keep `delta_gino` and `fno_voxel` as comparators**; keep `prior_only` as the control.
3. **Do not claim a real-world geometry-resolved result on synthetic evidence** — carry the
   recipe to real CityGML / scan geometry first (Exp 2.6).

## Consequences

- **The Block-2 null becomes a win on the irregular corpus (field rel-L2, 300 ep × 3 seeds).**
  On the *same* rotated-block corpus where Exp 2.2 was a null:

  | Model | Field rel-L2 | vs grid | Params |
  |---|---|---|---|
  | **delta_transolver** | **0.0444 ± 0.0009** | — | 978k |
  | fno_voxel (grid) | 0.0603 ± 0.0014 | −26 % | 2.41M |
  | delta_gino (Exp-2.2) | 0.0636 ± 0.0015 | −30 % | 2.81M |

  Seed bands non-overlapping (Δ ≈ 11σ); the win comes at **⅓ the parameters**.

- **Mechanism: gridless × prior, both needed.** Data-only `transolver` *collapses* on rotated
  geometry (0.1068, like data-only `gino` 0.1668) — rotation destroys its positional cue;
  +prior → 0.0444. Gridlessness handles the rotation that smears a voxel lattice; the prior
  supplies the bulk physics.

- **The "hard" sub-voxel-fin corpus was a recorded null — the grid won** (fno_voxel 0.0233 <
  gino 0.0258 < delta_transolver 0.0290). The fins were too thin for uniform sampling to cover,
  and the *axis-aligned* voxel baseline captured a coarse average. **Lesson, sharpened:** it is
  **non-axis-alignment (rotation), not sub-grid feature size, that breaks a voxel grid** — which
  is exactly the regime real building geometry lives in, and where `delta_transolver` won.

- **Caveat / housekeeping carried forward:** the irregular U-MAE is untrustworthy (world-frame
  estimator on rotated geometry) — field rel-L2 is the metric the win rests on; a body-frame
  U-fix is the next housekeeping item. Real-thermal validation still blocked by the absence of a
  calibrated 3-D envelope thermal-field dataset (see `docs/datasets.md`); the runnable real test
  is real **geometry** (CityGML) with physics-exact simulated GT.

- **Next (Exp 2.6):** real-CityGML per-surface FV corpus → run the same roster; the geometry is
  genuinely irregular, so this is the test that converts the synthetic win into a real-world one.
