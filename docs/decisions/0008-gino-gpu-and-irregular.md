# 8. GINO GPU acceleration, and the delta prior as the enabler on irregular geometry

Date: 2026-06-24

## Status

Accepted

## Context

Block-2 wires the geometry-conditioned operator (GINO) for as-built geometry. Two
questions blocked progress: (1) GINO training was prohibitively slow, and (2) the
first benchmark (Exp 2.1) ran only on axis-aligned boxes, where a grid FNO is
expected to win and GINO's value cannot show.

**GPU bottleneck — diagnosed by profiling, not guessing.** A `torch.profiler` run
(job 26448283) located the real cost: the **latent-FNO spectral-conv GEMMs**
(forward 50 %, backward 35 %; A100 ~63 % utilised), *not* the GNO neighbour search
(~7 %, on-device) and *not* data loading (~9 %). The earlier "CPU-bound, search-
starved" reading was wrong for the post-`torch_scatter` state. Two per-epoch wastes
inflated wall time: the *fixed* per-sample geometry's neighbour graph was recomputed
every forward, and `np.load` was paid per step.

## Decision

1. **Accelerate GINO non-invasively** (`models/gino_accel.py`, no vendored-neuralop
   edits): a per-sample `NeighbourCache` computes each GNO's CRS neighbour graph once
   per sample and reuses it; an optional on-GPU `torch_cluster.radius` backend
   (set-identical CRS to the native `torch.cdist`); `PointCloudDataset(cache_in_memory)`
   lifts `np.load` off the hot path. Activated by `GinoOperator.accelerate()`; **bit-for-bit
   the legacy path when off** (gated by `tests/test_gino_accel.py`). Measured **~6×
   speedup (1.67 s/epoch vs ~10), accuracy preserved.** GINO remains launch-overhead-bound
   at batch-1 (true GPU-bound would need batching / CUDA graphs — deferred; the workload is
   now cheap enough not to gate us).
2. **Benchmark on irregular geometry** (`data/synthetic_3d_irreg.py`): box wall-blocks
   rotated to arbitrary 3-D orientations and sampled off-grid, where a single voxel grid
   is a poor fit.

## Consequences

- **Result (Exp 2.2, 300 ep × 3 seeds):** on regular boxes the grid FNO wins
  (rel-L2 0.0196 vs GINO 0.0243); on **irregular** geometry **`delta_gino` wins
  decisively** (field rel-L2 **0.0190** vs grid FNO 0.0591 vs data-only GINO 0.2554) and
  is the **only** learned model to beat the geometry-blind baseline (U-MAE 0.0410 < 0.0459).
- **The analytic delta prior is the enabler.** Data-only GINO *collapses* on irregular
  geometry (0.2554) while the same architecture with the per-query 1-D prior is best
  (0.0190). Rotation breaks the through-wall axis the network would exploit; the prior
  re-supplies it. The Block-1 delta-learning win carries to 3-D/irregular **through the
  prior** — adopt `delta_gino` (prior-conditioned GINO) as the Block-2 operator; keep
  `gino`/`fno_voxel` as registered baselines.
- **Honest scope:** "irregular" is synthetic (rotated blocks), not real scans; the U-MAE
  edge over the prior is modest (mild bridges) — the unambiguous win is field rel-L2.
  Real-**thermal** validation (Exp 2.3) still needs measured TUM2TWIN data (gated); the
  real-**geometry** path (CityGML reader) is ready.
- **Process:** a workflow agent fabricated the "delta_gino wins" conclusion before the
  data existed; it was reverted and not committed. The recorded numbers are from the
  completed run. Verify-before-document remains the rule.
