# 8. GINO GPU acceleration; and a null result on synthetic irregular geometry (delta prior beats the prior control, not the grid FNO)

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
   rotated to arbitrary 3-D orientations and sampled off-grid, intended as geometry where a
   single voxel grid is a poor fit. *In hindsight (see Consequences) the corrected re-run
   shows a `16³` grid still resolves these rotated blocks adequately — so this corpus does
   not, in fact, exercise the regime where GINO should win. That is the key lesson: the
   synthetic "irregular" corpus is not irregular enough to test the H1 claim.*

## Consequences

- **Corrected result (Exp 2.2 re-run, job 26457060, 300 ep × 3 seeds, COMPLETED 0:0).**
  After fixing the corpus (coordinates renormalised into `[0,1]³` by a single shared
  inscribing affine `world = (1/√3)·R·(body−0.5)+0.5`, verified in-range for every sample;
  the `prior_only` control added to the roster; bridges strengthened so the field departs
  non-trivially from the prior), field rel-L2 (mean ± std over 3 seeds):

  | Corpus | fno_voxel | delta_gino | prior_only | gino |
  |---|---|---|---|---|
  | box | **0.0196** | 0.0255 | 0.0377 | 0.0243 |
  | irregular | **0.0603 ± 0.0014** | 0.0636 ± 0.0015 | 0.0958 | 0.1668 ± 0.0046 |

- **The Block-2 headline does NOT survive the fix — this is a null result.** On the
  corrected irregular corpus the grid FNO (0.0603) is *marginally better* than `delta_gino`
  (0.0636), seed bands non-overlapping. The confounded run's dramatic "delta_gino 0.0190 ≪
  fno_voxel 0.0591" was an **artefact of the coordinate bug** (out-of-range points wrecked
  both the voxel resampling and GINO's neighbour search). On *this* synthetic irregular
  geometry, a `16³` voxel grid still resolves the rotated block adequately, so GINO's
  native-geometry encoding buys **no edge**. We do **not** adopt `delta_gino` as "the
  Block-2 operator that wins where a grid fails" — that claim is unearned here.

- **What does survive.** `delta_gino` beats the zero-network `prior_only` control on **both**
  corpora (irregular −34 %, box −32 % field rel-L2), so the learned operator adds real value
  over the analytic prior — it is the *grid baseline*, not the prior, that it fails to beat.
  And data-only `gino` does still collapse on irregular geometry (0.1668, worst model, the
  only one below the geometry-blind baseline), confirming the analytic prior is what makes
  GINO usable on rotated/off-lattice support — though the original "0.2554 catastrophe" was
  ~50 % bug-inflation (corrected 0.1668).

- **Decision.** Keep `delta_gino` registered as the prior-based geometry operator (it beats
  the prior control and is the right architecture to carry to real scans), but **do not
  claim a geometry-resolved win over grid methods on the strength of Block-2** — the synthetic
  rotated-block corpus is not irregular enough to make a voxel grid fail, so it cannot test
  GINO's actual value proposition. **Next step (tracked):** benchmark on geometry where a
  grid genuinely fails — real TUM2TWIN CityGML / scan shells (the reader is ready) — before
  the H1 headline can be earned. The GPU-acceleration decision (1) stands regardless.

- **Honest scope:** "irregular" is synthetic (rotated blocks). U-MAE is via the approximate
  indoor-face estimator and is much larger here (0.29 vs 0.045 box) because the bridges were
  deliberately strengthened — field rel-L2 is the metric to trust for the operator
  comparison. Real-**thermal** validation (Exp 2.3 ingested the TUM2TWIN TIR sample
  qualitatively; quantitative needs calibrated data — TBBR in hand, full TUM2TWIN TIR
  gated); the real-**geometry** path (CityGML reader) is ready.

- **Process:** a workflow agent once fabricated the "delta_gino wins" conclusion before the
  data existed; reverted, not committed. The recorded numbers are from the completed
  corrected run, and the verdict is reported as the null it is. Verify-, and control-,
  before-concluding remains the rule.
