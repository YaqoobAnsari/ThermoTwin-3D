# Block-2 redesign — finding a 3-D heat-predictor operator that actually wins

> Authored 2026-06-25 after a six-spike research + diagnosis campaign (three literature,
> three code/feasibility). This is the authoritative plan for the overnight job batch and
> the follow-up. It supersedes the "delta_gino is the Block-2 operator" framing — Exp 2.2
> was a **null**, and the diagnosis below shows the null was largely an artefact of how the
> benchmark was constructed, not a property of geometry-conditioned operators.

## 1. Diagnosis — why Block-2 produced a null (ranked)

1. **Equal-resolution framing (PRIMARY).** `benchmark_block2.py` hardcodes `LATENT_GRID=16`,
   `VOXEL_GRID=16`, `FNO_MODES=(6,6,6)`. GINO scatters the cloud onto a **16³ latent grid**
   and runs an FNO there; the voxel baseline voxelises onto the **identical 16³ grid** with
   the **identical (6,6,6) modes**. Both bottleneck on the same spectral representation —
   GINO can only add GNO encode/decode error on top, never resolution. A geometry-native
   operator cannot beat a grid baseline *at the grid baseline's own resolution*. (Confirmed
   by GINO's own paper: grid-FNO beats every standalone graph/point method; GINO wins only
   by wrapping an FNO grid, never discarding it. Li et al., NeurIPS 2023, arXiv:2309.00583.)

2. **The "irregular" corpus is a shrunk, tilted box (PRIMARY).** `synthetic_3d_irreg.py`'s
   `INSCRIBE_SCALE = 1/√3 ≈ 0.577` leaves the rotated solid in **19.3% of the 16³ latent
   cells** (measured: `frac sdf<0 = 0.193`; points span `[0.167, 0.837]`). The through-wall
   profile that carries all the physics is resolved by ~9 latent cells; bridge footprints map
   to 0.55–4.2 cells (sub-cell at the low end). A rotated box still *is* a box — a 16³ grid
   voxelises it near-losslessly, so grids never fail and there is no sub-grid signal for a
   point operator to exploit.

3. **The irregular U-MAE is an artefact (METRIC BUG).** `u_from_indoor_face_cloud` selects
   the indoor face with `points[:,0] < band` (0.08) in **world** axis-0. Rotated blocks never
   reach world-x < 0.167, so the band selects ~1 point in 2048 → whole-cloud fallback → the
   0.29 U-MAE measures the interior, not the face. Field rel-L2 is the only trustworthy column
   on rotated geometry. Fix: apply the band in **body frame** (the rotation is stored per
   sample) or define the face from the SDF.

4. **The delta prior works; data-only GINO collapses for a knowable reason.** `delta_gino`
   beats the zero-network `prior_only` control on both corpora (−34% irregular, −32% box) —
   the residual-learning recipe transfers. Data-only `gino` collapses on rotation (0.1668)
   because the SDF box-distance is **sign-symmetric** (no indoor/outdoor cue) and rotation
   randomises the through-wall direction, removing the only positional signal. The prior is
   load-bearing, exactly as the 2-D OOD study predicted.

NOT factors: GNO radius / point density (over-populated, ~77 neighbours on irregular);
training/capacity (converged, std≈0.0015, fair 2.8M-vs-2.4M match). 96 samples is small but
identical for both models, so it does not explain GINO losing.

**Conclusion:** the null was a *rigged benchmark* (same grid for both models + box-like corpus
+ a broken building metric), not evidence that geometry-conditioned operators can't win. To
earn a real result we change the regime, the roster, and the metric.

## 2. What the literature says to build

- **Transolver** (Wu et al., ICML 2024 Spotlight, arXiv:2402.02366) — the only architecture
  with direct, independently-corroborated wins over GINO on irregular 3-D (Shape-Net Car
  0.0207 vs GINO 0.0386; AirfRANS ~8×; DrivAer++ 17.3% vs GINO 19.8% in the independent UIUC
  2025 study, arXiv:2510.05995). Learnable Physics-Attention "slices" align to physical-state
  boundaries (thermal bridges, material interfaces) — beats a fixed grid *even on gridded
  data* (its ablation: learned slices 0.0067 vs fixed squares 0.0088 on Darcy). No grid
  bottleneck, linear cost. **Runs in our env tonight** (smoke-tested on CPU; one-line
  `timm.trunc_normal_` shim; the irregular-mesh `forward(x_coords, fx_feats) -> (B,N,1)`
  matches our contract; SDF dropped — geometry enters through coords + slices).
- **GNOT** (Hao et al., ICML 2023) would be the ideal heterogeneous-input fit (coords +
  material features + SDF as typed streams; has Heat/Heatsink benchmarks) but is **blocked**
  tonight: every model file imports `dgl`, which is absent.
- **Physics:** hard delta-prior + correction is the right backbone for low-data/OOD
  (DeltaPhi, arXiv:2406.09795: **50% gain on irregular-domain heat transfer at ~100 samples**
  — our exact regime; ClawNO/HardNet: hard structure wins most when data is scarce). A
  **mollified-GNO (mGINO, arXiv:2504.08277)** makes a GINO decoder differentiable wrt query
  coords, enabling an exact 3-D autograd conduction residual on the point cloud — the
  principled way to add 3-D physics. Secondary to fixing the corpus + operator; queued as a
  follow-up variation.
- **Real thermal reality (corrected 2026-06-25 after an adversarial falsification spike):**
  calibrated, dense thermal paired with real-building 3-D geometry **does exist** at facade /
  small scale — **ThermoScenes** (absolute °C FLIR, 8 real facades + COLMAP geometry; Thermoxels
  → FEA meshes; in hand) and **TSDN/ThermalGS** (radiometric °C aerial over 5 buildings +
  photogrammetric mesh). So calibrated real-thermal validation is **feasible now**, not blocked.
  (TBBR is uncalibrated detection-only; TUM2TWIN TIR is uncalibrated intensity; Twin Houses is
  calibrated but point-wise.) The credible real story is therefore: physics-exact on synthetic +
  real-geometry, **plus a calibrated thermal-field check against ThermoScenes/TSDN**, with TBBR
  as a localisation auxiliary and Twin Houses as the point-calibrated U/heat-flux readout. The
  genuine surviving gap: a calibrated dense field over a *whole* real envelope *with* paired
  material/U/boundary GT at scale — a contribution gap, not a validation blocker. See
  `docs/data_inventory.md` §4.

## 3. The corpora (what makes grids fail)

- **`block2_hard` — hard synthetic where 16³ aliases.** Native FV in-plane lifted to 48–80²,
  `cells_per_layer` 4–6, **thin high-contrast fins/studs** with in-plane footprint ≤ L/16
  (k = steel/aluminium 50–160), optional L/U re-entrant voids via near-zero k, rotated
  off-lattice (existing irreg path) **with the 1/√3 shrink removed**, sampled at 4096–8192
  points; SDF latent grid stays modest (the point branch carries the fine info). Proven:
  16³-voxelising a thin-fin block gives ~24% U-error from aliasing while the field is
  point-resolvable.
- **`block2_realcg` — real-geometry CityGML.** 27 clean TUM2TWIN LoD2 buildings → per-surface
  structured FV patch (axis-0 layers from the surface's real construction) + injected bridges,
  points placed on the real surface polygon, SDF from the real building shell. Honest scope:
  per-surface 3-D conduction on real envelopes; whole-building bridge coupling not modelled.

## 4. The roster (comprehensive operator bake-off)

Point-cloud operators: `gino` (data-only), `delta_gino`, **`transolver`** (new), **`delta_transolver`** (new).
Grid reference: `fno_voxel` — but **memory-matched**, held to a resolution it cannot exceed,
*not* set equal to GINO's latent grid. Controls: `prior_only` (zero-network).
Key ablation: **GINO latent grid decoupled** (e.g. 16 vs 32) to test the bottleneck directly.

Metrics: **field rel-L2 (primary, trusted on all geometry)** + body-frame-fixed U-MAE
(secondary). Same support (sampled points) and same estimator for every model and the GT.

## 5. Overnight jobs (no monitors; dependency-chained gen → train)

1. **gen-hard** (CPU): build `block2_hard_{train,val}`.
2. **gen-realcg** (CPU): build `block2_realcg_{train,val}`.
3. **bake-hard** (GPU, afterok gen-hard): full roster × 3 seeds × 300 ep on `block2_hard`.
4. **bake-realcg** (GPU, afterok gen-realcg): full roster on `block2_realcg`.
5. (stretch) **latent-sweep / physics** variations.

Results → `results/block2_hard_benchmark.*`, `results/block2_realcg_benchmark.*` (new stems;
the Exp-2.2 null is preserved for provenance). Reviewed the morning after.

## 6. What was submitted (2026-06-25 night) — handoff

Implemented + validated tonight (166 tests green, ruff clean, CPU smoke of every model):
- **`src/thermotwin/models/transolver.py`** — gridless physics-attention operator
  (`TransolverOperator` + `DeltaTransolver`), self-contained port of the irregular-mesh
  Transolver (pure torch + einops). Wired into `models/registry.py` and the Block-2 runner.
- **`data/synthetic_3d.py`** — `random_block_hard` / `generate_corpus_hard`: fine-native
  blocks (cells_y/z ∈ {48,64,80}, ~18 through-wall) with thermal fins of 3–6 native cells
  (≈0.75–1.5 voxel cells at G=16 ⇒ grid aliases, cloud resolves). `--hard` in
  `scripts/generate_3d_gt.py`.
- **`scripts/benchmark_block2.py`** — roster extended to
  `{gino, delta_gino, transolver, delta_transolver, fno_voxel, prior_only}`; `--corpus hard`;
  `--out_stem` override (so re-running irreg does not clobber the committed Exp-2.2 null).

Slurm jobs (no monitors; review the morning after):

| Job | id | partition | depends | writes |
|---|---|---|---|---|
| hard-corpus generation | 26474379 | sapphire (CPU) | — | `data/processed/block2_hard_{train,val}` |
| hard bake-off (6 models × 3 seeds × 300 ep) | 26474380 | feit-gpu-a100 | afterok:26474379 | `results/block2_hard_benchmark.{json,md}` |
| irregular bake-off (+ Transolver) | 26474381 | feit-gpu-a100 | — | `results/block2_irreg_ops_benchmark.{json,md}` |

**What each job answers.**
- *hard*: does a gridless operator (Transolver) resolve the sub-voxel fin that the 16³
  voxel-FNO **and** the 16³-latent GINO both alias? Watch **U-MAE** (the fin's signal is
  local; field rel-L2 is bulk-dominated). If `transolver`/`delta_transolver` beat
  `fno_voxel` and `gino` on U-MAE, that is the mechanistic 3-D win — gridless > grid on
  sub-grid thermal features.
- *irreg*: Transolver vs GINO vs voxel-FNO on the rotated-block geometry (zero new data).
  **Field rel-L2 is the trusted metric** here (the U estimator is in the wrong frame on
  rotated geometry — see §1.3).

**Still designed-but-not-built (next session, in priority order):**
1. **Real-geometry corpus** `block2_realcg` — per-surface structured FV patches + injected
   bridges on the 27 TUM2TWIN LoD2 CityGML buildings (reader ready; recipe in §3 / the
   feasibility spike). This is the real-world anchor the project most needs.
2. **GINO latent-grid decoupling** (16 vs 32, memory-matched voxel baseline) — the direct
   test of root-cause #1; needs SDF stored at G=32 (cheap) + per-model grid config.
3. **Body-frame U-estimator fix** (root-cause #3) — pass the stored `rotation` so U-MAE is
   meaningful on rotated/real geometry.
4. **3-D physics residual** via a mollified-GNO autograd decoder (mGINO, arXiv:2504.08277)
   — the principled low-data/real-geometry consistency rail.

**Source changes are uncommitted** (working tree) — the jobs run from the working tree, so
this is intentional; commit after reviewing tomorrow's results.
