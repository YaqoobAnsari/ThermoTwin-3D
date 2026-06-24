# Block-2 Benchmark — 3-D wall blocks · rotated / off-lattice irregular corpus (GINO vs grid FNO)

- corpus: `irreg` · train: `data/processed/block2_irreg_train` · val: `data/processed/block2_irreg_val`
- device: `cuda` · epochs: 300 · batch: 1 · seeds: [1337, 1, 2] · latent/voxel grid: 16
- U-value derived from the predicted field at the indoor face (near-face band 0.08 in normalised coords); the same estimator is applied to every model and to the ground truth.
- field rel-L2 and U-MAE are scored on the **sampled points** for all models (the voxel FNO is resampled back to the cloud), a fair head-to-head.

| Model | Kind | Field rel-L2 ↓ | U-value MAE ↓ (W/m²K) | U-MAPE ↓ | vs 1-D clear ↑ | Infer (ms) | Params |
|---|---|---|---|---|---|---|---|
| delta_gino | delta_gino | 0.0190 ± 0.0002 | 0.0410 ± 0.0003 | 11.6% | 1.12× | 11.26 | 2,808,181 |
| fno_voxel | fno_voxel | 0.0591 ± 0.0001 | 0.0492 ± 0.0007 | 14.9% | 0.93× | 3.35 | 2,410,689 |
| gino | gino | 0.2554 ± 0.0040 | 0.1049 ± 0.0012 | 33.4% | 0.44× | 11.27 | 2,807,892 |

Geometry-blind 1-D clear-wall baseline U-MAE: 0.0459 W/m²K — the number any geometry-aware operator must beat (H1).
