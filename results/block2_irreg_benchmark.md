# Block-2 Benchmark — 3-D wall blocks · rotated / off-lattice irregular corpus (GINO vs grid FNO)

- corpus: `irreg` · train: `data/processed/block2_irreg_train` · val: `data/processed/block2_irreg_val`
- device: `cuda` · epochs: 300 · batch: 1 · seeds: [1337, 1, 2] · latent/voxel grid: 16
- U-value derived from the predicted field at the indoor face (near-face band 0.08 in normalised coords); the same estimator is applied to every model and to the ground truth.
- field rel-L2 and U-MAE are scored on the **sampled points** for all models (the voxel FNO is resampled back to the cloud), a fair head-to-head.

| Model | Kind | Field rel-L2 ↓ | U-value MAE ↓ (W/m²K) | U-MAPE ↓ | vs 1-D clear ↑ | Infer (ms) | Params |
|---|---|---|---|---|---|---|---|
| fno_voxel | fno_voxel | 0.0603 ± 0.0014 | 0.2918 ± 0.0027 | 44.6% | 1.10× | 3.10 | 2,410,689 |
| delta_gino | delta_gino | 0.0636 ± 0.0015 | 0.2974 ± 0.0042 | 46.2% | 1.08× | 10.61 | 2,808,181 |
| prior_only | prior_only | 0.0958 ± 0.0000 | 0.3211 ± 0.0000 | 50.2% | 1.00× | 0.00 | 0 |
| gino | gino | 0.1668 ± 0.0046 | 0.4798 ± 0.0429 | 82.1% | 0.67× | 10.67 | 2,807,892 |

Geometry-blind 1-D clear-wall baseline U-MAE: 0.3211 W/m²K — the number any geometry-aware operator must beat (H1).
