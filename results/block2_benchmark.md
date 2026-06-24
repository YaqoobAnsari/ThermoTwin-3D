# Block-2 Benchmark — 3-D wall blocks · axis-aligned box corpus (GINO vs grid FNO)

- corpus: `box` · train: `data/processed/block2_train` · val: `data/processed/block2_val`
- device: `cuda` · epochs: 300 · batch: 1 · seeds: [1337, 1, 2] · latent/voxel grid: 16
- U-value derived from the predicted field at the indoor face (near-face band 0.08 in normalised coords); the same estimator is applied to every model and to the ground truth.
- field rel-L2 and U-MAE are scored on the **sampled points** for all models (the voxel FNO is resampled back to the cloud), a fair head-to-head.

| Model | Kind | Field rel-L2 ↓ | U-value MAE ↓ (W/m²K) | U-MAPE ↓ | vs 1-D clear ↑ | Infer (ms) | Params |
|---|---|---|---|---|---|---|---|
| fno_voxel | fno_voxel | 0.0196 ± 0.0001 | 0.0450 ± 0.0004 | 10.5% | 1.72× | 4.04 | 2,410,689 |
| gino | gino | 0.0243 ± 0.0003 | 0.0493 ± 0.0037 | 11.2% | 1.58× | 11.67 | 2,807,892 |
| delta_gino | delta_gino | 0.0255 ± 0.0002 | 0.0486 ± 0.0012 | 11.5% | 1.60× | 11.40 | 2,808,181 |

Geometry-blind 1-D clear-wall baseline U-MAE: 0.0776 W/m²K — the number any geometry-aware operator must beat (H1).
