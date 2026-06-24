# Block-2 Benchmark — 3-D irregular wall blocks (GINO vs grid FNO)

- device: `cuda` · epochs: 60 · batch: 1 · seeds: [1337] · latent/voxel grid: 16
- U-value derived from the predicted field at the indoor face (near-face band 0.08 in normalised coords); the same estimator is applied to every model and to the ground truth.
- field rel-L2 and U-MAE are scored on the **sampled points** for all models (the voxel FNO is resampled back to the cloud), a fair head-to-head.

| Model | Kind | Field rel-L2 ↓ | U-value MAE ↓ (W/m²K) | U-MAPE ↓ | vs 1-D clear ↑ | Infer (ms) | Params |
|---|---|---|---|---|---|---|---|
| fno_voxel | fno_voxel | 0.0202 ± 0.0000 | 0.0475 ± 0.0000 | 11.3% | 1.63× | 18.26 | 2,410,689 |
| gino | gino | 0.0238 ± 0.0000 | 0.0539 ± 0.0000 | 12.4% | 1.44× | 8.53 | 2,807,892 |
| delta_gino | delta_gino | 0.0243 ± 0.0000 | 0.0516 ± 0.0000 | 11.8% | 1.50× | 8.23 | 2,808,181 |

Geometry-blind 1-D clear-wall baseline U-MAE: 0.0776 W/m²K — the number any geometry-aware operator must beat (H1).
