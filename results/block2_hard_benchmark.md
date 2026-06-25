# Block-2 Benchmark — 3-D wall blocks · hard (GINO vs grid FNO)

- corpus: `hard` · train: `data/processed/block2_hard_train` · val: `data/processed/block2_hard_val`
- device: `cuda` · epochs: 300 · batch: 1 · seeds: [1337, 1, 2] · latent/voxel grid: 16
- U-value derived from the predicted field at the indoor face (near-face band 0.08 in normalised coords); the same estimator is applied to every model and to the ground truth.
- field rel-L2 and U-MAE are scored on the **sampled points** for all models (the voxel FNO is resampled back to the cloud), a fair head-to-head.

| Model | Kind | Field rel-L2 ↓ | U-value MAE ↓ (W/m²K) | U-MAPE ↓ | vs 1-D clear ↑ | Infer (ms) | Params |
|---|---|---|---|---|---|---|---|
| fno_voxel | fno_voxel | 0.0233 ± 0.0002 | 0.0390 ± 0.0005 | 9.4% | 2.50× | 3.12 | 2,410,689 |
| gino | gino | 0.0258 ± 0.0010 | 0.0476 ± 0.0020 | 11.5% | 2.05× | 14.58 | 2,807,892 |
| delta_transolver | delta_transolver | 0.0290 ± 0.0001 | 0.0457 ± 0.0008 | 11.5% | 2.14× | 10.36 | 978,113 |
| delta_gino | delta_gino | 0.0314 ± 0.0004 | 0.0485 ± 0.0026 | 12.4% | 2.02× | 14.32 | 2,808,181 |
| transolver | transolver | 0.0368 ± 0.0001 | 0.0517 ± 0.0023 | 12.9% | 1.89× | 10.29 | 977,857 |
| prior_only | prior_only | 0.0541 ± 0.0000 | 0.0976 ± 0.0000 | 24.5% | 1.00× | 0.00 | 0 |

Geometry-blind 1-D clear-wall baseline U-MAE: 0.0976 W/m²K — the number any geometry-aware operator must beat (H1).
