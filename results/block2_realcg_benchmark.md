# Block-2 Benchmark — 3-D wall blocks · realcg (GINO vs grid FNO)

- corpus: `realcg` · train: `data/processed/block2_realcg_train` · val: `data/processed/block2_realcg_val`
- device: `cuda` · epochs: 300 · batch: 1 · seeds: [1337, 1, 2] · latent/voxel grid: 16
- U-value derived from the predicted field at the indoor face (near-face band 0.08 in normalised coords); the same estimator is applied to every model and to the ground truth.
- field rel-L2 and U-MAE are scored on the **sampled points** for all models (the voxel FNO is resampled back to the cloud), a fair head-to-head.

| Model | Kind | Field rel-L2 ↓ | U-value MAE ↓ (W/m²K) | U-MAPE ↓ | vs 1-D clear ↑ | Infer (ms) | Params |
|---|---|---|---|---|---|---|---|
| prior_only | prior_only | 0.0152 ± 0.0000 | 0.0000 ± 0.0000 | 0.0% | 0.00× | 0.00 | 0 |
| delta_transolver | delta_transolver | 0.0183 ± 0.0014 | 0.0051 ± 0.0006 | 1.8% | 0.00× | 10.42 | 978,113 |
| delta_gino | delta_gino | 0.0254 ± 0.0021 | 0.0058 ± 0.0020 | 2.0% | 0.00× | 14.66 | 2,808,181 |
| transolver | transolver | 0.2454 ± 0.0009 | 0.0269 ± 0.0024 | 9.5% | 0.00× | 10.33 | 977,857 |
| fno_voxel | fno_voxel | 0.4310 ± 0.0000 | 0.0530 ± 0.0001 | 18.7% | 0.00× | 3.17 | 2,410,689 |
| gino | gino | 0.8077 ± 0.0479 | 0.2120 ± 0.0560 | 74.8% | 0.00× | 14.60 | 2,807,892 |

Geometry-blind 1-D clear-wall baseline U-MAE: 0.0000 W/m²K — the number any geometry-aware operator must beat (H1).
