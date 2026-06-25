# Block-2 Benchmark — 3-D wall blocks · hard (GINO vs grid FNO)

- corpus: `hard` · train: `data/processed/block2_hard_train` · val: `data/processed/block2_hard_val`
- device: `cuda` · epochs: 300 · batch: 1 · seeds: [1337, 1, 2] · latent/voxel grid: 16
- U-value derived from the predicted field at the indoor face (near-face band 0.08 in normalised coords); the same estimator is applied to every model and to the ground truth.
- field rel-L2 and U-MAE are scored on the **sampled points** for all models (the voxel FNO is resampled back to the cloud), a fair head-to-head.

| Model | Kind | Field rel-L2 ↓ | U-value MAE ↓ (W/m²K) | U-MAPE ↓ | vs 1-D clear ↑ | Infer (ms) | Params |
|---|---|---|---|---|---|---|---|
| fno_voxel | fno_voxel | 0.0233 ± 0.0002 | 0.0390 ± 0.0005 | 9.4% | 2.50× | 3.55 | 2,410,689 |
| gino | gino | 0.0258 ± 0.0010 | 0.0476 ± 0.0020 | 11.5% | 2.05× | 14.75 | 2,807,892 |
| delta_transolver | delta_transolver | 0.0290 ± 0.0001 | 0.0457 ± 0.0008 | 11.5% | 2.14× | 10.51 | 978,113 |
| delta_gino | delta_gino | 0.0314 ± 0.0004 | 0.0485 ± 0.0026 | 12.4% | 2.02× | 14.43 | 2,808,181 |
| transolver | transolver | 0.0368 ± 0.0001 | 0.0517 ± 0.0023 | 12.9% | 1.89× | 10.41 | 977,857 |
| prior_only | prior_only | 0.0541 ± 0.0000 | 0.0976 ± 0.0000 | 24.5% | 1.00× | 0.00 | 0 |

Geometry-blind 1-D clear-wall baseline U-MAE: 0.0976 W/m²K — the number any geometry-aware operator must beat (H1).

## Bridge-focused — does the learned correction beat the prior?

`correction rel-L2` = ‖pred−true‖ / ‖prior−true‖ — **`prior_only` ≡ 1.000 by construction, so < 1.000 means the operator genuinely beats the analytic prior**; the *bridge* columns restrict this to points the bridge perturbs (|true−prior| > τ in θ). `clear rel-L2` guards that clear walls are not corrupted.
Bridge region (τ=0.02) covers ~28.9% of points.

| Model | correction rel-L2 ↓ | bridge corr-relL2 (τ=0.02) ↓ | bridge corr-relL2 (τ=0.05) ↓ | correction R² ↑ | clear rel-L2 ↓ |
|---|---|---|---|---|---|
| fno_voxel | 0.3889 ± 0.0041 | 0.3350 ± 0.0037 | 0.3041 ± 0.0038 | 0.8463 ± 0.0033 | 0.0164 ± 0.0001 |
| gino | 0.4361 ± 0.0169 | 0.3912 ± 0.0134 | 0.3648 ± 0.0117 | 0.8065 ± 0.0149 | 0.0158 ± 0.0011 |
| delta_transolver | 0.4741 ± 0.0012 | 0.4538 ± 0.0011 | 0.4568 ± 0.0014 | 0.7715 ± 0.0011 | 0.0124 ± 0.0002 |
| delta_gino | 0.5626 ± 0.0060 | 0.5455 ± 0.0069 | 0.5192 ± 0.0098 | 0.6783 ± 0.0068 | 0.0108 ± 0.0002 |
| transolver | 0.6063 ± 0.0013 | 0.5757 ± 0.0024 | 0.5749 ± 0.0027 | 0.6265 ± 0.0016 | 0.0169 ± 0.0001 |
| prior_only | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | -0.0162 ± 0.0000 | 0.0056 ± 0.0000 |
