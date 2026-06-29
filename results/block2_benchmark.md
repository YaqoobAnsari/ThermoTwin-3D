# Block-2 Benchmark — 3-D wall blocks · axis-aligned box corpus (GINO vs grid FNO)

- corpus: `box` · train: `data/processed/block2_train` · val: `data/processed/block2_val`
- device: `cuda` · epochs: 300 · batch: 1 · seeds: [1337, 1, 2] · latent/voxel grid: 16
- U-value derived from the predicted field at the indoor face (near-face band 0.08 in normalised coords); the same estimator is applied to every model and to the ground truth.
- field rel-L2 and U-MAE are scored on the **sampled points** for all models (the voxel FNO is resampled back to the cloud), a fair head-to-head.

| Model | Kind | Field rel-L2 ↓ | U-value MAE ↓ (W/m²K) | U-MAPE ↓ | vs 1-D clear ↑ | Infer (ms) | Params |
|---|---|---|---|---|---|---|---|
| delta_pointnet2 | delta_pointnet2 | 0.0178 ± 0.0001 | 0.0632 ± 0.0005 | 14.1% | 1.23× | 1.86 | 174,721 |
| fno_voxel | fno_voxel | 0.0196 ± 0.0001 | 0.0450 ± 0.0004 | 10.5% | 1.72× | 3.89 | 2,410,689 |
| delta_gnot | delta_gnot | 0.0204 ± 0.0007 | 0.0402 ± 0.0048 | 9.8% | 1.96× | 5.62 | 3,593,569 |
| delta_meshgraphnet | delta_meshgraphnet | 0.0227 ± 0.0012 | 0.0769 ± 0.0033 | 17.7% | 1.01× | 7.56 | 1,274,753 |
| gino | gino | 0.0243 ± 0.0003 | 0.0493 ± 0.0037 | 11.2% | 1.58× | 10.26 | 2,807,892 |
| delta_gino | delta_gino | 0.0255 ± 0.0002 | 0.0486 ± 0.0012 | 11.5% | 1.60× | 10.21 | 2,808,181 |
| delta_transolver | delta_transolver | 0.0265 ± 0.0068 | 0.0446 ± 0.0054 | 10.8% | 1.77× | 6.17 | 978,113 |
| meshgraphnet | meshgraphnet | 0.0282 ± 0.0004 | 0.0874 ± 0.0013 | 20.3% | 0.89× | 7.52 | 1,274,625 |
| gnot | gnot | 0.0296 ± 0.0017 | 0.0607 ± 0.0133 | 14.9% | 1.33× | 5.62 | 3,593,313 |
| transolver | transolver | 0.0307 ± 0.0012 | 0.0702 ± 0.0069 | 15.9% | 1.12× | 6.14 | 977,857 |
| delta_deeponet | delta_deeponet | 0.0377 ± 0.0000 | 0.0775 ± 0.0001 | 18.3% | 1.00× | 0.65 | 463,617 |
| prior_only | prior_only | 0.0377 ± 0.0000 | 0.0776 ± 0.0000 | 18.4% | 1.00× | 0.00 | 0 |
| deeponet | deeponet | 0.0394 ± 0.0001 | 0.0688 ± 0.0029 | 15.7% | 1.13× | 0.60 | 463,361 |
| pointnet2 | pointnet2 | 0.0776 ± 0.0008 | 0.0764 ± 0.0004 | 17.0% | 1.02× | 1.82 | 174,529 |

Geometry-blind 1-D clear-wall baseline U-MAE: 0.0776 W/m²K — the number any geometry-aware operator must beat (H1).

## Bridge-focused — does the learned correction beat the prior?

`correction rel-L2` = ‖pred−true‖ / ‖prior−true‖ — **`prior_only` ≡ 1.000 by construction, so < 1.000 means the operator genuinely beats the analytic prior**; the *bridge* columns restrict this to points the bridge perturbs (|true−prior| > τ in θ). `clear rel-L2` guards that clear walls are not corrupted.
Bridge region (τ=0.02) covers ~21.9% of points.

| Model | correction rel-L2 ↓ | bridge corr-relL2 (τ=0.02) ↓ | bridge corr-relL2 (τ=0.05) ↓ | correction R² ↑ | clear rel-L2 ↓ |
|---|---|---|---|---|---|
| fno_voxel | 0.3889 ± 0.0015 | 0.3274 ± 0.0021 | 0.3029 ± 0.0018 | 0.8462 ± 0.0012 | 0.0143 ± 0.0000 |
| delta_pointnet2 | 0.4203 ± 0.0031 | 0.3952 ± 0.0029 | 0.3465 ± 0.0032 | 0.8204 ± 0.0026 | 0.0077 ± 0.0001 |
| delta_gnot | 0.4364 ± 0.0206 | 0.4205 ± 0.0232 | 0.4046 ± 0.0236 | 0.8059 ± 0.0185 | 0.0070 ± 0.0004 |
| gino | 0.4754 ± 0.0075 | 0.3957 ± 0.0071 | 0.3473 ± 0.0073 | 0.7701 ± 0.0073 | 0.0173 ± 0.0004 |
| delta_meshgraphnet | 0.5576 ± 0.0129 | 0.5418 ± 0.0163 | 0.4870 ± 0.0217 | 0.6837 ± 0.0146 | 0.0061 ± 0.0008 |
| gnot | 0.5966 ± 0.0602 | 0.5537 ± 0.0678 | 0.5179 ± 0.0629 | 0.6343 ± 0.0756 | 0.0142 ± 0.0009 |
| delta_gino | 0.6240 ± 0.0053 | 0.6023 ± 0.0059 | 0.5824 ± 0.0069 | 0.6041 ± 0.0067 | 0.0102 ± 0.0001 |
| delta_transolver | 0.6315 ± 0.2223 | 0.6238 ± 0.2287 | 0.6226 ± 0.2444 | 0.5442 ± 0.3193 | 0.0060 ± 0.0008 |
| meshgraphnet | 0.6390 ± 0.0060 | 0.6049 ± 0.0051 | 0.5426 ± 0.0041 | 0.5848 ± 0.0078 | 0.0110 ± 0.0005 |
| transolver | 0.6464 ± 0.0404 | 0.6138 ± 0.0430 | 0.5793 ± 0.0356 | 0.5734 ± 0.0520 | 0.0131 ± 0.0004 |
| deeponet | 0.8167 ± 0.0037 | 0.7518 ± 0.0044 | 0.6993 ± 0.0041 | 0.3217 ± 0.0062 | 0.0196 ± 0.0003 |
| delta_deeponet | 0.9999 ± 0.0001 | 0.9999 ± 0.0001 | 0.9999 ± 0.0000 | -0.0167 ± 0.0001 | 0.0049 ± 0.0000 |
| prior_only | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | -0.0169 ± 0.0000 | 0.0050 ± 0.0000 |
| pointnet2 | 1.3297 ± 0.0129 | 0.6586 ± 0.0073 | 0.5188 ± 0.0124 | -0.7980 ± 0.0347 | 0.0840 ± 0.0013 |
