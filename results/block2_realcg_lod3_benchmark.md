# Block-2 Benchmark — 3-D wall blocks · realcg_lod3 (GINO vs grid FNO)

- corpus: `realcg_lod3` · train: `data/processed/block2_realcg_lod3_train` · val: `data/processed/block2_realcg_lod3_val`
- device: `cuda` · epochs: 300 · batch: 1 · seeds: [1337, 1, 2] · latent/voxel grid: 16
- U-value derived from the predicted field at the indoor face (near-face band 0.08 in normalised coords); the same estimator is applied to every model and to the ground truth.
- field rel-L2 and U-MAE are scored on the **sampled points** for all models (the voxel FNO is resampled back to the cloud), a fair head-to-head.

| Model | Kind | Field rel-L2 ↓ | U-value MAE ↓ (W/m²K) | U-MAPE ↓ | vs 1-D clear ↑ | Infer (ms) | Params |
|---|---|---|---|---|---|---|---|
| delta_pointnet2 | delta_pointnet2 | 0.0201 ± 0.0001 | 0.0028 ± 0.0001 | 1.0% | 0.00× | 2.08 | 174,721 |
| prior_only | prior_only | 0.0214 ± 0.0000 | 0.0000 ± 0.0000 | 0.0% | 0.00× | 0.00 | 0 |
| delta_deeponet | delta_deeponet | 0.0217 ± 0.0001 | 0.0024 ± 0.0001 | 0.8% | 0.00× | 0.84 | 463,617 |
| delta_transolver | delta_transolver | 0.0246 ± 0.0026 | 0.0023 ± 0.0003 | 0.8% | 0.00× | 10.12 | 978,113 |
| delta_gnot | delta_gnot | 0.0255 ± 0.0016 | 0.0030 ± 0.0010 | 1.0% | 0.00× | 9.33 | 3,593,569 |
| delta_meshgraphnet | delta_meshgraphnet | 0.0260 ± 0.0002 | 0.0026 ± 0.0003 | 0.9% | 0.00× | 13.16 | 1,274,753 |
| delta_gino | delta_gino | 0.0308 ± 0.0006 | 0.0044 ± 0.0009 | 1.5% | 0.00× | 13.48 | 2,808,181 |
| pointnet2 | pointnet2 | 0.2323 ± 0.0002 | 0.0117 ± 0.0011 | 4.1% | 0.00× | 1.99 | 174,529 |
| meshgraphnet | meshgraphnet | 0.2455 ± 0.0007 | 0.0172 ± 0.0034 | 6.0% | 0.00× | 13.24 | 1,274,625 |
| transolver | transolver | 0.2471 ± 0.0026 | 0.0161 ± 0.0012 | 5.6% | 0.00× | 10.14 | 977,857 |
| gnot | gnot | 0.2737 ± 0.0108 | 0.0309 ± 0.0038 | 10.6% | 0.00× | 9.30 | 3,593,313 |
| fno_voxel | fno_voxel | 0.4415 ± 0.0000 | 0.0378 ± 0.0001 | 13.1% | 0.00× | 3.92 | 2,410,689 |
| deeponet | deeponet | 0.4538 ± 0.0051 | 0.0478 ± 0.0089 | 16.9% | 0.00× | 0.82 | 463,361 |
| gino | gino | 0.8766 ± 0.0316 | 0.2276 ± 0.0036 | 79.5% | 0.00× | 13.54 | 2,807,892 |

Geometry-blind 1-D clear-wall baseline U-MAE: 0.0000 W/m²K — the number any geometry-aware operator must beat (H1).

## Bridge-focused — does the learned correction beat the prior?

`correction rel-L2` = ‖pred−true‖ / ‖prior−true‖ — **`prior_only` ≡ 1.000 by construction, so < 1.000 means the operator genuinely beats the analytic prior**; the *bridge* columns restrict this to points the bridge perturbs (|true−prior| > τ in θ). `clear rel-L2` guards that clear walls are not corrupted.
Bridge region (τ=0.02) covers ~7.1% of points.

| Model | correction rel-L2 ↓ | bridge corr-relL2 (τ=0.02) ↓ | bridge corr-relL2 (τ=0.05) ↓ | correction R² ↑ | clear rel-L2 ↓ |
|---|---|---|---|---|---|
| delta_pointnet2 | 0.9049 ± 0.0026 | 0.8424 ± 0.0032 | 0.8569 ± 0.0035 | 0.1642 ± 0.0049 | 0.0087 ± 0.0002 |
| prior_only | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | -0.0208 ± 0.0000 | 0.0028 ± 0.0000 |
| delta_deeponet | 1.0015 ± 0.0029 | 0.9855 ± 0.0025 | 0.9857 ± 0.0023 | -0.0238 ± 0.0060 | 0.0054 ± 0.0002 |
| delta_transolver | 1.1078 ± 0.1037 | 0.9308 ± 0.0078 | 0.9291 ± 0.0141 | -0.2637 ± 0.2371 | 0.0145 ± 0.0052 |
| delta_meshgraphnet | 1.1391 ± 0.0064 | 0.9091 ± 0.0072 | 0.8858 ± 0.0088 | -0.3245 ± 0.0148 | 0.0165 ± 0.0005 |
| delta_gnot | 1.1571 ± 0.0164 | 0.9087 ± 0.0130 | 0.9160 ± 0.0120 | -0.3669 ± 0.0387 | 0.0181 ± 0.0011 |
| delta_gino | 1.3742 ± 0.0337 | 1.0296 ± 0.0059 | 0.9974 ± 0.0015 | -0.9289 ± 0.0940 | 0.0225 ± 0.0012 |
| pointnet2 | 9.6120 ± 0.0093 | 2.6694 ± 0.0108 | 1.5989 ± 0.0095 | -93.3086 ± 0.1822 | 0.2310 ± 0.0003 |
| meshgraphnet | 10.1735 ± 0.0353 | 3.2112 ± 0.0091 | 2.0205 ± 0.0057 | -104.6486 ± 0.7332 | 0.2398 ± 0.0010 |
| transolver | 10.2279 ± 0.1065 | 2.8667 ± 0.0242 | 1.6039 ± 0.0224 | -105.7924 ± 2.2302 | 0.2452 ± 0.0027 |
| gnot | 11.4215 ± 0.4935 | 3.2705 ± 0.1581 | 1.9343 ± 0.1345 | -132.4060 ± 11.4189 | 0.2733 ± 0.0121 |
| fno_voxel | 18.2744 ± 0.0007 | 3.8706 ± 0.0003 | 2.3039 ± 0.0001 | -339.8848 ± 0.0277 | 0.4507 ± 0.0000 |
| deeponet | 18.7894 ± 0.2297 | 3.8839 ± 0.2263 | 2.1995 ± 0.1678 | -359.4225 ± 8.8181 | 0.4640 ± 0.0056 |
| gino | 36.4393 ± 1.2996 | 9.1436 ± 0.1925 | 5.8886 ± 0.2679 | -1356.1036 ± 97.8961 | 0.8876 ± 0.0328 |
