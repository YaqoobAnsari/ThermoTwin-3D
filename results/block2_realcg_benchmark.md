# Block-2 Benchmark — 3-D wall blocks · realcg (GINO vs grid FNO)

- corpus: `realcg` · train: `data/processed/block2_realcg_train` · val: `data/processed/block2_realcg_val`
- device: `cuda` · epochs: 300 · batch: 1 · seeds: [1337, 1, 2] · latent/voxel grid: 16
- U-value derived from the predicted field at the indoor face (near-face band 0.08 in normalised coords); the same estimator is applied to every model and to the ground truth.
- field rel-L2 and U-MAE are scored on the **sampled points** for all models (the voxel FNO is resampled back to the cloud), a fair head-to-head.

| Model | Kind | Field rel-L2 ↓ | U-value MAE ↓ (W/m²K) | U-MAPE ↓ | vs 1-D clear ↑ | Infer (ms) | Params |
|---|---|---|---|---|---|---|---|
| delta_pointnet2 | delta_pointnet2 | 0.0146 ± 0.0001 | 0.0028 ± 0.0000 | 1.0% | 0.00× | 2.07 | 174,721 |
| prior_only | prior_only | 0.0152 ± 0.0000 | 0.0000 ± 0.0000 | 0.0% | 0.00× | 0.01 | 0 |
| delta_deeponet | delta_deeponet | 0.0156 ± 0.0001 | 0.0025 ± 0.0001 | 0.9% | 0.00× | 0.85 | 463,617 |
| delta_gnot | delta_gnot | 0.0176 ± 0.0024 | 0.0039 ± 0.0006 | 1.4% | 0.00× | 9.51 | 3,593,569 |
| delta_transolver | delta_transolver | 0.0183 ± 0.0014 | 0.0051 ± 0.0006 | 1.8% | 0.00× | 10.39 | 978,113 |
| delta_meshgraphnet | delta_meshgraphnet | 0.0184 ± 0.0001 | 0.0032 ± 0.0004 | 1.1% | 0.00× | 13.49 | 1,274,753 |
| delta_gino | delta_gino | 0.0254 ± 0.0021 | 0.0058 ± 0.0020 | 2.0% | 0.00× | 14.49 | 2,808,181 |
| pointnet2 | pointnet2 | 0.2239 ± 0.0003 | 0.0143 ± 0.0004 | 5.0% | 0.00× | 2.05 | 174,529 |
| meshgraphnet | meshgraphnet | 0.2334 ± 0.0022 | 0.0170 ± 0.0007 | 6.0% | 0.00× | 13.40 | 1,274,625 |
| transolver | transolver | 0.2454 ± 0.0009 | 0.0269 ± 0.0024 | 9.5% | 0.00× | 10.35 | 977,857 |
| gnot | gnot | 0.2460 ± 0.0012 | 0.0267 ± 0.0011 | 9.4% | 0.00× | 9.49 | 3,593,313 |
| fno_voxel | fno_voxel | 0.4310 ± 0.0000 | 0.0530 ± 0.0001 | 18.7% | 0.00× | 3.62 | 2,410,689 |
| deeponet | deeponet | 0.4610 ± 0.0083 | 0.1769 ± 0.0543 | 62.4% | 0.00× | 0.82 | 463,361 |
| gino | gino | 0.8077 ± 0.0479 | 0.2120 ± 0.0560 | 74.8% | 0.00× | 14.75 | 2,807,892 |

Geometry-blind 1-D clear-wall baseline U-MAE: 0.0000 W/m²K — the number any geometry-aware operator must beat (H1).

## Bridge-focused — does the learned correction beat the prior?

`correction rel-L2` = ‖pred−true‖ / ‖prior−true‖ — **`prior_only` ≡ 1.000 by construction, so < 1.000 means the operator genuinely beats the analytic prior**; the *bridge* columns restrict this to points the bridge perturbs (|true−prior| > τ in θ). `clear rel-L2` guards that clear walls are not corrupted.
Bridge region (τ=0.02) covers ~4.5% of points.

| Model | correction rel-L2 ↓ | bridge corr-relL2 (τ=0.02) ↓ | bridge corr-relL2 (τ=0.05) ↓ | correction R² ↑ | clear rel-L2 ↓ |
|---|---|---|---|---|---|
| delta_pointnet2 | 0.8923 ± 0.0075 | 0.7680 ± 0.0059 | 0.7821 ± 0.0034 | 0.1905 ± 0.0136 | 0.0085 ± 0.0001 |
| prior_only | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | -0.0166 ± 0.0000 | 0.0028 ± 0.0000 |
| delta_deeponet | 1.0027 ± 0.0037 | 0.9811 ± 0.0007 | 0.9835 ± 0.0007 | -0.0220 ± 0.0075 | 0.0048 ± 0.0002 |
| delta_meshgraphnet | 1.0862 ± 0.0085 | 0.8337 ± 0.0067 | 0.8181 ± 0.0050 | -0.1995 ± 0.0188 | 0.0123 ± 0.0003 |
| delta_gnot | 1.0901 ± 0.1313 | 0.9116 ± 0.0092 | 0.9267 ± 0.0029 | -0.2256 ± 0.3027 | 0.0108 ± 0.0037 |
| delta_transolver | 1.1362 ± 0.0878 | 0.8976 ± 0.0163 | 0.9074 ± 0.0151 | -0.3201 ± 0.2061 | 0.0126 ± 0.0020 |
| delta_gino | 1.5252 ± 0.1208 | 1.0090 ± 0.0038 | 0.9830 ± 0.0017 | -1.3797 ± 0.3692 | 0.0206 ± 0.0028 |
| pointnet2 | 12.6575 ± 0.0163 | 2.8650 ± 0.0177 | 1.7659 ± 0.0281 | -161.8667 ± 0.4202 | 0.2238 ± 0.0002 |
| meshgraphnet | 13.2342 ± 0.1300 | 3.1712 ± 0.0214 | 2.0064 ± 0.0101 | -177.0629 ± 3.4876 | 0.2327 ± 0.0025 |
| transolver | 13.8686 ± 0.0524 | 3.1856 ± 0.0095 | 1.9020 ± 0.0011 | -194.5293 ± 1.4755 | 0.2443 ± 0.0010 |
| gnot | 13.8982 ± 0.0692 | 3.2307 ± 0.0387 | 1.9259 ± 0.0294 | -195.3645 ± 1.9559 | 0.2447 ± 0.0012 |
| fno_voxel | 24.3618 ± 0.0006 | 4.5336 ± 0.0012 | 2.8530 ± 0.0019 | -602.3348 ± 0.0295 | 0.4377 ± 0.0000 |
| deeponet | 26.1081 ± 0.5441 | 4.3466 ± 0.0460 | 2.4565 ± 0.0686 | -692.2300 ± 29.0221 | 0.4705 ± 0.0101 |
| gino | 46.0747 ± 2.9129 | 8.9174 ± 0.4334 | 5.5689 ± 0.3329 | -2165.6849 ± 277.1711 | 0.8242 ± 0.0537 |
