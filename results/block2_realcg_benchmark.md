# Block-2 Benchmark — 3-D wall blocks · realcg (GINO vs grid FNO)

- corpus: `realcg` · train: `data/processed/block2_realcg_train` · val: `data/processed/block2_realcg_val`
- device: `cuda` · epochs: 300 · batch: 1 · seeds: [1337, 1, 2] · latent/voxel grid: 16
- U-value derived from the predicted field at the indoor face (near-face band 0.08 in normalised coords); the same estimator is applied to every model and to the ground truth.
- field rel-L2 and U-MAE are scored on the **sampled points** for all models (the voxel FNO is resampled back to the cloud), a fair head-to-head.

| Model | Kind | Field rel-L2 ↓ | U-value MAE ↓ (W/m²K) | U-MAPE ↓ | vs 1-D clear ↑ | Infer (ms) | Params |
|---|---|---|---|---|---|---|---|
| prior_only | prior_only | 0.0152 ± 0.0000 | 0.0000 ± 0.0000 | 0.0% | 0.00× | 0.00 | 0 |
| delta_transolver | delta_transolver | 0.0183 ± 0.0014 | 0.0051 ± 0.0006 | 1.8% | 0.00× | 10.32 | 978,113 |
| delta_gino | delta_gino | 0.0254 ± 0.0021 | 0.0058 ± 0.0020 | 2.0% | 0.00× | 14.09 | 2,808,181 |
| transolver | transolver | 0.2454 ± 0.0009 | 0.0269 ± 0.0024 | 9.5% | 0.00× | 10.22 | 977,857 |
| fno_voxel | fno_voxel | 0.4310 ± 0.0000 | 0.0530 ± 0.0001 | 18.7% | 0.00× | 3.10 | 2,410,689 |
| gino | gino | 0.8077 ± 0.0479 | 0.2120 ± 0.0560 | 74.8% | 0.00× | 14.29 | 2,807,892 |

Geometry-blind 1-D clear-wall baseline U-MAE: 0.0000 W/m²K — the number any geometry-aware operator must beat (H1).

## Bridge-focused — does the learned correction beat the prior?

`correction rel-L2` = ‖pred−true‖ / ‖prior−true‖ — **`prior_only` ≡ 1.000 by construction, so < 1.000 means the operator genuinely beats the analytic prior**; the *bridge* columns restrict this to points the bridge perturbs (|true−prior| > τ in θ). `clear rel-L2` guards that clear walls are not corrupted.
Bridge region (τ=0.02) covers ~4.5% of points.

| Model | correction rel-L2 ↓ | bridge corr-relL2 (τ=0.02) ↓ | bridge corr-relL2 (τ=0.05) ↓ | correction R² ↑ | clear rel-L2 ↓ |
|---|---|---|---|---|---|
| prior_only | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | -0.0166 ± 0.0000 | 0.0028 ± 0.0000 |
| delta_transolver | 1.1362 ± 0.0878 | 0.8976 ± 0.0163 | 0.9074 ± 0.0151 | -0.3201 ± 0.2061 | 0.0126 ± 0.0020 |
| delta_gino | 1.5252 ± 0.1208 | 1.0090 ± 0.0038 | 0.9830 ± 0.0017 | -1.3797 ± 0.3692 | 0.0206 ± 0.0028 |
| transolver | 13.8686 ± 0.0524 | 3.1856 ± 0.0095 | 1.9020 ± 0.0011 | -194.5293 ± 1.4755 | 0.2443 ± 0.0010 |
| fno_voxel | 24.3618 ± 0.0006 | 4.5336 ± 0.0012 | 2.8530 ± 0.0019 | -602.3348 ± 0.0295 | 0.4377 ± 0.0000 |
| gino | 46.0747 ± 2.9129 | 8.9174 ± 0.4334 | 5.5689 ± 0.3329 | -2165.6849 ± 277.1711 | 0.8242 ± 0.0537 |
