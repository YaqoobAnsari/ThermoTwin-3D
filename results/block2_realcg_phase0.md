# Block-2 Benchmark — 3-D wall blocks · realcg (GINO vs grid FNO)

- corpus: `realcg` · train: `data/processed/block2_realcg_train` · val: `data/processed/block2_realcg_val`
- device: `cuda` · epochs: 300 · batch: 1 · seeds: [1337, 1, 2] · latent/voxel grid: 16
- U-value derived from the predicted field at the indoor face (near-face band 0.08 in normalised coords); the same estimator is applied to every model and to the ground truth.
- field rel-L2 and U-MAE are scored on the **sampled points** for all models (the voxel FNO is resampled back to the cloud), a fair head-to-head.

| Model | Kind | Field rel-L2 ↓ | U-value MAE ↓ (W/m²K) | U-MAPE ↓ | vs 1-D clear ↑ | Infer (ms) | Params |
|---|---|---|---|---|---|---|---|
| delta_pointnet2 | delta_pointnet2 | 0.0146 ± 0.0002 | 0.0026 ± 0.0001 | 0.9% | 0.00× | 2.01 | 174,721 |
| cond_pointnet2 | cond_pointnet2 | 0.0147 ± 0.0000 | 0.0026 ± 0.0001 | 0.9% | 0.00× | 1.99 | 174,721 |
| cond_transolver | cond_transolver | 0.0148 ± 0.0001 | 0.0033 ± 0.0002 | 1.2% | 0.00× | 10.25 | 978,113 |
| prior_only | prior_only | 0.0152 ± 0.0000 | 0.0000 ± 0.0000 | 0.0% | 0.00× | 0.00 | 0 |
| delta_deeponet | delta_deeponet | 0.0156 ± 0.0001 | 0.0025 ± 0.0001 | 0.9% | 0.00× | 0.81 | 463,617 |
| delta_transolver | delta_transolver | 0.0183 ± 0.0014 | 0.0051 ± 0.0006 | 1.8% | 0.00× | 10.28 | 978,113 |
| delta_gino | delta_gino | 0.0254 ± 0.0021 | 0.0058 ± 0.0020 | 2.0% | 0.00× | 14.07 | 2,808,181 |
| pointnet2 | pointnet2 | 0.2240 ± 0.0002 | 0.0142 ± 0.0007 | 5.0% | 0.00× | 1.99 | 174,529 |
| delta_const_pointnet2 | delta_const_pointnet2 | 0.2242 ± 0.0004 | 0.0145 ± 0.0002 | 5.1% | 0.00× | 2.06 | 174,721 |
| transolver | transolver | 0.2454 ± 0.0009 | 0.0269 ± 0.0024 | 9.5% | 0.00× | 10.21 | 977,857 |
| delta_const_transolver | delta_const_transolver | 0.2464 ± 0.0031 | 0.0246 ± 0.0008 | 8.7% | 0.00× | 10.32 | 978,113 |
| predict_mean | predict_mean | 0.4308 ± 0.0000 | 0.0665 ± 0.0000 | 23.4% | 0.00× | 0.01 | 0 |
| deeponet | deeponet | 0.4610 ± 0.0083 | 0.1769 ± 0.0543 | 62.4% | 0.00× | 0.81 | 463,361 |
| cond_deeponet | cond_deeponet | 0.4777 ± 0.0146 | 0.1789 ± 0.0170 | 63.2% | 0.00× | 0.78 | 463,617 |
| delta_const_deeponet | delta_const_deeponet | 0.5625 ± 0.0240 | 0.2013 ± 0.0145 | 70.9% | 0.00× | 0.83 | 463,617 |
| cond_gino | cond_gino | 0.6645 ± 0.0193 | 0.1761 ± 0.0625 | 62.2% | 0.00× | 14.05 | 2,808,181 |
| delta_const_gino | delta_const_gino | 0.6700 ± 0.0190 | 0.1262 ± 0.0049 | 44.5% | 0.00× | 14.11 | 2,808,181 |
| gino | gino | 0.8077 ± 0.0479 | 0.2120 ± 0.0560 | 74.8% | 0.00× | 14.08 | 2,807,892 |

Geometry-blind 1-D clear-wall baseline U-MAE: 0.0000 W/m²K — the number any geometry-aware operator must beat (H1).

## Bridge-focused — does the learned correction beat the prior?

`correction rel-L2` = ‖pred−true‖ / ‖prior−true‖ — **`prior_only` ≡ 1.000 by construction, so < 1.000 means the operator genuinely beats the analytic prior**; the *bridge* columns restrict this to points the bridge perturbs (|true−prior| > τ in θ). `clear rel-L2` guards that clear walls are not corrupted.
Bridge region (τ=0.02) covers ~4.5% of points.

| Model | correction rel-L2 ↓ | bridge corr-relL2 (τ=0.02) ↓ | bridge corr-relL2 (τ=0.05) ↓ | correction R² ↑ | clear rel-L2 ↓ |
|---|---|---|---|---|---|
| delta_pointnet2 | 0.8942 ± 0.0093 | 0.7683 ± 0.0105 | 0.7819 ± 0.0071 | 0.1871 ± 0.0169 | 0.0085 ± 0.0001 |
| cond_pointnet2 | 0.9134 ± 0.0023 | 0.8228 ± 0.0031 | 0.8470 ± 0.0040 | 0.1519 ± 0.0042 | 0.0078 ± 0.0000 |
| cond_transolver | 0.9377 ± 0.0043 | 0.8851 ± 0.0029 | 0.9113 ± 0.0028 | 0.1062 ± 0.0083 | 0.0066 ± 0.0003 |
| prior_only | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | -0.0166 ± 0.0000 | 0.0028 ± 0.0000 |
| delta_deeponet | 1.0027 ± 0.0037 | 0.9811 ± 0.0007 | 0.9835 ± 0.0007 | -0.0220 ± 0.0075 | 0.0048 ± 0.0002 |
| delta_transolver | 1.1362 ± 0.0878 | 0.8976 ± 0.0163 | 0.9074 ± 0.0151 | -0.3201 ± 0.2061 | 0.0126 ± 0.0020 |
| delta_gino | 1.5252 ± 0.1208 | 1.0090 ± 0.0038 | 0.9830 ± 0.0017 | -1.3797 ± 0.3692 | 0.0206 ± 0.0028 |
| pointnet2 | 12.6633 ± 0.0149 | 2.8606 ± 0.0174 | 1.7635 ± 0.0255 | -162.0176 ± 0.3837 | 0.2239 ± 0.0002 |
| delta_const_pointnet2 | 12.6717 ± 0.0199 | 2.8729 ± 0.0394 | 1.7538 ± 0.0388 | -162.2326 ± 0.5124 | 0.2240 ± 0.0002 |
| transolver | 13.8686 ± 0.0524 | 3.1856 ± 0.0095 | 1.9020 ± 0.0011 | -194.5293 ± 1.4755 | 0.2443 ± 0.0010 |
| delta_const_transolver | 13.9324 ± 0.1960 | 3.1718 ± 0.0343 | 1.8795 ± 0.0313 | -196.3670 ± 5.5798 | 0.2457 ± 0.0036 |
| predict_mean | 24.3336 ± 0.0000 | 3.6745 ± 0.0000 | 1.8079 ± 0.0000 | -600.9372 ± 0.0000 | 0.4397 ± 0.0000 |
| deeponet | 26.1081 ± 0.5441 | 4.3466 ± 0.0460 | 2.4565 ± 0.0686 | -692.2300 ± 29.0221 | 0.4705 ± 0.0101 |
| cond_deeponet | 27.1441 ± 0.9275 | 4.7377 ± 0.2796 | 2.8054 ± 0.2808 | -748.8886 ± 50.7305 | 0.4882 ± 0.0162 |
| delta_const_deeponet | 32.5071 ± 1.4365 | 6.6512 ± 0.4120 | 4.4245 ± 0.4263 | -1075.3184 ± 94.2425 | 0.5803 ± 0.0259 |
| delta_const_gino | 37.9269 ± 1.1030 | 7.1925 ± 0.4296 | 4.1847 ± 0.0896 | -1462.5241 ± 84.2092 | 0.6793 ± 0.0187 |
| cond_gino | 37.9465 ± 1.0753 | 7.3099 ± 0.5097 | 4.6125 ± 0.3813 | -1463.9764 ± 83.7703 | 0.6793 ± 0.0185 |
| gino | 46.0747 ± 2.9129 | 8.9174 ± 0.4334 | 5.5689 ± 0.3329 | -2165.6849 ± 277.1711 | 0.8242 ± 0.0537 |
