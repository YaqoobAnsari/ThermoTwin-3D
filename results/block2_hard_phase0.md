# Block-2 Benchmark — 3-D wall blocks · hard (GINO vs grid FNO)

- corpus: `hard` · train: `data/processed/block2_hard_train` · val: `data/processed/block2_hard_val`
- device: `cuda` · epochs: 300 · batch: 1 · seeds: [1337, 1, 2] · latent/voxel grid: 16
- U-value derived from the predicted field at the indoor face (near-face band 0.08 in normalised coords); the same estimator is applied to every model and to the ground truth.
- field rel-L2 and U-MAE are scored on the **sampled points** for all models (the voxel FNO is resampled back to the cloud), a fair head-to-head.

| Model | Kind | Field rel-L2 ↓ | U-value MAE ↓ (W/m²K) | U-MAPE ↓ | vs 1-D clear ↑ | Infer (ms) | Params |
|---|---|---|---|---|---|---|---|
| delta_pointnet2 | delta_pointnet2 | 0.0213 ± 0.0001 | 0.0599 ± 0.0005 | 14.8% | 1.63× | 2.06 | 174,721 |
| cond_pointnet2 | cond_pointnet2 | 0.0221 ± 0.0002 | 0.0567 ± 0.0008 | 13.9% | 1.72× | 2.07 | 174,721 |
| cond_gino | cond_gino | 0.0240 ± 0.0002 | 0.0440 ± 0.0005 | 10.7% | 2.22× | 15.33 | 2,808,181 |
| delta_const_gino | delta_const_gino | 0.0257 ± 0.0004 | 0.0463 ± 0.0011 | 11.2% | 2.11× | 15.32 | 2,808,181 |
| gino | gino | 0.0258 ± 0.0010 | 0.0476 ± 0.0020 | 11.5% | 2.05× | 15.40 | 2,807,892 |
| delta_transolver | delta_transolver | 0.0290 ± 0.0001 | 0.0457 ± 0.0008 | 11.5% | 2.14× | 10.59 | 978,113 |
| cond_transolver | cond_transolver | 0.0310 ± 0.0004 | 0.0506 ± 0.0048 | 12.7% | 1.94× | 10.55 | 978,113 |
| delta_gino | delta_gino | 0.0314 ± 0.0004 | 0.0485 ± 0.0026 | 12.4% | 2.02× | 14.84 | 2,808,181 |
| delta_const_transolver | delta_const_transolver | 0.0366 ± 0.0001 | 0.0508 ± 0.0022 | 12.7% | 1.93× | 10.61 | 978,113 |
| transolver | transolver | 0.0368 ± 0.0001 | 0.0517 ± 0.0023 | 12.9% | 1.89× | 10.54 | 977,857 |
| cond_deeponet | cond_deeponet | 0.0477 ± 0.0002 | 0.0522 ± 0.0005 | 12.7% | 1.87× | 0.82 | 463,617 |
| delta_const_deeponet | delta_const_deeponet | 0.0485 ± 0.0004 | 0.0569 ± 0.0009 | 13.8% | 1.72× | 0.90 | 463,617 |
| deeponet | deeponet | 0.0486 ± 0.0002 | 0.0558 ± 0.0024 | 13.5% | 1.75× | 0.83 | 463,361 |
| delta_deeponet | delta_deeponet | 0.0512 ± 0.0001 | 0.0519 ± 0.0002 | 12.8% | 1.88× | 0.85 | 463,617 |
| prior_only | prior_only | 0.0541 ± 0.0000 | 0.0976 ± 0.0000 | 24.5% | 1.00× | 0.00 | 0 |
| pointnet2 | pointnet2 | 0.0732 ± 0.0003 | 0.0582 ± 0.0009 | 14.0% | 1.68× | 2.11 | 174,529 |
| delta_const_pointnet2 | delta_const_pointnet2 | 0.0733 ± 0.0002 | 0.0578 ± 0.0004 | 13.8% | 1.69× | 2.18 | 174,721 |
| predict_mean | predict_mean | 0.4819 ± 0.0000 | 1.9273 ± 0.0000 | 536.7% | 0.05× | 0.01 | 0 |

Geometry-blind 1-D clear-wall baseline U-MAE: 0.0976 W/m²K — the number any geometry-aware operator must beat (H1).

## Bridge-focused — does the learned correction beat the prior?

`correction rel-L2` = ‖pred−true‖ / ‖prior−true‖ — **`prior_only` ≡ 1.000 by construction, so < 1.000 means the operator genuinely beats the analytic prior**; the *bridge* columns restrict this to points the bridge perturbs (|true−prior| > τ in θ). `clear rel-L2` guards that clear walls are not corrupted.
Bridge region (τ=0.02) covers ~28.9% of points.

| Model | correction rel-L2 ↓ | bridge corr-relL2 (τ=0.02) ↓ | bridge corr-relL2 (τ=0.05) ↓ | correction R² ↑ | clear rel-L2 ↓ |
|---|---|---|---|---|---|
| delta_pointnet2 | 0.3584 ± 0.0019 | 0.3362 ± 0.0021 | 0.3049 ± 0.0029 | 0.8695 ± 0.0014 | 0.0096 ± 0.0002 |
| cond_pointnet2 | 0.3712 ± 0.0041 | 0.3475 ± 0.0032 | 0.3170 ± 0.0020 | 0.8600 ± 0.0031 | 0.0103 ± 0.0003 |
| cond_gino | 0.3983 ± 0.0020 | 0.3448 ± 0.0028 | 0.3183 ± 0.0031 | 0.8388 ± 0.0016 | 0.0164 ± 0.0001 |
| delta_const_gino | 0.4330 ± 0.0056 | 0.3804 ± 0.0042 | 0.3510 ± 0.0039 | 0.8094 ± 0.0049 | 0.0171 ± 0.0003 |
| gino | 0.4361 ± 0.0169 | 0.3912 ± 0.0134 | 0.3648 ± 0.0117 | 0.8065 ± 0.0149 | 0.0158 ± 0.0011 |
| delta_transolver | 0.4741 ± 0.0012 | 0.4538 ± 0.0011 | 0.4568 ± 0.0014 | 0.7715 ± 0.0011 | 0.0124 ± 0.0002 |
| cond_transolver | 0.5082 ± 0.0084 | 0.4901 ± 0.0103 | 0.4968 ± 0.0079 | 0.7374 ± 0.0087 | 0.0128 ± 0.0000 |
| delta_gino | 0.5626 ± 0.0060 | 0.5455 ± 0.0069 | 0.5192 ± 0.0098 | 0.6783 ± 0.0068 | 0.0108 ± 0.0002 |
| delta_const_transolver | 0.6015 ± 0.0016 | 0.5689 ± 0.0007 | 0.5669 ± 0.0006 | 0.6323 ± 0.0020 | 0.0175 ± 0.0002 |
| transolver | 0.6063 ± 0.0013 | 0.5757 ± 0.0024 | 0.5749 ± 0.0027 | 0.6265 ± 0.0016 | 0.0169 ± 0.0001 |
| cond_deeponet | 0.7853 ± 0.0023 | 0.7198 ± 0.0038 | 0.6810 ± 0.0035 | 0.3733 ± 0.0037 | 0.0282 ± 0.0005 |
| delta_const_deeponet | 0.7988 ± 0.0073 | 0.7338 ± 0.0087 | 0.6908 ± 0.0070 | 0.3515 ± 0.0118 | 0.0285 ± 0.0002 |
| deeponet | 0.8011 ± 0.0042 | 0.7389 ± 0.0025 | 0.6952 ± 0.0017 | 0.3479 ± 0.0068 | 0.0279 ± 0.0005 |
| delta_deeponet | 0.9483 ± 0.0017 | 0.9409 ± 0.0023 | 0.9766 ± 0.0018 | 0.0861 ± 0.0033 | 0.0162 ± 0.0003 |
| prior_only | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | -0.0162 ± 0.0000 | 0.0056 ± 0.0000 |
| pointnet2 | 1.1539 ± 0.0056 | 0.5493 ± 0.0110 | 0.4221 ± 0.0046 | -0.3530 ± 0.0132 | 0.0896 ± 0.0002 |
| delta_const_pointnet2 | 1.1555 ± 0.0029 | 0.5478 ± 0.0082 | 0.4205 ± 0.0026 | -0.3568 ± 0.0068 | 0.0897 ± 0.0003 |
| predict_mean | 7.6543 ± 0.0000 | 3.5834 ± 0.0000 | 1.6653 ± 0.0000 | -58.5371 ± 0.0000 | 0.5873 ± 0.0000 |
