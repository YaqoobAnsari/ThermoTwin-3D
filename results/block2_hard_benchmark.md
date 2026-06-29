# Block-2 Benchmark — 3-D wall blocks · hard (GINO vs grid FNO)

- corpus: `hard` · train: `data/processed/block2_hard_train` · val: `data/processed/block2_hard_val`
- device: `cuda` · epochs: 300 · batch: 1 · seeds: [1337, 1, 2] · latent/voxel grid: 16
- U-value derived from the predicted field at the indoor face (near-face band 0.08 in normalised coords); the same estimator is applied to every model and to the ground truth.
- field rel-L2 and U-MAE are scored on the **sampled points** for all models (the voxel FNO is resampled back to the cloud), a fair head-to-head.

| Model | Kind | Field rel-L2 ↓ | U-value MAE ↓ (W/m²K) | U-MAPE ↓ | vs 1-D clear ↑ | Infer (ms) | Params |
|---|---|---|---|---|---|---|---|
| delta_meshgraphnet | delta_meshgraphnet | 0.0178 ± 0.0003 | 0.0596 ± 0.0007 | 14.6% | 1.64× | 13.71 | 1,274,753 |
| delta_pointnet2 | delta_pointnet2 | 0.0213 ± 0.0001 | 0.0601 ± 0.0008 | 14.8% | 1.63× | 2.03 | 174,721 |
| fno_voxel | fno_voxel | 0.0233 ± 0.0002 | 0.0390 ± 0.0005 | 9.4% | 2.50× | 3.31 | 2,410,689 |
| gino | gino | 0.0258 ± 0.0010 | 0.0476 ± 0.0020 | 11.5% | 2.05× | 14.51 | 2,807,892 |
| meshgraphnet | meshgraphnet | 0.0267 ± 0.0002 | 0.0625 ± 0.0024 | 15.4% | 1.56× | 13.58 | 1,274,625 |
| delta_gnot | delta_gnot | 0.0289 ± 0.0006 | 0.0440 ± 0.0014 | 11.1% | 2.22× | 9.58 | 3,593,569 |
| delta_transolver | delta_transolver | 0.0290 ± 0.0001 | 0.0457 ± 0.0008 | 11.5% | 2.14× | 10.41 | 978,113 |
| delta_gino | delta_gino | 0.0314 ± 0.0004 | 0.0485 ± 0.0026 | 12.4% | 2.02× | 14.30 | 2,808,181 |
| gnot | gnot | 0.0367 ± 0.0002 | 0.0529 ± 0.0014 | 13.3% | 1.85× | 9.52 | 3,593,313 |
| transolver | transolver | 0.0368 ± 0.0001 | 0.0517 ± 0.0023 | 12.9% | 1.89× | 10.36 | 977,857 |
| deeponet | deeponet | 0.0486 ± 0.0002 | 0.0558 ± 0.0024 | 13.5% | 1.75× | 0.77 | 463,361 |
| delta_deeponet | delta_deeponet | 0.0512 ± 0.0001 | 0.0519 ± 0.0002 | 12.8% | 1.88× | 0.82 | 463,617 |
| prior_only | prior_only | 0.0541 ± 0.0000 | 0.0976 ± 0.0000 | 24.5% | 1.00× | 0.01 | 0 |
| pointnet2 | pointnet2 | 0.0733 ± 0.0003 | 0.0580 ± 0.0007 | 13.9% | 1.68× | 2.00 | 174,529 |

Geometry-blind 1-D clear-wall baseline U-MAE: 0.0976 W/m²K — the number any geometry-aware operator must beat (H1).

## Bridge-focused — does the learned correction beat the prior?

`correction rel-L2` = ‖pred−true‖ / ‖prior−true‖ — **`prior_only` ≡ 1.000 by construction, so < 1.000 means the operator genuinely beats the analytic prior**; the *bridge* columns restrict this to points the bridge perturbs (|true−prior| > τ in θ). `clear rel-L2` guards that clear walls are not corrupted.
Bridge region (τ=0.02) covers ~28.9% of points.

| Model | correction rel-L2 ↓ | bridge corr-relL2 (τ=0.02) ↓ | bridge corr-relL2 (τ=0.05) ↓ | correction R² ↑ | clear rel-L2 ↓ |
|---|---|---|---|---|---|
| delta_meshgraphnet | 0.3119 ± 0.0060 | 0.2944 ± 0.0053 | 0.2582 ± 0.0031 | 0.9011 ± 0.0038 | 0.0083 ± 0.0004 |
| delta_pointnet2 | 0.3587 ± 0.0015 | 0.3364 ± 0.0016 | 0.3048 ± 0.0022 | 0.8692 ± 0.0011 | 0.0097 ± 0.0002 |
| fno_voxel | 0.3889 ± 0.0041 | 0.3350 ± 0.0037 | 0.3041 ± 0.0038 | 0.8463 ± 0.0033 | 0.0164 ± 0.0001 |
| gino | 0.4361 ± 0.0169 | 0.3912 ± 0.0134 | 0.3648 ± 0.0117 | 0.8065 ± 0.0149 | 0.0158 ± 0.0011 |
| meshgraphnet | 0.4476 ± 0.0048 | 0.4123 ± 0.0040 | 0.3748 ± 0.0032 | 0.7964 ± 0.0044 | 0.0147 ± 0.0013 |
| delta_gnot | 0.4726 ± 0.0100 | 0.4457 ± 0.0098 | 0.4494 ± 0.0093 | 0.7730 ± 0.0096 | 0.0144 ± 0.0007 |
| delta_transolver | 0.4741 ± 0.0012 | 0.4538 ± 0.0011 | 0.4568 ± 0.0014 | 0.7715 ± 0.0011 | 0.0124 ± 0.0002 |
| delta_gino | 0.5626 ± 0.0060 | 0.5455 ± 0.0069 | 0.5192 ± 0.0098 | 0.6783 ± 0.0068 | 0.0108 ± 0.0002 |
| gnot | 0.6020 ± 0.0022 | 0.5661 ± 0.0022 | 0.5663 ± 0.0027 | 0.6317 ± 0.0027 | 0.0184 ± 0.0008 |
| transolver | 0.6063 ± 0.0013 | 0.5757 ± 0.0024 | 0.5749 ± 0.0027 | 0.6265 ± 0.0016 | 0.0169 ± 0.0001 |
| deeponet | 0.8011 ± 0.0042 | 0.7389 ± 0.0025 | 0.6952 ± 0.0017 | 0.3479 ± 0.0068 | 0.0279 ± 0.0005 |
| delta_deeponet | 0.9483 ± 0.0017 | 0.9409 ± 0.0023 | 0.9766 ± 0.0018 | 0.0861 ± 0.0033 | 0.0162 ± 0.0003 |
| prior_only | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | -0.0162 ± 0.0000 | 0.0056 ± 0.0000 |
| pointnet2 | 1.1548 ± 0.0051 | 0.5497 ± 0.0100 | 0.4222 ± 0.0028 | -0.3552 ± 0.0119 | 0.0896 ± 0.0002 |
