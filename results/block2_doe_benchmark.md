# Block-2 Benchmark — 3-D wall blocks · doe (GINO vs grid FNO)

- corpus: `doe` · train: `data/processed/block2_doe_train` · val: `data/processed/block2_doe_val`
- device: `cuda` · epochs: 300 · batch: 1 · seeds: [1337, 1, 2] · latent/voxel grid: 16
- U-value derived from the predicted field at the indoor face (near-face band 0.08 in normalised coords); the same estimator is applied to every model and to the ground truth.
- field rel-L2 and U-MAE are scored on the **sampled points** for all models (the voxel FNO is resampled back to the cloud), a fair head-to-head.

| Model | Kind | Field rel-L2 ↓ | U-value MAE ↓ (W/m²K) | U-MAPE ↓ | vs 1-D clear ↑ | Infer (ms) | Params |
|---|---|---|---|---|---|---|---|
| prior_only | prior_only | 0.0075 ± 0.0000 | 0.0000 ± 0.0000 | 0.0% | 0.00× | 0.01 | 0 |
| delta_gnot | delta_gnot | 0.0075 ± 0.0001 | 0.0024 ± 0.0004 | 0.1% | 0.00× | 9.55 | 3,593,569 |
| delta_transolver | delta_transolver | 0.0077 ± 0.0003 | 0.0046 ± 0.0022 | 0.3% | 0.00× | 10.44 | 978,113 |
| delta_meshgraphnet | delta_meshgraphnet | 0.0078 ± 0.0003 | 0.0058 ± 0.0003 | 0.3% | 0.00× | 14.15 | 1,274,753 |
| delta_gino | delta_gino | 0.0082 ± 0.0003 | 0.0035 ± 0.0011 | 0.2% | 0.00× | 13.01 | 2,808,181 |
| delta_pointnet2 | delta_pointnet2 | 0.0089 ± 0.0009 | 0.0035 ± 0.0005 | 0.2% | 0.00× | 2.10 | 174,721 |
| delta_deeponet | delta_deeponet | 0.0089 ± 0.0001 | 0.0034 ± 0.0006 | 0.2% | 0.00× | 0.82 | 463,617 |
| meshgraphnet | meshgraphnet | 0.3205 ± 0.0012 | 0.0886 ± 0.0098 | 5.4% | 0.00× | 14.13 | 1,274,625 |
| pointnet2 | pointnet2 | 0.4222 ± 0.0014 | 0.0647 ± 0.0013 | 4.0% | 0.00× | 2.02 | 174,529 |
| transolver | transolver | 0.4742 ± 0.0674 | 0.1349 ± 0.0764 | 8.3% | 0.00× | 10.38 | 977,857 |
| fno_voxel | fno_voxel | 0.5083 ± 0.0000 | 0.1177 ± 0.0009 | 7.1% | 0.00× | 4.75 | 2,410,689 |
| gnot | gnot | 0.5191 ± 0.0470 | 0.1393 ± 0.0352 | 8.6% | 0.00× | 9.54 | 3,593,313 |
| deeponet | deeponet | 0.5218 ± 0.0080 | 0.1391 ± 0.0053 | 8.7% | 0.00× | 0.75 | 463,361 |
| gino | gino | 0.7119 ± 0.0092 | 0.2541 ± 0.0388 | 15.8% | 0.00× | 13.15 | 2,807,892 |

Geometry-blind 1-D clear-wall baseline U-MAE: 0.0000 W/m²K — the number any geometry-aware operator must beat (H1).

## Bridge-focused — does the learned correction beat the prior?

`correction rel-L2` = ‖pred−true‖ / ‖prior−true‖ — **`prior_only` ≡ 1.000 by construction, so < 1.000 means the operator genuinely beats the analytic prior**; the *bridge* columns restrict this to points the bridge perturbs (|true−prior| > τ in θ). `clear rel-L2` guards that clear walls are not corrupted.
Bridge region (τ=0.02) covers ~1.1% of points.

| Model | correction rel-L2 ↓ | bridge corr-relL2 (τ=0.02) ↓ | bridge corr-relL2 (τ=0.05) ↓ | correction R² ↑ | clear rel-L2 ↓ |
|---|---|---|---|---|---|
| delta_gnot | 0.9797 ± 0.0005 | 0.9576 ± 0.0143 | 0.9583 ± 0.0140 | 0.0352 ± 0.0010 | 0.0027 ± 0.0005 |
| prior_only | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | -0.0052 ± 0.0000 | 0.0018 ± 0.0000 |
| delta_transolver | 1.0026 ± 0.0137 | 0.8814 ± 0.0727 | 0.8907 ± 0.0652 | -0.0105 ± 0.0276 | 0.0047 ± 0.0019 |
| delta_meshgraphnet | 1.0125 ± 0.0252 | 0.6795 ± 0.0108 | 0.7131 ± 0.0149 | -0.0311 ± 0.0517 | 0.0072 ± 0.0002 |
| delta_deeponet | 1.0458 ± 0.0034 | 0.9872 ± 0.0006 | 0.9831 ± 0.0007 | -0.0994 ± 0.0071 | 0.0038 ± 0.0001 |
| delta_gino | 1.0920 ± 0.0484 | 0.9862 ± 0.0044 | 0.9765 ± 0.0098 | -0.2011 ± 0.1048 | 0.0048 ± 0.0012 |
| delta_pointnet2 | 1.1006 ± 0.0707 | 0.8251 ± 0.0052 | 0.8557 ± 0.0037 | -0.2226 ± 0.1599 | 0.0073 ± 0.0010 |
| meshgraphnet | 32.7170 ± 0.1474 | 4.9374 ± 0.1737 | 2.9544 ± 0.2318 | -1075.0042 ± 9.7067 | 0.3201 ± 0.0017 |
| pointnet2 | 42.9923 ± 0.1424 | 4.6720 ± 0.1936 | 2.7503 ± 0.0660 | -1856.9961 ± 12.3142 | 0.4244 ± 0.0016 |
| transolver | 48.6948 ± 7.2628 | 4.4757 ± 0.0971 | 1.9500 ± 0.2283 | -2435.5726 ± 746.5557 | 0.4819 ± 0.0729 |
| fno_voxel | 51.9215 ± 0.0015 | 6.3334 ± 0.0057 | 4.2405 ± 0.0034 | -2708.9017 ± 0.1614 | 0.5109 ± 0.0000 |
| gnot | 53.2265 ± 4.8756 | 4.2942 ± 0.2919 | 1.7687 ± 0.0441 | -2870.7285 ± 504.8321 | 0.5276 ± 0.0486 |
| deeponet | 53.4182 ± 0.7890 | 5.4503 ± 0.1245 | 3.4855 ± 0.0855 | -2868.0132 ± 85.1748 | 0.5277 ± 0.0079 |
| gino | 73.2580 ± 0.7218 | 8.6454 ± 0.4263 | 5.6441 ± 0.1755 | -5394.2482 ± 105.9767 | 0.7221 ± 0.0072 |
