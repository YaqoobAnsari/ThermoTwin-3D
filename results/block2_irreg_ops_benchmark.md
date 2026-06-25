# Block-2 Benchmark — 3-D wall blocks · rotated / off-lattice irregular corpus (GINO vs grid FNO)

- corpus: `irreg` · train: `data/processed/block2_irreg_train` · val: `data/processed/block2_irreg_val`
- device: `cuda` · epochs: 300 · batch: 1 · seeds: [1337, 1, 2] · latent/voxel grid: 16
- U-value derived from the predicted field at the indoor face (near-face band 0.08 in normalised coords); the same estimator is applied to every model and to the ground truth.
- field rel-L2 and U-MAE are scored on the **sampled points** for all models (the voxel FNO is resampled back to the cloud), a fair head-to-head.

| Model | Kind | Field rel-L2 ↓ | U-value MAE ↓ (W/m²K) | U-MAPE ↓ | vs 1-D clear ↑ | Infer (ms) | Params |
|---|---|---|---|---|---|---|---|
| delta_transolver | delta_transolver | 0.0444 ± 0.0009 | 0.2993 ± 0.0024 | 47.0% | 1.07× | 6.15 | 978,113 |
| fno_voxel | fno_voxel | 0.0603 ± 0.0014 | 0.2918 ± 0.0027 | 44.6% | 1.10× | 3.09 | 2,410,689 |
| delta_gino | delta_gino | 0.0636 ± 0.0015 | 0.2974 ± 0.0042 | 46.2% | 1.08× | 10.62 | 2,808,181 |
| prior_only | prior_only | 0.0958 ± 0.0000 | 0.3211 ± 0.0000 | 50.2% | 1.00× | 0.00 | 0 |
| transolver | transolver | 0.1068 ± 0.0171 | 0.2850 ± 0.0066 | 43.8% | 1.13× | 6.14 | 977,857 |
| gino | gino | 0.1668 ± 0.0046 | 0.4798 ± 0.0429 | 82.1% | 0.67× | 10.76 | 2,807,892 |

Geometry-blind 1-D clear-wall baseline U-MAE: 0.3211 W/m²K — the number any geometry-aware operator must beat (H1).

## Bridge-focused — does the learned correction beat the prior?

`correction rel-L2` = ‖pred−true‖ / ‖prior−true‖ — **`prior_only` ≡ 1.000 by construction, so < 1.000 means the operator genuinely beats the analytic prior**; the *bridge* columns restrict this to points the bridge perturbs (|true−prior| > τ in θ). `clear rel-L2` guards that clear walls are not corrupted.
Bridge region (τ=0.02) covers ~46.6% of points.

| Model | correction rel-L2 ↓ | bridge corr-relL2 (τ=0.02) ↓ | bridge corr-relL2 (τ=0.05) ↓ | correction R² ↑ | clear rel-L2 ↓ |
|---|---|---|---|---|---|
| delta_transolver | 0.3677 ± 0.0122 | 0.3475 ± 0.0137 | 0.3325 ± 0.0129 | 0.8612 ± 0.0092 | 0.0227 ± 0.0011 |
| fno_voxel | 0.5235 ± 0.0188 | 0.4557 ± 0.0206 | 0.4028 ± 0.0197 | 0.7187 ± 0.0203 | 0.0435 ± 0.0003 |
| delta_gino | 0.6013 ± 0.0182 | 0.5762 ± 0.0182 | 0.5523 ± 0.0177 | 0.6290 ± 0.0227 | 0.0257 ± 0.0009 |
| transolver | 0.8673 ± 0.1639 | 0.6674 ± 0.0938 | 0.5595 ± 0.0641 | 0.2011 ± 0.3057 | 0.1010 ± 0.0274 |
| prior_only | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | -0.0253 ± 0.0000 | 0.0064 ± 0.0000 |
| gino | 1.4642 ± 0.0554 | 0.8669 ± 0.0012 | 0.7085 ± 0.0074 | -1.2012 ± 0.1644 | 0.2375 ± 0.0148 |
