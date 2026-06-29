# Block-2 Benchmark — 3-D wall blocks · rotated / off-lattice irregular corpus (GINO vs grid FNO)

- corpus: `irreg` · train: `data/processed/block2_irreg_train` · val: `data/processed/block2_irreg_val`
- device: `cuda` · epochs: 300 · batch: 1 · seeds: [1337, 1, 2] · latent/voxel grid: 16
- U-value derived from the predicted field at the indoor face (near-face band 0.08 in normalised coords); the same estimator is applied to every model and to the ground truth.
- field rel-L2 and U-MAE are scored on the **sampled points** for all models (the voxel FNO is resampled back to the cloud), a fair head-to-head.

| Model | Kind | Field rel-L2 ↓ | U-value MAE ↓ (W/m²K) | U-MAPE ↓ | vs 1-D clear ↑ | Infer (ms) | Params |
|---|---|---|---|---|---|---|---|
| delta_meshgraphnet | delta_meshgraphnet | 0.0304 ± 0.0006 | 0.3036 ± 0.0013 | 47.0% | 1.06× | 7.41 | 1,274,753 |
| delta_pointnet2 | delta_pointnet2 | 0.0364 ± 0.0002 | 0.3059 ± 0.0024 | 47.4% | 1.05× | 1.83 | 174,721 |
| delta_gnot | delta_gnot | 0.0418 ± 0.0011 | 0.2898 ± 0.0027 | 45.0% | 1.11× | 5.58 | 3,593,569 |
| delta_transolver | delta_transolver | 0.0444 ± 0.0009 | 0.2993 ± 0.0024 | 47.0% | 1.07× | 6.16 | 978,113 |
| fno_voxel | fno_voxel | 0.0603 ± 0.0014 | 0.2918 ± 0.0027 | 44.6% | 1.10× | 3.11 | 2,410,689 |
| delta_gino | delta_gino | 0.0636 ± 0.0015 | 0.2974 ± 0.0042 | 46.2% | 1.08× | 10.53 | 2,808,181 |
| meshgraphnet | meshgraphnet | 0.0639 ± 0.0017 | 0.3234 ± 0.0035 | 50.6% | 0.99× | 7.40 | 1,274,625 |
| delta_deeponet | delta_deeponet | 0.0897 ± 0.0005 | 0.3038 ± 0.0032 | 48.2% | 1.06× | 0.64 | 463,617 |
| prior_only | prior_only | 0.0958 ± 0.0000 | 0.3211 ± 0.0000 | 50.2% | 1.00× | 0.00 | 0 |
| gnot | gnot | 0.0993 ± 0.0135 | 0.2910 ± 0.0172 | 45.5% | 1.11× | 5.59 | 3,593,313 |
| pointnet2 | pointnet2 | 0.1026 ± 0.0039 | 0.3069 ± 0.0071 | 47.7% | 1.05× | 1.79 | 174,529 |
| transolver | transolver | 0.1068 ± 0.0171 | 0.2850 ± 0.0066 | 43.8% | 1.13× | 6.13 | 977,857 |
| deeponet | deeponet | 0.1509 ± 0.0041 | 0.3968 ± 0.0208 | 66.0% | 0.81× | 0.59 | 463,361 |
| gino | gino | 0.1668 ± 0.0046 | 0.4798 ± 0.0429 | 82.1% | 0.67× | 10.66 | 2,807,892 |

Geometry-blind 1-D clear-wall baseline U-MAE: 0.3211 W/m²K — the number any geometry-aware operator must beat (H1).

## Bridge-focused — does the learned correction beat the prior?

`correction rel-L2` = ‖pred−true‖ / ‖prior−true‖ — **`prior_only` ≡ 1.000 by construction, so < 1.000 means the operator genuinely beats the analytic prior**; the *bridge* columns restrict this to points the bridge perturbs (|true−prior| > τ in θ). `clear rel-L2` guards that clear walls are not corrupted.
Bridge region (τ=0.02) covers ~46.6% of points.

| Model | correction rel-L2 ↓ | bridge corr-relL2 (τ=0.02) ↓ | bridge corr-relL2 (τ=0.05) ↓ | correction R² ↑ | clear rel-L2 ↓ |
|---|---|---|---|---|---|
| delta_meshgraphnet | 0.3266 ± 0.0091 | 0.3186 ± 0.0086 | 0.3092 ± 0.0080 | 0.8905 ± 0.0061 | 0.0104 ± 0.0007 |
| delta_gnot | 0.3395 ± 0.0149 | 0.3076 ± 0.0220 | 0.2856 ± 0.0252 | 0.8816 ± 0.0102 | 0.0262 ± 0.0022 |
| delta_pointnet2 | 0.3668 ± 0.0048 | 0.3531 ± 0.0049 | 0.3395 ± 0.0049 | 0.8620 ± 0.0036 | 0.0153 ± 0.0004 |
| delta_transolver | 0.3677 ± 0.0122 | 0.3475 ± 0.0137 | 0.3325 ± 0.0129 | 0.8612 ± 0.0092 | 0.0227 ± 0.0011 |
| fno_voxel | 0.5235 ± 0.0188 | 0.4557 ± 0.0206 | 0.4028 ± 0.0197 | 0.7187 ± 0.0203 | 0.0435 ± 0.0003 |
| meshgraphnet | 0.5947 ± 0.0260 | 0.5379 ± 0.0285 | 0.5016 ± 0.0298 | 0.6366 ± 0.0314 | 0.0432 ± 0.0006 |
| delta_gino | 0.6013 ± 0.0182 | 0.5762 ± 0.0182 | 0.5523 ± 0.0177 | 0.6290 ± 0.0227 | 0.0257 ± 0.0009 |
| gnot | 0.7866 ± 0.1115 | 0.5914 ± 0.0610 | 0.4914 ± 0.0445 | 0.3528 ± 0.1759 | 0.0963 ± 0.0228 |
| pointnet2 | 0.8185 ± 0.0314 | 0.6108 ± 0.0215 | 0.5166 ± 0.0167 | 0.3121 ± 0.0531 | 0.1018 ± 0.0038 |
| transolver | 0.8673 ± 0.1639 | 0.6674 ± 0.0938 | 0.5595 ± 0.0641 | 0.2011 ± 0.3057 | 0.1010 ± 0.0274 |
| delta_deeponet | 0.9317 ± 0.0068 | 0.9197 ± 0.0064 | 0.9207 ± 0.0062 | 0.1100 ± 0.0130 | 0.0278 ± 0.0011 |
| prior_only | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | -0.0253 ± 0.0000 | 0.0064 ± 0.0000 |
| deeponet | 1.2467 ± 0.0380 | 0.9258 ± 0.0239 | 0.7128 ± 0.0082 | -0.5951 ± 0.0960 | 0.1488 ± 0.0062 |
| gino | 1.4642 ± 0.0554 | 0.8669 ± 0.0012 | 0.7085 ± 0.0074 | -1.2012 ± 0.1644 | 0.2375 ± 0.0148 |
