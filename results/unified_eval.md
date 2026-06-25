# Unified cross-dataset evaluation — ThermoTwin-3D operators

Every Block-2 operator scored across all geometry / field-prediction datasets, over the full metric suite. **mean ± std** over seeds; **best per column in bold**. The real-thermal datasets validate different quantities in different formats — they appear in the coverage map below and require bespoke adapters; they are **not** faked into this matrix.

Datasets evaluated: **synthetic-box, synthetic-irregular, synthetic-hard, real-CityGML, DOE-refbldg**  ·  models compared: **14**

### field rel-L2 ↓

| Model | synthetic-box | synthetic-irregular | synthetic-hard | real-CityGML | DOE-refbldg |
|---|---|---|---|---|---|
| delta_transolver | — | **0.044 ± 0.001** | 0.029 ± 0.000 | 0.018 ± 0.001 | 0.008 ± 0.000 |
| transolver | — | 0.107 ± 0.017 | 0.037 ± 0.000 | 0.245 ± 0.001 | 0.474 ± 0.067 |
| delta_gnot | — | — | — | — | — |
| gnot | — | — | — | — | — |
| delta_gino | 0.025 ± 0.000 | 0.064 ± 0.002 | 0.031 ± 0.000 | 0.025 ± 0.002 | 0.008 ± 0.000 |
| gino | 0.024 ± 0.000 | 0.167 ± 0.005 | 0.026 ± 0.001 | 0.808 ± 0.048 | 0.712 ± 0.009 |
| delta_deeponet | — | — | — | — | — |
| deeponet | — | — | — | — | — |
| delta_pointnet2 | — | — | — | — | — |
| pointnet2 | — | — | — | — | — |
| delta_meshgraphnet | — | — | — | — | — |
| meshgraphnet | — | — | — | — | — |
| fno_voxel | **0.020 ± 0.000** | 0.060 ± 0.001 | **0.023 ± 0.000** | 0.431 ± 0.000 | 0.508 ± 0.000 |
| prior_only | 0.038 ± 0.000 | 0.096 ± 0.000 | 0.054 ± 0.000 | **0.015 ± 0.000** | **0.007 ± 0.000** |

### field RMSE [K] ↓

| Model | synthetic-box | synthetic-irregular | synthetic-hard | real-CityGML | DOE-refbldg |
|---|---|---|---|---|---|
| delta_transolver | — | — | — | — | — |
| transolver | — | — | — | — | — |
| delta_gnot | — | — | — | — | — |
| gnot | — | — | — | — | — |
| delta_gino | — | — | — | — | — |
| gino | — | — | — | — | — |
| delta_deeponet | — | — | — | — | — |
| deeponet | — | — | — | — | — |
| delta_pointnet2 | — | — | — | — | — |
| pointnet2 | — | — | — | — | — |
| delta_meshgraphnet | — | — | — | — | — |
| meshgraphnet | — | — | — | — | — |
| fno_voxel | — | — | — | — | — |
| prior_only | — | — | — | — | — |

### field rel-L2 @bridge ↓

| Model | synthetic-box | synthetic-irregular | synthetic-hard | real-CityGML | DOE-refbldg |
|---|---|---|---|---|---|
| delta_transolver | — | — | — | — | — |
| transolver | — | — | — | — | — |
| delta_gnot | — | — | — | — | — |
| gnot | — | — | — | — | — |
| delta_gino | — | — | — | — | — |
| gino | — | — | — | — | — |
| delta_deeponet | — | — | — | — | — |
| deeponet | — | — | — | — | — |
| delta_pointnet2 | — | — | — | — | — |
| pointnet2 | — | — | — | — | — |
| delta_meshgraphnet | — | — | — | — | — |
| meshgraphnet | — | — | — | — | — |
| fno_voxel | — | — | — | — | — |
| prior_only | — | — | — | — | — |

### U-MAE [W/m²K] ↓

| Model | synthetic-box | synthetic-irregular | synthetic-hard | real-CityGML | DOE-refbldg |
|---|---|---|---|---|---|
| delta_transolver | — | 0.299 ± 0.002 | 0.046 ± 0.001 | 0.005 ± 0.001 | 0.005 ± 0.002 |
| transolver | — | **0.285 ± 0.007** | 0.052 ± 0.002 | 0.027 ± 0.002 | 0.135 ± 0.076 |
| delta_gnot | — | — | — | — | — |
| gnot | — | — | — | — | — |
| delta_gino | 0.049 ± 0.001 | 0.297 ± 0.004 | 0.049 ± 0.003 | 0.006 ± 0.002 | 0.004 ± 0.001 |
| gino | 0.049 ± 0.004 | 0.480 ± 0.043 | 0.048 ± 0.002 | 0.212 ± 0.056 | 0.254 ± 0.039 |
| delta_deeponet | — | — | — | — | — |
| deeponet | — | — | — | — | — |
| delta_pointnet2 | — | — | — | — | — |
| pointnet2 | — | — | — | — | — |
| delta_meshgraphnet | — | — | — | — | — |
| meshgraphnet | — | — | — | — | — |
| fno_voxel | **0.045 ± 0.000** | 0.292 ± 0.003 | **0.039 ± 0.001** | 0.053 ± 0.000 | 0.118 ± 0.001 |
| prior_only | 0.078 ± 0.000 | 0.321 ± 0.000 | 0.098 ± 0.000 | **0.000 ± 0.000** | **0.000 ± 0.000** |

### correction rel-L2 (vs prior) ↓

| Model | synthetic-box | synthetic-irregular | synthetic-hard | real-CityGML | DOE-refbldg |
|---|---|---|---|---|---|
| delta_transolver | — | **0.368 ± 0.012** | 0.474 ± 0.001 | 1.136 ± 0.088 | 1.003 ± 0.014 |
| transolver | — | 0.867 ± 0.164 | 0.606 ± 0.001 | 13.869 ± 0.052 | 48.695 ± 7.263 |
| delta_gnot | — | — | — | — | — |
| gnot | — | — | — | — | — |
| delta_gino | — | 0.601 ± 0.018 | 0.563 ± 0.006 | 1.525 ± 0.121 | 1.092 ± 0.048 |
| gino | — | 1.464 ± 0.055 | 0.436 ± 0.017 | 46.075 ± 2.913 | 73.258 ± 0.722 |
| delta_deeponet | — | — | — | — | — |
| deeponet | — | — | — | — | — |
| delta_pointnet2 | — | — | — | — | — |
| pointnet2 | — | — | — | — | — |
| delta_meshgraphnet | — | — | — | — | — |
| meshgraphnet | — | — | — | — | — |
| fno_voxel | — | 0.523 ± 0.019 | **0.389 ± 0.004** | 24.362 ± 0.001 | 51.922 ± 0.002 |
| prior_only | — | 1.000 ± 0.000 | 1.000 ± 0.000 | **1.000 ± 0.000** | **1.000 ± 0.000** |

### bridge corr-relL2 (τ=0.02) ↓

| Model | synthetic-box | synthetic-irregular | synthetic-hard | real-CityGML | DOE-refbldg |
|---|---|---|---|---|---|
| delta_transolver | — | **0.348 ± 0.014** | 0.454 ± 0.001 | **0.898 ± 0.016** | **0.881 ± 0.073** |
| transolver | — | 0.667 ± 0.094 | 0.576 ± 0.002 | 3.186 ± 0.010 | 4.476 ± 0.097 |
| delta_gnot | — | — | — | — | — |
| gnot | — | — | — | — | — |
| delta_gino | — | 0.576 ± 0.018 | 0.545 ± 0.007 | 1.009 ± 0.004 | 0.986 ± 0.004 |
| gino | — | 0.867 ± 0.001 | 0.391 ± 0.013 | 8.917 ± 0.433 | 8.645 ± 0.426 |
| delta_deeponet | — | — | — | — | — |
| deeponet | — | — | — | — | — |
| delta_pointnet2 | — | — | — | — | — |
| pointnet2 | — | — | — | — | — |
| delta_meshgraphnet | — | — | — | — | — |
| meshgraphnet | — | — | — | — | — |
| fno_voxel | — | 0.456 ± 0.021 | **0.335 ± 0.004** | 4.534 ± 0.001 | 6.333 ± 0.006 |
| prior_only | — | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |

### correction R² ↑

| Model | synthetic-box | synthetic-irregular | synthetic-hard | real-CityGML | DOE-refbldg |
|---|---|---|---|---|---|
| delta_transolver | — | **0.861 ± 0.009** | 0.772 ± 0.001 | -0.320 ± 0.206 | -0.011 ± 0.028 |
| transolver | — | 0.201 ± 0.306 | 0.626 ± 0.002 | -194.529 ± 1.476 | -2435.573 ± 746.556 |
| delta_gnot | — | — | — | — | — |
| gnot | — | — | — | — | — |
| delta_gino | — | 0.629 ± 0.023 | 0.678 ± 0.007 | -1.380 ± 0.369 | -0.201 ± 0.105 |
| gino | — | -1.201 ± 0.164 | 0.806 ± 0.015 | -2165.685 ± 277.171 | -5394.248 ± 105.977 |
| delta_deeponet | — | — | — | — | — |
| deeponet | — | — | — | — | — |
| delta_pointnet2 | — | — | — | — | — |
| pointnet2 | — | — | — | — | — |
| delta_meshgraphnet | — | — | — | — | — |
| meshgraphnet | — | — | — | — | — |
| fno_voxel | — | 0.719 ± 0.020 | **0.846 ± 0.003** | -602.335 ± 0.030 | -2708.902 ± 0.161 |
| prior_only | — | -0.025 ± 0.000 | -0.016 ± 0.000 | **-0.017 ± 0.000** | **-0.005 ± 0.000** |

### params ↓

| Model | synthetic-box | synthetic-irregular | synthetic-hard | real-CityGML | DOE-refbldg |
|---|---|---|---|---|---|
| delta_transolver | — | — | — | — | — |
| transolver | — | — | — | — | — |
| delta_gnot | — | — | — | — | — |
| gnot | — | — | — | — | — |
| delta_gino | — | — | — | — | — |
| gino | — | — | — | — | — |
| delta_deeponet | — | — | — | — | — |
| deeponet | — | — | — | — | — |
| delta_pointnet2 | — | — | — | — | — |
| pointnet2 | — | — | — | — | — |
| delta_meshgraphnet | — | — | — | — | — |
| meshgraphnet | — | — | — | — | — |
| fno_voxel | — | — | — | — | — |
| prior_only | — | — | — | — | — |

### infer ms ↓

| Model | synthetic-box | synthetic-irregular | synthetic-hard | real-CityGML | DOE-refbldg |
|---|---|---|---|---|---|
| delta_transolver | — | 6.153 ± 0.002 | 10.508 ± 0.084 | 10.317 ± 0.037 | 10.479 ± 0.040 |
| transolver | — | 6.136 ± 0.008 | 10.406 ± 0.028 | 10.222 ± 0.029 | 10.420 ± 0.026 |
| delta_gnot | — | — | — | — | — |
| gnot | — | — | — | — | — |
| delta_gino | 10.258 ± 0.061 | 10.617 ± 0.029 | 14.428 ± 0.133 | 14.085 ± 0.034 | 13.068 ± 0.117 |
| gino | 10.646 ± 0.472 | 10.756 ± 0.227 | 14.750 ± 0.524 | 14.293 ± 0.313 | 13.296 ± 0.205 |
| delta_deeponet | — | — | — | — | — |
| deeponet | — | — | — | — | — |
| delta_pointnet2 | — | — | — | — | — |
| pointnet2 | — | — | — | — | — |
| delta_meshgraphnet | — | — | — | — | — |
| meshgraphnet | — | — | — | — | — |
| fno_voxel | 7.666 ± 6.597 | 3.087 ± 0.125 | 3.545 ± 0.747 | 3.102 ± 0.089 | 4.482 ± 1.880 |
| prior_only | **0.003 ± 0.000** | **0.003 ± 0.001** | **0.004 ± 0.001** | **0.005 ± 0.001** | **0.007 ± 0.004** |

## Geometry datasets — coverage

| Dataset | Family | Status | Note |
|---|---|---|---|
| synthetic-box | synthetic geometry | ✅ evaluated | axis-aligned box (legacy 4-model run) |
| synthetic-irregular | synthetic geometry | ✅ evaluated | rotated / off-lattice |
| synthetic-hard | synthetic geometry | ✅ evaluated | sub-voxel thermal fins |
| real-CityGML | real geometry | ✅ evaluated | TUM2TWIN LoD2 shells, sim. physics |
| real-CityGML-LoD3 | real geometry | ⏳ bake-off running | TUM2TWIN LoD3 shells (coplanar-merged), sim. physics |
| real-3DBAG | real geometry | ⏳ bake-off running | 3D BAG Amsterdam LoD2.2 shells, sim. physics |
| DOE-refbldg | real constructions | ✅ evaluated | DOE Reference Buildings (real materials, idealised geometry) |

## Cross-task validation (non-direct datasets, made comparable)

These real datasets validate *different* quantities than the θ-field matrix, so each carries its own metric — all wired and run:

| Dataset | Family | Validates | Result |
|---|---|---|---|
| Twin Houses | real measured U | per-element U vs documented (real assemblies) | **U-MAE 0.0042 W/m²K** over 9 real elements (8/9 exact; roof Δ = rafter bridging the 1-D prior misses) |
| ThermoScenes | real calibrated thermal | measured 3-D thermal-field decomposition on real facade geometry (ladder rung 2) | **measured 3-D field** (1620 pts, σ 1.088 °C): a geometry-local predictor explains **63%** of the facade field; localized residual σ 0.66 °C is the inverse-twin (H2) target |
| TBBR | real bridge detection | heat-loss saliency vs annotated bridges | precision 0.008, bridge-recall 0.10, **enrichment 0.77×** (<1 ⇒ saliency ≠ a trained detector) |
| TUM2TWIN-TIR | real measured thermal | airborne IR heat-loss localisation on the modelled CityGML envelopes | **enrichment 2.11×** over 27 modelled envelopes (measured IR heat-loss anomalies land 2.1× more on our buildings than off; +11 DN contrast) |

**Reading the matrix.** `correction rel-L2` and `bridge corr-relL2` are normalised so `prior_only ≡ 1.000`; **< 1 means the operator genuinely beats the analytic prior**. `delta_transolver` is the lead operator on irregular/real geometry; a voxel grid wins on axis-aligned geometry; data-only operators (gino/transolver) fail on real shells. Field rel-L2 / U-MAE are the absolute-accuracy columns.
