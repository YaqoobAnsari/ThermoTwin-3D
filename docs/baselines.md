# Baselines & Metrics — ThermoTwin-3D

## Baselines (everything we compare against)

### Numerical / physics (references + ground truth)
- **High-fidelity FEM/CFD** — ground truth and the slow reference we beat on speed.
- **EnergyPlus / RC-zone model** — the building-domain status quo (lumped geometry).
- **1D analytical U-value** — the standard building-physics envelope estimate.

### Building-domain ML (lumped / zone — show geometry-resolution wins)
- Physically-consistent NN for multi-zone buildings — *Di Natale et al.*
- Physics-constrained graph model for building thermal dynamics — *Yang et al.*
- Control-oriented building PINN — *Gokhale et al.*

### SciML / geometry neural-operator (the real competition — adapt to conduction)
The Block-2 bake-off runs a **14-model roster** = six backbone families, each in a *data-only*
and a *delta-prior* variant, plus a voxel-FNO grid reference and the analytic prior control:

| Family | data-only | delta-prior (ours) | params |
|---|---|---|---|
| operator-transformer (slice attention) | `transolver` | `delta_transolver` | 0.98 M |
| operator-transformer (linear attn + geo-MoE) | `gnot` | `delta_gnot` | 3.59 M |
| graph neural operator (latent FNO) | `gino` | `delta_gino` | 2.81 M |
| branch/trunk operator | `deeponet` | `delta_deeponet` | 0.46 M |
| point-set (set abstraction + FP) | `pointnet2` | `delta_pointnet2` | **0.17 M** |
| mesh GNN (kNN message passing) | `meshgraphnet` | `delta_meshgraphnet` | 1.27 M |
| voxel grid reference | `fno_voxel` | — | 2.41 M |
| analytic 1-D control | — | `prior_only` (≡ 1.000) | 0 |

The six families span the four major architecture classes (transformer / GNN / point / branch-trunk)
so the question *"is there a great model we missed?"* is answered by construction.

### Our model + ablations
The contribution is the **delta-prior recipe**, not a single backbone: hand the network the
closed-form 1-D clear-wall conduction field and train it to predict only the *correction*
(thermal bridges, corners, where 1-D breaks). Ablations: data-only vs delta (the recipe's effect,
run for all six families) · prior-quality sensitivity · physics loss on/off · geometry encoding ·
with/without thermal-data assimilation.

## Metrics

| Family | Metrics |
|--------|---------|
| **Field accuracy** | Temperature RMSE / MAE, heat-flux error, relative L2 (the operator-learning standard). |
| **Building-relevant** | Per-surface U-value error, whole-building transmission heat-loss error, thermal-bridge localisation **and quantification**. |
| **Efficiency** | Speedup and inference time vs FEM (GINO-class methods report ~10³–10⁴× CFD speedups — a real headline). |
| **Generalisation** | Unseen geometries, unseen boundary conditions, discretisation-invariance across mesh resolutions. |
| **Real-data validation** | Predictions vs measured thermal fields on TUM2TWIN / TBBR. |
| **Uncertainty (if added)** | Calibration error / coverage on the inferred properties. |

> Reporting rule: always pair an ML metric (relative L2) with a building metric (U-value /
> heat-loss error) — the venue cares about the latter.

## Results so far (Block-2 / H1) — the verdict

Across 6 direct corpora (3 synthetic + real-CityGML LoD2/LoD3 + DOE; **3D-BAG re-running** after a
walltime-truncated job) × 14 models × 9 metrics, scored over seeds 1337/1/2:

- **The delta-prior recipe is the decisive, backbone-agnostic factor.** Every data-only operator
  *fails* on real geometry — field rel-L2 0.22–0.88 and bridge corr-relL2 **3–9** (≫ 1, i.e. *worse*
  than the analytic prior). Wrap the same backbone in the delta-prior and it snaps to rel-L2
  **0.015–0.026** and bridge corr-relL2 **< 1** (beats the prior). All six families improve; the
  recipe — not the architecture — is what makes real as-built geometry work.
- **`delta_pointnet2` is the best backbone on real geometry — and the smallest (0.17 M).** Wins
  field rel-L2, field rel-L2@bridge, bridge corr-relL2, and is the *only* model with positive
  correction R² on real shells (real-CityGML 0.768 bridge corr-relL2; LoD3 0.842; 3D-BAG **0.659**
  from the truncated log — our strongest real-geometry result).
- **`delta_meshgraphnet` owns the sharp/sub-voxel-bridge regime** (synthetic-hard sweep; best at the
  DOE bridges). `delta_transolver`/`delta_gnot` remain strong (beat the prior everywhere) but are no
  longer the leaders. `fno_voxel` wins only on axis-aligned synthetic-box and collapses on real
  geometry (bridge corr-relL2 ~4.5).

Honest caveat: on real shells the prior is already near-exact (these envelopes are mostly clear
wall, U-MAE ~0.002–0.005), so the correction R² is small for everyone — `delta_pointnet2` being the
only positive one is the meaningful separation. The defensible claim is at the bridges
(corr-relL2 < 1 = genuinely improves on the prior where geometry matters), not global R².

One consolidated matrix + coverage map: `results/unified_eval.{md,json}`
(`scripts/unified_eval.py`); per-corpus detail in `results/block2_*_benchmark.{md,json}`.
