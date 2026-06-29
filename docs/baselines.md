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

## Results so far (Block-2 / H1) — status: UNDER AUDIT, decision gate running

> A 5-angle adversarial self-audit (2026-06-29; GT/circularity, data-hygiene, metric-fairness,
> training-fairness, dataset/baseline sufficiency) walked back the earlier "delta beats physics,
> backbone-agnostically" verdict. See **[ADR 0010](decisions/0010-phase0-deconfound-gate.md)** for
> the full finding list and the Phase-0 decision gate now running. Read this section as the honest
> current state, not a settled result.

**What holds.** The delta-prior recipe *rescues operators from total failure*: every data-only
operator collapses on real geometry (field rel-L2 0.22–0.88), and the same backbone wrapped in the
delta-prior reconstructs the field (rel-L2 0.015–0.026). The clean-comparison facts also hold — no
geometry leakage (by-building splits), leak-free normalisation, faithful (non-strawman) competitor
backbones.

**What does NOT hold (the walk-back).**
- **"Beats the prior on real geometry" is not established.** On the real corpora the *global*
  `correction_rel_l2` is **≥ 1 for 5 of 6 delta backbones** (they do not beat the zero-parameter
  prior); only `delta_pointnet2` is < 1, and by ~4%. On DOE, `prior_only` wins field rel-L2 outright.
  The prior already explains ~98% of the real-geometry field — possibly because real envelopes are
  mostly clear wall (physics), not a bug to fix.
- **The data-only vs delta comparison is confounded.** "Delta" changes two things at once — it gains
  the prior as an extra *input channel* AND a residual *target*. Phase-0 (`cond_*` / `delta_const_*`,
  see `scripts/benchmark_block2.py`) deconfounds this.
- **The GT is welded to the prior** (the prior is the FV solver's own per-column integral), the
  "real geometry" corpora are synthetic per-surface conduction with the cloud rotated into building
  coords (DOE never runs EnergyPlus), U-MAE is vacuous on the real corpora (prior U-MAE = 0), and the
  **trained operator is never run on measured data** — the cross-task rungs validate the prior /
  heuristics, not the network.

**The gate (ADR 0010).** Phase-0 decides H1-rescue vs pivot-to-H2 on three falsifiable questions:
does the residual structure matter (`cond` vs `delta`); does the *physics* of the prior matter
(`delta_const` vs `delta`); and does any backbone beat `prior_only` on real geometry under the
mean-removed `field_rel_l2_fluct`. Per-corpus detail: `results/block2_*_phase0.{md,json}`; the
(now-superseded) 14-model matrix: `results/unified_eval.{md,json}`.
