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
- **GINO** — primary backbone and headline baseline.
- **FNO**, **DeepONet**, **GNOT**, **Transolver**, **MeshGraphNet** — the standard
  operator / transformer / graph PDE-solver roster.
- Pure data-driven: **3D U-Net / CNN on voxels**, **PointNet++ regression** (no physics)
  — to prove the physics loss earns its keep.

### Our model + ablations
physics loss on/off · geometry encoding on/off · PINN vs neural-operator ·
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
