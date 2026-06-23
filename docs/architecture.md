# Architecture — ThermoTwin-3D

How the five pipeline stages map onto code, and the design intent behind each. This is a
living document; update it (and add an ADR under `decisions/`) when a significant choice lands.

## Data flow

```
            scan (point cloud + SDF + mesh, material layers per surface)
                                   │
        ┌──────────────────────────▼──────────────────────────┐
        │  geometry/   normalise · SDF features · material ids │   (Stage 1)
        └──────────────────────────┬──────────────────────────┘
                                   │  (geometry features, BC encodings)
        ┌──────────────────────────▼──────────────────────────┐
        │  models/   GINO: GNO encode → latent grid → Fourier  │   (Stage 3)
        │            layers → GNO decode to query points       │
        └──────────────────────────┬──────────────────────────┘
                                   │  predicted T / heat-flux field
        ┌──────────────────────────▼──────────────────────────┐
        │  losses/   data loss + heat-equation residual (PINN) │   (Stage 2 ∩ 3)
        │  physics/  Fourier conduction operator + BCs         │
        └──────────────────────────┬──────────────────────────┘
                                   │
   ┌────────────────────────────────▼────────────────────────────────┐
   │ calibration/  assimilate measured IR → spatially-varying U-values │ (Stage 4)
   └────────────────────────────────┬────────────────────────────────┘
                                   │
        ┌──────────────────────────▼──────────────────────────┐
        │  eval/ + viz/   U-values · heat-loss map · bridges · │   (Stage 5)
        │                 retrofit optimisation                │
        └─────────────────────────────────────────────────────┘
```

## Module responsibilities

- **`geometry/`** — ingest point cloud / SDF / mesh; compute SDF and geometric features;
  attach per-surface material layers; produce the input tensors the operator consumes.
- **`physics/`** — the transient heat-conduction operator (Fourier), boundary-condition
  application (indoor/outdoor T, solar, convection), optional zone air-balance node; and the
  FEM/EnergyPlus ground-truth generators.
- **`models/`** — the **GINO** backbone (graph-neural-operator encode → latent regular grid →
  spectral/Fourier layers → decode to arbitrary query points) and the baseline operators
  (FNO, DeepONet, GNOT, Transolver, MeshGraphNet) + data-driven controls.
- **`losses/`** — supervised field loss + the **PDE residual** (heat equation) and BC penalties
  that make the prediction physics-consistent.
- **`calibration/`** — the inverse problem: differentiate through the forward twin to fit
  spatially-varying envelope properties to measured IR; carry uncertainty.
- **`training/` · `eval/` · `viz/` · `utils/`** — loops/optimisers; the metric suite
  (field + building-relevant + speedup); heat-loss/field rendering; shared config/log/seed/IO.

## Key design decisions (open)

- **Backbone:** GINO first (it already speaks point-cloud + SDF). Keep `models/` pluggable so
  FNO/DeepONet/etc. slot in for the baseline table. → candidate ADR.
- **Steady vs transient:** start steady-state conduction for the benchmark, add transient +
  zone coupling once the field accuracy story is solid. → candidate ADR.
- **GT generator:** FEniCS (open, scriptable, HPC-friendly) as default; COMSOL as a
  higher-fidelity cross-check. → candidate ADR.

## Next implementation steps

1. `physics/` steady-state conduction operator + a tiny analytic test case (1D slab → known U-value).
2. `geometry/` SDF + material-layer featuriser on a single DOE Reference Building.
3. `scripts/generate_fem_groundtruth.py` → first FEM fields into `data/processed/`.
4. `models/gino.py` minimal forward; overfit one sample; then the Block-1 sweep.
