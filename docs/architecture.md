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

1. ~~`physics/` steady-state conduction operator + a tiny analytic test case~~ ✅
   `physics/conduction.py` (1-D oracle) + `physics/steady_fv.py` (geometry-resolved
   FV solver, machine-precision match to the oracle). See ADR `0002`.
2. ~~`geometry/` material-layer featuriser on a DOE Reference Building~~ ✅
   `geometry/idf.py` (dependency-free IDF reader) + `geometry/envelope.py`
   (Material/Construction/Surface → per-surface U-values, case-insensitive name
   resolution, polygon area/normal). Bridges to `steady_fv` and is gated by
   `tests/test_envelope.py` (incl. a real-DOE integration test). *SDF / point-cloud
   featurisation still to come — needed for the operator input on curved geometry.*
3. ~~`scripts/generate_fem_groundtruth.py` → first fields into `data/processed/`~~ ✅
   `data/synthetic_fem.py` generates layered walls + thermal bridges, solved by
   `steady_fv`; the script writes a seeded corpus (`block1_train`/`block1_val`) with
   a manifest. Bridges shift effective U 40–50% (max 4×) from the 1-D clear-wall
   value — the geometry signal that motivates the operator (H1).
4. ~~Minimal operator forward; overfit one sample; Block-1 training~~ ✅ (FNO first)
   `models/fno.py` (neuraloperator FNO; GINO follows on Block-2 point clouds),
   `data/dataset.py` (predicts dimensionless θ), `eval/metrics.py` (relative L2),
   and a Hydra `scripts/train.py`. Smoke run: **val relative L2 0.65 → 0.033 in 15
   CPU epochs.** Gated by `tests/test_models.py`.

### Now next
5. ~~Building-metric eval (recover U from θ, pair with relative L2)~~ ✅
   `eval/building.py` + `scripts/evaluate.py` — paired metrics at native resolution,
   operator vs 1-D clear-wall baseline (H1). `scripts/slurm/train.slurm` ready.
   **← run the GPU Block-1 sweep on Slurm (full epochs/batched) to sharpen the numbers.**
   NB: the login node kills sustained CPU training — Slurm only.
6. ~~Baseline comparison wiring — registry + no-operator CNN control~~ ✅
   `models/registry.py` + `models/cnn.py`; `model=fno`/`model=cnn` config-selectable.
   Point-cloud operators registered as deferred (Block 2). **← run the FNO-vs-CNN
   comparison on Slurm to populate the table.**
7. `geometry/` SDF / point-cloud featurisation, then `models/gino.py` — the bridge to
   real scans (Block 2); wire the deferred operators there.
