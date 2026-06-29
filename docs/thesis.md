# Thesis — ThermoTwin-3D

> The authoritative statement of what this project is and why it is novel.
> (Supersedes the archived robustness-benchmark framing under `_archive/`.)

## Thesis

Learn a **fast, differentiable, physics-consistent model of heat transfer** through a
*real building's reconstructed 3D envelope*, so we can predict heat loss, U-values, and
thermal bridges — and run retrofit what-ifs — on the **actual building** instead of a
shoebox or an RC graph.

## The pipeline, end to end

1. **Geometry in.** The scan-to-BIM envelope, represented as a point cloud + signed-distance
   function (SDF) and/or a mesh, with material layers per surface. *This is the input nobody
   else feeds a thermal model.*
2. **Physics core.** The transient heat-conduction equation (Fourier's law) through the
   envelope, with boundary conditions (indoor/outdoor temperatures, solar gain, convection
   coefficients), optionally coupled to a zone air-balance node.
3. **The learner.** A geometry-conditioned neural operator that maps
   *(geometry, material properties, boundary conditions) → temperature/heat-flux field*.
   **GINO** is a natural backbone: it encodes an irregular point cloud plus SDF features through a
   graph neural operator onto a latent grid, applies Fourier layers, and decodes back onto arbitrary
   query points — i.e. it already speaks "point cloud + SDF," which is exactly our scan output.

   > **Correction (2026-06-29, per [ADR 0010](decisions/0010-phase0-deconfound-gate.md)).** The
   > original phrasing "with the heat equation enforced in the loss" is **not** what carries the
   > result. The PDE-residual loss was empirically dead (it lost to plain training in Block-1) and is
   > used in *no* Block-2 run. The mechanism that works is **delta learning on a closed-form analytic
   > 1-D conduction prior** — predict the geometry-coupled *correction* on the prior, not the field
   > from scratch. The forward operator's value over the prior alone is currently under audit (the
   > prior is near-optimal on real envelopes); see ADR 0010. Frame H1 as the hybrid analytic+
   > correction surrogate, not as physics-in-the-loss.
4. **Calibration / inverse.** Assimilate measured thermal data (thermal point clouds, UAV IR)
   to infer *spatially varying* envelope properties (per-surface U-value, thermal-bridge
   conductance), turning today's descriptive thermal maps into a predictive, calibrated twin.
   This is the second novelty hook and a clean uncertainty story.
5. **Outputs.** Heat-loss map, per-surface U-values, thermal-bridge quantification (not just
   detection), and gradient-based retrofit optimisation on the real geometry.

## The contribution, stated plainly

The **first geometry-resolved, physics-informed thermal twin of building envelopes from
as-built scans** — bridging:

- the **building-thermal-ML** literature (which uses lumped / zone geometry),
- the **SciML operator** literature (which never touches buildings), and
- **thermal point clouds** (which only visualise).

## Two novelty hooks

- **H1 — Geometry resolution.** A thermal field solved on the real as-built envelope, not a
  lumped zone. The ablations must show geometry-resolution *earns its keep* vs zone/RC baselines.
- **H2 — Calibrated inverse twin.** Assimilating measured IR to recover spatially varying
  envelope properties with uncertainty — descriptive thermography becomes a predictive twin.

## Framing note (venue)

Target: **Automation in Construction** (IF 11.5, Q1). Building-physics relevance (U-values,
transmission heat loss, thermal bridges, retrofit ROI) must carry as much weight as the
operator-learning novelty. See [`baselines.md`](baselines.md) and [`experiment-plan.md`](experiment-plan.md).
