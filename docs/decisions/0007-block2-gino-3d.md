# 7. Block-2 — carry the delta prior into a 3-D geometry-conditioned operator (GINO)

Date: 2026-06-24

## Status

Accepted (benchmark in flight — results section to be backfilled from the real JSON)

## Context

Block-1 closed with one robust finding (ADRs [`0004`](0004-block1-model-ablation.md),
[`0005`](0005-ood-generalization.md)): the lever for the venue's headline metric is a
**hard analytic 1-D clear-wall θ prior wired into the architecture as an additive
correction**. `delta_fno` — which predicts only the residual the prior misses (the
lateral spreading near thermal bridges) — cut U-MAE 56% in-distribution **and**
generalised, posting the lowest U-MAE and the smallest generalization gap on all 8 OOD
cells (4 axes × 2 regimes), 1.9–5.0× ahead of every prior-less variant. The OOD study
also named the binding risk for as-built scans: **unseen envelope assemblies**, not
films or mesh resolution.

But all of that lives on **2-D regular grids**. ThermoTwin-3D's target is the as-built
envelope — irregular, tessellated, three-dimensional geometry reconstructed from scans.
The Block-1 mandate to Block-2 was therefore explicit: carry the delta-prior recipe onto
**irregular 3-D geometry** with a geometry-conditioned operator, and make the
assembly/material-layer encoding the design focus. This ADR records how we wired that
operator (**GINO**), how we generate exact 3-D ground truth to test it, how we bring in
**real** TUM2TWIN geometry, and what the first 3-D benchmark (Exp 2.1) is set up to
decide.

## Decision

### 1. Backbone: GINO (Geometry-Informed Neural Operator), with the verified wiring

GINO (Li et al., 2023) is the natural backbone for scattered as-built clouds: an input
GNO graph-encodes per-point features from arbitrary coordinates onto a regular latent
grid, an FNO mixes globally on that grid, and an output GNO graph-decodes back to *any*
query coordinates. Geometry enters **twice** — as the input/output point coordinates and
as a signed-distance field on the latent grid — so the operator is conditioned on the
as-built shape rather than assuming a box. `models/gino.py` wraps the installed
`neuralop.models.GINO` (2.0.0) and fixes the construction that recon proved correct:

- **`in_gno_transform_type="nonlinear"`** (not the default `"linear"`). This is the
  trap: with the linear transform GINO sets the input-GNO output width to
  `fno_in_channels` and elementwise-multiplies it by the per-point features, raising a
  `RuntimeError` whenever `in_channels ≠ fno_in_channels`. Our feature width is 3–4 and
  the latent width is 32, so the nonlinear transform — which decouples the two, as the
  official car example does — is **required**, not a tuning choice.
- **SDF as `latent_features`** (`latent_feature_channels=1`). The signed-distance field
  on the `G³` latent grid is the second geometry channel; for the synthetic
  axis-aligned blocks it is analytic.
- **`gno_use_torch_scatter=False`** (the package is absent; GINO falls back silently —
  this just silences the warning) and **`gno_use_open3d=True`** (present on CPU and GPU).
- **Mode/grid invariants guarded** (`_validate_modes`): `len(fno_n_modes) == gno_coord_dim
  == 3` and each mode `< latent_grid // 2`. All forward arguments are normalised to a
  shared `[0,1]³` box, in which the GNO radius lives.

### 2. The delta structure on irregular geometry (`delta_gino`)

`DeltaGino` mirrors `DeltaFNO` exactly, but for scattered queries. The forward is
`query_prior + GinoOperator(correction)`: the analytic 1-D clear-wall prior is evaluated
at **each output query's** through-wall coordinate and supplied *per query by the
dataset*, then added back after the network predicts only the residual. The prior is
**not** read from a feature channel here — output queries need not coincide with input
points — which keeps the additive structure explicit and PINN-friendly (gradients flow
through the output queries). The data-only `gino` is the ablation: same network, prior
**dropped** from the features (3 channels, `[logk_std, r_si, r_se]`), forced to learn θ
from scratch on the cloud.

### 3. Exact 3-D ground truth (`data/synthetic_3d.py`)

The 3-D analogue of the Block-1 generator. A wall **block** layered through axis 0,
homogeneous across the two in-plane axes, punctured by **rectangular-prism** thermal
bridges finite in *both* in-plane axes (a stud/nib, not a 2-D strip) that target the
insulation layer (ADR `0006`). Solved by `physics/steady_fv` (axis-0 Dirichlet/film, all
other faces adiabatic — the existing N-dimensional solver, no new code), giving a
genuinely 3-D θ field. Each block becomes a GINO sample: ~2k interior points in `[0,1]³`
with features `[logk_std, r_si, r_se, theta1d]`, per-point target θ (trilinear from the
FV field), and the analytic box SDF on a `16³` latent grid. Seeded; per-sample `.npz` +
`manifest.json`, mirroring Block-1.

### 4. Fair scoring across heterogeneous representations

The roster mixes a scattered operator (GINO) and a dense grid operator (`fno_voxel`), so
both are scored on the **same support** — the original sampled points; the voxel field is
trilinearly resampled back to the cloud — with the **same** U-from-indoor-face estimator,
`eval/building.u_from_indoor_face_cloud`. U is read as the indoor-face dimensionless-deficit
ratio against the analytic prior, `U ≈ U_clear · mean(1−θ_face)/mean(1−θ1d_face)` over a
near-face band (0.08 normalised): **exact on a clear column** (θ ≡ θ1d ⇒ U = U_clear), so
it estimates only the bridge-driven excess and **never touches the target U**
(leakage-free, identical on every model and on the ground truth).

### 5. Real-geometry on-ramp: CityGML → Envelope (`geometry/citygml.py`)

To exercise the operator on genuine tessellated shells rather than synthetic boxes, a
TUM2TWIN CityGML 2.0 reader lifts each building into our `Envelope`: thematic surfaces
(`WallSurface`/`RoofSurface`/`GroundSurface` → Wall/Roof/Floor, boundary Outdoors/Ground),
`gml:posList` exterior rings → `(n,3)` vertices (closing duplicate dropped), coordinates
made local-metric by subtracting the per-building min corner (EPSG:25832, parsed with
stdlib `xml.etree`). LoD2 = one polygon per surface; LoD3 = tessellated multi-polygon
surfaces (one `Surface` per exterior ring) with window/door interior rings **dropped for
v1** (opaque wall). Truncated files (`ET.parse` raises) re-raise clearly (single building)
or skip-and-log (directory). CityGML carries **geometry only**, so materials come from a
small default library keyed by surface type. The `Envelope` flows **unchanged** into the
existing point-cloud featuriser and SDF mesher.

## Results

**Pending — not fabricated.** The Block-2 benchmark (`scripts/benchmark_block2.py`,
3 models × seeds `[1337, 1]` × 150 epochs, AdamW + cosine) is running as Slurm job
**26440168** (`tt3d-block2`, A100 `spartan-gpgpu120`, CUDA confirmed) and writes
`results/block2_benchmark.{json,md}` only after all 6 runs finish; at the time of writing
it had emitted no per-seed result line. The two questions the JSON will settle, with the
bar each model must clear, are recorded so the verdict is mechanical when the artefact
lands:

1. **`delta_prior_carries_to_3d`** = TRUE iff `delta_gino`'s field rel-L2 **and** U-MAE
   are materially below data-only `gino`'s (within-σ ⇒ report as "did not carry").
2. **GINO vs voxel** — GINO justifies itself only if `delta_gino` ≤ `fno_voxel` on both
   metrics **despite never seeing a grid**; if the voxel-FNO ties/wins on these simple
   boxes, the honest read is that the grid baseline suffices here and GINO's value must
   be argued on irregular/real scans.
3. Both geometry-aware models must clear the geometry-blind 1-D clear-wall U-MAE (H1).

The **CityGML reader is landed and validated**: all 27 TUM2TWIN LoD2 buildings parse
(899 surfaces, 0 skipped); a sample envelope feeds the point cloud and watertight SDF
mesh unchanged. Gated by `tests/test_citygml.py` (9/9).

## Limitations

- **The 3-D benchmark is still synthetic.** Clouds are ~2k clean interior points with no
  occlusion, noise, registration error, or missing faces — far easier than a real
  LiDAR/photogrammetry scan (the GNO radius is tuned generously to 0.12 to compensate).
  Geometries are simple wall blocks, not the tessellated LoD3 envelopes with window/door
  holes the project ultimately targets.
- **Ground truth is linear steady conduction** (`steady_fv`, dimensionless θ): no
  convection, radiation, moisture, transient or thermal-mass effects, and **no measured
  IR**. The U-from-face estimator's own bias is shared across models but unvalidated
  against real metering.
- **Real thermal labels remain gated.** CityGML carries geometry only, so materials are
  library placeholders; the assembly-shift robustness claim from ADR `0005` is a
  *hypothesis about real scans* that Block-2 must still confirm on measured IR (TUM2TWIN,
  TBBR) before the paper can lead with it.
- **Coverage is modest:** 2 seeds and a single `16³` latent/voxel grid, so the
  resolution-generalisation axis that was decisive in Block-1 OOD is **untested** in this
  run.

## Consequences

- **+** The Block-1 recipe (`delta` head on a hard analytic prior) is now implemented on
  irregular 3-D geometry with a geometry-conditioned backbone, plus a data-only GINO
  ablation and a regular-grid voxel-FNO reference — the comparison that decides whether
  the prior carries and whether GINO earns its place is set up to be read straight off
  the JSON, leakage-free and on a shared support.
- **+** A real-geometry on-ramp exists end-to-end: TUM2TWIN CityGML → `Envelope` → point
  cloud + SDF, reusing the whole Block-1 featuriser unchanged. The 3-D operator can be
  pointed at genuine as-built shells the moment thermal labels are available.
- **−** No physics validation against measured data yet, and the synthetic 3-D corpus
  omits the resolution axis. Next steps: backfill Exp 2.1 from the real JSON; add a
  resolution-generalisation eval (train `G=16`, test larger); then drive the LoD3 CityGML
  envelopes (with window/door holes) and, ultimately, calibrate against measured IR.
