# Data inventory — what works for ThermoTwin-3D, what we have, what we need

> Snapshot 2026-06-25. Our use case needs three data roles: **real geometry** (as-built
> envelope shape), **real thermal** (measured heat / temperature to validate + calibrate
> against), and **physics ground truth** (the exact temperature field — which we generate).
> **Correction (2026-06-25, after an adversarial falsification spike):** an earlier draft of
> this doc claimed *no* public calibrated 3-D envelope thermal-field dataset paired with
> geometry exists. **That was overstated.** Calibrated, dense thermal paired with real-building
> 3-D geometry *does* exist at facade / small scale — **ThermoScenes/ThermoNeRF** (absolute °C
> FLIR, 8 real facades, multi-view + COLMAP geometry; Thermoxels bakes it to FEA meshes) and
> **TSDN/ThermalGS** (radiometric °C aerial over 5 buildings + photogrammetric mesh). So
> real-thermal validation is **feasible now** (we hold ThermoScenes). The *narrower* gap that
> genuinely survives: no public dataset pairs a calibrated, spatially-dense thermal field over a
> **whole real building exterior envelope** with as-built 3-D geometry **and** physics-grade
> material-layer / U-value / boundary-condition ground truth. That gap (scale + paired physics
> GT), not the absence of any real calibrated thermal, is the contribution. See §4.

## 1. What we HAVE on disk (usable now)

| Dataset | On disk | Role | License | Usable for | Limitation |
|---|---|---|---|---|---|
| **TUM2TWIN CityGML** | `data/raw/tum2twin-datasets/` (5.0 GB) — **27 LoD2** + 27 LoD3 `.gml` | real geometry | TUM2TWIN terms | **Real-geometry operator test** (per-surface FV GT → bake-off, Exp 2.6). Reader verified (0.1 s/bldg). | LoD2 = coarse shells (no windows as holes); LoD3 heavy (≤174 MB/file) |
| **TBBR (Flug1_100)** | `data/raw/tbbr/` (5.7 GB) — `Flug1_100.tar.zst` = 200 × (2680×3370×5) RGB+thermal+height `.npy` + 2 COCO jsons (1,313 + more anns) | real thermal | CC-BY-4.0 | **Thermal-bridge localisation** (detection/segmentation); real-world anomaly check for predicted high-flux regions | **Uncalibrated** (ε=1.0, no reflected-T) → *no absolute temperature / U*; **detection only**; only the 100 flight downloaded |
| **TUM2TWIN street TIR** | `data/raw/tum2twin/` (32 MB) — 73 frames + vehicle pose | real thermal | TUM2TWIN terms | **Qualitative** warm-region saliency, trajectory context | Uncalibrated counts; carrier-not-sensor pose; no intrinsics → **no pixel→surface, no field GT** |
| **DOE Reference Buildings** | `data/raw/doe/` (12 MB) — 16 EnergyPlus IDFs + TMY3 | geometry + constructions | public | Synthetic-corpus substrate; real material/construction libraries; envelope featuriser tests | Idealised boxes, not as-built scans |
| **Self-generated synthetic FEM** | `data/processed/` — block1, block2 box/irreg/**hard**, OOD | physics GT | ours | All Block-1/2 training + the operator bake-offs | Synthetic geometry + physics (exact, but not measured) |

## 2. What we have STUBBED (registered, gated, NOT downloaded)

`data/raw/{building3d,matterport3d,scannetpp,structured3d}/` are `SOURCE.md` placeholders only
(EULA-gated). Of these, **Building3D** (real ALS envelope geometry at scale) is the one most
worth acquiring; ScanNet++/Matterport3D are indoor (less aligned to envelopes); Structured3D is
synthetic layout.

## 3. What we NEED to get (ranked by value to the project)

| # | Dataset / asset | Why we need it | How to get it | Effort |
|---|---|---|---|---|
| 1 | **Full TBBR / TBBRv2** (all flights + val split) | Our only *real* thermal asset; needed for a credible thermal-bridge-localisation benchmark (currently only Flug1_100) | Zenodo record 7022736 (CC-BY); `scripts/download_data.py tbbr` extended | Low (download) |
| 2 | **KU Leuven HAM benchmark** (Building Simulation 2024) | The *only* genuinely **calibrated** real heat-flux + T + material-layer data — a physics sanity anchor for our conduction GT (1-D component scale) | Journal supplement (s12273-024-1176-8) | Low–med |
| 3 | **Full TUM2TWIN TIR + calibration** (camera intrinsics, boresight/lever-arm extrinsics, radiometric calibration) | The *only* path to turn TUM2TWIN street TIR into quantitative real thermal on the geometry we already have | Request from TUM2TWIN authors (EULA) | Med (gated request) |
| 4 | **Building3D** (real ALS envelope geometry ↔ mesh GT) | Real as-built geometry at scale → harder real-geometry corpus beyond 27 CityGML buildings | Registration (gated) | Med |
| 5 | **Self-captured radiometric IR campaign** on 1–3 buildings (ISO 9869 heat-flux-meter reference, logged ε / reflected-T / steady-state, paired RGB + scan) | The **only** way to get a real, *quantitative* U-value ground truth on real geometry — there is no public substitute | Own field capture | High (logistics) |

## 4. The "real validation" ladder (given the data)

Real validation is staged from strongest-physics to strongest-realism — and, contrary to the
earlier draft, **calibrated real-thermal validation is available now (rung 2)**:

1. **Real geometry + physics-exact GT (runnable now).** Per-surface FV conduction on the 27 real
   TUM2TWIN CityGML envelopes → the operator bake-off on genuinely irregular real shells
   (**Exp 2.6**). Real geometry, simulated-but-exact physics.
2. **Calibrated real-thermal field on real geometry (FEASIBLE NOW — we hold ThermoScenes).**
   Feed a real facade's geometry + (assumed/defaulted) materials → predict the surface-temperature
   field → compare against the **measured absolute °C** of ThermoScenes (8 EPFL facades, FLIR raw,
   ±3 °C, COLMAP geometry; Thermoxels gives the FEA mesh). **TSDN/ThermalGS** adds an aerial,
   5-building case (radiometric °C + photogrammetric mesh, non-commercial). This is the genuine
   "measured reality" check — validates the predicted *thermal field*, though not U directly
   (no measured U / material layers in these sets).
3. **Real thermal anomaly localisation (TBBR).** Predicted high-flux regions vs annotated bridges
   (detection metrics). Real, uncalibrated, wrong-task-for-U-value. Honest auxiliary.
4. **Point-calibrated U / heat flux (Twin Houses).** Measured U-values + ψ-bridges + point heat
   flux on two real houses → validates the U-value/heat-flux *readout* (sparse, not a field).
5. **Qualitative pattern agreement (TUM2TWIN TIR)** and, if needed, a self-captured ISO-9869
   case study for a whole-envelope quantitative U-value.

**Bottom line:** we can validate (1) the operator on real geometry, (2) the **thermal field
against calibrated real measurements** (ThermoScenes/TSDN), (3) bridge localisation (TBBR), and
(4) the U-value readout against point-calibrated reference (Twin Houses) — **all without gated
data.** What no public dataset offers is a calibrated dense thermal field over a *whole* real
envelope *with* paired material/U/boundary ground truth at scale — a contribution gap, not a
validation blocker.
