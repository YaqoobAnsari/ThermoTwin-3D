# Datasets — ThermoTwin-3D

Three roles: **real thermal + geometry** (validation / calibration), **geometry corpus**
(scan realism + a substrate to generate physics GT over), and **physics ground truth**
(which we generate — also a release asset).

> **Audit caveats (2026-06-29, [ADR 0010](decisions/0010-phase0-deconfound-gate.md)) — read before
> citing dataset breadth.** (1) Every *direct* training corpus uses **simulated** per-surface FV
> conduction GT, not measured fields; the trained operator has **never** been scored against a
> measured thermal field (the TUM2TWIN/TBBR/Twin-Houses/ThermoScenes rungs currently validate the
> analytic prior or hand-crafted heuristics, not the network). (2) The "real-CityGML LoD2" and
> "LoD3" corpora are the **same 27 TUM2TWIN buildings** at two fidelities — one building set, not
> two datasets. (3) The DOE corpus reads EnergyPlus IDF *constructions* but **does not run
> EnergyPlus**; its GT is the same FV solve. (4) Effective real-geometry breadth is ~3 correlated
> sources (Munich + Amsterdam + idealised DOE), one climate — too narrow for a general claim without
> broader geometry. These are open items, not settled capabilities.

## Real thermal + geometry (validation + calibration)

| Dataset | What it provides | Role |
|---------|------------------|------|
| **TUM2TWIN** (ISPRS JPRS 2026) | MLS point clouds + LoD2/LoD3 CityGML + street-level TIR fused into **3D thermal point clouds**. | **The premier validation anchor** — real building, real thermal field. |
| **TBBR / TBBRv2** (Thermal Bridges on Building Rooftops) | Public Zenodo dataset: **926 UAV images** over Karlsruhe, **6,927 thermal-bridge annotations**; each image has RGB + thermographic + height channels. Code/data via `Helmholtz-AI-Energy/TBBRDet`. | Thermal-bridge ground truth + real heat-loss patterns. |

## Real thermal validation datasets

Spatially-resolved, **calibrated** thermal data on real building envelopes is genuinely
scarce — most public IR is either uncalibrated, indoor, or a single facade — so the
near-term validation substrate is what we already hold (TBBR), with TUM2TWIN street-level
TIR as a characterised-but-gated qualitative anchor.

| Dataset | Status | License | What it gives | Quantitative? |
|---------|--------|---------|---------------|---------------|
| **TBBR / TBBRv2** | **In hand** (`data/raw/tbbr/`) | CC-BY-4.0 | 926 UAV scenes over Karlsruhe; per-scene **RGB + thermographic + height** channels; **6,927 thermal-bridge polygon annotations** (COCO). Paper: Mayer et al., *Automation in Construction* 2022 (our venue). | **Yes** — annotated heat-loss features → the practical near-term substrate for **H2** (thermal-bridge detection / heat-loss patterns). |
| **TUM2TWIN street-level TIR** | **Sample characterised** (`data/raw/tum2twin/thermal_tir_2016/`); full corpus **gated** | TUM2TWIN terms | 73-frame Jenoptik IR-TCM 640 sequence (16-bit 640×480 raw counts), per-frame **vehicle-carrier** pose (ENU) + ENU→ECEF 4×4. | **No** — see sample caveats below. |

### TUM2TWIN TIR sample — what the sample can and cannot do

Ingested by `src/thermotwin/data/thermal_tir.py`; characterised by
`scripts/analyse_thermal_sample.py` (figures + `results/thermal_sample/summary.json`).

* **Can:** qualitative within-frame warm/cold structure and heat-loss saliency (windows
  / thermal bridges as warm anomalies), and sequence/trajectory characterisation. The
  sample is a real Munich facade drive-by (recovered ENU origin ≈ 11.569°E, 48.149°N;
  vehicle path ≈ 20.4 m).
* **Cannot:** (1) **no radiometric calibration** — values are uncalibrated microbolometer
  counts, no count→Kelvin / emissivity / reflected-temperature correction; (2) **carrier
  pose, not sensor pose** — no camera intrinsics or boresight/lever-arm extrinsics, so no
  pixel→surface back-projection; (3) **no thermal ground-truth field**. Hence the sample
  supports characterisation + qualitative saliency only — never absolute-temperature or
  U-value validation.

### Geometry-fusion feasibility (TIR ENU ↔ CityGML UTM32N)

*Feasibility note only — not a fusion.* The TIR sample is in a **local ENU** frame; the
TUM2TWIN CityGML is in **EPSG:25832 (ETRS89 / UTM32N)**. They are relatable: the
readme's ENU→ECEF 4×4 places the ENU origin in Munich (verified ECEF→geodetic by hand →
the CityGML extent), so the chain is **ENU → ECEF (readme 4×4) → geographic → UTM32N**.
`pyproj` is currently **absent** from the env but **pip-installable** here (`pip install
pyproj` dry-run resolves `pyproj-3.7.1`, `certifi` already present); with it the
geographic→UTM32N step is one `Transformer`. The pure-geometry chain is therefore
feasible. The blocker for an *image*↔geometry fusion is unchanged: **missing camera
intrinsics/extrinsics and carrier-not-sensor pose** mean pixels still cannot be projected
onto CityGML surfaces. So fusion is feasible only at trajectory/frame granularity, not
pixel→surface — and we do **not** force it.

### Acquisition notes (actionable)

* **Request the full TUM2TWIN TIR + calibration** (intrinsics, boresight/lever-arm
  extrinsics, and any radiometric calibration) — without these the street-level TIR
  stays qualitative.
* **Lean on TBBR for H2 now**: it is in hand, calibrated-thermography-grade, CC-BY-4.0,
  and published in our target venue. Extract the 6,927 annotations into our eval format.
* **Consider a controlled own-capture** (FLIR/calibrated handheld with logged emissivity,
  reflected temperature, paired RGB + a survey of one envelope) if a quantitative
  spatially-resolved U-value reference is needed — the cleanest path to H1 validation.

## Geometry sources (scan realism + generating physics GT)

| Source | What it provides | Role |
|--------|------------------|------|
| **DOE Commercial Prototype / Reference Buildings** | Ready EnergyPlus models. | Corpus for large-scale **synthetic training data**. |
| **ScanNet++ / Matterport3D** | Indoor as-built reconstructions. | Realistic reconstructed geometry, incl. realistic defects. |
| **Structured3D** | Layout. | Layout priors / structured geometry. |
| **BIMNet** | IFC + point cloud. | Paired BIM ↔ scan geometry. |
| **Building3D** | Envelope / roof (real ALS ↔ wireframe/mesh GT). | Real envelope geometry at scale. |

## Physics ground truth (we generate)

There is **no** off-the-shelf "envelope heat-conduction field on real building geometry"
dataset. Building it is part of the contribution (exactly as GINO generated its own
aerodynamics set) and a **release asset**.

- **FEM heat-conduction solutions** (FEniCS / COMSOL) over the geometry corpus, and/or
- **EnergyPlus** runs for whole-building / zone references.

Generation entry point: `scripts/generate_fem_groundtruth.py`. Outputs land in
`data/processed/` (git-ignored). Persist a manifest per sample (geometry id, material
assignment, BCs, solver settings, seed) so every field is reproducible.

## Storage & hygiene

- All raw/processed data is **git-ignored** (see `.gitignore`); only READMEs + `.gitkeep`
  are tracked.
- Keep provenance: for each dataset note the version, license, and download date in
  `data/raw/<dataset>/SOURCE.md` when you fetch it.
- Re-confirm licences before redistribution (TUM2TWIN, TBBR terms in particular).
