# Datasets — ThermoTwin-3D

Three roles: **real thermal + geometry** (validation / calibration), **geometry corpus**
(scan realism + a substrate to generate physics GT over), and **physics ground truth**
(which we generate — also a release asset).

## Real thermal + geometry (validation + calibration)

| Dataset | What it provides | Role |
|---------|------------------|------|
| **TUM2TWIN** (ISPRS JPRS 2026) | MLS point clouds + LoD2/LoD3 CityGML + street-level TIR fused into **3D thermal point clouds**. | **The premier validation anchor** — real building, real thermal field. |
| **TBBR / TBBRv2** (Thermal Bridges on Building Rooftops) | Public Zenodo dataset: **926 UAV images** over Karlsruhe, **6,927 thermal-bridge annotations**; each image has RGB + thermographic + height channels. Code/data via `Helmholtz-AI-Energy/TBBRDet`. | Thermal-bridge ground truth + real heat-loss patterns. |

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
