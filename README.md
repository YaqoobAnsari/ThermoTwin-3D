<div align="center">

# ThermoTwin-3D

**A geometry-resolved, physics-informed thermal twin of building envelopes — learned from as-built scans.**

*Status: 🚧 early research scaffold · Target venue: Automation in Construction (Q1)*

</div>

---

## What this is

ThermoTwin-3D learns a **fast, differentiable, physics-consistent** model of heat transfer through a *real building's reconstructed 3D envelope*. Instead of collapsing a building into a shoebox or an RC graph, we feed the model the actual as-built geometry (point cloud + signed-distance function + mesh, with material layers per surface) and predict the **temperature and heat-flux field** through the envelope — then read off heat loss, per-surface U-values, and thermal bridges, and run **gradient-based retrofit what-ifs** on the real geometry.

The headline contribution: **the first geometry-resolved, physics-informed thermal twin of building envelopes from as-built scans** — bridging three literatures that have not met:

- **building-thermal ML**, which uses lumped / zone geometry;
- **SciML operator learning**, which never touches buildings; and
- **thermal point clouds**, which only *visualise* heat, never *predict* it.

## The pipeline

| # | Stage | What happens | Code |
|---|-------|--------------|------|
| 1 | **Geometry in** | As-built envelope as point cloud + SDF and/or mesh, with per-surface material layers. *The input nobody else feeds a thermal model.* | `src/thermotwin/geometry/` |
| 2 | **Physics core** | Transient heat conduction (Fourier) through the envelope; BCs = indoor/outdoor temperature, solar gain, convection; optional zone air-balance node. | `src/thermotwin/physics/` |
| 3 | **The learner** | A geometry-conditioned neural operator maps *(geometry, materials, BCs) → temperature/heat-flux field*, with the heat equation enforced in the loss. **GINO** is the backbone (graph neural operator over point cloud + SDF → latent grid → Fourier layers → decode to query points). | `src/thermotwin/models/`, `src/thermotwin/losses/` |
| 4 | **Calibration / inverse** | Assimilate measured thermal data (thermal point clouds, UAV IR) to infer *spatially varying* envelope properties (per-surface U-value, thermal-bridge conductance). Second novelty hook + uncertainty story. | `src/thermotwin/calibration/` |
| 5 | **Outputs** | Heat-loss map, per-surface U-values, thermal-bridge **quantification** (not just detection), differentiable retrofit optimisation. | `src/thermotwin/eval/`, `src/thermotwin/viz/` |

## Repository layout

```
ThermoTwin-3D/
├── src/thermotwin/        # the package — one subpackage per pipeline stage
│   ├── geometry/  physics/  data/  models/  losses/
│   ├── calibration/  training/  eval/  viz/  utils/
├── configs/               # Hydra YAML (data / model / physics / train / experiment)
├── data/                  # datasets (git-ignored; see data/README.md)
├── docs/                  # thesis, datasets, baselines, experiment plan, ADRs
├── experiments/           # one sub-dir per experiment block (see docs/experiment-plan.md)
├── scripts/               # train / evaluate / data / FEM-GT entry points + Slurm templates
├── notebooks/  tests/  results/  env/
├── README.md  CLAUDE.md  CHANGELOG.md  Makefile  pyproject.toml
```

## Datasets

| Role | Dataset | Use |
|------|---------|-----|
| Real thermal + geometry | **TUM2TWIN** (ISPRS JPRS 2026) | Validation anchor: MLS point clouds + LoD2/3 CityGML + TIR fused into 3D thermal point clouds. |
| Thermal-bridge GT | **TBBR / TBBRv2** | 926 UAV images / 6,927 thermal-bridge annotations (RGB + thermal + height). |
| Geometry corpus | **DOE Reference Buildings**, ScanNet++, Structured3D, BIMNet, **Building3D** | Realistic as-built geometry + EnergyPlus models for synthetic training data. |
| Physics GT (we generate) | **FEM heat-conduction fields** (FEniCS/COMSOL) and/or EnergyPlus | No off-the-shelf "envelope conduction field on real geometry" exists — building it is part of the contribution and a release asset. |

Full catalogue: [`docs/datasets.md`](docs/datasets.md).

## Baselines

Numerical (FEM/CFD, EnergyPlus/RC, 1D U-value) · building-domain ML (Di Natale, Yang, Gokhale) · SciML operators (**GINO**, FNO, DeepONet, GNOT, Transolver, MeshGraphNet) · pure data-driven (3D U-Net, PointNet++). See [`docs/baselines.md`](docs/baselines.md).

## Quickstart

```bash
# 1. Environment (conda)
make setup                      # creates the `thermotwin` env, installs the package + hooks
#   or: conda env create -f env/environment.yml && pip install -e .

# 2. Sanity
make test                       # smoke tests
make lint                       # ruff

# 3. Train / evaluate (stubs today — see docs/architecture.md)
python scripts/train.py experiment=block1_synthetic_fem
python scripts/evaluate.py
#   on Spartan: sbatch scripts/slurm/train.slurm experiment=block1_synthetic_fem
```

## Experiment blocks

1. **Controlled synthetic FEM benchmark** — field accuracy + speedup vs FEM.
2. **Real-building validation** — predictions vs measured fields on TUM2TWIN / TBBR.
3. **Ablations** — physics-loss on/off · geometry-encoding on/off · PINN vs operator · assimilation on/off.
4. **Retrofit what-if** — the differentiable twin optimising an intervention on real geometry.

Detail + metrics: [`docs/experiment-plan.md`](docs/experiment-plan.md).

## Citation

```bibtex
@misc{thermotwin3d,
  title  = {ThermoTwin-3D: A Geometry-Resolved, Physics-Informed Thermal Twin of Building Envelopes from As-Built Scans},
  author = {Ansari and collaborators},
  year   = {2026},
  note   = {In preparation, Automation in Construction}
}
```

## License

Pre-publication research code — **all rights reserved** for now (see [`LICENSE`](LICENSE)). A permissive license will be chosen before public release.
