# TSDN / ThermalGS — Thermal Scene Day-and-Night (aerial radiometric thermal + geometry)

- **key:** `tsdn`
- **role:** real_thermal (calibrated °C) + geometry — counterexample dataset (see data_inventory §4)
- **license:** code MIT; **dataset = non-commercial research use only** ⚠️
- **paper:** Liu et al., *Remote Sensing* 17(2):335, 2025 — DOI 10.3390/rs17020335
- **repo:** https://github.com/porcofly/ThermalGS-and-TSDN-Dataset
- **download:** OneDrive — https://1drv.ms/u/s!AuxIu5p3iyOxiGeExKUHh35V4HWB?e=8UA0IM

## What it gives

One real aerial scene (Changsha, China; DJI Matrice 300 RTK + H20T LWIR + oblique RGB) with
**5 real buildings**, captured at 5 times of day/night. **Radiometric °C** TIR (via DJI Thermal
SDK) + a **photogrammetric 3-D model of the same scene** (`points3d.ply` + `mesh.obj`/`.mtl`).
A genuine calibrated-thermal + geometry pairing — the aerial/multi-building complement to
ThermoScenes' close-range facades.

## ⚠️ Manual download required

The OneDrive `1drv.ms` link has migrated to SharePoint Online and **403s all headless
requests** (share-API, resolved-redirect, `download=1` — all blocked; personal SPO shares need
a browser session). `scripts/slurm/download_extra.slurm` attempts it best-effort and falls back
to these instructions.

**To install:** open the OneDrive link in a browser, download the archive, and place it in this
directory (`data/raw/tsdn/`). Then extract. We'll wire the loader once the files are here.

## Limitation

n=5 buildings, single scene, non-commercial license; delivered as multi-view °C TIR + geometry
(reconstruct the field, not pre-baked). No material layers / U-values / boundary conditions.
