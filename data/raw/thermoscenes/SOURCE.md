# ThermoScenes (ThermoNeRF) — calibrated-°C building-facade IR

- **key:** `thermoscenes`
- **role:** real_thermal (dense-field, calibrated temperature)
- **license:** CC-BY-4.0
- **homepage / DOI:** https://doi.org/10.5281/zenodo.10835108
- **download date:** 2026-06-25

## What it gives

The only *verified* calibrated-°C facade IR corpus we found: ~8 building facades captured with
a FLIR radiometric camera (true surface temperature, ±3 °C), multi-view RGB + thermal. Unlike
TBBR (uncalibrated DN) and TUM2TWIN street TIR (uncalibrated intensity), these are **absolute
temperatures** — usable for a quantitative, dense surface-temperature pattern check. Follow-up
work (Thermoxels) converts these scenes into FEA-ready heat-conduction meshes, so it can also
seed a real-geometry conduction case.

## Files fetched

- ThermoScenes.zip (~819 MB)

## Limitation

Handheld/close-range facades (not UAV/whole-building, not as-built MLS); no material-layer or
U-value ground truth. Complements Twin Houses (which has U/heat-flux but sparse sensing) by
providing the **dense calibrated-temperature field** Twin Houses lacks.
