# Twin Houses — IEA EBC Annex 58/71 (Holzkirchen)

- **key:** `twin_houses`
- **role:** real_thermal + geometry (H1/H2 validation backbone)
- **license:** CC-BY-SA-4.0  (⚠️ ShareAlike — constrains redistribution of derivatives)
- **homepage:** https://fordatis.fraunhofer.de/handle/fordatis/161.2
- **DOI:** 10.24406/fordatis/76.2
- **download date:** 2026-06-25

## What it gives

Two *real* full-scale single-family houses (Holzkirchen), winter experiments. Documented
envelope **geometry** (layout drawings, wall layer build-ups), per-component **U-values**
(e.g. exterior wall 0.24, window 1.2 W/m²K), thermal-bridge **ψ-values** (wall–ceiling /
wall–wall / wall–floor joints), and paired **wall surface-temperature + heat-flux** sensor
time series. The only open dataset matching our actual claim (geometry + U + bridges + heat
flux on a real building) — the validation anchor for H1 (geometry-resolved) and the U-value /
thermal-bridge readout, plus zone energy balance.

## Files fetched (~410 MB)

- 00_File_Structure.pdf
- 01_Experimental_specification.pdf
- 02_Additional_Documents.zip   (geometry drawings, ~306 MB)
- 03_Data_Main_Experiment.zip   (~45 MB, XLSX sensor data)
- 04_Data_Extended_Experiment.zip (~52 MB, XLSX sensor data)

## Limitation

Thermal sensing is **sparse / point-wise** (a few sensors, mainly the west wall), not a dense
full-field surface map — validates the U-value/heat-flux readout and energy balance, not a
dense-field prediction. Pair with ThermoScenes IR for spatial field ground truth.
