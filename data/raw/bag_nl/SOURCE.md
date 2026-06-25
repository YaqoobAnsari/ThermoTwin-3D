# 3D BAG (Netherlands) — real reconstructed building shells, representative subset

- **key:** `bag_nl`
- **role:** geometry (real as-built shells at scale, Need B)
- **license:** CC-BY-4.0
- **homepage:** https://3dbag.nl/en/download  ·  docs: https://docs.3dbag.nl/en/
- **version:** v20240420
- **download date:** 2026-06-25

## What it gives

~10 M real reconstructed Dutch buildings, CityJSON with **wall / roof / ground surfaces
semantically separated** (our per-surface conditioning + per-surface U-value unit) and a
construction year (`oorspronkelijkbouwjaar`) for vintage-archetype material keying. Real,
irregular, multi-orientation shells — the regime where the gridless `delta_transolver` won
(Exp 2.5/2.6), at ~370× the scale of our 27 TUM2TWIN buildings.

## Subset fetched (NOT the full nationwide set)

A **representative subset of 8 CityJSON tiles over central Amsterdam** (RD New bbox
~119000–123000 E, 486000–489000 N), ≈ a few thousand real building shells — sufficient for
the cross-dataset evaluation without pulling the hundreds-of-GB national set. Tiles selected
from `tile_index.fgb`; direct `cj_download` URLs. To extend, re-run the selection over a
larger bbox (see `scripts/slurm/download_datasets.slurm`).

## Limitation

Geometry only — **no materials, no thermal**. Materials stay exogenous (TABULA/IWU keyed off
build-year). LoD2.2 walls are reconstructed/extruded (not as-scanned MLS fidelity like
TUM2TWIN LoD3) — use for scale + irregularity; keep TUM2TWIN as the high-fidelity patch.
Ingest **CityJSON** (the `.city.json.gz` tiles), not OBJ (OBJ strips the wall/roof semantics).
