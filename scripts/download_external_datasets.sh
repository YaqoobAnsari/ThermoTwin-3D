#!/bin/bash
# Fetch the external datasets identified in the 2026-06-30 dataset spike into the gitignored
# data/raw/external/. See docs/datasets.md "External dataset spike" and ADR 0010.
#
# Priority = the SciML conduction-with-inclusions benchmarks: the thermal bridge is
# mathematically a high-conductivity inclusion in a matrix, so these let us run the SAME
# Phase-0 deconfound gate (homogenised prior <-> our 1-D prior) on credible PUBLISHED field
# data — no collection. Building-bridge field data does not exist openly and must be
# regenerated (EUROKOBRA/ISO 10211 geometry + THERM solver); that is a separate effort.
#
#   bash scripts/download_external_datasets.sh
set -uo pipefail
ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
DEST="$ROOT/data/raw/external"
mkdir -p "$DEST"
cd "$DEST"

clone() {  # clone() <dir> <url>  — shallow, idempotent
  local dir="$1" url="$2"
  if [ -d "$dir/.git" ]; then echo "[skip] $dir already present"; return 0; fi
  echo "[clone] $url -> $dir"
  git clone --depth 1 "$url" "$dir" && echo "[ok] $dir" || echo "[FAIL] $dir (network? clone via '! bash ...' on a login node)"
}

# --- SciML conduction-with-inclusions (the methods-test data) ---
clone WHNO         https://github.com/gmcavallazzi/WHNO.git          # rect/disk/Voronoi inclusions + generator + forward field GT
clone learning-eit https://github.com/nickhnelsen/learning-eit.git   # EIT three-phase inclusions, 100x contrast (forward solver bundled)

echo "=================================================================="
echo "Cloned under $DEST:"; ls -d "$DEST"/*/ 2>/dev/null || echo "  (none — likely no network on this node)"
cat <<'NOTE'
------------------------------------------------------------------
NOT auto-fetched (package-based / large / manual landing page):
  * Darcy flow  : `pip`-installed `neuraloperator` auto-downloads, or PDEBench DaRUS
                  doi:10.18419/darus-2986 (steady diffusion, sharp 2-phase coefficients).
  * Dublin U    : Mendeley doi:10.17632/4kbb93bx32.1  (H2/calibration: flux + both BCs + U).
  * IRIS        : Zenodo doi:10.5281/zenodo.7463995   (LARGE ~100s GB; grab a subset only).
  * TSDN        : OneDrive via github.com/porcofly/ThermalGS-and-TSDN-Dataset (manual).
  * ISO 10211   : 4 validated reference cases (geometry+BC+ref temps/fluxes) — quickfield.com
                  mirrors; the physics-validation anchor, transcribe by hand.
------------------------------------------------------------------
NOTE
