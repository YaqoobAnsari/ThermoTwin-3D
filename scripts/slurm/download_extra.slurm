#!/bin/bash
# Extra real-thermal datasets for the evaluation matrix (resumable; survives disconnect).
#   - ThermoScenes FULL (5.4 GB, Zenodo 15609062) — calibrated-°C facades + COLMAP geometry,
#     the larger release superseding the 819 MB subset. Direct, CC-BY-4.0.
#   - TSDN/ThermalGS — best-effort OneDrive fetch (likely 403: SharePoint-migrated personal
#     share needs a browser; falls back to printing manual instructions, job still succeeds).
#
#   sbatch scripts/slurm/download_extra.slurm
#SBATCH --job-name=tt3d-download-extra
#SBATCH --account=punim2769
#SBATCH --partition=sapphire
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=results/logs/%x-%j.out
#SBATCH --error=results/logs/%x-%j.err

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
RAW=data/raw
mkdir -p results/logs "$RAW/thermoscenes" "$RAW/tsdn"

dl() {  # url dest
  local url="$1" dest="$2" exp have
  exp=$(curl -sIL --max-time 90 "$url" | grep -i '^content-length' | tail -1 | tr -dc '0-9')
  have=$(stat -c%s "$dest" 2>/dev/null || echo 0)
  if [ -n "$exp" ] && [ "$have" = "$exp" ]; then echo "[$(date '+%T')] [skip] $dest ($have B)"; return 0; fi
  echo "[$(date '+%T')] [get ] $dest (have $have / exp ${exp:-?})"
  curl -fL --no-progress-meter -C - --retry 10 --retry-delay 20 -o "$dest" "$url"
}

echo "host=$(hostname)  start=$(date '+%F %T')"

echo "=== ThermoScenes FULL (Zenodo 15609062, 5.4 GB) ==="
dl "https://zenodo.org/api/records/15609062/files/ThermoScenes.zip/content" "$RAW/thermoscenes/ThermoScenes_full.zip"

echo "=== TSDN/ThermalGS — best-effort OneDrive (manual fallback) ==="
SHARE="https://1drv.ms/u/s!AuxIu5p3iyOxiGeExKUHh35V4HWB?e=8UA0IM"
B64=$(printf '%s' "$SHARE" | base64 -w0 | tr '/+' '_-' | tr -d '=')
ok=0
for u in \
  "https://api.onedrive.com/v1.0/shares/u!${B64}/root/content" \
  "https://api.onedrive.com/v1.0/shares/u!${B64}/driveItem/content" ; do
  echo "[$(date '+%T')] try $u"
  if curl -fL --no-progress-meter -C - --retry 3 -o "$RAW/tsdn/ThermalGS_TSDN.zip" "$u"; then ok=1; break; fi
done
if [ "$ok" = 1 ]; then
  echo "[$(date '+%T')] TSDN fetched ($(du -h "$RAW/tsdn/ThermalGS_TSDN.zip"|cut -f1))"
else
  echo "[$(date '+%T')] TSDN headless fetch FAILED (SharePoint-migrated personal share, 403)."
  echo "  MANUAL: open $SHARE in a browser, download the archive, place it in $RAW/tsdn/."
  echo "  (See data/raw/tsdn/SOURCE.md.) Job continues — this is expected."
fi

echo "=== DONE $(date '+%F %T'); sizes ==="
du -sh "$RAW/thermoscenes" "$RAW/tsdn" 2>/dev/null
