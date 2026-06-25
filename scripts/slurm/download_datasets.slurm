#!/bin/bash
# Download ALL real-world datasets to the project drive, as a resumable Slurm job that
# survives disconnect (Spartan compute nodes have outbound internet — verified). Idempotent:
# skips files already complete (local size == server content-length), resumes partials
# (curl -C -). Re-run safely any time. Everything lands under data/raw/ (project storage).
#
#   sbatch scripts/slurm/download_datasets.slurm
#
# Contents: TBBR/TBBRv2 all 6 flights (Zenodo 6517768, ~68.5 GB) · Twin Houses (FORDATIS,
# ~410 MB) · ThermoScenes (Zenodo 10835108, ~819 MB) · 3D BAG representative subset
# (8 Amsterdam CityJSON tiles ≈ a few thousand real irregular shells).
#SBATCH --job-name=tt3d-download
#SBATCH --account=punim2769
#SBATCH --partition=sapphire
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --output=results/logs/%x-%j.out
#SBATCH --error=results/logs/%x-%j.err

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
RAW=data/raw
mkdir -p results/logs "$RAW/tbbr" "$RAW/twin_houses" "$RAW/thermoscenes" "$RAW/bag_nl"

dl() {  # url dest
  local url="$1" dest="$2"
  local exp have
  exp=$(curl -sIL --max-time 90 "$url" | grep -i '^content-length' | tail -1 | tr -dc '0-9')
  have=$(stat -c%s "$dest" 2>/dev/null || echo 0)
  if [ -n "$exp" ] && [ "$have" = "$exp" ]; then
    echo "[$(date '+%T')] [skip] $dest ($have B)"; return 0
  fi
  echo "[$(date '+%T')] [get ] $dest (have $have / exp ${exp:-?})"
  curl -fL --no-progress-meter -C - --retry 10 --retry-delay 20 -o "$dest" "$url"
  echo "[$(date '+%T')] [done] $dest ($(stat -c%s "$dest" 2>/dev/null || echo 0) B)"
}

echo "host=$(hostname)  start=$(date '+%F %T')"

echo "=== TBBR/TBBRv2 — all flights (Zenodo 6517768) ==="
REC="https://zenodo.org/records/6517768/files"
for n in 100 101 102 103 104 105; do dl "$REC/Flug1_$n.tar.zst?download=1" "$RAW/tbbr/Flug1_$n.tar.zst"; done
dl "$REC/Flug1_100-104Media_coco.json?download=1" "$RAW/tbbr/Flug1_100-104Media_coco.json"
dl "$REC/Flug1_105Media_coco.json?download=1"     "$RAW/tbbr/Flug1_105Media_coco.json"

echo "=== Twin Houses (FORDATIS fordatis/161.2) ==="
TH="https://fordatis.fraunhofer.de/bitstream/fordatis/161.2"
dl "$TH/1/00_File_Structure.pdf" "$RAW/twin_houses/00_File_Structure.pdf"
dl "$TH/2/01_Experimental%20specification%20BESmodVAL_annex71_final_v1_0.pdf" "$RAW/twin_houses/01_Experimental_specification.pdf"
dl "$TH/3/02_Additional%20Documents.zip" "$RAW/twin_houses/02_Additional_Documents.zip"
dl "$TH/4/03_Data_Main_Experiment.zip" "$RAW/twin_houses/03_Data_Main_Experiment.zip"
dl "$TH/5/04_Data_Extended_Experiment.zip" "$RAW/twin_houses/04_Data_Extended_Experiment.zip"

echo "=== ThermoScenes (Zenodo 10835108) ==="
dl "https://zenodo.org/api/records/10835108/files/ThermoScenes.zip/content" "$RAW/thermoscenes/ThermoScenes.zip"

echo "=== 3D BAG representative subset (8 Amsterdam CityJSON tiles) ==="
for u in \
  "https://data.3dbag.nl/v20240420/tiles/9/436/716/9-436-716.city.json.gz" \
  "https://data.3dbag.nl/v20240420/tiles/10/432/716/10-432-716.city.json.gz" \
  "https://data.3dbag.nl/v20240420/tiles/10/434/716/10-434-716.city.json.gz" \
  "https://data.3dbag.nl/v20240420/tiles/10/434/718/10-434-718.city.json.gz" \
  "https://data.3dbag.nl/v20240420/tiles/10/432/718/10-432-718.city.json.gz" \
  "https://data.3dbag.nl/v20240420/tiles/10/430/718/10-430-718.city.json.gz" \
  "https://data.3dbag.nl/v20240420/tiles/10/430/716/10-430-716.city.json.gz" \
  "https://data.3dbag.nl/v20240420/tiles/10/428/716/10-428-716.city.json.gz" ; do
  dl "$u" "$RAW/bag_nl/$(basename "$u")"
done

echo "=== DONE $(date '+%F %T'); sizes ==="
du -sh "$RAW/tbbr" "$RAW/twin_houses" "$RAW/thermoscenes" "$RAW/bag_nl" 2>/dev/null
