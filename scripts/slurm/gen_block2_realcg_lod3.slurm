#!/bin/bash
# Generate the Block-2 REAL-geometry corpus from TUM2TWIN **LoD3** CityGML buildings — the
# higher-fidelity counterpart to the LoD2 realcg corpus. LoD3 is photogrammetric (tens of
# thousands of triangles per shell); each shell is coplanar-merged into planar faces
# (thermotwin.geometry.coplanar) before the same per-surface FV + bridge physics runs on it.
# Split by building (train 0–19 / val 20–26) so val geometry is unseen. CPU job; LoD3 files are
# large (up to 78 MB) so parsing dominates — allow generous time/mem. ~150 samples -> ~30 min.
#
#   sbatch scripts/slurm/gen_block2_realcg_lod3.slurm
#SBATCH --job-name=tt3d-gen-realcg-lod3
#SBATCH --account=punim2769
#SBATCH --partition=sapphire
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=results/logs/%x-%j.out
#SBATCH --error=results/logs/%x-%j.err

set -euo pipefail
ENV=/data/gpfs/projects/punim2769/envs/thermotwin
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p results/logs
CG=data/raw/tum2twin-datasets/citygml/lod3-building-datasets

echo "host=$(hostname)  [$(date '+%T')] generating REAL-geometry (CityGML LoD3) corpus"
srun "$ENV/bin/python" scripts/generate_3d_gt.py --real-citygml-lod3 "$CG" \
  --building-start 0 --building-end 20 --n-per-building 6 \
  --seed 1337 --grid 16 --npts 4096 --cells-per-layer 4 --name block2_realcg_lod3_train
srun "$ENV/bin/python" scripts/generate_3d_gt.py --real-citygml-lod3 "$CG" \
  --building-start 20 --building-end 27 --n-per-building 5 \
  --seed 99 --grid 16 --npts 4096 --cells-per-layer 4 --name block2_realcg_lod3_val
echo "[$(date '+%T')] REAL-geometry LoD3 corpus generation complete."
