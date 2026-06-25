#!/bin/bash
# Generate the Block-2 REAL-geometry corpus from TUM2TWIN LoD2 CityGML buildings (Exp 2.6):
# each building -> a whole-shell point cloud at REAL orientations, per-surface FV + strong
# bridges. Split by building (train 0–19 / val 20–26) so val geometry is unseen. CPU job;
# ~5 s/sample, ~150 samples -> ~15 min. See data/real_citygml_3d.py + docs/block2_redesign.md.
#
#   sbatch scripts/slurm/gen_block2_realcg.slurm
#SBATCH --job-name=tt3d-gen-realcg
#SBATCH --account=punim2769
#SBATCH --partition=sapphire
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=results/logs/%x-%j.out
#SBATCH --error=results/logs/%x-%j.err

set -euo pipefail
ENV=/data/gpfs/projects/punim2769/envs/thermotwin
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p results/logs
CG=data/raw/tum2twin-datasets/citygml/lod2-building-datasets

echo "host=$(hostname)  [$(date '+%T')] generating REAL-geometry (CityGML) corpus"
srun "$ENV/bin/python" scripts/generate_3d_gt.py --real-citygml "$CG" \
  --building-start 0 --building-end 20 --n-per-building 6 \
  --seed 1337 --grid 16 --npts 4096 --cells-per-layer 4 --name block2_realcg_train
srun "$ENV/bin/python" scripts/generate_3d_gt.py --real-citygml "$CG" \
  --building-start 20 --building-end 27 --n-per-building 5 \
  --seed 99 --grid 16 --npts 4096 --cells-per-layer 4 --name block2_realcg_val
echo "[$(date '+%T')] REAL-geometry corpus generation complete."
