#!/bin/bash
# Generate the Block-2 REAL-geometry corpus from 3D BAG (Netherlands) CityJSON tiles — real
# LoD2.2 building shells at scale, per-surface FV + bridges. Split by building (train 0–119 /
# val 120–159, disjoint) so val geometry is unseen. CPU job; ~5 s/building. See
# data/real_citygml_3d.py::generate_corpus_bag + geometry/cityjson.py.
#
#   sbatch scripts/slurm/gen_block2_bag.slurm
#SBATCH --job-name=tt3d-gen-bag
#SBATCH --account=punim2769
#SBATCH --partition=sapphire
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=results/logs/%x-%j.out
#SBATCH --error=results/logs/%x-%j.err

set -uo pipefail
ENV=/data/gpfs/projects/punim2769/envs/thermotwin
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p results/logs
BAG=data/raw/bag_nl

echo "host=$(hostname)  [$(date '+%T')] generating 3D BAG real-geometry corpus"
srun "$ENV/bin/python" scripts/generate_3d_gt.py --bag-cityjson "$BAG" \
  --building-start 0 --building-end 120 --n-per-building 3 \
  --seed 1337 --grid 16 --npts 4096 --cells-per-layer 4 --name block2_bag_train
srun "$ENV/bin/python" scripts/generate_3d_gt.py --bag-cityjson "$BAG" \
  --building-start 120 --building-end 160 --n-per-building 3 \
  --seed 99 --grid 16 --npts 4096 --cells-per-layer 4 --name block2_bag_val
echo "[$(date '+%T')] 3D BAG corpus generation complete."
