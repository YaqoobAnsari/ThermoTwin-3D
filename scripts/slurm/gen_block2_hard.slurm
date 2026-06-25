#!/bin/bash
# Generate the Block-2 HARD corpus: fine-native 3-D wall blocks with sub-voxel thermal
# fins (the diagnosis-driven corpus where a 16³ voxel grid genuinely aliases the bridge,
# so a gridless / fine-latent operator can out-resolve the voxel-FNO baseline). CPU job;
# ~30 s per FV solve, ~128 blocks -> ~70 min serial. See docs/block2_redesign.md.
#
#   sbatch scripts/slurm/gen_block2_hard.slurm
#SBATCH --job-name=tt3d-gen-hard
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

echo "host=$(hostname)  [$(date '+%T')] generating Block-2 HARD corpus"
srun "$ENV/bin/python" scripts/generate_3d_gt.py --hard \
  --n 96 --seed 1337 --grid 16 --npts 4096 --cells-per-layer 6 --name block2_hard_train
srun "$ENV/bin/python" scripts/generate_3d_gt.py --hard \
  --n 32 --seed 99   --grid 16 --npts 4096 --cells-per-layer 6 --name block2_hard_val
echo "[$(date '+%T')] HARD corpus generation complete."
