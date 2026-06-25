#!/bin/bash
# Block-2 operator bake-off on the HARD corpus (sub-voxel fins): the experiment that asks
# whether a gridless operator (Transolver) resolves the bridge a 16³ voxel-FNO and a
# 16³-latent GINO both alias. Full roster x 3 seeds x 300 ep. Depends on the HARD corpus
# (submit gen_block2_hard.slurm first, or chain with --dependency=afterok:<genjob>).
#
#   sbatch scripts/slurm/block2_hard.slurm
# Writes results/block2_hard_benchmark.{json,md}. See docs/block2_redesign.md.
#SBATCH --job-name=tt3d-block2-hard
#SBATCH --account=punim2769
#SBATCH --partition=feit-gpu-a100
#SBATCH --qos=feit
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=results/logs/%x-%j.out
#SBATCH --error=results/logs/%x-%j.err

set -euo pipefail
ENV=/data/gpfs/projects/punim2769/envs/thermotwin
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export PYTHONUNBUFFERED=1
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p results/logs

echo "host=$(hostname)"
"$ENV/bin/python" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

echo "=================================================================="
echo "[$(date '+%T')] Block-2 HARD bake-off — full roster incl. Transolver"
echo "=================================================================="
srun "$ENV/bin/python" scripts/benchmark_block2.py --corpus hard \
  --models gino delta_gino transolver delta_transolver fno_voxel prior_only \
  --epochs 300 --seeds 1337 1 2 --device cuda "$@"
echo "[$(date '+%T')] Block-2 HARD bake-off complete."
