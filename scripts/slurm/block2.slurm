#!/bin/bash
# Block-2 benchmark on Spartan GPU: GINO / delta_gino / fno_voxel on the 3-D corpus.
# Submit from the repo root:
#   sbatch scripts/slurm/block2.slurm
# Pass extra args through, e.g.:
#   sbatch scripts/slurm/block2.slurm --epochs 200
#SBATCH --job-name=tt3d-block2
#SBATCH --account=punim2769
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=results/logs/%x-%j.out
#SBATCH --error=results/logs/%x-%j.err

set -euo pipefail
ENV=/data/gpfs/projects/punim2769/envs/thermotwin
export PYTHONNOUSERSITE=1

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p results/logs

echo "host=$(hostname)"
"$ENV/bin/python" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

srun "$ENV/bin/python" scripts/benchmark_block2.py --epochs 150 --device cuda "$@"
