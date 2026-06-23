#!/bin/bash
# Block-1 ablation sweep on Spartan GPU. Submit from the repo root:
#   sbatch scripts/slurm/ablate.slurm
#SBATCH --job-name=tt3d-ablate
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

srun "$ENV/bin/python" scripts/ablate.py --epochs 300 --device cuda "$@"
