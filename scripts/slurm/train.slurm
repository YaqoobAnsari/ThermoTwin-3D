#!/bin/bash
# Block-1 GPU training on Spartan. Submit from the repo root, e.g.:
#   sbatch scripts/slurm/train.slurm experiment=block1_synthetic_fem
#   sbatch scripts/slurm/train.slurm experiment=block1_synthetic_fem train.epochs=300 train.batch_size=64
#SBATCH --job-name=tt3d-train
#SBATCH --account=punim2769
#SBATCH --partition=gpu-a100          # adjust to an available Spartan GPU partition
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=results/logs/%x-%j.out
#SBATCH --error=results/logs/%x-%j.err

set -euo pipefail

# Use the project-disk env's interpreter directly. We deliberately do NOT
# `conda activate` — conda's MKL activation script trips `set -u` (unbound
# MKL_INTERFACE_LAYER). PYTHONNOUSERSITE shields us from a stale ~/.local.
ENV=/data/gpfs/projects/punim2769/envs/thermotwin
export PYTHONNOUSERSITE=1

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p results/logs

echo "host=$(hostname)  cwd=$(pwd)"
"$ENV/bin/python" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

srun "$ENV/bin/python" scripts/train.py device=cuda "$@"
