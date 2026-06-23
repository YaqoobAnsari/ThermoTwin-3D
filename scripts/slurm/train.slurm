#!/bin/bash
#SBATCH --job-name=tt3d-train
#SBATCH --account=punim2769
#SBATCH --partition=gpu-a100          # adjust to an available Spartan GPU partition
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=results/logs/%x-%j.out
#SBATCH --error=results/logs/%x-%j.err

module purge
# module load Anaconda3/2024.02-1      # adjust to current Spartan module names
# source activate thermotwin

set -euo pipefail
srun python scripts/train.py "$@"
