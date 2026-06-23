#!/bin/bash
#SBATCH --job-name=tt3d-femgt
#SBATCH --account=punim2769
#SBATCH --partition=cascade            # CPU partition for FEM solves; adjust as needed
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=results/logs/%x-%j.out
#SBATCH --error=results/logs/%x-%j.err

module purge
# source activate thermotwin

set -euo pipefail
srun python scripts/generate_fem_groundtruth.py "$@"
