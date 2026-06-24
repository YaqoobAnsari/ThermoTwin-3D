#!/bin/bash
# Block-2 benchmark on Spartan GPU: GINO / delta_gino / fno_voxel on the 3-D corpus.
# Submit from the repo root:
#   sbatch scripts/slurm/block2.slurm
# Pass extra args through, e.g.:
#   sbatch scripts/slurm/block2.slurm --epochs 200
# H100 via the public-GPU QoS (account punim2769 holds publicgpu + feit). If gpu-h100
# is queued, the immediate-start fallback is our faculty partition:
#   sbatch --partition=feit-gpu-a100 --qos=feit scripts/slurm/block2.slurm
#SBATCH --job-name=tt3d-block2
#SBATCH --account=punim2769
#SBATCH --partition=gpu-h100
#SBATCH --qos=publicgpu
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=results/logs/%x-%j.out
#SBATCH --error=results/logs/%x-%j.err

set -euo pipefail
ENV=/data/gpfs/projects/punim2769/envs/thermotwin
export PYTHONNOUSERSITE=1
# GINO's GNO neighbour search runs in native torch on the GPU (gno_use_open3d=false);
# the CPU threads cover data loading and any host-side ops. Match cpus-per-task.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
# Force GINO to flush per-line so the leaderboard progress is visible while running.
export PYTHONUNBUFFERED=1

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p results/logs

echo "host=$(hostname)"
"$ENV/bin/python" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

srun "$ENV/bin/python" scripts/benchmark_block2.py --epochs 150 --device cuda "$@"
