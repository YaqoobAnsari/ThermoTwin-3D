#!/bin/bash
# Block-2 FULL benchmark on Spartan GPU: GINO / delta_gino / fno_voxel, full config
# (epochs 300, seeds 1337/1/2, all 3 models) run on BOTH 3-D corpora in one job —
# the axis-aligned box corpus and the rotated / off-lattice irregular corpus.
#
# Submit from the repo root, immediate start on our faculty A100 partition:
#   sbatch --partition=feit-gpu-a100 --qos=feit --gres=gpu:A100:1 scripts/slurm/block2_full.slurm
# H100 (faster per-epoch, but queues on Resources):
#   sbatch --partition=gpu-h100 --qos=publicgpu --gres=gpu:H100:1 scripts/slurm/block2_full.slurm
#
# Extra args are appended to BOTH benchmark invocations (argparse last-wins), e.g.
#   sbatch ... scripts/slurm/block2_full.slurm --epochs 150
#SBATCH --job-name=tt3d-block2-full
#SBATCH --account=punim2769
#SBATCH --partition=feit-gpu-a100
#SBATCH --qos=feit
#SBATCH --gres=gpu:A100:1
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
# Flush per-line so the leaderboard progress is visible while running.
export PYTHONUNBUFFERED=1

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p results/logs

echo "host=$(hostname)"
"$ENV/bin/python" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

FULL=(--epochs 300 --seeds 1337 1 2 --device cuda)

echo "=================================================================="
echo "[$(date '+%T')] Block-2 FULL — BOX corpus (block2_train/val)"
echo "=================================================================="
srun "$ENV/bin/python" scripts/benchmark_block2.py --corpus box "${FULL[@]}" "$@"

echo "=================================================================="
echo "[$(date '+%T')] Block-2 FULL — IRREGULAR corpus (block2_irreg_train/val)"
echo "=================================================================="
srun "$ENV/bin/python" scripts/benchmark_block2.py --corpus irreg "${FULL[@]}" "$@"

echo "[$(date '+%T')] Block-2 FULL benchmark complete."
