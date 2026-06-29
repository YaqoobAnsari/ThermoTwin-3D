#!/bin/bash
# Cross-PDE Phase-0 deconfound gate on the WHNO conduction-with-inclusions benchmark.
# A thermal bridge == a high-conductivity inclusion; the homogenised (uniform-k) solution is
# the analytic prior. Runs the SAME gate as Block-2 (data-only / cond / delta / delta_const +
# prior_only + predict_mean) across gridless backbones, on credible PUBLISHED field data.
# Decides whether the physics-prior recipe is necessary AND beats the prior where inclusions
# (bridges) are severe — the cross-PDE evidence the ICLR pitch needs.
#
#   sbatch scripts/slurm/whno_phase0.slurm
#
#SBATCH --job-name=tt3d-whno
#SBATCH --account=punim2769
#SBATCH --partition=feit-gpu-a100
#SBATCH --qos=feit
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
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
"$ENV/bin/python" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

echo "=================================================================="
echo "[$(date '+%T')] WHNO cross-PDE deconfound gate (3 backbones x {data,cond,delta,delta_const})"
echo "=================================================================="
srun "$ENV/bin/python" scripts/benchmark_whno.py \
  --n-train 1000 --n-val 200 --epochs 150 --seeds 1337 1 \
  --backbones pointnet2 transolver deeponet --device cuda
echo "[$(date '+%T')] WHNO gate complete."
