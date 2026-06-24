#!/bin/bash
# Profile GINO's per-step cost on the GPU to find the TRUE bottleneck (no optimisation).
# Faculty partition for immediate start:
#   sbatch --partition=feit-gpu-a100 --qos=feit --gres=gpu:A100:1 scripts/slurm/profile_gino.slurm
#SBATCH --job-name=tt3d-prof-gino
#SBATCH --account=punim2769
#SBATCH --partition=feit-gpu-a100
#SBATCH --qos=feit
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:25:00
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
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# Profile the data-only GINO (the campaign's headline model). A second pass on
# delta_gino confirms the same shape with the prior fed + added back.
for KIND in gino delta_gino; do
  echo "======== PROFILE kind=$KIND ========"
  srun "$ENV/bin/python" scripts/profile_gino.py \
    --kind "$KIND" --n-samples 8 --warmup 5 --iters 20 --device cuda
done
