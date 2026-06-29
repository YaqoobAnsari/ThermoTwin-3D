#!/bin/bash
# Boundary-layer A/B ablation on one corpus (the AiC system-paper method evidence).
# prior_only / delta (additive) / delta_input (window off, +d feature) / bl (window on) /
# blu (window + heteroscedastic UQ). Tests whether the SDF-keyed boundary-layer window beats
# plain additive residual learning, and whether the uncertainty head is calibrated.
# Writes results/bl_<corpus>.json.
#
#   sbatch scripts/slurm/bl.slurm <corpus>     # box irreg hard realcg realcg_lod3 bag doe
#
#SBATCH --job-name=tt3d-bl
#SBATCH --account=punim2769
#SBATCH --partition=feit-gpu-a100
#SBATCH --qos=feit
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=results/logs/%x-%j.out
#SBATCH --error=results/logs/%x-%j.err

set -euo pipefail
CORPUS=${1:?usage: sbatch bl.slurm <corpus>}
ENV=/data/gpfs/projects/punim2769/envs/thermotwin
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export PYTHONUNBUFFERED=1
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p results/logs

echo "host=$(hostname)  corpus=$CORPUS"
"$ENV/bin/python" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
echo "[$(date '+%T')] Boundary-layer A/B ablation — corpus=$CORPUS"
srun "$ENV/bin/python" scripts/benchmark_bl.py --corpus "$CORPUS" --epochs 200 --seeds 1337 1 2 --device cuda
echo "[$(date '+%T')] corpus=$CORPUS BL ablation complete."
