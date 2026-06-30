#!/bin/bash
# Inverse-twin benchmark on a Block-2 corpus: train a frozen differentiable forward surrogate,
# then recover the conductivity / thermal-bridge field from observed theta three ways
# (optimization / amortized / hybrid) with regularisation + ensemble-UQ variants. Scores field
# recovery, bridge localisation (P/R/IoU), U-MAE, and UQ calibration. Also runs the Exp-3
# identifiability study (observation masking + integration-scale U + per-point field dump).
# Writes results/inverse_<corpus>.json and results/inverse_fields_<corpus>.npz.
#
#   sbatch scripts/slurm/inverse.slurm <corpus>      # corpus: hard | realcg | realcg_lod3 | box
#
#SBATCH --job-name=tt3d-inverse
#SBATCH --account=punim2769
#SBATCH --partition=feit-gpu-a100
#SBATCH --qos=feit
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=results/logs/%x-%j.out
#SBATCH --error=results/logs/%x-%j.err

set -euo pipefail
CORPUS=${1:?usage: sbatch inverse.slurm <corpus>}
ENV=/data/gpfs/projects/punim2769/envs/thermotwin
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export PYTHONUNBUFFERED=1
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p results/logs

echo "host=$(hostname)  corpus=$CORPUS"
"$ENV/bin/python" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

echo "=================================================================="
echo "[$(date '+%T')] Inverse-twin benchmark — corpus=$CORPUS"
echo "=================================================================="
srun "$ENV/bin/python" scripts/benchmark_inverse.py --corpus "$CORPUS" \
  --fwd-epochs 200 --amort-epochs 200 --inv-steps 400 --ensemble 5 --device cuda
echo "[$(date '+%T')] corpus=$CORPUS inverse benchmark complete."
