#!/bin/bash
# Block-2 operator bake-off on the EXISTING irregular (rotated / off-lattice) corpus, with
# the full roster INCLUDING the new gridless Transolver contender — a zero-new-data test of
# Transolver vs GINO vs voxel-FNO on the rotated-block geometry. Writes a NEW stem
# (block2_irreg_ops_benchmark) so the committed Exp-2.2 null artefact is preserved.
#
# NOTE: on rotated geometry the U-from-indoor-face estimator is in the wrong frame (world
# axis-0), so the U-MAE column is unreliable here (see docs/block2_redesign.md §diagnosis) —
# **field rel-L2 is the metric to trust** for the operator comparison.
#
#   sbatch scripts/slurm/block2_irreg_ops.slurm
#SBATCH --job-name=tt3d-block2-irreg-ops
#SBATCH --account=punim2769
#SBATCH --partition=feit-gpu-a100
#SBATCH --qos=feit
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
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
echo "[$(date '+%T')] Block-2 IRREGULAR bake-off — full roster incl. Transolver"
echo "=================================================================="
srun "$ENV/bin/python" scripts/benchmark_block2.py --corpus irreg \
  --out_stem block2_irreg_ops_benchmark \
  --models gino delta_gino transolver delta_transolver fno_voxel prior_only \
  --epochs 300 --seeds 1337 1 2 --device cuda "$@"
echo "[$(date '+%T')] Block-2 IRREGULAR bake-off complete."
