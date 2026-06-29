#!/bin/bash
# Phase-0 decision gate — the experiment that decides H1-rescue vs pivot-to-H2.
# For four backbones spanning the architecture families (point / transformer / branch-trunk
# / graph-neural-operator), run the full ablation that DECONFOUNDS the delta-prior recipe:
#
#   <bb>             data-only      : 3-ch features (no prior), predict full field
#   cond_<bb>        prior-as-input : 4-ch features (incl theta1d), predict full field
#   delta_<bb>       the recipe     : 4-ch features + analytic prior added back (residual)
#   delta_const_<bb> non-physics    : delta wrapper, prior replaced by a constant
#
# plus prior_only (zero-param physics baseline) and predict_mean (trivial floor). Reading:
#   - cond ~= delta            => the residual structure adds nothing beyond prior-as-input
#   - delta_const ~= data-only => the *physics* of the prior is the workhorse, not the recipe
#   - everyone ~= prior_only   => H1 forward is a null; pivot to H2
# Writes results/block2_<corpus>_phase0.{json,md} (own stem; never clobbers committed runs).
#
#   sbatch scripts/slurm/block2_phase0.slurm <corpus>     # corpus: realcg | hard | doe ...
#
#SBATCH --job-name=tt3d-phase0
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
CORPUS=${1:?usage: sbatch block2_phase0.slurm <corpus>}
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
echo "[$(date '+%T')] Phase-0 deconfound ablation — corpus=$CORPUS"
echo "=================================================================="
srun "$ENV/bin/python" scripts/benchmark_block2.py --corpus "$CORPUS" \
  --out_stem "block2_${CORPUS}_phase0" \
  --models predict_mean prior_only \
           pointnet2 cond_pointnet2 delta_pointnet2 delta_const_pointnet2 \
           transolver cond_transolver delta_transolver delta_const_transolver \
           deeponet cond_deeponet delta_deeponet delta_const_deeponet \
           gino cond_gino delta_gino delta_const_gino \
  --epochs 300 --seeds 1337 1 2 --device cuda
echo "[$(date '+%T')] corpus=$CORPUS Phase-0 ablation complete."
