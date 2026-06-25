#!/bin/bash
# Block-2 operator bake-off on the DOE real-geometry corpus — the second real-geometry
# source (after TUM2TWIN realcg), Reference Buildings at scale. Full roster x 3 seeds x 300 ep.
# Depends on the bag corpus (chain with --dependency=afterok:<genjob>). Writes
# results/block2_bag_benchmark.{json,md}; folds into scripts/unified_eval.py automatically.
#
# NOTE: multi-normal real geometry -> the indoor-face U estimator is a frame artefact;
# **field rel-L2 + bridge-focused correction are the metrics to trust** (cf. realcg).
#SBATCH --job-name=tt3d-block2-doe
#SBATCH --account=punim2769
#SBATCH --partition=feit-gpu-a100
#SBATCH --qos=feit
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=results/logs/%x-%j.out
#SBATCH --error=results/logs/%x-%j.err

set -uo pipefail
ENV=/data/gpfs/projects/punim2769/envs/thermotwin
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export PYTHONUNBUFFERED=1
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p results/logs

echo "host=$(hostname)"
"$ENV/bin/python" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

echo "[$(date '+%T')] Block-2 3D-BAG real-geometry bake-off — full roster incl. Transolver"
srun "$ENV/bin/python" scripts/benchmark_block2.py --corpus doe \
  --models gino delta_gino transolver delta_transolver fno_voxel prior_only \
  --epochs 300 --seeds 1337 1 2 --device cuda "$@"
echo "[$(date '+%T')] Block-2 3D-BAG bake-off complete."
