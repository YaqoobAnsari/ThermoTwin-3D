#!/bin/bash
# Block-2 operator bake-off on the REAL-geometry CityGML corpus (Exp 2.6) — the test that
# converts the synthetic delta_transolver win into a real-world one. Full roster x 3 seeds x
# 300 ep on whole-building shells at real orientations (train/val on disjoint buildings).
# Depends on the realcg corpus (chain with --dependency=afterok:<genjob>).
#
#   sbatch scripts/slurm/block2_realcg.slurm
# Writes results/block2_realcg_benchmark.{json,md}.
#
# NOTE: like the irregular corpus, the multi-normal real geometry breaks the world-frame
# U-from-indoor-face estimator, so **field rel-L2 is the metric to trust** (U-MAE is a
# frame artefact). See docs/experiments.md Exp 2.5/2.6.
#SBATCH --job-name=tt3d-block2-realcg
#SBATCH --account=punim2769
#SBATCH --partition=feit-gpu-a100
#SBATCH --qos=feit
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
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
echo "[$(date '+%T')] Block-2 REAL-GEOMETRY bake-off — full roster incl. Transolver"
echo "=================================================================="
srun "$ENV/bin/python" scripts/benchmark_block2.py --corpus realcg \
  --models gino delta_gino transolver delta_transolver fno_voxel prior_only \
  --epochs 300 --seeds 1337 1 2 --device cuda "$@"
echo "[$(date '+%T')] Block-2 REAL-GEOMETRY bake-off complete."
