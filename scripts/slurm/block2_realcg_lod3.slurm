#!/bin/bash
# Block-2 operator bake-off on the REAL-geometry **LoD3** CityGML corpus — the higher-fidelity
# real-geometry column (more planar faces / orientation diversity than LoD2). Full roster x 3
# seeds x 300 ep on coplanar-merged whole-building shells (train/val on disjoint buildings).
# Depends on the LoD3 corpus (chain with --dependency=afterok:<genjob>).
#
#   sbatch scripts/slurm/block2_realcg_lod3.slurm
# Writes results/block2_realcg_lod3_benchmark.{json,md}.
#
# NOTE: as on realcg/irregular, multi-normal real geometry breaks the world-frame U estimator,
# so **field rel-L2 + bridge corr-relL2 are the metrics to trust** (U-MAE is a frame artefact).
#SBATCH --job-name=tt3d-block2-realcg-lod3
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
echo "[$(date '+%T')] Block-2 REAL-GEOMETRY LoD3 bake-off — full roster incl. Transolver"
echo "=================================================================="
srun "$ENV/bin/python" scripts/benchmark_block2.py --corpus realcg_lod3 \
  --models gino delta_gino transolver delta_transolver fno_voxel prior_only \
  --epochs 300 --seeds 1337 1 2 --device cuda "$@"
echo "[$(date '+%T')] Block-2 REAL-GEOMETRY LoD3 bake-off complete."
