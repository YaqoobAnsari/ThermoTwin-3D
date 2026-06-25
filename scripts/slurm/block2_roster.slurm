#!/bin/bash
# Full-roster Block-2 bake-off for ONE corpus — the expanded competitor field.
# 14 models: our delta_transolver + transolver, GNOT (gnot/delta_gnot), GINO (gino/delta_gino),
# DeepONet (deeponet/delta_deeponet), PointNet++ (pointnet2/delta_pointnet2),
# MeshGraphNet (meshgraphnet/delta_meshgraphnet), the fno_voxel grid reference, and the
# prior_only control. Covers the four major architecture families (operator-transformer,
# GNN, point-based, branch/trunk) so the "is there a great model we missed?" question is
# answered, plus the expanded metric suite (physical-K RMSE, bridge-region rel-L2, tail, params).
#
#   sbatch scripts/slurm/block2_roster.slurm <corpus> [out_stem]
#     <corpus>   one of: box irreg hard realcg realcg_lod3 bag doe
#     [out_stem] optional results stem override (e.g. block2_irreg_ops_benchmark)
#
#SBATCH --job-name=tt3d-roster
#SBATCH --account=punim2769
#SBATCH --partition=feit-gpu-a100
#SBATCH --qos=feit
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --output=results/logs/%x-%j.out
#SBATCH --error=results/logs/%x-%j.err

set -euo pipefail
CORPUS=${1:?usage: sbatch block2_roster.slurm <corpus> [out_stem]}
OUT_STEM=${2:-}
ENV=/data/gpfs/projects/punim2769/envs/thermotwin
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export PYTHONUNBUFFERED=1
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p results/logs

echo "host=$(hostname)  corpus=$CORPUS  out_stem=${OUT_STEM:-<default>}"
"$ENV/bin/python" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

STEM_ARG=()
[ -n "$OUT_STEM" ] && STEM_ARG=(--out_stem "$OUT_STEM")

echo "=================================================================="
echo "[$(date '+%T')] Block-2 FULL-ROSTER (14 models) bake-off — corpus=$CORPUS"
echo "=================================================================="
srun "$ENV/bin/python" scripts/benchmark_block2.py --corpus "$CORPUS" "${STEM_ARG[@]}" \
  --models gino delta_gino transolver delta_transolver gnot delta_gnot \
           deeponet delta_deeponet pointnet2 delta_pointnet2 \
           meshgraphnet delta_meshgraphnet fno_voxel prior_only \
  --epochs 300 --seeds 1337 1 2 --device cuda
echo "[$(date '+%T')] corpus=$CORPUS full-roster bake-off complete."
