#!/usr/bin/env bash
# Build the ThermoTwin-3D conda env on PROJECT disk (home is near-full).
# Idempotent-ish: re-running will update the existing prefix.
set -uo pipefail

CONDA_ROOT=/apps/easybuild-2022/easybuild/software/Core/Anaconda3/2024.02-1
PROJECT=/data/gpfs/projects/punim2769
ENV_PREFIX="$PROJECT/envs/thermotwin"
REPO="$PROJECT/ThermoTwin-3D"

# Keep package cache off home.
export CONDA_PKGS_DIRS="$PROJECT/conda/pkgs"
mkdir -p "$CONDA_PKGS_DIRS" "$PROJECT/envs"

source "$CONDA_ROOT/etc/profile.d/conda.sh"

echo "=== [1/3] create env at $ENV_PREFIX (python 3.10 + pytorch cuda 12.1) ==="
conda create -y -p "$ENV_PREFIX" \
  -c pytorch -c nvidia -c conda-forge \
  python=3.10 pip numpy scipy pytorch pytorch-cuda=12.1 \
  || { echo "CONDA CREATE FAILED"; exit 1; }

# Use the env's interpreters directly — `conda activate` trips `set -u` on conda's
# own MKL activation script (unbound MKL_INTERFACE_LAYER), so we avoid it.
PY="$ENV_PREFIX/bin/python"
PIP="$ENV_PREFIX/bin/pip"
echo "python: $PY  ->  $($PY --version)"

echo "=== [2/3] pip install core requirements (open3d handled separately) ==="
# Install everything except open3d (heavy/fragile on HPC) so one bad wheel can't sink the env.
grep -vE '^\s*#|^\s*$|open3d' "$REPO/env/requirements.txt" > /tmp/tt_req_core.txt
"$PIP" install -r /tmp/tt_req_core.txt || { echo "PIP CORE FAILED"; exit 1; }

echo "=== [3/3] editable install of thermotwin + optional open3d ==="
"$PIP" install -e "$REPO" || echo "WARN: editable install failed (check pyproject)"
"$PIP" install open3d || echo "WARN: open3d failed — fine, geometry IO can fall back to trimesh/pyvista"

echo "=== DONE ==="
"$PY" -c "import torch; print('torch', torch.__version__, 'cuda_build', torch.version.cuda, 'avail', torch.cuda.is_available())"
"$PY" -c "import neuralop; print('neuraloperator OK', neuralop.__version__)" 2>&1 | tail -1
echo "ENV_PREFIX=$ENV_PREFIX"
