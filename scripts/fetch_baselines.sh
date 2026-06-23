#!/usr/bin/env bash
# Vendor baseline / reference implementations into vendored/ (git-ignored).
# Shallow clones (depth 1); per-repo failures are non-fatal. Records the exact
# checked-out commit of each repo into vendored/MANIFEST.txt for reproducibility.
#
# These are the methods the Block-1/3 comparison table runs against. None ship
# transferable weights for building heat conduction — we train them on our own
# FEM corpus — so this fetches CODE only.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENDOR="$REPO_ROOT/vendored"
MANIFEST="$VENDOR/MANIFEST.txt"
mkdir -p "$VENDOR"
: > "$MANIFEST"
{ echo "# Vendored baselines for ThermoTwin-3D"
  echo "# fetched: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "# format: <dir> <commit> <url>"; } >> "$MANIFEST"

# name|url|sparse_subdir(optional)
REPOS=(
  # --- SciML / geometry neural operators (the real competition) ---
  "neuraloperator|https://github.com/neuraloperator/neuraloperator|"   # GINO + FNO (also pip-installed)
  "Transolver|https://github.com/thuml/Transolver|"
  "GNOT|https://github.com/HaoZhongkai/GNOT|"
  "deepxde|https://github.com/lululxvi/deepxde|"                       # DeepONet (maintained)
  # --- mesh / graph PDE solvers ---
  "meshgraphnets|https://github.com/google-deepmind/deepmind-research|meshgraphnets"
  "modulus|https://github.com/NVIDIA/modulus|"                         # MeshGraphNet + operators (PyTorch)
  # --- pure data-driven controls (prove the physics loss earns its keep) ---
  "pytorch-3dunet|https://github.com/wolny/pytorch-3dunet|"
  "Pointnet_Pointnet2_pytorch|https://github.com/yanx27/Pointnet_Pointnet2_pytorch|"
  # --- building-domain ML (lumped/zone; show geometry-resolution wins) ---
  "PhysNet_Thermal_Models|https://github.com/GargyaGokhale/PhysNet_Thermal_Models|"  # Gokhale PINN
)

clone_one() {
  local name="$1" url="$2" sparse="$3"
  local dest="$VENDOR/$name"
  if [ -d "$dest/.git" ]; then
    echo "= $name already vendored, skipping (rm -rf to refresh)"
  elif [ -n "$sparse" ]; then
    echo "+ $name (sparse: $sparse) <- $url"
    git clone --depth 1 --filter=blob:none --sparse "$url" "$dest" \
      && git -C "$dest" sparse-checkout set "$sparse" \
      || { echo "  ! clone failed: $name"; return 1; }
  else
    echo "+ $name <- $url"
    git clone --depth 1 "$url" "$dest" || { echo "  ! clone failed: $name"; return 1; }
  fi
  local sha; sha="$(git -C "$dest" rev-parse HEAD 2>/dev/null || echo UNKNOWN)"
  echo "$name $sha $url" >> "$MANIFEST"
}

for spec in "${REPOS[@]}"; do
  IFS='|' read -r name url sparse <<< "$spec"
  clone_one "$name" "$url" "$sparse"
done

echo
echo "=== vendored manifest ==="
cat "$MANIFEST"
echo
echo "NOTE: two building-ML baselines have NO public implementation located:"
echo "  - Di Natale et al. PCNN (Physically Consistent NN, multi-zone) -> reimplement from paper"
echo "  - Yang et al. physics-constrained graph thermal dynamics       -> reimplement from paper"
echo "See docs/baselines.md."
