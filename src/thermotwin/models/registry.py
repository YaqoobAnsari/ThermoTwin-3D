"""Model registry — build any baseline from its config behind one contract.

Every model maps ``(B, C, H, W) -> (B, out, H, W)``. The Block-1 benchmark compares
the FNO operator against a no-spectral CNN control. The vendored point-cloud
operators (GNOT, Transolver, MeshGraphNet, PointNet++) consume irregular geometry,
not grids — they enter as Block-2 competitors once the SDF / point-cloud
featuriser lands, so they are registered here as explicit "not yet wired" to fail
loudly rather than silently.
"""

from __future__ import annotations

from collections.abc import Mapping

from torch import nn

from .cnn import build_cnn
from .fno import build_fno

__all__ = ["build_model", "WIRED_MODELS", "DEFERRED_MODELS"]

WIRED_MODELS = ("fno", "cnn")
# Vendored under vendored/; wire when point-cloud featurisation exists (Block 2).
DEFERRED_MODELS = ("gino", "gnot", "transolver", "meshgraphnet", "deeponet", "pointnet2")


def build_model(model_cfg: Mapping) -> nn.Module:
    """Build a model from its config mapping (must contain ``name``)."""
    name = str(model_cfg["name"]).lower()
    in_ch = int(model_cfg.get("in_channels", 3))
    out_ch = int(model_cfg.get("out_channels", 1))
    hidden = int(model_cfg.get("hidden_channels", 32))
    n_layers = int(model_cfg.get("n_layers", 4))

    if name == "fno":
        return build_fno(
            in_channels=in_ch,
            out_channels=out_ch,
            n_modes=tuple(model_cfg.get("n_modes", (8, 16))),
            hidden_channels=hidden,
            n_layers=n_layers,
        )
    if name == "cnn":
        return build_cnn(
            in_channels=in_ch, out_channels=out_ch, hidden_channels=hidden, n_layers=n_layers
        )
    if name in DEFERRED_MODELS:
        raise NotImplementedError(
            f"model '{name}' is vendored but not yet wired — it consumes point clouds, "
            f"not grids (Block 2). Wired models: {WIRED_MODELS}."
        )
    raise KeyError(f"unknown model '{name}'. Wired: {WIRED_MODELS}; deferred: {DEFERRED_MODELS}.")
