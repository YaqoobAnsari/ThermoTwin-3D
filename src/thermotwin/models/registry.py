"""Model registry — build any baseline from its config behind one contract.

The grid models map ``(B, C, H, W) -> (B, out, H, W)``. The Block-1 benchmark compares
the FNO operator against a no-spectral CNN control.

The geometry-conditioned operators (``gino``, ``delta_gino``) consume *irregular
point clouds*, not grids: their forward is
``(input_geom, x, latent_queries, sdf, output_queries) -> (B, n_out, 1)`` and cannot
honour the ``(B, C, H, W)`` grid contract. They are therefore built via
:func:`build_gino` / :func:`build_delta_gino` (exposed here and re-exported) and called
directly by the Block-2 runner. Routing them through :func:`build_model` raises with a
pointer to those builders, so the grid path stays clean and fails loudly rather than
silently mis-shaping a point-cloud model. The remaining vendored point-cloud operators
(GNOT, Transolver, MeshGraphNet, PointNet++) are likewise registered as "not yet wired".
"""

from __future__ import annotations

from collections.abc import Mapping

from torch import nn

from .cnn import build_cnn
from .delta_fno import build_delta_fno
from .fno import build_fno
from .gino import DeltaGino, GinoOperator, build_delta_gino, build_gino
from .transolver import (
    DeltaTransolver,
    TransolverOperator,
    build_delta_transolver,
    build_transolver,
)
from .ufno import build_ufno
from .unet import build_unet

__all__ = [
    "build_model",
    "build_gino",
    "build_delta_gino",
    "build_transolver",
    "build_delta_transolver",
    "GinoOperator",
    "DeltaGino",
    "TransolverOperator",
    "DeltaTransolver",
    "WIRED_MODELS",
    "GEOMETRY_MODELS",
    "DEFERRED_MODELS",
]

WIRED_MODELS = ("fno", "cnn", "unet", "delta_fno", "ufno")
# Geometry-conditioned operators built directly via their builders (point-cloud forward,
# not the (B,C,H,W) grid contract): gino/delta_gino (models/gino.py, latent-grid GNO+FNO)
# and transolver/delta_transolver (models/transolver.py, gridless physics-attention).
GEOMETRY_MODELS = ("gino", "delta_gino", "transolver", "delta_transolver")
# Vendored under vendored/; wire when point-cloud featurisation exists (Block 2+).
DEFERRED_MODELS = ("gnot", "transolver", "meshgraphnet", "deeponet", "pointnet2")


def build_model(model_cfg: Mapping) -> nn.Module:
    """Build a model from its config mapping (must contain ``name``)."""
    name = str(model_cfg["name"]).lower()
    in_ch = int(model_cfg.get("in_channels", 3))
    out_ch = int(model_cfg.get("out_channels", 1))
    hidden = int(model_cfg.get("hidden_channels", 32))
    n_layers = int(model_cfg.get("n_layers", 4))

    if name == "fno":
        domain_padding = model_cfg.get("domain_padding", None)
        if domain_padding is not None and not isinstance(domain_padding, (int, float)):
            domain_padding = list(domain_padding)
        return build_fno(
            in_channels=in_ch,
            out_channels=out_ch,
            n_modes=tuple(model_cfg.get("n_modes", (8, 16))),
            hidden_channels=hidden,
            n_layers=n_layers,
            domain_padding=domain_padding,
            positional_embedding=model_cfg.get("positional_embedding", "grid"),
        )
    if name == "delta_fno":
        domain_padding = model_cfg.get("domain_padding", None)
        if domain_padding is not None and not isinstance(domain_padding, (int, float)):
            domain_padding = list(domain_padding)
        return build_delta_fno(
            in_channels=in_ch,
            out_channels=out_ch,
            n_modes=tuple(model_cfg.get("n_modes", (8, 16))),
            hidden_channels=hidden,
            n_layers=n_layers,
            clearwall_index=int(model_cfg.get("clearwall_index", 3)),
            domain_padding=domain_padding,
            positional_embedding=model_cfg.get("positional_embedding", "grid"),
        )
    if name == "ufno":
        return build_ufno(
            in_channels=in_ch,
            out_channels=out_ch,
            n_modes=tuple(model_cfg.get("n_modes", (8, 16))),
            hidden_channels=hidden,
            n_layers=n_layers,
            local_kernel=int(model_cfg.get("local_kernel", 3)),
        )
    if name == "cnn":
        return build_cnn(
            in_channels=in_ch, out_channels=out_ch, hidden_channels=hidden, n_layers=n_layers
        )
    if name == "unet":
        return build_unet(
            in_channels=in_ch,
            out_channels=out_ch,
            base_channels=int(model_cfg.get("base_channels", 32)),
            depth=int(model_cfg.get("depth", 2)),
        )
    if name in GEOMETRY_MODELS:
        builder = {
            "gino": "build_gino",
            "delta_gino": "build_delta_gino",
            "transolver": "build_transolver",
            "delta_transolver": "build_delta_transolver",
        }[name]
        raise NotImplementedError(
            f"model '{name}' is a geometry-conditioned operator with a point-cloud "
            f"forward (input_geom, x, latent_queries, sdf, output_queries) -> (B, n_out, 1); "
            f"it does not fit build_model's (B, C, H, W) grid contract. Build it directly "
            f"with thermotwin.models.{builder} and call it from the Block-2 runner. "
            f"Grid models: {WIRED_MODELS}."
        )
    if name in DEFERRED_MODELS:
        raise NotImplementedError(
            f"model '{name}' is vendored but not yet wired — it consumes point clouds, "
            f"not grids (Block 2). Wired models: {WIRED_MODELS}."
        )
    raise KeyError(
        f"unknown model '{name}'. Grid: {WIRED_MODELS}; geometry: {GEOMETRY_MODELS}; "
        f"deferred: {DEFERRED_MODELS}."
    )
