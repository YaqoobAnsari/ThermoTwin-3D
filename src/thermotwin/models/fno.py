"""FNO backbone for the Block-1 synthetic benchmark (regular-grid operator).

Our synthetic corpus lives on a **regular grid**, so a Fourier Neural Operator is
the natural first operator — GINO's graph encode/decode only earns its keep on the
*irregular* point clouds of real scans (Block 2). We keep ``models/`` pluggable
(``docs/architecture.md``): FNO here, GINO and the rest slot in behind the same
``(B, C, H, W) -> (B, 1, H, W)`` contract.

Thin wrapper over ``neuralop.models.FNO`` so experiment configs stay declarative and
the rest of the codebase doesn't import the third-party API directly.
"""

from __future__ import annotations

from neuralop.models import FNO
from torch import nn

__all__ = ["build_fno"]


def build_fno(
    in_channels: int = 3,
    out_channels: int = 1,
    n_modes: tuple[int, int] = (8, 16),
    hidden_channels: int = 32,
    n_layers: int = 4,
) -> nn.Module:
    """Construct an FNO mapping input fields to the dimensionless temperature field.

    Args:
        in_channels: input field channels (default ``[log k, r_si, r_se]``).
        out_channels: output channels (1 = θ).
        n_modes: retained Fourier modes per axis (through-wall, along-wall).
        hidden_channels: width of the spectral-convolution channels.
        n_layers: number of Fourier layers.
    """
    return FNO(
        n_modes=n_modes,
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_channels=hidden_channels,
        n_layers=n_layers,
    )
