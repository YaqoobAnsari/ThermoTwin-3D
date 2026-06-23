"""A plain convolutional baseline — the "no neural-operator" control.

`docs/baselines.md` calls for a data-driven CNN with no spectral/operator
machinery, to show the operator (and later the physics loss) earns its keep. This
is a fully convolutional, size-agnostic network: a lifting 1×1 conv, a stack of
residual 3×3 blocks with **growing dilation** (to enlarge the receptive field
without pooling, so it accepts any grid size like the FNO does), and a 1×1
projection. Same ``(B, C, H, W) -> (B, out, H, W)`` contract as the operators.
"""

from __future__ import annotations

import torch
from torch import nn

__all__ = ["build_cnn", "SimpleCNN"]


class _ResBlock(nn.Module):
    def __init__(self, channels: int, dilation: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.conv1(x))
        h = self.conv2(h)
        return self.act(x + h)


class SimpleCNN(nn.Module):
    """Conv-only encoder→decoder with dilation-grown receptive field."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        hidden_channels: int = 32,
        n_layers: int = 4,
    ):
        super().__init__()
        self.lift = nn.Conv2d(in_channels, hidden_channels, 1)
        # Dilations cycle 1,2,4,8,... to cover the wall without downsampling.
        self.blocks = nn.ModuleList(
            _ResBlock(hidden_channels, dilation=2**i) for i in range(n_layers)
        )
        self.project = nn.Conv2d(hidden_channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.lift(x)
        for block in self.blocks:
            h = block(h)
        return self.project(h)


def build_cnn(
    in_channels: int = 3,
    out_channels: int = 1,
    hidden_channels: int = 32,
    n_layers: int = 4,
) -> nn.Module:
    """Construct the convolutional baseline (see :class:`SimpleCNN`)."""
    return SimpleCNN(in_channels, out_channels, hidden_channels, n_layers)
