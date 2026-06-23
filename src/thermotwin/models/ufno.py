"""U-FNO — spectral operator with a parallel *local* convolution per block.

The plain FNO has two weaknesses that directly inflate U-MAE on our walls:

1. **Spectral bias.** The truncated Fourier basis smears sharp features — exactly
   the high-k bridge edges whose near-boundary gradient sets the U-value.
2. **Periodic FFT on non-periodic faces.** The axis-0 (through-wall) faces are
   Dirichlet/film, not periodic; the FFT's implicit wraparound contaminates the
   boundary the U-value is read from.

U-FNO (Wen et al., 2022) fixes the first by running a **local** path alongside the
global spectral path *inside every block* and summing them on the hidden
representation: ``h <- gelu( spectral(h) + local(h) ) + h``. The local Conv2d adds
boundary-aware, high-frequency capacity the FFT path lacks, while the spectral path
keeps the cheap global mixing. We build the block directly from neuralop's
``SpectralConv`` (rather than nesting whole FNOs) so the local and global paths
share one hidden state — the faithful U-FNO topology.

Every operation is a conv or an FFT (no flatten, no fixed Ny), so the model honours
the strict ``(B, C, H, W) -> (B, 1, H, W)`` contract and is size-agnostic for
H = 24, W in {32, 48, 64}.
"""

from __future__ import annotations

import torch
from neuralop.layers.spectral_convolution import SpectralConv
from torch import Tensor, nn
from torch.nn import functional as F

__all__ = ["UFNO", "build_ufno"]


class _UFNOBlock(nn.Module):
    """One U-FNO block: spectral (global) + local conv, summed, with a residual."""

    def __init__(
        self,
        channels: int,
        n_modes: tuple[int, int],
        local_kernel: int = 3,
    ) -> None:
        super().__init__()
        self.spectral = SpectralConv(channels, channels, tuple(n_modes))
        # 1x1 "W" path mirrors the pointwise channel-mixing FNO keeps next to its
        # spectral conv; the k>1 local conv is the U-FNO addition for sharp edges.
        self.pointwise = nn.Conv2d(channels, channels, 1)
        self.local = nn.Conv2d(channels, channels, local_kernel, padding=local_kernel // 2)

    def forward(self, h: Tensor) -> Tensor:
        out = self.spectral(h) + self.pointwise(h) + self.local(h)
        return F.gelu(out) + h


class UFNO(nn.Module):
    """FNO-style operator with a parallel local conv path in every block.

    Args:
        in_channels: input field channels.
        out_channels: output channels (1 = theta).
        n_modes: retained Fourier modes per axis (through-wall, along-wall).
        hidden_channels: width of the spectral / conv hidden representation.
        n_layers: number of U-FNO blocks.
        local_kernel: spatial kernel size of the local conv path (odd; padded to
            preserve grid size, hence size-agnostic).
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        n_modes: tuple[int, int] = (8, 16),
        hidden_channels: int = 32,
        n_layers: int = 4,
        local_kernel: int = 3,
    ) -> None:
        super().__init__()
        if local_kernel % 2 == 0:
            raise ValueError(f"local_kernel must be odd to preserve grid size, got {local_kernel}")
        # Lift: a normalized through-wall coord channel (applied once) then 1x1.
        self.lift = nn.Conv2d(in_channels + 1, hidden_channels, 1)
        self.blocks = nn.ModuleList(
            _UFNOBlock(hidden_channels, n_modes, local_kernel) for _ in range(n_layers)
        )
        self.proj = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, 1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, out_channels, 1),
        )

    @staticmethod
    def _coord_channel(x: Tensor) -> Tensor:
        """Normalized through-wall (axis-0/H) coordinate in [-1, 1] as a channel."""
        b, _, h, w = x.shape
        coord = torch.linspace(-1.0, 1.0, h, device=x.device, dtype=x.dtype)
        return coord.view(1, 1, h, 1).expand(b, 1, h, w)

    def forward(self, x: Tensor) -> Tensor:
        """Map input fields to theta; size-agnostic in H, W."""
        h = self.lift(torch.cat([x, self._coord_channel(x)], dim=1))
        for block in self.blocks:
            h = block(h)
        return self.proj(h)


def build_ufno(
    in_channels: int = 3,
    out_channels: int = 1,
    n_modes: tuple[int, int] = (8, 16),
    hidden_channels: int = 32,
    n_layers: int = 4,
    local_kernel: int = 3,
) -> nn.Module:
    """Construct a :class:`UFNO`. See the class for argument semantics."""
    return UFNO(
        in_channels=in_channels,
        out_channels=out_channels,
        n_modes=n_modes,
        hidden_channels=hidden_channels,
        n_layers=n_layers,
        local_kernel=local_kernel,
    )
