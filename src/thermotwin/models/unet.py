"""A 2D U-Net baseline — the multiscale-encoder/decoder control for Block 1.

The FNO mixes scales spectrally and the CNN grows its receptive field by
dilation; a U-Net instead pools to a coarse latent and reconstructs with skip
connections, which is the standard image-to-image workhorse and a useful third
data point for the benchmark. It keeps the same ``(B, C, H, W) -> (B, out, H, W)``
contract as the operators.

The Block-1 grids are awkward for pooling: ``H = 24`` and ``W in {32, 48, 64}``
are not powers of two, so naive ``2x`` downsampling loses the odd halves. We
handle arbitrary sizes by recording each level's input size, padding feature maps
up to an even size (replicate padding) before every stride-2 downsample, and
cropping the upsampled maps back to the recorded size before each skip
concatenation. Activations are GELU throughout, to match the other baselines.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

__all__ = ["build_unet", "UNet"]


class _DoubleConv(nn.Module):
    """Two 3x3 convolutions (padding-preserving) with GELU after each."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


def _pad_to_even(x: torch.Tensor) -> torch.Tensor:
    """Replicate-pad the last two dims up to even sizes (0 or 1 each)."""
    h, w = x.shape[-2], x.shape[-1]
    pad_h, pad_w = h % 2, w % 2
    if pad_h or pad_w:
        # F.pad order is (left, right, top, bottom).
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
    return x


def _crop_to(x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """Crop the last two dims of ``x`` down to ``size`` (centred at top-left)."""
    return x[..., : size[0], : size[1]]


class UNet(nn.Module):
    """Size-agnostic encoder/decoder U-Net with skip connections."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        base_channels: int = 32,
        depth: int = 2,
    ):
        super().__init__()
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")
        self.depth = depth

        # Encoder: a DoubleConv per level; channels double each level.
        self.encoders = nn.ModuleList()
        ch = in_channels
        enc_channels: list[int] = []
        for level in range(depth):
            out = base_channels * (2**level)
            self.encoders.append(_DoubleConv(ch, out))
            enc_channels.append(out)
            ch = out
        self.pool = nn.MaxPool2d(2)

        # Bottleneck at the coarsest resolution.
        bottleneck_ch = base_channels * (2**depth)
        self.bottleneck = _DoubleConv(ch, bottleneck_ch)

        # Decoder: transpose-conv up, concat skip, DoubleConv down.
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        ch = bottleneck_ch
        for level in reversed(range(depth)):
            skip_ch = enc_channels[level]
            self.upconvs.append(nn.ConvTranspose2d(ch, skip_ch, 2, stride=2))
            self.decoders.append(_DoubleConv(skip_ch * 2, skip_ch))
            ch = skip_ch

        self.head = nn.Conv2d(ch, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips: list[torch.Tensor] = []
        sizes: list[tuple[int, int]] = []

        h = x
        for encoder in self.encoders:
            h = encoder(h)
            skips.append(h)
            sizes.append((h.shape[-2], h.shape[-1]))
            h = self.pool(_pad_to_even(h))

        h = self.bottleneck(h)

        for upconv, decoder in zip(self.upconvs, self.decoders, strict=True):
            skip = skips.pop()
            size = sizes.pop()
            h = upconv(h)
            h = _crop_to(h, size)
            h = torch.cat([h, skip], dim=1)
            h = decoder(h)

        return self.head(h)


def build_unet(
    in_channels: int = 3,
    out_channels: int = 1,
    base_channels: int = 32,
    depth: int = 2,
) -> nn.Module:
    """Construct the U-Net baseline (see :class:`UNet`)."""
    return UNet(in_channels, out_channels, base_channels, depth)
