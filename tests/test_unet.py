"""U-Net baseline: shape contract, size-agnosticism, and an overfit sanity check.

The U-Net pools to a coarse latent, so the awkward Block-1 grids (``H = 24``,
``W in {32, 48, 64}`` — not powers of two) are the interesting case: the pad-up /
crop-down bookkeeping must round-trip every size exactly. The overfit test mirrors
the FNO/CNN milestone — if it can drive relative-L2 on one fixed pair well down,
the model→loss plumbing is sound.
"""

from __future__ import annotations

import pytest
import torch

torch.manual_seed(1337)

from thermotwin.eval.metrics import relative_l2  # noqa: E402
from thermotwin.models.unet import build_unet  # noqa: E402


def test_shape_contract():
    model = build_unet()
    out = model(torch.randn(2, 3, 24, 48))
    assert out.shape == (2, 1, 24, 48)


@pytest.mark.parametrize("width", [32, 48, 64])
def test_size_agnostic(width: int):
    model = build_unet()
    out = model(torch.randn(1, 3, 24, width))
    assert out.shape == (1, 1, 24, width)


def test_overfits_single_pair():
    torch.manual_seed(0)
    x = torch.randn(1, 3, 24, 48)
    y = torch.randn(1, 1, 24, 48)

    model = build_unet()
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)

    start = relative_l2(model(x), y).item()
    for _ in range(150):
        opt.zero_grad()
        loss = relative_l2(model(x), y)
        loss.backward()
        opt.step()
    end = relative_l2(model(x), y).item()

    assert end < 0.1 * start, f"failed to overfit: {start:.3f} -> {end:.3f}"
