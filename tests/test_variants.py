"""DeltaFNO + U-FNO variants: contract, size-agnosticism, overfit, prior wiring.

These two models target U-MAE from opposite angles — DeltaFNO learns a correction on
the analytic 1-D clear-wall theta prior, U-FNO adds a local conv path to each
spectral block — but both must obey the strict ``(B, C, H, W) -> (B, 1, H, W)``
contract and stay size-agnostic across the along-wall widths W in {32, 48, 64}.
"""

from __future__ import annotations

import pytest
import torch

torch.manual_seed(1337)

from thermotwin.eval.metrics import relative_l2  # noqa: E402
from thermotwin.models.registry import WIRED_MODELS, build_model  # noqa: E402

DELTA_CFG = {
    "name": "delta_fno",
    "in_channels": 6,
    "out_channels": 1,
    "n_modes": [8, 16],
    "hidden_channels": 16,
    "n_layers": 2,
    "clearwall_index": 3,
}
UFNO_CFG = {
    "name": "ufno",
    "in_channels": 3,
    "out_channels": 1,
    "n_modes": [8, 16],
    "hidden_channels": 16,
    "n_layers": 2,
}
_VARIANTS = [DELTA_CFG, UFNO_CFG]


def _ids(cfg: dict) -> str:
    return cfg["name"]


def test_variants_are_registered():
    assert "delta_fno" in WIRED_MODELS
    assert "ufno" in WIRED_MODELS


@pytest.mark.parametrize("cfg", _VARIANTS, ids=_ids)
def test_contract_shapes(cfg):
    model = build_model(cfg)
    c = cfg["in_channels"]
    x = torch.randn(2, c, 24, 48)
    assert model(x).shape == (2, 1, 24, 48)


@pytest.mark.parametrize("cfg", _VARIANTS, ids=_ids)
@pytest.mark.parametrize("w", [32, 64])
def test_size_agnostic(cfg, w):
    """One model instance must accept every along-wall width without rebuild."""
    model = build_model(cfg)
    c = cfg["in_channels"]
    x = torch.randn(1, c, 24, w)
    assert model(x).shape == (1, 1, 24, w)


@pytest.mark.parametrize("cfg", _VARIANTS, ids=_ids)
def test_overfits_tiny_batch(cfg):
    torch.manual_seed(0)
    model = build_model(cfg)
    c = cfg["in_channels"]
    x = torch.randn(2, c, 24, 48)
    # A smooth, structured target (not white noise) so the residual the DeltaFNO must
    # fit is a learnable field — representative of the real smooth-bulk + sharp-bridge
    # theta, and well inside the spectral basis both operators rely on.
    yy, xx = torch.meshgrid(torch.linspace(0, 1, 24), torch.linspace(0, 1, 48), indexing="ij")
    y = (torch.sin(3 * xx) * torch.cos(2 * yy)).expand(2, 1, 24, 48).contiguous()
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    start = relative_l2(model(x), y).item()
    for _ in range(150):
        opt.zero_grad()
        loss = relative_l2(model(x), y)
        loss.backward()
        opt.step()
    end = relative_l2(model(x), y).item()
    assert end < 0.1 * start, f"{cfg['name']} failed to overfit: {start:.3f} -> {end:.3f}"


def test_delta_fno_adds_clearwall_prior():
    """Perturbing the clearwall channel must shift the output by exactly that delta.

    DeltaFNO returns ``x[:, idx] + fno(x)``. The fno also reads that channel, so to
    isolate the additive prior we hold the fno input fixed (a separate channel) and
    only swap what the prior reads. We do this by comparing the model output to the
    pure-fno output: their difference must equal the clearwall channel everywhere.
    """
    model = build_model(DELTA_CFG).eval()
    idx = DELTA_CFG["clearwall_index"]
    x = torch.randn(2, DELTA_CFG["in_channels"], 24, 48)
    with torch.no_grad():
        out = model(x)
        fno_only = model.fno(x)
    prior = x[:, idx : idx + 1]
    assert torch.allclose(out - fno_only, prior, atol=1e-5), (
        "DeltaFNO output minus the fno contribution must equal the clearwall prior"
    )


def test_delta_fno_prior_is_additive():
    """Changing only the clearwall channel changes the output by the same delta."""
    model = build_model(DELTA_CFG).eval()
    idx = DELTA_CFG["clearwall_index"]
    x = torch.randn(2, DELTA_CFG["in_channels"], 24, 48)
    bump = torch.full((2, 1, 24, 48), 0.3)
    x2 = x.clone()
    x2[:, idx : idx + 1] += bump
    with torch.no_grad():
        out1 = model(x)
        out2 = model(x2)
    # The fno also consumes the channel, so the change is NOT purely the bump; but the
    # prior term contributes exactly +bump on top of the fno's (nonzero) response.
    # Verify the prior path is live: zeroing the fno-side effect via the closed-form
    # decomposition, out2 - out1 must contain the bump from the prior plus the fno
    # delta — at minimum the prior must make the two outputs differ.
    assert not torch.allclose(out1, out2), "clearwall channel does not affect the output"
    with torch.no_grad():
        fno1 = model.fno(x)
        fno2 = model.fno(x2)
    # (out2 - out1) - (fno2 - fno1) isolates the additive prior delta == bump.
    prior_delta = (out2 - out1) - (fno2 - fno1)
    assert torch.allclose(prior_delta, bump, atol=1e-5)
