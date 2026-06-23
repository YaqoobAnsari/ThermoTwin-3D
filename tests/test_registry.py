"""Model registry + CNN baseline: contract, size-agnosticism, overfit, errors."""

from __future__ import annotations

import pytest
import torch

torch.manual_seed(0)

from thermotwin.eval.metrics import relative_l2  # noqa: E402
from thermotwin.models.registry import DEFERRED_MODELS, build_model  # noqa: E402

FNO_CFG = {"name": "fno", "in_channels": 3, "out_channels": 1, "n_modes": [8, 16],
           "hidden_channels": 16, "n_layers": 2}
CNN_CFG = {"name": "cnn", "in_channels": 3, "out_channels": 1, "hidden_channels": 16, "n_layers": 4}


@pytest.mark.parametrize("cfg", [FNO_CFG, CNN_CFG])
def test_contract_shapes(cfg):
    model = build_model(cfg)
    x = torch.randn(2, 3, 24, 48)
    assert model(x).shape == (2, 1, 24, 48)


@pytest.mark.parametrize("hw", [(24, 32), (24, 64), (24, 48)])
def test_cnn_is_size_agnostic(hw):
    """The conv baseline must accept arbitrary grid sizes, like the FNO."""
    model = build_model(CNN_CFG)
    x = torch.randn(1, 3, *hw)
    assert model(x).shape == (1, 1, *hw)


def test_cnn_overfits_one_sample():
    model = build_model(CNN_CFG)
    x = torch.randn(1, 3, 24, 48)
    y = torch.rand(1, 1, 24, 48)
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    start = relative_l2(model(x), y).item()
    for _ in range(150):
        opt.zero_grad()
        loss = relative_l2(model(x), y)
        loss.backward()
        opt.step()
    assert relative_l2(model(x), y).item() < 0.1 * start


def test_deferred_models_raise():
    for name in DEFERRED_MODELS:
        with pytest.raises(NotImplementedError):
            build_model({"name": name})


def test_unknown_model_raises():
    with pytest.raises(KeyError):
        build_model({"name": "definitely-not-a-model"})
