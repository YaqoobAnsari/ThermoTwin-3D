"""The differentiable U-value loss must agree with the numpy building reporter.

The decisive checks: ``U`` computed from a field by the torch ``u_from_theta``
matches :func:`thermotwin.eval.building.effective_u_from_theta` on a real solved
sample to <1e-5; the loss is zero when ``pred == target``, strictly positive for a
perturbed prediction, and differentiable wrt the prediction (with the gradient
reaching only the indoor-face row, confirming U-MAE is a near-boundary quantity).
"""

from __future__ import annotations

import numpy as np
import torch

from thermotwin.data.synthetic_fem import (
    ThermalBridge,
    WallSample,
    build_k_field,
    solve_sample,
)
from thermotwin.eval.building import effective_u_from_theta
from thermotwin.losses.building_loss import u_from_theta, u_value_loss
from thermotwin.physics.conduction import Layer


def _bridged_sample() -> WallSample:
    layers = (
        Layer("stucco", 0.025, 0.72),
        Layer("concrete", 0.20, 1.95),
        Layer("eps_insulation", 0.10, 0.035),
        Layer("gypsum", 0.0127, 0.16),
    )
    bridge = ThermalBridge(x0=0.225, x1=0.325, y0=0.1, y1=0.2, conductivity_w_mk=50.0)
    return WallSample(
        layers=layers,
        width_m=0.6,
        t_indoor=21.0,
        t_outdoor=-5.0,
        r_si=0.13,
        r_se=0.04,
        bridges=(bridge,),
        lateral_cells=48,
    )


def _gt_inputs(dtype=torch.float64):
    """Solve a bridged sample -> (theta, k, dx0, dy, r_si) torch tensors + numpy field."""
    sample = _bridged_sample()
    field = solve_sample(sample)
    k, spacing = build_k_field(sample)
    theta_np = (field.temperature - sample.t_outdoor) / (sample.t_indoor - sample.t_outdoor)

    theta = torch.as_tensor(theta_np.astype(np.float64), dtype=dtype)
    k_t = torch.as_tensor(np.asarray(k, dtype=np.float64), dtype=dtype)
    dx0 = torch.as_tensor(np.asarray(spacing[0], dtype=np.float64), dtype=dtype)
    dy = float(spacing[1])
    return theta, k_t, dx0, dy, sample.r_si, field, k, spacing


def test_u_from_theta_matches_numpy():
    """Torch U on the GT field matches the numpy reporter (and the solver U)."""
    theta, k_t, dx0, dy, r_si, field, k, spacing = _gt_inputs()
    u_torch = float(u_from_theta(theta, k_t, dx0, dy, r_si)[0])
    u_numpy = effective_u_from_theta(theta.numpy(), np.asarray(k), spacing[0], spacing[1], r_si)
    assert abs(u_torch - u_numpy) < 1e-5
    assert abs(u_torch - float(field.u_value)) < 1e-5


def test_loss_zero_when_equal():
    theta, k_t, dx0, dy, r_si, *_ = _gt_inputs()
    loss = u_value_loss(theta, theta, k_t, dx0, dy, r_si)
    assert float(loss) == 0.0


def test_loss_positive_when_perturbed():
    theta, k_t, dx0, dy, r_si, *_ = _gt_inputs()
    torch.manual_seed(1337)
    # Perturb the indoor-face row so U actually changes.
    pred = theta.clone()
    pred[0, :] = pred[0, :] + 0.05
    loss = u_value_loss(pred, theta, k_t, dx0, dy, r_si)
    assert float(loss) > 0.0


def test_gradient_flows_only_on_indoor_face():
    """Loss is differentiable wrt the prediction; grad lives only on row 0."""
    theta, k_t, dx0, dy, r_si, *_ = _gt_inputs()
    pred = (theta + 0.05).clone().requires_grad_(True)
    loss = u_value_loss(pred, theta, k_t, dx0, dy, r_si)
    loss.backward()

    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()
    assert float(pred.grad[0, :].abs().sum()) > 0.0
    # U reads only the indoor face -> the rest of the field gets no gradient.
    assert float(pred.grad[1:, :].abs().sum()) == 0.0


def test_smooth_l1_variant_runs():
    theta, k_t, dx0, dy, r_si, *_ = _gt_inputs()
    pred = (theta + 0.05).clone()
    loss = u_value_loss(pred, theta, k_t, dx0, dy, r_si, smooth_l1=True)
    assert float(loss) > 0.0


def test_batched_matches_single():
    theta, k_t, dx0, dy, r_si, *_ = _gt_inputs()
    u_single = u_from_theta(theta, k_t, dx0, dy, r_si)
    u_batch = u_from_theta(
        torch.stack([theta, theta]),
        torch.stack([k_t, k_t]),
        torch.stack([dx0, dx0]),
        torch.tensor([dy, dy], dtype=torch.float64),
        torch.tensor([r_si, r_si], dtype=torch.float64),
    )
    assert torch.allclose(u_batch, u_single.expand(2))
