"""The differentiable PDE residual must match the FV solver's discretisation.

The decisive test: a sample solved by :mod:`thermotwin.data.synthetic_fem` at its
*native* resolution produces a temperature field that exactly satisfies the
discrete finite-volume steady equations. Mapped to dimensionless ``theta`` and fed
to :func:`heat_residual_loss`, the residual must therefore vanish to solver
tolerance. Perturbing the field must strictly raise it, and gradients must flow.
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
from thermotwin.losses.heat_residual import heat_residual, heat_residual_loss
from thermotwin.physics.conduction import Layer


def _bridged_sample() -> WallSample:
    """A layered wall with a steel-stud thermal bridge through the insulation."""
    layers = (
        Layer("stucco", 0.025, 0.72),
        Layer("concrete", 0.20, 1.95),
        Layer("eps_insulation", 0.10, 0.035),
        Layer("gypsum", 0.0127, 0.16),
    )
    # Bridge spans the insulation layer (x in [0.225, 0.325]) over part of the width.
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


def _gt_theta_inputs(dtype=torch.float64):
    """Solve a bridged sample and return (theta_gt, k, dx0, dy, r_si, r_se) tensors."""
    sample = _bridged_sample()
    field = solve_sample(sample)
    k, spacing = build_k_field(sample)

    t = field.temperature.astype(np.float64)
    theta = (t - sample.t_outdoor) / (sample.t_indoor - sample.t_outdoor)

    theta_t = torch.as_tensor(theta, dtype=dtype)
    k_t = torch.as_tensor(np.asarray(k, dtype=np.float64), dtype=dtype)
    dx0_t = torch.as_tensor(np.asarray(spacing[0], dtype=np.float64), dtype=dtype)
    dy_v = float(spacing[1])
    return theta_t, k_t, dx0_t, dy_v, sample.r_si, sample.r_se


def test_gt_field_has_zero_residual():
    """The native-resolution GT theta satisfies the discrete PDE to solver tol."""
    theta, k, dx0, dy, r_si, r_se = _gt_theta_inputs(dtype=torch.float64)
    loss = heat_residual_loss(theta, k, dx0, dy, r_si, r_se)
    assert float(loss) < 1e-6, f"GT residual loss too large: {float(loss):.3e}"

    # The per-cell residual itself must be uniformly tiny, not just on average.
    res = heat_residual(theta, k, dx0, dy, r_si, r_se)
    assert float(res.abs().max()) < 1e-4


def test_perturbation_increases_residual():
    """Any departure from the steady field strictly raises the residual."""
    theta, k, dx0, dy, r_si, r_se = _gt_theta_inputs(dtype=torch.float64)
    base = float(heat_residual_loss(theta, k, dx0, dy, r_si, r_se))

    torch.manual_seed(1337)
    perturbed = theta + 0.05 * torch.randn_like(theta)
    bumped = float(heat_residual_loss(perturbed, k, dx0, dy, r_si, r_se))
    assert bumped > base
    assert bumped > 1e-3  # a real, non-trivial increase


def test_gradient_flows():
    """Loss is differentiable wrt theta with finite, nonzero gradient."""
    theta, k, dx0, dy, r_si, r_se = _gt_theta_inputs(dtype=torch.float64)
    theta = (theta + 0.1 * torch.ones_like(theta)).clone().requires_grad_(True)

    loss = heat_residual_loss(theta, k, dx0, dy, r_si, r_se)
    loss.backward()

    assert theta.grad is not None
    assert torch.isfinite(theta.grad).all()
    assert float(theta.grad.abs().sum()) > 0.0


def test_shape_variants_agree():
    """Unbatched (Nx,Ny), batched (B,Nx,Ny) and channel (B,1,Nx,Ny) all match."""
    theta, k, dx0, dy, r_si, r_se = _gt_theta_inputs(dtype=torch.float64)

    loss_2d = heat_residual_loss(theta, k, dx0, dy, r_si, r_se)

    theta_b = theta.unsqueeze(0)
    k_b = k.unsqueeze(0)
    loss_3d = heat_residual_loss(theta_b, k_b, dx0, dy, r_si, r_se)

    theta_c = theta.unsqueeze(0).unsqueeze(0)
    k_c = k.unsqueeze(0).unsqueeze(0)
    loss_4d = heat_residual_loss(theta_c, k_c, dx0.unsqueeze(0), torch.tensor([dy]), r_si, r_se)

    assert torch.allclose(loss_2d, loss_3d)
    assert torch.allclose(loss_2d, loss_4d)


def test_batched_per_sample_params():
    """A batch with per-sample dx0/dy/films equals stacking single solves."""
    theta, k, dx0, dy, r_si, r_se = _gt_theta_inputs(dtype=torch.float64)
    theta_b = torch.stack([theta, theta], dim=0)
    k_b = torch.stack([k, k], dim=0)
    dx0_b = torch.stack([dx0, dx0], dim=0)
    dy_b = torch.tensor([dy, dy], dtype=torch.float64)
    r_si_b = torch.tensor([r_si, r_si], dtype=torch.float64)
    r_se_b = torch.tensor([r_se, r_se], dtype=torch.float64)

    res = heat_residual(theta_b, k_b, dx0_b, dy_b, r_si_b, r_se_b)
    assert res.shape == (2, theta.shape[0], theta.shape[1])
    assert float(res.abs().max()) < 1e-4
