"""Tests for the boundary-layer (matched-asymptotic) uncertainty-aware residual operator."""

from __future__ import annotations

import torch
from torch import nn

from thermotwin.models.boundary_layer import (
    BoundaryLayerOperator,
    boundary_window,
    build_bl_pointnet2,
    build_blu_pointnet2,
    heteroscedastic_loss,
    interface_distance,
)


class _DummyOp(nn.Module):
    """Deterministic stand-in operator: out channel c = c * sum(features). Lets us check the
    wrapper's window/prior math exactly (the real PointNet++ is stochastic)."""

    def __init__(self, out_channels: int) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.scale = nn.Parameter(torch.ones(1))  # a real param so grad-flow tests have a target

    def forward(self, input_geom, x, latent_queries, sdf, output_queries):
        base = x.sum(dim=-1, keepdim=True) * self.scale  # (b, n, 1)
        mult = torch.arange(1, self.out_channels + 1, dtype=x.dtype, device=x.device)
        return base * mult  # (b, n, out_channels)


def _toy(n=20, c=4):
    torch.manual_seed(0)
    ig = torch.rand(1, n, 3)
    feats = torch.rand(1, n, c)
    prior = torch.rand(1, n)
    d = torch.rand(n)
    return ig, feats, prior, d


# ---- interface distance -------------------------------------------------------------------

def test_interface_distance_locates_the_jump():
    # 6 colinear points; conductivity jumps between index 2 and 3.
    x = torch.tensor([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    pts = torch.stack([x, torch.zeros(6), torch.zeros(6)], dim=-1)
    logk = torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    d = interface_distance(pts, logk, jump=0.5, max_d=1.0)
    assert d.shape == (6,)
    # points adjacent to the interface are closest to it; interior-clear points are farther.
    assert d[2] < d[0]
    assert torch.allclose(d[2], torch.tensor(0.1), atol=1e-5)
    assert torch.allclose(d[3], torch.tensor(0.1), atol=1e-5)


def test_interface_distance_no_bridge_is_max():
    pts = torch.rand(10, 3)
    logk = torch.zeros(10)  # uniform conductivity -> no interface
    d = interface_distance(pts, logk, jump=0.5, max_d=1.0)
    assert torch.allclose(d, torch.ones(10))


# ---- window + loss ------------------------------------------------------------------------

def test_boundary_window_decays():
    eps = 0.1
    assert torch.allclose(boundary_window(torch.tensor(0.0), eps), torch.tensor(1.0))
    assert boundary_window(torch.tensor(1.0), eps) < 1e-6  # d >> eps -> ~0
    d = torch.tensor([0.0, 0.05, 0.1, 0.5])
    w = boundary_window(d, eps)
    assert torch.all(w[:-1] >= w[1:])  # monotonically non-increasing


def test_heteroscedastic_loss_values_and_grad():
    pred = torch.tensor([[1.0, 2.0]], requires_grad=True)
    target = torch.tensor([[1.0, 2.0]])
    log_var = torch.zeros(1, 2, requires_grad=True)
    # perfect prediction, s=0 -> loss 0
    assert torch.allclose(heteroscedastic_loss(pred, log_var, target), torch.tensor(0.0), atol=1e-6)
    # unit error, s=0 -> 0.5
    loss = heteroscedastic_loss(pred + 1.0, log_var, target)
    assert torch.allclose(loss, torch.tensor(0.5), atol=1e-6)
    loss.backward()
    assert log_var.grad is not None


# ---- wrapper math (deterministic dummy) ---------------------------------------------------

def test_window_off_is_additive_residual():
    ig, feats, prior, d = _toy()
    op = BoundaryLayerOperator(_DummyOp(1), uncertainty=False, window=False)
    pred = op(ig, feats, None, None, ig, prior, d)
    # corr = sum([feats, d]) ; with window off, pred = prior + corr exactly.
    x = torch.cat([feats, d.unsqueeze(0).unsqueeze(-1)], dim=-1)
    corr = x.sum(dim=-1, keepdim=True)
    assert torch.allclose(pred, prior.reshape(corr.shape) + corr, atol=1e-6)


def test_window_on_gates_the_correction():
    ig, feats, prior, d = _toy()
    op = BoundaryLayerOperator(_DummyOp(1), uncertainty=False, eps_init=0.1, window=True)
    pred = op(ig, feats, None, None, ig, prior, d)
    x = torch.cat([feats, d.unsqueeze(0).unsqueeze(-1)], dim=-1)
    corr = x.sum(dim=-1, keepdim=True)
    w = boundary_window(d.unsqueeze(0).unsqueeze(-1), op.eps)
    assert torch.allclose(pred, prior.reshape(corr.shape) + w * corr, atol=1e-5)
    # the window genuinely changes the output vs additive (some d are far from 0).
    assert not torch.allclose(pred, prior.reshape(corr.shape) + corr, atol=1e-3)


def test_uncertainty_head_returns_logvar():
    ig, feats, prior, d = _toy()
    op = BoundaryLayerOperator(_DummyOp(2), uncertainty=True, window=True)
    out = op(ig, feats, None, None, ig, prior, d)
    assert isinstance(out, tuple) and len(out) == 2
    pred, log_var = out
    assert pred.shape == (1, feats.shape[1], 1)
    assert log_var.shape == (1, feats.shape[1], 1)
    assert torch.isfinite(log_var).all()


def test_eps_and_uncertainty_receive_gradient():
    ig, feats, prior, d = _toy()
    op = BoundaryLayerOperator(_DummyOp(2), uncertainty=True, window=True)
    pred, log_var = op(ig, feats, None, None, ig, prior, d)
    target = torch.rand_like(pred)
    heteroscedastic_loss(pred, log_var, target).backward()
    assert op.raw_eps.grad is not None and torch.isfinite(op.raw_eps.grad)  # ε is learned
    assert op.operator.scale.grad is not None


# ---- real builders (shape smoke; PointNet++ is stochastic so seed) -------------------------

def test_build_bl_pointnet2_shapes():
    torch.manual_seed(1)
    ig, feats, prior, d = _toy(n=64, c=4)
    model = build_bl_pointnet2(feat_channels=4, width=32)
    pred = model(ig, feats, None, None, ig, prior, d)
    assert pred.shape == (1, 64, 1)


def test_build_blu_pointnet2_shapes_and_loss():
    torch.manual_seed(1)
    ig, feats, prior, d = _toy(n=64, c=4)
    model = build_blu_pointnet2(feat_channels=4, width=32)
    pred, log_var = model(ig, feats, None, None, ig, prior, d)
    assert pred.shape == (1, 64, 1) and log_var.shape == (1, 64, 1)
    loss = heteroscedastic_loss(pred, log_var, torch.rand_like(pred))
    loss.backward()
    assert model.raw_eps.grad is not None
