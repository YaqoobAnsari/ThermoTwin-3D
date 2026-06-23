"""Field-accuracy metrics — the operator-learning side of the reporting pair.

The reporting rule (``docs/baselines.md``) is to always pair an ML metric with a
building metric. This module holds the ML side; building metrics (U-value /
heat-loss error) live with the calibration / eval pipeline as it lands.
"""

from __future__ import annotations

import torch

__all__ = ["relative_l2", "rmse"]


def relative_l2(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Mean relative L2 error ``||pred - target|| / ||target||`` over the batch.

    Norms are taken per sample over all non-batch dims, then averaged — the standard
    neural-operator metric.
    """
    b = pred.shape[0]
    p, t = pred.reshape(b, -1), target.reshape(b, -1)
    num = torch.linalg.norm(p - t, dim=1)
    den = torch.linalg.norm(t, dim=1).clamp_min(eps)
    return (num / den).mean()


def rmse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Root-mean-square error over all elements."""
    return torch.sqrt(torch.mean((pred - target) ** 2))
