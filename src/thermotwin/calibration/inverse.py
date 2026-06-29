"""Differentiable inverse twin — recover the conductivity / thermal-bridge field from an
observed temperature field by inverting a frozen differentiable forward operator.

The forward operator ``f`` maps a per-point property field (conductivity ``logk`` plus the
fixed nominal construction) to the dimensionless temperature field ``θ``. The inverse problem
is: given an observed ``θ_obs`` on known geometry with a known *nominal* clear-wall
construction, recover the conductivity field — i.e. **where and how strongly thermal bridges
depart from the clear wall** — and hence per-surface U / bridge conductance.

This is ill-posed (surface θ under-determines the interior field), so we (a) recover only the
conductivity departure from the known clear wall, (b) regularise toward a sparse, spatially
coherent bridge structure, and (c) quantify the residual non-uniqueness with an ensemble.

Three inverse routes, all here:
  * :func:`optimize_inverse`   — gradient descent on the conductivity field through the frozen
    forward (PDE-constrained / adjoint-style; the differentiable-twin core).
  * amortized inverse          — a learned ``θ → logk`` network (trained in the runner); this
    module supplies :func:`ensemble_inverse` and the metrics it is scored with.
  * hybrid                     — amortized init handed to :func:`optimize_inverse` to refine.

The forward is passed as a closure ``forward_fn(logk) -> θ_pred`` so this module is agnostic
to the operator's call signature (data-only vs delta, with/without a prior channel).
"""

from __future__ import annotations

from collections.abc import Callable

import torch

__all__ = [
    "optimize_inverse",
    "ensemble_inverse",
    "knn_edges",
    "bridge_localization",
    "uq_calibration",
]


def knn_edges(coords: torch.Tensor, k: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
    """``(i, j)`` index pairs of a symmetric kNN graph over ``coords`` ``(N, 3)``.

    Used for the spatial-coherence (total-variation) regulariser on an unordered point cloud.
    Computed once per sample and reused across optimisation steps.
    """
    with torch.no_grad():
        d = torch.cdist(coords, coords)
        d.fill_diagonal_(float("inf"))
        nn = d.topk(k, largest=False).indices  # (N, k)
        n = coords.shape[0]
        i = torch.arange(n, device=coords.device).unsqueeze(1).expand(-1, k).reshape(-1)
        j = nn.reshape(-1)
    return i, j


def optimize_inverse(
    forward_fn: Callable[[torch.Tensor], torch.Tensor],
    theta_obs: torch.Tensor,
    logk_init: torch.Tensor,
    clear_ref: torch.Tensor | float,
    *,
    n_steps: int = 300,
    lr: float = 5e-2,
    l1: float = 0.0,
    l2: float = 0.0,
    tv: float = 0.0,
    tv_edges: tuple[torch.Tensor, torch.Tensor] | None = None,
    return_history: bool = False,
):
    """Recover the conductivity field by minimising data-fit + regularisation.

    ``forward_fn(logk)`` must return the predicted θ for a candidate ``logk`` field ``(N,)`` or
    ``(1, N)`` (it builds the full feature tensor and runs the *frozen* operator). The loss is

        ‖f(logk) − θ_obs‖ / ‖θ_obs‖    (relative-L2 data fit)
        + l1 · mean|logk − clear_ref|              (bridges are *sparse* departures)
        + l2 · mean(logk − clear_ref)²             (stay near the known clear wall)
        + tv · mean|logk_i − logk_j| over kNN edges (spatially coherent bridges)

    Args:
        theta_obs: observed field, shape broadcastable to the forward output.
        logk_init: initial conductivity field (e.g. the clear-wall reference, i.e. no bridges).
        clear_ref: the known nominal clear-wall conductivity (scalar or per-point).
        tv_edges: ``(i, j)`` from :func:`knn_edges`; required iff ``tv > 0``.

    Returns:
        ``logk_hat`` ``(N,)`` (detached); plus a ``history`` dict if ``return_history``.
    """
    logk = logk_init.detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([logk], lr=lr)
    t_obs = theta_obs.reshape(-1)
    denom = torch.linalg.norm(t_obs).clamp_min(1e-8)
    hist = {"data": [], "total": []}
    for _ in range(n_steps):
        opt.zero_grad()
        pred = forward_fn(logk).reshape(-1)
        data = torch.linalg.norm(pred - t_obs) / denom
        loss = data
        if l1:
            loss = loss + l1 * (logk - clear_ref).abs().mean()
        if l2:
            loss = loss + l2 * ((logk - clear_ref) ** 2).mean()
        if tv and tv_edges is not None:
            i, j = tv_edges
            loss = loss + tv * (logk.reshape(-1)[i] - logk.reshape(-1)[j]).abs().mean()
        loss.backward()
        opt.step()
        if return_history:
            hist["data"].append(float(data.detach()))
            hist["total"].append(float(loss.detach()))
    out = logk.detach().reshape(-1)
    return (out, hist) if return_history else out


def ensemble_inverse(
    run_once: Callable[[int], torch.Tensor],
    n: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run an inverse ``n`` times (``run_once(member_idx) -> logk_hat (N,)``) for UQ.

    Returns ``(mean, std)`` over the ensemble — the posterior estimate and the per-point
    spread that quantifies residual non-uniqueness.
    """
    members = torch.stack([run_once(m).reshape(-1) for m in range(n)], dim=0)  # (n, N)
    return members.mean(0), members.std(0)


def bridge_localization(
    logk_hat: torch.Tensor,
    logk_true: torch.Tensor,
    clear_ref: torch.Tensor | float,
    *,
    margin: float = 0.25,
) -> dict:
    """Precision / recall / IoU of the recovered bridge map vs the true high-conductivity mask.

    A point is a *bridge* if its conductivity exceeds the clear-wall reference by ``margin``.
    Thresholding both the recovered and the true field at the same rule makes it a fair,
    geometry-defined localisation metric (not the ``|true−prior|`` field-defined mask).
    """
    pred_b = (logk_hat.reshape(-1) - clear_ref) > margin
    true_b = (logk_true.reshape(-1) - clear_ref) > margin
    tp = float((pred_b & true_b).sum())
    fp = float((pred_b & ~true_b).sum())
    fn = float((~pred_b & true_b).sum())
    union = tp + fp + fn
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    rec = tp / (tp + fn) if (tp + fn) else float("nan")
    return {
        "bridge_precision": prec,
        "bridge_recall": rec,
        "bridge_iou": (tp / union) if union else float("nan"),
        "bridge_frac_true": float(true_b.float().mean()),
        "bridge_frac_pred": float(pred_b.float().mean()),
    }


def uq_calibration(mean: torch.Tensor, std: torch.Tensor, true: torch.Tensor) -> dict:
    """Does the ensemble spread track the error? (coverage + correlation).

    Returns the fraction of points whose true value lies within ±1σ / ±2σ of the mean
    (well-calibrated ≈ 0.68 / 0.95) and the |error|–σ rank correlation (spread should be
    larger where the estimate is worse).
    """
    err = (mean.reshape(-1) - true.reshape(-1)).abs()
    s = std.reshape(-1).clamp_min(1e-8)
    cov1 = float((err <= s).float().mean())
    cov2 = float((err <= 2 * s).float().mean())
    e, ss = err - err.mean(), s - s.mean()
    corr = float((e * ss).mean() / (e.std().clamp_min(1e-8) * ss.std().clamp_min(1e-8)))
    return {"uq_cov_1sigma": cov1, "uq_cov_2sigma": cov2, "uq_err_std_corr": corr}
