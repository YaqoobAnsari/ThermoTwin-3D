"""Bridge-focused metrics — does the operator's *correction* earn its keep?

Exp 2.6 exposed the problem: on a whole-building cloud the thermal-bridge residual is a
small fraction of the field, so a **global** field rel-L2 is dominated by clear-wall points
where the analytic 1-D prior is already exact — and the operator, which exists to predict the
*localized bridge correction the prior misses*, looks no better (or worse) than the
zero-parameter prior. The global metric washes out exactly the signal the operator is for.

This module scores the operator on the **correction** instead of the field, and **where the
bridges actually are**. The central quantity is the *correction relative-L2*:

    correction_rel_l2 = ‖θ_pred − θ_true‖ / ‖θ_prior − θ_true‖.

Because the denominator is the prior's own error (the size of the true correction), the
zero-network ``prior_only`` (whose θ_pred = θ_prior) scores **exactly 1.0** by construction.
So **< 1 means the operator genuinely beats the prior**, and = 0 means it recovers the
correction perfectly — a leakage-free, model-agnostic "earns its keep" number that a global
field rel-L2 cannot give.

The **bridge region** is defined from the ground truth only (identical for every model, so the
comparison is fair): the points where the true field departs from the 1-D prior by more than a
threshold τ in dimensionless θ, ``|θ_true − θ_prior| > τ`` — i.e. the points the bridge's
lateral spreading actually perturbs. We report the correction skill over the whole cloud and,
**focused**, over the bridge region at a sweep of thresholds, plus the clear-region error (a
guard that the operator is not corrupting the easy clear-wall majority) and correction R² /
correlation (does it put the correction in the right places?).
"""

from __future__ import annotations

import numpy as np

__all__ = ["bridge_focused_metrics", "DEFAULT_THRESHOLDS"]

# Bridge-region thresholds in dimensionless θ (θ ∈ [0,1]); a sweep from "any departure"
# to "strong bridge core". 0.02 is the headline focus.
DEFAULT_THRESHOLDS = (0.01, 0.02, 0.05)


def _norm(a: np.ndarray) -> float:
    return float(np.sqrt(np.sum(a * a)))


def bridge_focused_metrics(
    pred: np.ndarray,
    true: np.ndarray,
    prior: np.ndarray,
    thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS,
    eps: float = 1e-8,
) -> dict[str, float]:
    """Bridge-focused correction metrics for one prediction (or a concatenated corpus).

    Args:
        pred: predicted dimensionless θ at each point, shape ``(N,)``.
        true: ground-truth θ at each point, ``(N,)``.
        prior: analytic 1-D clear-wall θ at each point, ``(N,)`` (the same prior the
            ``delta_*`` operators add back; for ``prior_only`` ``pred == prior``).
        thresholds: bridge-region thresholds τ on ``|true − prior|`` (dimensionless θ).
        eps: denominator floor.

    Returns:
        Flat dict. Key entries:

        * ``correction_rel_l2`` — global ‖pred−true‖ / ‖prior−true‖. **< 1 beats the prior**;
          ``prior_only`` ≡ 1.0.
        * ``correction_r2`` / ``correction_corr`` — R² and Pearson corr of the predicted
          correction (pred−prior) vs the true correction (true−prior).
        * ``true_correction_rms`` / ``true_correction_max`` — size of the signal (context).
        * per τ: ``bridge_frac_t{τ}`` (region size), ``bridge_corr_rel_l2_t{τ}`` (correction
          rel-L2 *on the bridge region* — the focused "beats the prior where it matters"
          number), ``bridge_field_rel_l2_t{τ}`` (plain field rel-L2 on the region).
        * ``clear_field_rel_l2`` — field rel-L2 where the prior is ~exact (guard: should stay
          tiny; large ⇒ the operator is corrupting clear walls).
    """
    pred = np.asarray(pred, dtype=np.float64).ravel()
    true = np.asarray(true, dtype=np.float64).ravel()
    prior = np.asarray(prior, dtype=np.float64).ravel()
    if not (pred.shape == true.shape == prior.shape):
        raise ValueError(
            f"pred/true/prior must share shape; got {pred.shape}, {true.shape}, {prior.shape}"
        )

    err = pred - true  # operator error
    r_true = true - prior  # the true bridge correction (what the operator must learn)
    r_pred = pred - prior  # the predicted correction

    out: dict[str, float] = {}
    nrt = _norm(r_true)
    out["correction_rel_l2"] = _norm(err) / (nrt + eps)
    # R² of the predicted correction vs the true correction (note r_pred − r_true == err).
    denom_var = float(np.sum((r_true - r_true.mean()) ** 2))
    out["correction_r2"] = 1.0 - float(np.sum(err**2)) / (denom_var + eps)
    # Pearson correlation of the two correction fields (placement, scale-free).
    if r_true.std() > eps and r_pred.std() > eps:
        out["correction_corr"] = float(np.corrcoef(r_pred, r_true)[0, 1])
    else:
        out["correction_corr"] = 0.0
    out["true_correction_rms"] = float(np.sqrt(np.mean(r_true**2)))
    out["true_correction_max"] = float(np.max(np.abs(r_true))) if r_true.size else 0.0

    for tau in thresholds:
        mask = np.abs(r_true) > tau
        tag = f"t{tau:g}".replace(".", "")
        out[f"bridge_frac_{tag}"] = float(np.mean(mask)) if mask.size else 0.0
        if np.any(mask):
            eb, rb, tb = err[mask], r_true[mask], true[mask]
            out[f"bridge_corr_rel_l2_{tag}"] = _norm(eb) / (_norm(rb) + eps)
            out[f"bridge_field_rel_l2_{tag}"] = _norm(eb) / (_norm(tb) + eps)
        else:
            out[f"bridge_corr_rel_l2_{tag}"] = float("nan")
            out[f"bridge_field_rel_l2_{tag}"] = float("nan")

    clear = np.abs(r_true) <= thresholds[0]
    if np.any(clear):
        out["clear_field_rel_l2"] = _norm(err[clear]) / (_norm(true[clear]) + eps)
    else:
        out["clear_field_rel_l2"] = float("nan")
    return out
