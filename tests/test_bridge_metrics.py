"""Tests for the bridge-focused correction metrics."""

from __future__ import annotations

import numpy as np
import pytest

from thermotwin.eval.bridge_metrics import bridge_focused_metrics


def _case():
    """A cloud where 20% of points carry a known bridge correction of 0.1."""
    rng = np.random.default_rng(0)
    prior = rng.uniform(0.1, 0.9, size=500)
    r_true = np.zeros(500)
    r_true[:100] = 0.1  # 100/500 = 20% bridge points
    true = prior + r_true
    return prior, true, r_true


def test_perfect_prediction_is_zero():
    prior, true, _ = _case()
    m = bridge_focused_metrics(true, true, prior)  # pred == true
    assert m["correction_rel_l2"] == pytest.approx(0.0, abs=1e-6)
    assert m["correction_r2"] == pytest.approx(1.0, abs=1e-6)
    for k, v in m.items():
        if k.startswith("bridge_corr_rel_l2"):
            assert v == pytest.approx(0.0, abs=1e-6)


def test_prior_only_scores_exactly_one():
    """The zero-network prior must score 1.0 — the whole point of the normalisation."""
    prior, true, _ = _case()
    m = bridge_focused_metrics(prior, true, prior)  # pred == prior
    assert m["correction_rel_l2"] == pytest.approx(1.0, abs=1e-6)
    assert m["bridge_corr_rel_l2_t002"] == pytest.approx(1.0, abs=1e-6)
    assert m["correction_corr"] == 0.0  # predicted correction is all zeros


def test_half_correction_scores_half():
    prior, true, r_true = _case()
    pred = prior + 0.5 * r_true
    m = bridge_focused_metrics(pred, true, prior)
    assert m["correction_rel_l2"] == pytest.approx(0.5, abs=1e-6)
    assert m["bridge_corr_rel_l2_t002"] == pytest.approx(0.5, abs=1e-6)
    # The half-correction beats the prior (1.0) on the bridge region.
    assert m["bridge_corr_rel_l2_t002"] < 1.0


def test_bridge_fraction():
    prior, true, _ = _case()
    m = bridge_focused_metrics(true, true, prior)
    # correction 0.1 sits above τ=0.02 and 0.05 on exactly 20% of points.
    assert m["bridge_frac_t002"] == pytest.approx(0.2, abs=1e-9)
    assert m["bridge_frac_t005"] == pytest.approx(0.2, abs=1e-9)
    assert m["true_correction_max"] == pytest.approx(0.1, abs=1e-9)


def test_no_bridge_points_gives_nan():
    prior = np.linspace(0.1, 0.9, 50)
    true = prior.copy()  # no correction anywhere
    m = bridge_focused_metrics(prior + 0.001, true, prior, thresholds=(0.1,))
    assert np.isnan(m["bridge_corr_rel_l2_t01"])
    assert m["bridge_frac_t01"] == 0.0


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        bridge_focused_metrics(np.zeros(3), np.zeros(4), np.zeros(3))
