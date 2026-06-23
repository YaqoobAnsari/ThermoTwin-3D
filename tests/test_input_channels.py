"""Input-channel featuriser: layout, the analytic prior, and dataset wiring.

``build_input_channels`` is shared by the dataset and the ablation eval, so its
contract is pinned here: the 'base' set is the original 3 channels; 'enriched' adds
the verified clear-wall 1-D prior (at the documented channel index), a normalised
through-wall coordinate and a normalised lateral coordinate. The prior must match
the FV solver to machine precision on a no-bridge wall (where the 1-D assumption is
exact at cell centres).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from thermotwin.data.dataset import (
    ENRICHED_CLEARWALL_CHANNEL,
    LOGK_MEAN,
    LOGK_STD,
    build_input_channels,
    clearwall_theta,
)
from thermotwin.data.synthetic_fem import WallSample, build_k_field, solve_sample
from thermotwin.physics.conduction import Layer

WALL = (
    Layer("concrete", 0.20, 1.95),
    Layer("eps_insulation", 0.10, 0.035),
    Layer("gypsum", 0.0127, 0.16),
)

_CORPUS = Path(__file__).resolve().parents[1] / "data/processed/block1_train"


def test_base_channel_count_and_values():
    k = np.full((24, 32), 0.5, dtype=np.float32)
    dx0 = np.full(24, 0.01, dtype=np.float32)
    x = build_input_channels(k, dx0, 0.02, 0.13, 0.04, feature_set="base")
    assert x.shape == (3, 24, 32)
    expected_logk = (np.log10(0.5) - LOGK_MEAN) / LOGK_STD
    assert np.allclose(x[0], expected_logk)
    assert np.allclose(x[1], 0.13)
    assert np.allclose(x[2], 0.04)


def test_enriched_channel_count_and_layout():
    k = np.linspace(0.1, 2.0, 24 * 48).reshape(24, 48).astype(np.float32)
    dx0 = np.full(24, 0.01, dtype=np.float32)
    x = build_input_channels(k, dx0, 0.02, 0.13, 0.04, feature_set="enriched")
    assert x.shape == (6, 24, 48)
    # Documented: clear-wall theta lives at ENRICHED_CLEARWALL_CHANNEL (== 3).
    assert ENRICHED_CLEARWALL_CHANNEL == 3
    theta1d = clearwall_theta(k, dx0, 0.13, 0.04)
    assert np.allclose(x[ENRICHED_CLEARWALL_CHANNEL], theta1d.astype(np.float32))
    # Through-wall coord: constant along axis 1, spans [0,1] along axis 0.
    assert np.allclose(x[4][:, 0], np.linspace(0, 1, 24, dtype=np.float32))
    assert np.allclose(x[4], x[4][:, :1])
    # Lateral coord: constant along axis 0, spans [0,1] along axis 1.
    assert np.allclose(x[5][0, :], np.linspace(0, 1, 48, dtype=np.float32))
    assert np.allclose(x[5], x[5][:1, :])
    # The first three channels match 'base'.
    base = build_input_channels(k, dx0, 0.02, 0.13, 0.04, feature_set="base")
    assert np.allclose(x[:3], base)


def test_enriched_matches_base_prefix_check_logk_unchanged():
    """LOGK_STD scaling is applied consistently across feature sets."""
    k = np.full((24, 16), 10.0, dtype=np.float32)
    dx0 = np.full(24, 0.01, dtype=np.float32)
    x = build_input_channels(k, dx0, 0.02, 0.1, 0.05, feature_set="enriched")
    assert np.allclose(x[0], (np.log10(10.0) - LOGK_MEAN) / LOGK_STD)


def test_unknown_feature_set_raises():
    k = np.full((24, 8), 1.0, dtype=np.float32)
    dx0 = np.full(24, 0.01, dtype=np.float32)
    with pytest.raises(ValueError):
        build_input_channels(k, dx0, 0.02, 0.1, 0.05, feature_set="nope")


def test_clearwall_theta_matches_solver_no_bridge():
    """On a clear (no-bridge) wall, the 1-D prior reproduces the FV field exactly."""
    s = WallSample(WALL, width_m=0.6, t_indoor=21.0, t_outdoor=-4.0, bridges=(), lateral_cells=32)
    field = solve_sample(s)
    k, spacing = build_k_field(s)
    theta_gt = (field.temperature - s.t_outdoor) / (s.t_indoor - s.t_outdoor)
    theta1d = clearwall_theta(np.asarray(k), spacing[0], s.r_si, s.r_se)
    assert np.max(np.abs(theta1d - theta_gt)) < 1e-9


@pytest.mark.skipif(
    not (_CORPUS / "manifest.json").exists(),
    reason="corpus not generated (run scripts/generate_fem_groundtruth.py)",
)
def test_dataset_feature_sets():
    from thermotwin.data.dataset import SyntheticFEMDataset

    base = SyntheticFEMDataset(_CORPUS, target_width=48, feature_set="base")
    enriched = SyntheticFEMDataset(_CORPUS, target_width=48, feature_set="enriched")
    xb, yb = base[0]
    xe, ye = enriched[0]
    assert xb.shape[0] == 3
    assert xe.shape[0] == 6
    assert xb.shape[1:] == xe.shape[1:] == yb.shape[1:]
    # 'base' is the prefix of 'enriched'.
    assert np.allclose(xb.numpy(), xe[:3].numpy())
    # default stays 'base' and physics bundle is unchanged.
    default = SyntheticFEMDataset(_CORPUS, target_width=48, return_physics=True)
    x, y, phys = default[0]
    assert x.shape[0] == 3
    assert set(phys) == {"k", "dx0", "dy", "r_si", "r_se"}
