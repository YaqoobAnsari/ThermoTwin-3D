"""Dataset + FNO wiring: shapes, and an overfit sanity check.

The overfit test is the "minimal forward; overfit one sample" milestone from the
architecture next-steps: if the FNO can drive the relative-L2 on a handful of fixed
samples toward zero, the data→model→loss plumbing is correct end to end. It runs on
CPU with a tiny model and is skipped if the corpus has not been generated.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

torch.manual_seed(1337)

from thermotwin.eval.metrics import relative_l2  # noqa: E402

_CORPUS = Path(__file__).resolve().parents[1] / "data/processed/block1_train"
_pytestmark_reason = "corpus not generated (run scripts/generate_fem_groundtruth.py)"


@pytest.mark.skipif(not (_CORPUS / "manifest.json").exists(), reason=_pytestmark_reason)
def test_dataset_shapes():
    from thermotwin.data.dataset import SyntheticFEMDataset

    ds = SyntheticFEMDataset(_CORPUS, target_width=48)
    assert len(ds) > 0
    x, y = ds[0]
    assert x.shape[0] == 3 and y.shape[0] == 1
    assert x.shape[1:] == y.shape[1:]
    assert x.shape[2] == 48
    # theta is a dimensionless field, broadly within [0, 1] for these BCs.
    assert float(y.min()) > -0.2 and float(y.max()) < 1.2


@pytest.mark.skipif(not (_CORPUS / "manifest.json").exists(), reason=_pytestmark_reason)
def test_fno_overfits_small_batch():
    from torch.utils.data import DataLoader, Subset

    from thermotwin.data.dataset import SyntheticFEMDataset
    from thermotwin.models.fno import build_fno

    ds = Subset(SyntheticFEMDataset(_CORPUS, target_width=48), list(range(4)))
    x, y = next(iter(DataLoader(ds, batch_size=4)))

    model = build_fno(
        in_channels=3, out_channels=1, n_modes=(8, 16), hidden_channels=24, n_layers=4
    )
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)

    start = relative_l2(model(x), y).item()
    for _ in range(250):
        opt.zero_grad()
        loss = relative_l2(model(x), y)
        loss.backward()
        opt.step()
    end = relative_l2(model(x), y).item()

    assert end < start, f"no learning: {start:.3f} -> {end:.3f}"
    assert end < 0.1, f"failed to overfit 4 samples: relative L2 {end:.3f}"
