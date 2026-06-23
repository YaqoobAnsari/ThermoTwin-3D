"""Reproducibility helpers. Default seed is 1337 (CLAUDE.md)."""

from __future__ import annotations

import os
import random

import numpy as np

__all__ = ["seed_everything"]


def seed_everything(seed: int = 1337) -> int:
    """Seed Python, NumPy and (if present) PyTorch RNGs. Returns the seed."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    # Intentionally seed the legacy global RNG too, for any code/library that uses
    # np.random.* directly; our own samplers use np.random.default_rng(seed).
    np.random.seed(seed)  # noqa: NPY002
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    return seed
