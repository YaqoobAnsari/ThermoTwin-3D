#!/usr/bin/env python
"""Measure the finite-volume (GT) solve wall-clock — the denominator of the speedup claim.

The thesis advertises a ~10³–10⁴× speedup vs the numerical reference, but the 3-D / real-
geometry results never timed the FV solver they are compared against. This times
:func:`thermotwin.data.synthetic_3d.solve_block` (the exact per-sample conduction solve used
to generate the Block-2 ground truth) over a batch of representative blocks, so the speedup
is ``fem_ms / infer_ms`` with a *measured* numerator and denominator (the model's
``infer_ms_per_sample`` comes from the benchmark JSON). Writes
``results/phase0/fem_speedup.json``.

    python scripts/fem_speedup.py --n 32
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from thermotwin.data.synthetic_3d import (  # noqa: E402
    random_block,
    random_block_hard,
    solve_block,
)


def _time_solves(gen, n: int, seed: int) -> dict:
    """Build ``n`` blocks with ``gen(rng)`` and time ``solve_block`` on each (ms)."""
    rng = np.random.default_rng(seed)
    blocks = [gen(rng) for _ in range(n)]  # build first; time only the solve
    ms = []
    for b in blocks:
        t0 = time.perf_counter()
        solve_block(b)
        ms.append((time.perf_counter() - t0) * 1e3)
    ms = np.array(ms)
    return {
        "n": n,
        "ms_mean": float(ms.mean()),
        "ms_median": float(np.median(ms)),
        "ms_p95": float(np.percentile(ms, 95)),
        "ms_min": float(ms.min()),
        "ms_max": float(ms.max()),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=32, help="blocks to time per generator")
    p.add_argument("--seed", type=int, default=1337)
    a = p.parse_args()

    out = {
        "note": (
            "FV (finite-volume) per-sample conduction solve time — the GT generator's cost, "
            "the numerator of the speedup vs a learned operator. Divide by a model's "
            "infer_ms_per_sample (benchmark JSON) for the speedup. CPU single-thread."
        ),
        "standard_block": _time_solves(random_block, a.n, a.seed),
        "hard_block": _time_solves(random_block_hard, a.n, a.seed),
    }
    odir = _REPO / "results" / "phase0"
    odir.mkdir(parents=True, exist_ok=True)
    (odir / "fem_speedup.json").write_text(json.dumps(out, indent=2))
    for name in ("standard_block", "hard_block"):
        s = out[name]
        print(f"{name:16s} FV solve: {s['ms_median']:.1f} ms median "
              f"({s['ms_mean']:.1f} mean, {s['ms_p95']:.1f} p95) over {s['n']} blocks")
    print(f"wrote {odir / 'fem_speedup.json'}")


if __name__ == "__main__":
    main()
