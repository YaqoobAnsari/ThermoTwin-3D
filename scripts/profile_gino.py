#!/usr/bin/env python
"""Profile one GINO model's per-step cost on the Block-2 corpus to find the TRUE
bottleneck before any optimisation.

For a handful of ``block2_train`` samples we time, per training step:
  (a) data load (.npz -> tensors, via the real DataLoader)
  (b) host->device transfer of every forward tensor
  (c) forward, decomposed into the GINO's three internal phases:
        - input GNO  (in turn: fixed-radius neighbour search vs integral transform)
        - latent FNO
        - output GNO (search vs integral transform)
  (d) backward
  (e) optimizer step

GPU timings use ``torch.cuda.Event`` with ``synchronize()`` so we measure device
wall-time, not enqueue time. A coarse ``nvidia-smi`` utilisation sampler runs in a
background thread for the duration so we can say whether the A100 was actually busy.
A short ``torch.profiler`` trace at the end gives the device-time / CPU-time split and
the top kernels.

This script does NOT change any model or training code — it only wraps it. Run on
Slurm (feit-gpu-a100); it prints a JSON blob the caller collects.

    python scripts/profile_gino.py --kind gino --n-samples 6 --warmup 3 --iters 12
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from thermotwin.data.pointcloud_dataset import (  # noqa: E402
    PointCloudDataset,
    collate_pointcloud,
)
from thermotwin.eval.metrics import relative_l2  # noqa: E402
from thermotwin.models.gino import build_delta_gino, build_gino  # noqa: E402
from thermotwin.utils.seed import seed_everything  # noqa: E402

LATENT_GRID = 16
FNO_MODES = (6, 6, 6)
GNO_RADIUS = 0.12


class _SkipProfiler(Exception):
    """Sentinel to skip the torch.profiler trace without it looking like an error."""


# --------------------------------------------------------------------------------------
# CUDA-event timing helpers
# --------------------------------------------------------------------------------------
class CudaTimer:
    """Accumulates elapsed-ms across calls using CUDA events (or perf_counter on CPU)."""

    def __init__(self, device: str) -> None:
        self.cuda = device == "cuda"
        self.samples: dict[str, list[float]] = {}

    def time(self, name: str):
        return _Section(self, name)

    def record(self, name: str, ms: float) -> None:
        self.samples.setdefault(name, []).append(ms)

    def summary(self) -> dict[str, dict[str, float]]:
        out = {}
        for k, v in self.samples.items():
            a = np.asarray(v, dtype=np.float64)
            out[k] = {
                "mean_ms": float(a.mean()),
                "std_ms": float(a.std()),
                "p50_ms": float(np.percentile(a, 50)),
                "n": int(a.size),
            }
        return out


class _Section:
    def __init__(self, timer: CudaTimer, name: str) -> None:
        self.timer, self.name = timer, name

    def __enter__(self):
        if self.timer.cuda:
            self._s = torch.cuda.Event(enable_timing=True)
            self._e = torch.cuda.Event(enable_timing=True)
            self._s.record()
        else:
            self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if self.timer.cuda:
            self._e.record()
            torch.cuda.synchronize()
            self.timer.record(self.name, self._s.elapsed_time(self._e))
        else:
            self.timer.record(self.name, (time.perf_counter() - self._t0) * 1e3)
        return False


# --------------------------------------------------------------------------------------
# Instrument the GNOBlock to split neighbour-search vs integral-transform, attributing
# each to the input vs output GNO. We monkeypatch GNOBlock.forward (non-invasive).
# --------------------------------------------------------------------------------------
def _install_gno_probe(timer: CudaTimer):
    from neuralop.layers.gno_block import GNOBlock

    orig = GNOBlock.forward
    # tag in_p / out_p by radius is fragile; instead tag by call order within a step.
    state = {"calls": 0}

    def forward(self, y, x, f_y=None):  # mirrors the upstream body, with timing
        if f_y is not None and f_y.ndim == 3 and f_y.shape[0] == -1:
            f_y = f_y.squeeze(0)
        phase = "in_gno" if state["calls"] % 2 == 0 else "out_gno"
        state["calls"] += 1
        with timer.time(f"{phase}.neighbor_search"):
            neighbors_dict = self.neighbor_search(data=y, queries=x, radius=self.radius)
        # record neighbour count so we can report graph size
        ns = neighbors_dict["neighbors_index"].numel()
        timer.record(f"{phase}.n_neighbors", float(ns))
        if self.pos_embedding is not None:
            y_embed, x_embed = self.pos_embedding(y), self.pos_embedding(x)
        else:
            y_embed, x_embed = y, x
        with timer.time(f"{phase}.integral_transform"):
            out_features = self.integral_transform(
                y=y_embed, x=x_embed, neighbors=neighbors_dict, f_y=f_y
            )
        return out_features

    GNOBlock.forward = forward
    return lambda: setattr(GNOBlock, "forward", orig)


def _build(kind: str, device: str) -> torch.nn.Module:
    common = dict(
        fno_in_channels=32,
        fno_n_modes=FNO_MODES,
        fno_hidden_channels=64,
        fno_n_layers=4,
        in_gno_radius=GNO_RADIUS,
        out_gno_radius=GNO_RADIUS,
        latent_grid=LATENT_GRID,
    )
    if kind == "gino":
        m = build_gino(in_channels=3, **common)
    elif kind == "delta_gino":
        m = build_delta_gino(in_channels=4, **common)
    else:
        raise KeyError(kind)
    return m.to(device)


def _to_device(batch: dict, kind: str, device: str, timer: CudaTimer) -> dict:
    """Time the host->device transfer of exactly the tensors the forward consumes."""
    keys = ["input_geom", "sdf", "latent_queries", "output_queries", "theta"]
    keys += ["gino_feats"] if kind == "gino" else ["feats", "prior"]
    with timer.time("h2d_transfer"):
        d = {k: batch[k].to(device, non_blocking=True) for k in keys}
        if device == "cuda":
            torch.cuda.synchronize()
    return d


def _forward(model, kind: str, d: dict, timer: CudaTimer) -> torch.Tensor:
    feats = d["gino_feats"] if kind == "gino" else d["feats"]
    with timer.time("forward_total"):
        if kind == "gino":
            out = model(d["input_geom"], feats, d["latent_queries"], d["sdf"], d["output_queries"])
        else:
            out = model(
                d["input_geom"],
                feats,
                d["latent_queries"],
                d["sdf"],
                d["output_queries"],
                d["prior"],
            )
    return out


# --------------------------------------------------------------------------------------
# Background nvidia-smi utilisation sampler
# --------------------------------------------------------------------------------------
class SmiSampler(threading.Thread):
    def __init__(self, period_s: float = 0.1) -> None:
        super().__init__(daemon=True)
        self.period_s, self._halt = period_s, threading.Event()
        self.util, self.mem = [], []

    def run(self) -> None:
        while not self._halt.is_set():
            try:
                q = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=utilization.gpu,memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=2,
                )
                line = q.stdout.strip().splitlines()[0]
                u, m = (x.strip() for x in line.split(","))
                self.util.append(float(u))
                self.mem.append(float(m))
            except Exception:
                pass
            self._halt.wait(self.period_s)

    def stop(self) -> dict:
        self._halt.set()
        self.join(timeout=2)
        if not self.util:
            return {"available": False}
        u = np.asarray(self.util)
        return {
            "available": True,
            "util_mean_pct": float(u.mean()),
            "util_p50_pct": float(np.percentile(u, 50)),
            "util_max_pct": float(u.max()),
            "frac_samples_busy_gt10pct": float((u > 10).mean()),
            "mem_used_mb_max": float(np.max(self.mem)) if self.mem else None,
            "n_samples": int(u.size),
        }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--kind", default="gino", choices=["gino", "delta_gino"])
    p.add_argument("--n-samples", type=int, default=6)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=12)
    p.add_argument("--device", default="cuda")
    p.add_argument("--train-root", default="data/processed/block2_train")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--no-profiler", action="store_true", help="skip the torch.profiler trace")
    a = p.parse_args()

    device = a.device if (a.device == "cpu" or torch.cuda.is_available()) else "cpu"
    seed_everything(1337)

    ds = PointCloudDataset(_REPO / a.train_root, latent_grid=LATENT_GRID, voxelise=False)
    # restrict to a handful of samples for a quick, representative profile
    ds.files = ds.files[: a.n_samples]
    loader = DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_pointcloud,
        num_workers=a.num_workers,
    )

    model = _build(a.kind, device)
    n_params = sum(q.numel() for q in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    timer = CudaTimer(device)
    restore = _install_gno_probe(timer)

    info = {
        "device": device,
        "gpu_name": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "kind": a.kind,
        "n_params": int(n_params),
        "n_points": 2048,
        "latent_grid": LATENT_GRID,
        "gno_radius": GNO_RADIUS,
    }

    # ---- step loop: separately time dataload, h2d, fwd phases, bwd, opt -------------
    def one_epoch(record: bool):
        model.train()
        t_load0 = time.perf_counter()
        for batch in loader:  # dataloading happens here (np.load inside __getitem__)
            load_ms = (time.perf_counter() - t_load0) * 1e3
            if record:
                timer.record("dataload", load_ms)
            d = _to_device(batch, a.kind, device, timer if record else CudaTimer(device))
            opt.zero_grad(set_to_none=True)
            tt = timer if record else CudaTimer(device)
            out = _forward(model, a.kind, d, tt)
            target = d["theta"].unsqueeze(-1)
            with tt.time("loss"):
                loss = relative_l2(out, target)
            with tt.time("backward"):
                loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            with tt.time("optimizer_step"):
                opt.step()
                if device == "cuda":
                    torch.cuda.synchronize()
            t_load0 = time.perf_counter()

    # warmup (compiles cuda kernels, fills caches) — not recorded
    saved = timer.samples
    timer.samples = {}
    for _ in range(a.warmup):
        one_epoch(record=False)
    timer.samples = saved
    if device == "cuda":
        torch.cuda.synchronize()

    smi = SmiSampler()
    smi.start()
    wall0 = time.perf_counter()
    for _ in range(a.iters):
        one_epoch(record=True)
    if device == "cuda":
        torch.cuda.synchronize()
    wall_total_s = time.perf_counter() - wall0
    smi_stats = smi.stop()

    comp = timer.summary()

    # ---- torch.profiler short trace for device/CPU time split ----------------------
    prof_summary: dict = {"skipped": True} if a.no_profiler else {}
    try:
        if a.no_profiler:
            raise _SkipProfiler
        from torch.profiler import ProfilerActivity, profile

        acts = [ProfilerActivity.CPU]
        if device == "cuda":
            acts.append(ProfilerActivity.CUDA)
        with profile(activities=acts, record_shapes=False) as prof:
            for _ in range(3):
                one_epoch(record=False)
            if device == "cuda":
                torch.cuda.synchronize()
        ka = prof.key_averages()
        total_cuda_us = sum(getattr(k, "self_device_time_total", 0) or 0 for k in ka)
        total_cpu_us = sum(getattr(k, "self_cpu_time_total", 0) or 0 for k in ka)
        top = sorted(
            ka,
            key=lambda k: (
                (getattr(k, "self_device_time_total", 0) or 0)
                if device == "cuda"
                else (getattr(k, "self_cpu_time_total", 0) or 0)
            ),
            reverse=True,
        )[:12]
        prof_summary = {
            "total_self_cuda_ms": total_cuda_us / 1e3,
            "total_self_cpu_ms": total_cpu_us / 1e3,
            "cuda_over_cpu_ratio": (total_cuda_us / total_cpu_us) if total_cpu_us else None,
            "top_ops": [
                {
                    "name": k.key[:48],
                    "cpu_ms": (getattr(k, "self_cpu_time_total", 0) or 0) / 1e3,
                    "cuda_ms": (getattr(k, "self_device_time_total", 0) or 0) / 1e3,
                    "count": int(k.count),
                }
                for k in top
            ],
        }
    except _SkipProfiler:
        pass
    except Exception as e:  # profiler is best-effort
        prof_summary = {"error": repr(e)}

    restore()

    # ---- assemble a per-component ms table (mean per step) --------------------------
    def mean(name: str) -> float:
        return round(comp.get(name, {}).get("mean_ms", 0.0), 4)

    per_component_ms = {
        "dataload": mean("dataload"),
        "h2d_transfer": mean("h2d_transfer"),
        "forward_total": mean("forward_total"),
        "in_gno.neighbor_search": mean("in_gno.neighbor_search"),
        "in_gno.integral_transform": mean("in_gno.integral_transform"),
        "latent_fno": round(
            mean("forward_total")
            - mean("in_gno.neighbor_search")
            - mean("in_gno.integral_transform")
            - mean("out_gno.neighbor_search")
            - mean("out_gno.integral_transform"),
            4,
        ),
        "out_gno.neighbor_search": mean("out_gno.neighbor_search"),
        "out_gno.integral_transform": mean("out_gno.integral_transform"),
        "loss": mean("loss"),
        "backward": mean("backward"),
        "optimizer_step": mean("optimizer_step"),
    }
    # per-step wall budget the dominant share is measured against
    step_components = [
        "dataload",
        "h2d_transfer",
        "forward_total",
        "loss",
        "backward",
        "optimizer_step",
    ]
    step_total = round(sum(per_component_ms[k] for k in step_components), 4)
    shares = {k: round(100 * per_component_ms[k] / step_total, 1) for k in step_components}

    result = {
        "info": info,
        "per_component_ms": per_component_ms,
        "step_total_ms": step_total,
        "component_share_pct": shares,
        "neighbor_counts": {
            "in_gno": int(comp.get("in_gno.n_neighbors", {}).get("mean_ms", 0)),
            "out_gno": int(comp.get("out_gno.n_neighbors", {}).get("mean_ms", 0)),
        },
        "wall_total_s": round(wall_total_s, 3),
        "steps_recorded": a.iters * len(ds.files),
        "gpu_utilisation_nvidia_smi": smi_stats,
        "torch_profiler": prof_summary,
        "raw_component_stats": comp,
    }
    print("PROFILE_JSON_BEGIN")
    print(json.dumps(result, indent=2))
    print("PROFILE_JSON_END")


if __name__ == "__main__":
    main()
