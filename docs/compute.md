# Compute — Spartan GPU reference & job log

Operational reference for running ThermoTwin-3D on Spartan (account **`punim2769`**).
GPU work goes through Slurm; this is where we record what's available, how to target
it, and what we've submitted.

## GPU partitions we can use

QOS held by `punim2769`: **`feit`, `publicgpu`, `normal`, `publiccpu`**. Every GPU
node = 64 CPUs, ~1 TB RAM, 4 GPUs.

| Partition | GPU | Walltime | `--qos` | Notes |
|---|---|---|---|---|
| `gpu-h100` | H100 ×64 | 7 d | `publicgpu` | Fastest for FNO/training. **Contended → usually queues on `(Resources)`.** |
| `feit-gpu-a100` | A100 80GB ×84 | 7 d | `feit` | **Our faculty partition — strong priority, starts ~immediately.** Default target. |
| `gpu-a100` | A100 ×108 | 7 d | `publicgpu` | Public A100s. |
| `gpu-l40s` | L40S 48GB ×40 | 7 d | `publicgpu` | Ada, strong; some availability. |
| `gpu-a100-short` / `-mig` / `*-preempt` | A100 / slices | 4 h / — | `publicgpu` | quick / tiny (MIG) / preemptible. |

**Not accessible:** `feit-geoandco` (H100; needs a `feit-geoandco` QOS we don't hold).

### sbatch flags
```bash
# Faculty A100 — immediate start (DEFAULT for now)
sbatch --partition=feit-gpu-a100 --qos=feit --gres=gpu:A100:1 scripts/slurm/<job>.slurm
# H100 — fastest GPU, but expect a queue
sbatch --partition=gpu-h100 --qos=publicgpu --gres=gpu:H100:1 scripts/slurm/<job>.slurm
# L40S
sbatch --partition=gpu-l40s --qos=publicgpu --gres=gpu:L40S:1 scripts/slurm/<job>.slurm
```
Check live idle before choosing: `sinfo -p <part> -o "%t %D %G" -h`. Job start estimate:
`squeue -j <id> --start`. Is a running job GPU-bound? `sstat -j <id>.0 --format=AveCPU` —
if `AveCPU ≈ Elapsed` it's pinned to ~1 CPU core (GPU starved); if `AveCPU ≪ Elapsed`
it's GPU-bound.

## GINO performance notes (hard-won)

GINO is the geometry-conditioned backbone, so its GPU efficiency matters. Lessons:
1. **Open3D in our env is CPU-only** (no CUDA build) → `gno_use_open3d=True` pins the
   GNO neighbour search to one CPU core. Use **`gno_use_open3d=false`** (native torch
   search, follows tensors to CUDA). Set in `configs/model/{gino,delta_gino}.yaml`.
2. **`torch_scatter` / `torch_cluster`** must be the CUDA builds, or the GNO
   scatter-reduce falls back to CPU. Install the prebuilt wheels (no compile):
   `pip install torch_scatter torch_cluster -f https://data.pyg.org/whl/torch-2.5.1+cu121.html`
   then `gno_use_torch_scatter=True`.
3. **Residual bottleneck — batch-1 launch overhead.** Even with (1)+(2), GINO on tiny
   per-sample clouds (≈2 k points, batch 1) stays CPU-bound: thousands of microsecond
   CUDA kernels are driven one-by-one by the per-sample Python loop, so the GPU is
   starved and the host CPU is the limiter (`AveCPU ≈ Elapsed`). **Proper fix (TODO):**
   batch multiple samples per step (non-trivial — GINO shares one geometry per batch),
   and/or `torch.compile` / CUDA graphs to cut launch overhead. Until then, keep the
   benchmark workload modest (fewer epochs/seeds) so it finishes within walltime.

## Job log

Most recent first. `R`=running, `C`=completed, `X`=cancelled, `TO`=timeout.

| Job ID | Date | Partition | What | Config | Status | Result |
|---|---|---|---|---|---|---|
| 26446105 | 2026-06-24 | feit-gpu-a100 | Block-2 GINO benchmark | 60 ep, 1 seed (reduced) | R | `results/block2_benchmark.{json,md}` |
| 26445982 | 2026-06-24 | feit-gpu-a100 | Block-2 benchmark | 150 ep, 2 seed | X | CPU-bound; would exceed walltime |
| 26444333 | 2026-06-24 | feit-gpu-a100 | Block-2 benchmark | 150 ep (post open3d/scatter fix) | X | still CPU-bound (batch-1 overhead) |
| 26442698 | 2026-06-24 | gpu-h100 | Block-2 benchmark | — | X | queued on `(Resources)`, switched to FEIT |
| 26440168 | 2026-06-24 | gpu-a100 | Block-2 benchmark (1st) | 150 ep, 2 seed | X | CPU-bound (Open3D CPU search) |
| 26439080 | 2026-06-24 | gpu-a100 | Block-1 OOD sweep | 5 var × 2 reg × 3 seed | C | `results/block1_ood.{json,md}` |
| 26437785 | 2026-06-23 | gpu-a100 | Block-1 benchmark | fno/cnn | C | `results/block1_benchmark.{json,md}` |

To reconnect and check the current job:
```bash
squeue -u $USER                                    # is it still running?
sacct -j <id> --format=State,Elapsed,ExitCode      # final state
tail results/logs/tt3d-block2-<id>.out             # progress / leaderboard
cat results/block2_benchmark.md                     # results once written
```
