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
   and/or `torch.compile` / CUDA graphs to cut launch overhead.

### GPU optimisation outcome (Exp 2.2 campaign, ADR 0008)

Profiling (job 26448283, `torch.profiler` + per-region CUDA timing) **overturned the
"search-bound" guess**: the cost is the **latent-FNO GEMMs** (forward 50 %, backward 35 %;
A100 ~63 % utilised), not the neighbour search (~7 %, on-device) or data load (~9 %). Two
per-epoch wastes inflated wall time: the *fixed* per-sample geometry's neighbour graph was
recomputed every forward, and `np.load` was paid per step. Fix in `models/gino_accel.py`
(non-invasive) + `PointCloudDataset(cache_in_memory=True)`: a per-sample `NeighbourCache`
computes each GNO's CRS graph once and reuses it; optional on-GPU `torch_cluster.radius`
search (set-identical CRS); RAM-cached corpus. Activated by `GinoOperator.accelerate()`,
**bit-for-bit the legacy path when off** (`tests/test_gino_accel.py`). **Measured ~6×
speedup (1.67 s/epoch vs ~10), accuracy preserved** — making the full 300-ep × 3-seed run
on both corpora affordable (~2.3 h). GINO is **still not classically GPU-bound** (batch-1
launch overhead remains; would need batching / CUDA graphs), but it no longer gates us.
Note: live `sstat -j <id>.0` returns no data on some GPU nodes mid-run — use post-hoc
`sacct -j <id> --format=AveCPU,Elapsed,TotalCPU` for the same signal.

## Job log

Most recent first. `R`=running, `C`=completed, `X`=cancelled, `TO`=timeout.

| Job ID | Date | Partition | What | Config | Status | Result |
|---|---|---|---|---|---|---|
| 26450191 | 2026-06-24 | feit-gpu-a100 | Block-2 FULL benchmark | 300 ep, 3 seed, box + irregular | C (2:17) | delta_gino 0.0190 > voxel 0.0591 > gino 0.2554 on irregular — but **PRELIMINARY**: coord bug + missing prior-alone control (Exp 2.2 audit), re-run pending |
| 26449117 | 2026-06-24 | feit-gpu-a100 | GINO speedup confirmation | 60 ep, accel on | C | 1.67 s/epoch, rel-L2 0.0243 — accuracy preserved |
| 26449084 | 2026-06-24 | feit-gpu-a100 | GINO speedup (primary) | 10 ep, accel on | C | 1.70 s/epoch — ~6× vs pre-opt |
| 26448283 | 2026-06-24 | feit-gpu-a100 | GINO profile | torch.profiler | C | FNO-GEMM-bound, not search; see GPU-optimisation note |
| 26446105 | 2026-06-24 | feit-gpu-a100 | Block-2 GINO benchmark | 60 ep, 1 seed (reduced) | C (30 min) | voxel-FNO won; GINO undertrained on box geom — see Exp 2.1 |
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
