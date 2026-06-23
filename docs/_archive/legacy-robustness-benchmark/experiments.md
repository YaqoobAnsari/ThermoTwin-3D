# Experiments — BuildTrust-3D (execution core)

The one question that matters for a 5-week build: **what do we actually run, and is it tractable?** Yes. Here's why and the exact matrix.

---

## Why it is NOT "too many experiments"

1. **Block-diagonal, not a cross-product.** Segmentation models run only on segmentation datasets; reconstruction models only on reconstruction datasets. The halves never multiply together. So it is never `7 × 3 × 8 × 10`.
2. **Inference is cheap.** One "experiment" = one model, one corrupted variant = a single forward pass over a test split = minutes. Hundreds of these = a few GPU-hours, run as a scripted sweep, not by hand.
3. **The real cost is environment setup per codebase** — and one umbrella (Pointcept) runs ~5 segmentation models from a single environment. See `models-and-code.md`.
4. **Checkpoint availability prunes the dataset list.** Inference-only ⇒ we can only use datasets with released checkpoints: **S3DIS** + **ScanNet/ScanNet++** (seg), **Building3D**/**Structured3D** (recon). ArCH / H3D / Matterport3D have ~no off-the-shelf checkpoints → fine-tune-later extensions, **not** MVP. **We do not run all 7 datasets.**

---

## Counting the runs

**Conditions per (model, dataset):** `8 corruptions × 3 severities = 24` corrupted + `1` clean = **25**.
With **3 seeds** on the ~4 stochastic corruptions (noise, ghost/specular, cross-sensor, dynamic): adds `4 × 3 × 2 = 24` extra runs → ≈ **37** runs per (model, dataset) at most. Deterministic corruptions (voxelisation, occlusion-by-fixed-pose) need 1 seed.

| Block | Datasets | Models | Conditions | Runs (≈) |
|-------|----------|--------|-----------|----------|
| **Seg — MVP** | S3DIS (Area-5) | 6 | 25–37 | **150–220** |
| **Recon — MVP** | Building3D (test subset) | 2 | 25 | **~50** |
| **MVP total** | | | | **~200–270 inference runs** |
| Seg — extension | ScanNet/ScanNet++ val | +4 | 25–37 | +100–150 |
| Recon — extension | Structured3D | +1–2 | 25 | +25–50 |

**~200–270 runs at minutes each ≈ a few GPU-hours of compute.** The schedule risk is engineering (corruption pipeline, metrics, per-codebase setup), not GPU time.

---

## The MVP matrix (freeze this)

**Datasets (MVP):** S3DIS (Area-5, seg) · Building3D (recon). *ScanNet++ pulled in for B3/B4 calibration pairs even before it's a full benchmark dataset.*

**Corruptions (MVP = 8):** from `corruption-taxonomy.md` — `A1` occlusion, `A2` coverage gaps, `A3` density, `B1` specular dropout, `B3` anisotropic noise, `B4` cross-sensor, `C1` registration drift, `C2` voxelisation. Each × S1/S2/S3.

**Models (MVP):**

| Block | Models | Source env |
|-------|--------|-----------|
| Seg (Pointcept) | PTv3 · PTv2 · SparseUNet (MinkUNet) · PointNet++ | **one** Pointcept env |
| Seg (instance) | Mask3D | own env |
| Seg (foundation) | Sonata (linear-probe / frozen features) | Sonata env (PTv3-based) |
| Recon | Point2Roof · Building3D baseline | Point2Roof / Building3D env |

→ **~4–5 environments total**, ~8 models, ~200–270 runs.

---

## Per-(model,dataset) loop (the only loop you script)

```
for model in MODELS[block]:
  for corruption in CORRUPTIONS:           # 8
    for severity in [1,2,3]:               # 3
      for seed in SEEDS[corruption]:       # 1 or 3
        variant = corrupt(clean_split, corruption, severity, seed)   # cached
        preds   = model.infer(variant)                               # forward pass
        metrics = evaluate(preds, gt, block)                         # Tier 1 or Tier 2
        log(model, corruption, severity, seed, metrics)
  preds_clean = model.infer(clean_split); log(..., "clean")          # reference
```

Corrupted variants are **generated once and cached** (keyed by `dataset/scene/corruption/severity/seed`), so all models reuse the same inputs — no recomputation.

---

## Execution order (swift path)

1. **Corruption engine first, on S3DIS.** Implement the 8 MVP corruptions; cache `S3DIS-C-Built` variants. (Biggest novel-code chunk; everything depends on it.)
2. **Stand up Pointcept**, pull S3DIS Area-5 checkpoints (PTv3, PTv2, SparseUNet, PointNet++), run clean → confirm published mIoU reproduces (sanity gate).
3. **Sweep Pointcept models × 8 corruptions × 3 severities.** Compute Tier-1 (mCE, mRR, per-class, severity curves).
4. **Add Mask3D + Sonata** (own envs), same sweep.
5. **Reconstruction:** Building3D + Point2Roof; cache `Building3D-C`; run Tier-2 (geometry, topology, IFC validity, calibration).
6. **Calibration realism check** vs ScanNet++ laser↔phone pairs for B3/B4.
7. **One baseline defence** (e.g. augment-with-corruptions or density-insensitive inference) → show mCE/ECE move.
8. Everything else (ScanNet/ScanNet++ full, KPConv/PointNeXt, RoomFormer/Structured3D, B2/C3/D1, ArCH/H3D via fine-tuning) = **extension backlog**.

---

## Compute & determinism notes
- Single modern GPU (24 GB, e.g. RTX 3090/4090/A6000) is enough for inference on all MVP models (PTv2/PTv3 fit in 24 GB per Pointcept notes).
- Cache variants to disk; budget storage (corrupted S3DIS Area-5 × 24 conditions is the main footprint — point data, not huge).
- Seed everything; persist a manifest per variant so results are exactly reproducible.
- **Sanity gate before trusting any corrupted number:** reproduce each model's *clean* accuracy within tolerance of its published figure.

> The point: the matrix is small and cheap once you see it's block-diagonal and inference-only. Spend the 5 weeks on the corruption engine and the metrics, not on compute.
