# Models & Code — BuildTrust-3D (verified repos)

Every model we plan to run, with its open-source repository, whether **pretrained checkpoints** exist (so we can stay inference-only), which datasets those checkpoints cover, and environment notes. ✅ = repo + URL **directly verified**; ⚠️ = well-known repo, **confirm URL/checkpoints** before relying on it.

> **The swiftness lever:** [Pointcept](https://github.com/Pointcept/Pointcept) is a single codebase that runs PTv3, PTv2, SparseUNet/MinkUNet, and PointNet++ with released S3DIS/ScanNet weights from **one** environment. Build the corruption→inference→metrics loop against Pointcept first; it covers most of the segmentation lineup with no per-model setup.

---

## Segmentation — primary (run via Pointcept, one environment)

| Model | Repo | Pretrained? | Datasets w/ ckpts | Notes |
|-------|------|-------------|-------------------|-------|
| **Point Transformer V3 (PTv3)** | ✅ https://github.com/Pointcept/PointTransformerV3 · in Pointcept | Yes | ScanNet, S3DIS, (more) | CVPR'24 oral; needs FlashAttention; 24 GB OK |
| **Point Transformer V2 (PTv2)** | ✅ https://github.com/Pointcept/Pointcept | Yes | ScanNet, S3DIS | Tuned to run on 4×3090/24 GB per repo |
| **SparseUNet / MinkUNet** | ✅ https://github.com/Pointcept/Pointcept | Yes | ScanNet, S3DIS | SpConv version recommended (easier than MinkowskiEngine) |
| **PointNet++** | ✅ in Pointcept; orig https://github.com/charlesq34/pointnet2 | Yes (Pointcept) | ScanNet, S3DIS | Canonical baseline; good mCE normaliser |

## Segmentation — standalone (own environments)

| Model | Repo | Pretrained? | Datasets w/ ckpts | Notes |
|-------|------|-------------|-------------------|-------|
| **PointNeXt** | ✅ https://github.com/guochengqian/PointNeXt | Yes | S3DIS (Area-5 + 6-fold) | Explicit `mode=test --pretrained_path …`; CUDA 11.3 |
| **KPConv** (PyTorch) | ✅ https://github.com/HuguesTHOMAS/KPConv-PyTorch · TF: https://github.com/HuguesTHOMAS/KPConv | Yes | S3DIS, ScanNet, Semantic3D, NPM3D | Pretrained weights + load instructions provided |
| **RandLA-Net** | ⚠️ https://github.com/QingyongHu/RandLA-Net | Yes (orig TF) | S3DIS, Semantic3D, SemanticKITTI | TF original; PyTorch ports exist; density-sensitive = good stress case |
| **Mask3D** (instance) | ✅ https://github.com/JonasSchult/Mask3D | Yes | ScanNet, ScanNet200, S3DIS, STPLS3D | Inference: `general.train_mode=false`; needs MinkowskiEngine |
| **OctFormer** | ⚠️ https://github.com/octree-nn/octformer | Yes | ScanNet, (S3DIS) | Octree attention |
| **OneFormer3D** | ⚠️ https://github.com/oneformer3d/oneformer3d | Yes | ScanNet | Unified sem/inst/panoptic |

## Segmentation — foundation / self-supervised

| Model | Repo | Pretrained? | Notes |
|-------|------|-------------|-------|
| **Sonata** | ✅ https://github.com/facebookresearch/sonata | Yes (PTv3 encoder) | Self-supervised "reliable" PTv3 features — *literally about reliability*, strong trust-angle baseline; linear-probe/frozen for inference |
| **Concerto** | ✅ https://github.com/Pointcept/Concerto | Yes (PTv3 encoder) | Joint 2D-3D SSL; Pointcept-compatible |
| **Point-MAE** | ⚠️ https://github.com/Pang-Yatian/Point-MAE | Yes | Masked-autoencoder pretraining (object-level origin) |

**Umbrella alternative:** [torch-points3d](https://github.com/torch-points3d/torch-points3d) ✅ packages KPConv, RandLA-Net, RSConv, etc. with S3DIS/ScanNet loaders — another single-env option if Pointcept coverage is short.

---

## Reconstruction (scan-to-BIM / roof / floorplan)

| Model | Repo | Pretrained? | GT pairing | Notes |
|-------|------|-------------|-----------|-------|
| **Point2Roof** | ✅ https://github.com/Li-Li-Whu/Point2Roof | Code + datasets (train fast) | Building3D / RoofN3D | ISPRS-J; PointNet++ backbone; synthetic + small real set provided |
| **Building3D baselines** | ✅ dataset+baselines (arXiv:2307.11914) — confirm repo on the Building3D project page | Baselines provided | **Real** ALS ↔ wireframe/mesh GT (Tallinn) | The linchpin: real point-cloud→model GT at scale |
| **RoomFormer** (indoor floorplan) | ⚠️ https://github.com/ywyue/RoomFormer | Yes | Structured3D | Topology / room-enclosure metrics |
| **City3D** (LoD2 buildings) | ⚠️ https://github.com/tudelft3d/City3D | Code (classical) | ALS | Pairs with Building3D; non-learned baseline |
| **PolyFit** (poly surface) | ⚠️ https://github.com/LiangliangNan/PolyFit | Code (classical) | objects/buildings | Hypothesis-and-selection classical baseline |
| **Points2Poly / PolyGNN** | ⚠️ https://github.com/chenzhaiyu/points2poly · https://github.com/chenzhaiyu/polygnn | Yes | buildings | Learned polyhedral reconstruction |

**Generative (qualitative only — no quantitative validity eval):** Text2BIM, NeRF-to-BIM → show as failure-case illustrations, not leaderboard rows.

---

## Environment reality (what actually eats time)

| Env | Covers | Pain points |
|-----|--------|-------------|
| **Pointcept** | PTv3, PTv2, SparseUNet, PointNet++ | FlashAttention (PTv3), SpConv build; once up, many models free |
| **Mask3D** | Mask3D | MinkowskiEngine + detectron2 + specific torch/CUDA (1.12/cu113) — finicky; budget a day |
| **Sonata** | Sonata/Concerto | PTv3-based; HuggingFace weight load; relatively clean |
| **PointNeXt / torch-points3d** | PointNeXt, KPConv, RandLA-Net | CUDA 11.3 custom ops |
| **Point2Roof / Building3D** | reconstruction | PointNet++ custom `pc_util` build; dataset download |

**Rule:** stand up Pointcept + one reconstruction env first (covers the MVP). Add Mask3D and PointNeXt only after the loop works end-to-end on Pointcept.

---

## Checkpoint-availability summary (drives dataset choice)

| Dataset | Off-the-shelf seg ckpts? | Recon GT? | MVP role |
|---------|--------------------------|-----------|----------|
| **S3DIS** | **Yes** (PTv2/PTv3/Mink/PointNeXt/KPConv/Mask3D) | — | **Primary seg** |
| **ScanNet / ScanNet++** | **Yes** (most Pointcept models) | — | Seg extension + **B3/B4 calibration pairs** |
| **Building3D** | — | **Yes** (wireframe/mesh) | **Primary recon** |
| **Structured3D** | — | Yes (layout) | Recon extension |
| ArCH | No (niche TLS heritage) | — | Fine-tune-later extension |
| H3D | No / sparse | — | Fine-tune-later extension |
| Matterport3D | Limited | mesh | Extension |

→ **Inference-only MVP = S3DIS + Building3D** (+ ScanNet++ for calibration). The rest need training and are out of scope for the first pass.

> ⚠️ **Verify before relying:** for every ⚠️ row, open the repo and confirm (a) it's the official/maintained one, (b) checkpoints for your target dataset are actually downloadable, (c) the licence permits your use. The ✅ rows were checked but re-confirm checkpoint URLs at build time.
