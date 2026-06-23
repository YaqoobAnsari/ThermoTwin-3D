# 5. Block-1 out-of-distribution generalization — the prior must travel

Date: 2026-06-24

## Status

Accepted

## Context

ADR [`0004`](0004-block1-model-ablation.md) adopted `delta_fno` — an FNO that
predicts a *correction* to a hard analytic per-column 1-D clear-wall θ prior — as the
Block-1 default, on the strength of an **in-distribution** ablation: U-MAE
0.0105 ± 0.0009 W/m²K, a −56% cut on the data-only FNO (0.0242 ± 0.0034), with the
best field rel-L2 in the sweep. Two strong non-prior challengers (`ufno` 0.0196,
`fno_uloss` 0.0200) trailed it but beat the reference.

In-distribution accuracy is not the claim the paper needs to make. ThermoTwin-3D's
target is the **as-built envelope**: every real building is a *new* construction, with
material assemblies, surface films, and bridge geometries the model was never trained
on, reconstructed at whatever mesh resolution the scan yields. The contribution only
holds if the winner's lead **survives distribution shift** — otherwise the −56% is an
artefact of an i.i.d. benchmark. We therefore need OOD evidence before committing
`delta_fno` as the Block-2 backbone and as the paper's lead architecture.

Two questions, framed as falsifiable hypotheses carried over from Exp 1.4:

1. Does `delta_fno`'s lead *widen* under shift, or does the hard prior turn into a
   liability when the geometry/assembly leaves the training band?
2. Does the soft PDE-residual loss — a deliberate **negative** result in-distribution
   (ADR `0003`: it did not improve U-MAE with abundant matched data) — finally pay off
   in the **low-data** regime it was always argued to be for?

### OOD axes, and why temperatures are *excluded*

The target θ=(T−T_out)/(T_in−T_out) is **invariant to the absolute indoor/outdoor
temperatures** under linear steady conduction: scaling or shifting the boundary
temperatures (provided they stay unequal) leaves θ unchanged, because θ is the
normalized solution of a linear BVP whose only free data are the *geometry, the
material conductivities, and the surface resistances*. Shifting BC temperatures is
therefore a **no-op OOD test** — it does not move the well-posed target — and would
manufacture a false sense of robustness. The honest covariate shifts live in the
inputs that actually set θ:

| OOD set | Shift (one axis each) | Why it is a real shift for as-built scans |
|---|---|---|
| `ood_walls` | unseen wall assemblies (layer materials / thicknesses) | each scanned building is a new construction |
| `ood_films` | `r_si` / `r_se` outside the training band | unseen interior/exterior convective regimes |
| `ood_bridges` | denser / wider thermal-bridge regime | more aggressive geometry than trained on |
| `ood_res` | finer through/along-wall discretisation | the operator must be mesh-agnostic |

Each set is 64 held-out samples; θ stays well-posed throughout because the
temperatures are untouched.

## Decision

Run a controlled OOD study (`scripts/ablate.py` OOD path → `scripts/slurm/ablate.slurm`,
artefact `results/block1_ood.{json,md}`) over five variants carried forward from the
ablation — `delta_fno` (winner), `fno` (reference), `fno_uloss` and `ufno` (the two
strongest non-prior in-distribution challengers), and `fno_physics` (the soft-physics
hypothesis to retest) — across:

- **seeds** `{1337, 1, 2}`, 300 epochs · batch 64 · lr 1e-3 · A100;
- **two data regimes** to separate "more data" from "better prior": **full** (all 256
  of `block1_train`) and **lowdata** (a fixed seeded 64-sample subset, ¼ of the data);
- **five test sets** (`in_dist`, `ood_walls`, `ood_films`, `ood_bridges`, `ood_res`),
  scored at **native resolution**, each featurised with the variant's own feature set;

30 training runs total. **U-MAE** (W/m²K) is primary; the **generalization gap**
(OOD U-MAE − same-variant in-distribution U-MAE) measures how much accuracy a model
*keeps* under shift.

## Results

Mean ± std over 3 seeds. Full tables in `results/block1_ood.{json,md}` and
`docs/experiments.md` (Exp 1.4). Headline numbers, **full** regime unless noted:

| OOD axis | `delta_fno` U-MAE | next-best | lead | `delta_fno` gap | next-best gap |
|---|---|---|---|---|---|
| `ood_walls` | **0.0680 ± 0.0089** | 0.3026 (ufno) | **4.4×** | **+0.0575** | +0.2830 |
| `ood_films` | **0.0617 ± 0.0054** | 0.1183 (fno_uloss) | 1.9× | **+0.0511** | +0.0983 |
| `ood_bridges` | **0.0491 ± 0.0081** | 0.1669 (fno_uloss) | **3.4×** | **+0.0386** | +0.1469 |
| `ood_res` | **0.0143 ± 0.0008** | 0.0713 (ufno) | **5.0×** | **+0.0037** | +0.0517 |

1. **`delta_fno` does not lose anywhere.** It posts the lowest U-MAE *and* the smallest
   gap on **all 8 OOD cells** (4 axes × 2 regimes), in both regimes. Gaps stay nearly
   flat (full `ood_res` +0.0037; it is essentially resolution-invariant) while every
   prior-less variant roughly triples its U-MAE on the geometry/assembly shifts. The
   *only* cell any competitor edges it is the **secondary** field rel-L2,
   **in-distribution**, in the lowdata regime (`ufno` 0.0206 vs 0.0244) — not OOD, not
   the primary metric.

2. **Unseen wall assemblies are the binding risk.** `ood_walls` is the hardest axis for
   everyone (mean gap +0.2419 full / +0.2639 lowdata), and prior-less models collapse
   toward the geometry-blind regime there. Ordering of difficulty:
   walls ≫ bridges > films ≫ res.

3. **In-distribution auxiliary winners overfit.** `ufno` (in-dist 0.0196) records the
   single largest gap in the study (lowdata `ood_walls` **+0.3791**) and collapses on
   bridges; `fno_uloss` (0.0200) carries a +0.2852 wall gap. Their gains do not survive
   shift.

4. **Soft PDE-residual loss: marginal low-data keep only.** In **full** data
   `fno_physics` vs `fno` is a wash (worse on 3 of 5 cells). In **lowdata** it flips to
   a consistent win on all 5 (in_dist −0.0028, ood_walls −0.0293, ood_films −0.0120,
   ood_bridges −0.0051, ood_res −0.0083) — but every delta is small relative to seed σ
   (~0.03–0.10), so it is **directional, not decisive**, and it is dominated ~8× by the
   hard prior in lowdata OOD. Prior strength ranking: hard θ-channel (`delta_fno`) ≫
   U-supervision (`fno_uloss`) > soft residual (`fno_physics`).

## Decision outcome

1. **`delta_fno` is confirmed as the Block-1 default and the paper's lead architecture.**
   Its lead is not an i.i.d. artefact — it widens under every physically meaningful
   shift. The hard analytic prior is a property of the conduction physics, not of the
   training distribution, which is exactly why it travels.

2. **The PDE-residual loss is *not* re-enabled by default, even for low data.** It earns
   a marginal, directional low-data keep, but the gain is within seed noise and is
   dominated by the hard prior. It remains a *consistency rail* available for
   data-scarce / OOD regimes (ADR `0003`), not an accuracy claim — no headline rests
   on it.

## Consequences

- **+** OOD evidence the paper can lead with: per-axis U-MAE under controlled
  single-axis shift, winner ahead 1.9–5.0× with near-flat generalization gaps. The
  temperature-invariance argument also pre-empts the obvious "did you just rescale the
  BCs?" review question by construction.
- **+** A clear design mandate for **Block-2 (GINO on as-built point clouds + SDF):**
  unseen **envelope assemblies** — not films or mesh resolution — are the binding
  generalization risk, so the **material-layer / assembly encoding** must be the design
  and OOD-evaluation focus. The `delta`-head pattern (predict a correction to a hard
  analytic clear-wall prior) carries forward once a clear-wall prior exists on
  irregular geometry; report U-MAE under per-axis OOD splits with assemblies as the
  headline stressor.
- **+** Honest negatives preserved: the in-distribution auxiliary winners (`ufno`,
  `fno_uloss`) are documented overfitters under shift, and the soft physics loss is
  recorded as a low-data nicety rather than a robustness lever. Both are retained as
  registered, config-selectable alternatives.
- **−** This is still a *synthetic* OOD study (Block-1 FEM corpus). The assembly-shift
  finding is a hypothesis about real scans that Block-2 must confirm on measured IR
  (TUM2TWIN, TBBR); the ADR's design mandate is conditioned on that validation.
