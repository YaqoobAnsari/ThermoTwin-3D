# Experiments & Results — running record

The authoritative log of every experiment run, for reproducibility and to feed the
paper. Each entry records the setup, **all** variants (winners and losers alike —
the losers document the design space we explored), the numbers, and the conclusion.
Raw artefacts live in `results/*.json` / `results/*.md` (git-tracked); the planning
side is in [`experiment-plan.md`](experiment-plan.md); significant decisions get an
ADR under [`decisions/`](decisions/).

Metrics throughout: **field relative-L2** (operator-learning standard) and
**U-value MAE** [W/m²K] (building-relevant; the venue's headline). The dimensionless
target is θ=(T−T_out)/(T_in−T_out). U-value is set by the θ-gradient at the indoor
face, so it is the *sensitive* metric and our primary objective.

---

## Block 1 — controlled synthetic FEM benchmark

Corpus: parametric layered walls (2 base assemblies) punctured by 0–3 thermal
bridges (steel/concrete/timber/aluminium), solved by the finite-volume engine
`physics/steady_fv.py`. Train 256 / val 64 samples, seeded. Thermal bridges shift
the effective U-value 40–50 % (up to 4×) off the 1-D clear-wall value — the geometry
signal a lumped model cannot see (novelty hook **H1**).

Reference floor — **geometry-blind 1-D clear-wall** U-value MAE: **0.1168 W/m²K**.
Every geometry-aware model must beat this.

### Exp 1.1 — first operator vs no-operator control (1 seed, A100, 300 ep)
Artefact: `results/block1_benchmark.json` (superseded by Exp 1.3 for the multi-seed
numbers; retained for provenance).

| Model | Field rel-L2 | U-MAE | vs 1-D clear |
|---|---|---|---|
| FNO | 0.0144 | 0.0205 | 5.70× |
| CNN (no-operator control) | 0.0170 | 0.0254 | 4.59× |

Conclusion: the operator beats the plain CNN, and both crush the geometry-blind
baseline → **H1 validated**. Speedup-vs-FV is *not* a win on these tiny 2-D grids
(the FV solve is 2.6 ms); that headline belongs to large 3-D FEM (Block 2+).

### Exp 1.2 — physics-residual loss + UNet baseline (1 seed, A100, 300 ep)
ADR [`0003`](decisions/0003-physics-residual-loss.md). Added the discrete
heat-equation residual (`losses/heat_residual.py`) as a soft penalty, weight 0.1.

| Model | Field rel-L2 | U-MAE |
|---|---|---|
| fno | 0.0144 | 0.0205 |
| fno_physics (residual, w=0.1) | 0.0143 | 0.0218 |
| unet | 0.0167 | 0.0343 |
| cnn | 0.0170 | 0.0254 |

Conclusion (**negative result, recorded deliberately**): the physics loss did *not*
help in-distribution — it tied on field error and was *worse* on U-MAE. Diagnosis:
with abundant matched data the data loss already suffices, and the residual was
evaluated on the resampled training grid (where even ground truth has a ~8e-4
residual floor). Physics priors are expected to pay off in low-data / OOD regimes,
which Block-1 in-distribution does not test → deferred to the generalization study.

### Exp 1.3 — model ablation: bridging where FNO lacks (3 seeds, A100, 300 ep) ⭐
ADR [`0004`](decisions/0004-block1-model-ablation.md). Artefact:
`results/block1_ablations.{json,md}`. Eight variants, each a hypothesis about a
*specific* FNO weakness on this problem; mean±std over seeds [1337, 1, 2]; U-MAE
stratified by thermal-bridge presence (15 clear / 49 bridged val samples).

Diagnosed weaknesses of data-only FNO:
- **field-L2 ≠ U-MAE** — U depends on the near-boundary θ-gradient, not the bulk;
- **spectral bias** — truncated Fourier modes smear the *sharp, local* bridge response;
- **non-periodicity** — the FFT assumes periodic boundaries, but our through-wall
  faces are Dirichlet, corrupting the very region that sets U.

| Variant | Field rel-L2 | **U-MAE** | U-MAE clear | U-MAE bridge | Targets | Robust win |
|---|---|---|---|---|---|---|
| **delta_fno** | 0.0131±0.0002 | **0.0105±0.0009** | 0.0017 | 0.0133 | wasted capacity on bulk | **yes** |
| delta_fno_uloss | 0.0132±0.0002 | 0.0111±0.0005 | 0.0015 | 0.0141 | delta + metric supervision | yes |
| fno_enriched | 0.0139±0.0004 | 0.0162±0.0017 | 0.0058 | 0.0194 | prior as feature (not additive) | yes |
| ufno | 0.0136±0.0002 | 0.0196±0.0010 | 0.0032 | 0.0247 | spectral bias on bridges | yes |
| fno_uloss | 0.0157±0.0009 | 0.0200±0.0009 | 0.0072 | 0.0240 | field-L2 ≠ U-MAE | yes |
| **fno** *(reference)* | 0.0147±0.0004 | 0.0242±0.0034 | 0.0076 | 0.0293 | — | — |
| fno_physics (w=0.05) | 0.0144±0.0003 | 0.0248±0.0035 | 0.0084 | 0.0299 | PDE consistency | no |
| fno_padded ([0.25,0]) | 0.0156±0.0008 | 0.0256±0.0064 | 0.0115 | 0.0299 | non-periodicity | no |

**Winner: `delta_fno`** — predicts a *correction* to the analytic per-column 1-D θ
prior (the prior is an input channel derived purely from k/dx0, not the target).
It **cuts U-MAE 0.0242 → 0.0105 (−57 %), robust across seeds** (Δ 0.0137 ≫ pooled
σ 0.0035), and improves field error 11 %. The gain concentrates where it should:
**−55 % on bridged walls, −78 % on clear walls**. ~11× better than the 1-D baseline.

What we learned (all of it useful for the paper as "alternatives explored"):
1. **The additive structure is the lever, not just the feature.** `fno_enriched`
   sees the same prior as a channel but predicts the field directly → 0.0162;
   `delta_fno` predicts only the correction → 0.0105. Isolating the hard 2-D bridge
   physics from the trivial 1-D bulk is the win.
2. **U-value supervision is redundant once you have the prior** (delta_fno_uloss ≈
   delta_fno). On its own it helps a little (fno_uloss 0.0200).
3. **Local-conv hybrid (ufno) helps modestly** (0.0196) — sharp-feature capacity matters.
4. **Physics-residual loss and domain padding did *not* beat the baseline** in this
   in-distribution regime (padding's symmetric zero-pad even hurt, with high variance).
5. **Multi-seed was essential** — the single-seed "0.0205" (Exp 1.1/1.2) was an
   optimistic draw; the honest reference is 0.0242±0.0034.

**Decision:** adopt `delta_fno` as the Block-1 experimental default
(`configs/experiment/block1_synthetic_fem.yaml`). The other seven remain registered,
config-selectable competitive alternatives.

### Exp 1.4 — out-of-distribution generalization (3 seeds × 2 regimes, A100, 300 ep) ⭐
ADR [`0005`](decisions/0005-ood-generalization.md). Artefact:
`results/block1_ood.{json,md}`. The in-distribution winner has to *travel* before it
earns a place in the paper, so this study asks one question: **does `delta_fno`'s lead
survive distribution shift, and does the soft physics-residual loss finally pay off
when data is scarce?**

Five variants carried forward from the ablation — `delta_fno` (winner), `fno`
(reference), `fno_uloss` and `ufno` (the strongest non-prior in-distribution
challengers), and `fno_physics` (the soft-physics hypothesis we want to retest
out-of-distribution). Each is trained once per (regime × seed), seeds `[1337, 1, 2]`,
and scored at **native resolution** on five held-out 64-sample test sets, featurised
with its own feature set. Mean±std over seeds.

**Two data regimes**, to separate "more data" from "better prior":
- **full** — train on all 256 of `block1_train`;
- **lowdata** — a fixed seeded 64-sample subset (¼ of the data).

**Four OOD axes, one physically meaningful shift each.** Crucially, the boundary
*temperatures* are held fixed across every split. θ=(T−T_out)/(T_in−T_out) is
**invariant to the absolute indoor/outdoor temperatures** under linear steady
conduction — only their being unequal matters — so shifting BC temperatures would be
a *no-op* OOD test (the well-posed target does not move). The genuine covariate shifts
are therefore in the geometry, the material assembly, the surface films, and the
discretisation:

| OOD set | Shift | Why it is OOD |
|---|---|---|
| `ood_walls` | unseen wall assemblies (layer materials / thicknesses) | the as-built envelope is a new construction |
| `ood_films` | `r_si`/`r_se` surface resistances outside the training band | unseen interior/exterior convective regimes |
| `ood_bridges` | denser / wider thermal-bridge regime | more aggressive geometry than trained on |
| `ood_res` | finer through/along-wall discretisation | the operator must be mesh-agnostic |

#### In-distribution (`block1_val`) — the reference point each gap is measured from

| Variant | U-MAE [full] | rel-L2 [full] | U-MAE [lowdata] | rel-L2 [lowdata] |
|---|---|---|---|---|
| **delta_fno** | **0.0105±0.0009** | 0.0131±0.0002 | **0.0168±0.0008** | 0.0244±0.0010 |
| ufno | 0.0196±0.0010 | 0.0136±0.0002 | 0.0376±0.0022 | **0.0206±0.0014** |
| fno_uloss | 0.0200±0.0009 | 0.0157±0.0009 | 0.0393±0.0023 | 0.0298±0.0025 |
| fno | 0.0242±0.0034 | 0.0147±0.0004 | 0.0657±0.0063 | 0.0286±0.0025 |
| fno_physics | 0.0248±0.0035 | 0.0144±0.0003 | 0.0628±0.0044 | 0.0282±0.0027 |

#### OOD · `ood_walls` (the hardest shift)

| Variant | U-MAE [full] | rel-L2 [full] | Gap [full] | U-MAE [lowdata] | rel-L2 [lowdata] | Gap [lowdata] |
|---|---|---|---|---|---|---|
| **delta_fno** | **0.0680±0.0089** | **0.0542±0.0041** | **+0.0575** | **0.0405±0.0141** | **0.0529±0.0012** | **+0.0236** |
| ufno | 0.3026±0.0598 | 0.1353±0.0078 | +0.2830 | 0.4167±0.1167 | 0.1642±0.0200 | +0.3791 |
| fno_uloss | 0.3052±0.1410 | 0.1670±0.0075 | +0.2852 | 0.3445±0.1501 | 0.1997±0.0146 | +0.3052 |
| fno_physics | 0.3103±0.0915 | 0.1649±0.0042 | +0.2854 | 0.3554±0.0945 | 0.1965±0.0098 | +0.2926 |
| fno | 0.3228±0.1008 | 0.1662±0.0046 | +0.2986 | 0.3848±0.1055 | 0.1958±0.0091 | +0.3191 |

#### OOD · `ood_films`

| Variant | U-MAE [full] | rel-L2 [full] | Gap [full] | U-MAE [lowdata] | rel-L2 [lowdata] | Gap [lowdata] |
|---|---|---|---|---|---|---|
| **delta_fno** | **0.0617±0.0054** | **0.0209±0.0004** | **+0.0511** | **0.0410±0.0003** | **0.0326±0.0013** | **+0.0241** |
| fno_uloss | 0.1183±0.0435 | 0.0402±0.0007 | +0.0983 | 0.1676±0.0532 | 0.0509±0.0036 | +0.1283 |
| ufno | 0.1367±0.0274 | 0.0410±0.0030 | +0.1170 | 0.1907±0.0285 | 0.0471±0.0056 | +0.1531 |
| fno | 0.1424±0.0687 | 0.0396±0.0008 | +0.1182 | 0.1881±0.0571 | 0.0510±0.0045 | +0.1225 |
| fno_physics | 0.1465±0.0764 | 0.0398±0.0008 | +0.1217 | 0.1761±0.0593 | 0.0506±0.0047 | +0.1133 |

#### OOD · `ood_bridges`

| Variant | U-MAE [full] | rel-L2 [full] | Gap [full] | U-MAE [lowdata] | rel-L2 [lowdata] | Gap [lowdata] |
|---|---|---|---|---|---|---|
| **delta_fno** | **0.0491±0.0081** | **0.0435±0.0014** | **+0.0386** | **0.0440±0.0024** | **0.0590±0.0011** | **+0.0271** |
| fno_uloss | 0.1669±0.0080 | 0.0678±0.0011 | +0.1469 | 0.2714±0.0180 | 0.1100±0.0075 | +0.2321 |
| ufno | 0.1834±0.0231 | 0.0686±0.0015 | +0.1638 | 0.2769±0.0393 | 0.0880±0.0059 | +0.2392 |
| fno_physics | 0.2183±0.0278 | 0.0667±0.0027 | +0.1935 | 0.3334±0.0347 | 0.1113±0.0058 | +0.2706 |
| fno | 0.2205±0.0176 | 0.0664±0.0017 | +0.1963 | 0.3385±0.0326 | 0.1093±0.0064 | +0.2728 |

#### OOD · `ood_res` (the easiest — `delta_fno` is essentially resolution-invariant)

| Variant | U-MAE [full] | rel-L2 [full] | Gap [full] | U-MAE [lowdata] | rel-L2 [lowdata] | Gap [lowdata] |
|---|---|---|---|---|---|---|
| **delta_fno** | **0.0143±0.0008** | **0.0133±0.0003** | **+0.0037** | **0.0228±0.0019** | **0.0270±0.0007** | **+0.0060** |
| ufno | 0.0713±0.0111 | 0.0303±0.0036 | +0.0517 | 0.0921±0.0101 | 0.0336±0.0024 | +0.0544 |
| fno | 0.0965±0.0093 | 0.0189±0.0007 | +0.0723 | 0.1688±0.0165 | 0.0331±0.0034 | +0.1032 |
| fno_uloss | 0.0988±0.0155 | 0.0194±0.0011 | +0.0788 | 0.1588±0.0253 | 0.0345±0.0037 | +0.1195 |
| fno_physics | 0.0990±0.0091 | 0.0188±0.0007 | +0.0742 | 0.1605±0.0190 | 0.0322±0.0036 | +0.0977 |

**What the study shows.**

1. **`delta_fno` sweeps every OOD set on the primary metric — it does not lose
   anywhere.** It has the lowest U-MAE *and* the smallest generalization gap on **all
   8 OOD cells** (4 axes × 2 regimes). The leads are not marginal: full-regime
   `ood_walls` 0.0680 vs next-best 0.3026 (**4.4×**), `ood_bridges` 0.0491 vs 0.1669
   (**3.4×**), `ood_films` 0.0617 vs 0.1183, `ood_res` 0.0143 vs 0.0713. Its gaps stay
   nearly flat (full `ood_res` **+0.0037** vs ufno +0.0517; full `ood_walls` +0.0575
   vs +0.2830), whereas every prior-less variant roughly *triples* its U-MAE on the
   geometry/assembly shifts. The hard analytic 1-D θ prior is a property of the
   physics, not of the training distribution, so it carries across shifts that wreck
   the data-only models.

2. **Unseen wall assemblies are the binding generalization risk.** `ood_walls` is the
   hardest axis for everyone (mean gap +0.2419 full / +0.2639 lowdata, vs `ood_res`
   +0.0561 / +0.0762), and the prior-less models effectively collapse to the
   geometry-blind regime there. For as-built scans, where every building is a *new*
   assembly, this is the axis the architecture must be designed and evaluated against.

3. **The in-distribution auxiliary winners are overfitters.** `ufno` (in-dist U-MAE
   0.0196, 2nd only to `delta_fno`) posts the single largest gap in the whole study
   (lowdata `ood_walls` **+0.3791**) and collapses on `ood_bridges`; `fno_uloss`
   (0.0200 in-dist) carries a +0.2852 wall gap. Their auxiliary objectives buy
   in-distribution accuracy that does **not** survive shift — a cautionary tale for
   reporting only in-distribution numbers.

4. **The soft physics-residual loss earns a marginal keep only in low data.** Retesting
   the ADR-`0003` hypothesis: in the **full** regime `fno_physics` vs data-only `fno`
   is a wash — *worse* on 3 of 5 cells (in_dist +0.0006, ood_films +0.0041, ood_res
   +0.0025), better only on ood_walls (−0.0126) and ood_bridges (−0.0023). In the
   **lowdata** regime it flips to a **consistent win on all 5 cells**: in_dist −0.0028,
   ood_walls −0.0293, ood_films −0.0120, ood_bridges −0.0051, ood_res −0.0083. So the
   residual does help when data is scarce — but every delta is small relative to the
   seed σ (~0.03–0.10), so this is a **directional, not decisive** signal. And it is
   dominated: the *hard* analytic prior in `delta_fno` is ~8× better in lowdata OOD
   than the *soft* residual buys on top of plain FNO. Ranked by how much a prior
   helps in low data: hard θ-channel (`delta_fno`) ≫ U-value supervision (`fno_uloss`)
   > soft PDE residual (`fno_physics`).

**Hypothesis verdict.** Both Exp-1.4 hypotheses are confirmed, but with a sharpened
mechanism: `delta_fno` does widen its lead OOD (decisively), and the physics-residual
loss does help in low data (but only marginally, and only the soft variant — the
hard analytic prior, not the soft loss term, is what actually buys OOD robustness).

**Conclusion.** A *hard, verified analytic prior wired into the architecture* (the
additive 1-D θ channel) is what generalizes; *soft auxiliary objectives* (U-loss,
PDE-residual) buy in-distribution accuracy that mostly evaporates under shift.
`delta_fno` is confirmed as the Block-1 default and the lead contribution to carry
into Block-2: report U-MAE under **per-axis OOD splits with unseen assemblies as the
headline stressor**, keep the PDE-residual loss only as a low-data consistency rail
(not an accuracy claim). See ADR [`0005`](decisions/0005-ood-generalization.md).

---

## Block 2 — 3-D geometry-conditioned operator (GINO + delta prior)

Block-1 settled the *recipe* — a hard analytic 1-D clear-wall θ prior wired into the
architecture as an additive correction (`delta_fno`) — and proved it **travels**
(Exp 1.4: lowest U-MAE and smallest generalization gap on all 8 OOD cells, 1.9–5.0×
ahead). Block-2's question is whether that recipe **carries off the regular grid** onto
the irregular point clouds of real as-built scans, with **GINO** (Li et al., 2023) as
the geometry-conditioned backbone. Two strands:

1. **A synthetic 3-D benchmark** (Exp 2.1) that keeps the physics exact and the
   geometry simple enough to isolate the operator's behaviour: does the delta prior
   still beat a data-only GINO, and does either point-cloud operator match a
   regular-grid voxel-FNO that *does* see a grid?
2. **A real-geometry on-ramp** — a CityGML reader that lifts TUM2TWIN LoD2/LoD3 city
   models into the same `Envelope` the featuriser already consumes, so the operator can
   be exercised on genuine tessellated shells (thermal labels still gated on measured
   IR; see *Limitations*).

ADR [`0007`](decisions/0007-block2-gino-3d.md). See also [`experiment-plan.md`](experiment-plan.md).

### Exp 2.1 — GINO vs delta-GINO vs voxel-FNO on 3-D wall blocks (2 seeds, A100, 150 ep)

**Corpus.** `data/synthetic_3d.py` is the 3-D analogue of the Block-1 generator: a wall
*block* layered through axis 0 (the through-wall direction), homogeneous across the two
in-plane axes, punctured by **rectangular-prism** thermal bridges that are now finite in
*both* in-plane axes (a stud/nib, not a 2-D strip) and target the insulation layer
(ADR `0006`). Each block is solved by `physics/steady_fv` (axis-0 Dirichlet/film, all
other faces adiabatic) into a genuinely 3-D θ field, then turned into a **GINO sample**:
~2k points drawn uniformly inside the block (coords normalised to `[0,1]³`), each carrying
features `[logk_std, r_si, r_se, theta1d]` (the analytic 1-D prior evaluated at the point's
through-wall position from its local k-column), the per-point target θ (trilinearly
interpolated from the FV field), and the analytic box SDF on a regular `G³` (G=16) latent
grid. Train 96 / val 32 samples, seeded. Artefact: `data/processed/block2_{train,val}`.

**Roster** (all kept — alternatives are part of the record):

| Model | Features | Structure | Role |
|---|---|---|---|
| **delta_gino** | `[logk_std, r_si, r_se, theta1d]` (4) | predicts a *correction* added to the per-query 1-D prior | the Block-1 winning recipe, now on irregular geometry |
| gino | `[logk_std, r_si, r_se]` (3, **prior dropped**) | predicts θ directly from the cloud | data-only ablation — must learn θ from scratch |
| fno_voxel | voxelised `[logk_std, r_si, r_se, theta1d]` (4) | 3-D FNO over the dense `16³` voxel field | the regular-grid reference the cloud operators must match without seeing a grid |

All three score **field relative-L2** and **U-value MAE** on the **same support** — the
original sampled points (the voxel field is trilinearly resampled back to the cloud) —
with the **same U-from-indoor-face estimator** (`eval/building.u_from_indoor_face_cloud`,
near-face band 0.08 in normalised coords) applied identically to every model and to the
ground truth. U is read as the indoor-face dimensionless-deficit ratio against the
analytic prior: `U ≈ U_clear · mean(1−θ_face) / mean(1−θ1d_face)` — exact on a clear
column (where θ ≡ θ1d ⇒ U = U_clear), so it estimates only the bridge-driven excess and
never touches the target U (leakage-free). Each model is run over seeds `[1337, 1]`,
150 epochs, AdamW + cosine, batch 1. Runner: `scripts/benchmark_block2.py` →
`results/block2_benchmark.{json,md}`.

#### Results — first pass (job 26446105, feit-gpu-a100, **60 ep, 1 seed**)

A *directional* run only — the GINO GPU bottleneck (below) capped what we could
afford, so this is undertrained (60 ep vs Block-1's 300) and single-seed (`±0.0000`
is *one seed*, not consistency). Reported as-is.

| Model | Field rel-L2 ↓ | U-MAE ↓ (W/m²K) | U-MAPE | vs 1-D clear |
|---|---|---|---|---|
| **fno_voxel** (grid baseline) | **0.0202** | **0.0475** | 11.3% | 1.63× |
| gino | 0.0238 | 0.0539 | 12.4% | 1.44× |
| delta_gino | 0.0243 | 0.0516 | 11.8% | 1.50× |

1-D clear-wall baseline U-MAE: 0.0776 W/m²K.

**Answers to the two questions:**
- **Did the delta prior carry to 3-D? — Marginally / not yet.** `delta_gino` improves
  U-MAE over `gino` by only ~4 % (0.0516 vs 0.0539) and is slightly *worse* on field
  error — nothing like the −57 % the additive prior bought on the 2-D grid. Single
  seed + 60 ep means even this could be noise. Verdict: the prior does **not** carry
  strongly to the cloud/GINO setting in this configuration.
- **GINO vs voxel-FNO — the grid baseline wins.** `fno_voxel` beats both GINO models
  on both metrics. This is the *expected* outcome and the reason the baseline is in the
  roster: **a grid FNO suffices on axis-aligned box geometries — GINO has no edge where
  the geometry is regular.** Its value proposition is *irregular* as-built geometry,
  which this synthetic block does not exercise.

**What this does and does not establish.** It establishes that the machinery works
end-to-end (GINO + delta_gino train, the point-cloud→U-value pipeline is sane, the
metrics are well-behaved) and that all three only modestly beat the geometry-blind
baseline (1.4–1.6×, vs 5–11× in 2-D — the 3-D bridges carry a smaller U-penalty here,
~13–27 % vs 40–50 %). It does **not** yet test GINO's actual advantage. Two structural
limits to lift before any real conclusion:
1. **GINO GPU efficiency.** Training was CPU-bound (~10 min/model for 60 ep): Open3D in
   this env is CPU-only and, even after installing the CUDA `torch_scatter`/`torch_cluster`
   wheels, batch-1 GINO on ~2 k-point clouds is *launch-overhead-bound* (thousands of
   µs kernels driven by the per-sample Python loop). A full 300-ep × multi-seed run is
   ~5 h as-is. Fix (TODO, ADR-worthy): batch samples per step / `torch.compile` / CUDA
   graphs. See `docs/compute.md`.
2. **Geometry that needs GINO.** Benchmark on the **real CityGML buildings** (below) or
   genuinely irregular synthetic geometry — Exp 2.2 — where a voxel grid is a poor fit
   and GINO's point-cloud/SDF encoding should finally pay off.

### CityGML real-geometry ingestion (TUM2TWIN) — landed

`geometry/citygml.py` lifts a TUM2TWIN CityGML 2.0 building (EPSG:25832, parsed with
stdlib `xml.etree` — no lxml/pyproj) into our `Envelope`: each `bldg:boundedBy` thematic
surface (`WallSurface`/`RoofSurface`/`GroundSurface` → Wall/Roof/Floor, boundary
Outdoors/Ground), its `gml:posList` exterior rings read to `(n,3)` vertices (closing
duplicate dropped), coordinates made **local-metric** by subtracting the per-building min
corner. LoD2 gives one clean polygon per surface; LoD3 tessellates into many triangles
(one `Surface` per exterior ring, synthesised names) and carries window/door holes as
interior rings (**dropped for v1** — treated as opaque wall). A truncated file
(e.g. `DEBY_LOD3_4907506.gml`) makes `ET.parse` raise; the single-building API re-raises
a clear error, the directory API skips-and-logs.

CityGML carries **geometry only**, so each surface is paired with a material/construction
from a small default library keyed by surface type (plausible mid-European constructions,
**not measured values** — placeholders until real material assignment lands). The
resulting `Envelope` flows **unchanged** into `geometry/pointcloud.py` (feature-tagged
cloud) and `geometry/sdf.py` (mesh + SDF), exactly as the synthetic DOE/IDF envelopes do.

Status: **all 27 LoD2 buildings parse** (0 skipped, 899 surfaces total). Sample
`DEBY_LOD2_4906980`: 7 surfaces (5 Wall, 1 Roof, 1 Floor), 6 Outdoors + 1 Ground,
local-metric (max |coord| ~12 m), watertight shell mesh (`is_watertight=True`), envelope
UA ~62.4 W/K, feeds the point cloud (512 finite points) and SDF unchanged. Gated by
`tests/test_citygml.py` (9/9: poslist parsing incl. ragged-input rejection, inline-CityGML
lift to Envelope, truncated-file error, directory skip-and-log, and three integration
tests on the real LoD2 corpus). LoD3 (`.gml` files 3–80 MB) is handled by the same
code path but not exercised in the local <30 s test budget.
