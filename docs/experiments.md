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

### Exp 1.4 — out-of-distribution generalization (planned)
Stress-test the roster on unseen BC ranges, unseen wall assemblies, cross-resolution,
and low-data splits. Hypothesis: `delta_fno` widens its lead, and the physics-residual
loss finally helps *on top of it* in the low-data regime. Results to be appended here.

---

## Block 2 — real-building validation (in progress)

Wiring the geometry-conditioned operator (GINO) onto the point-cloud + SDF featuriser
(`geometry/pointcloud.py`, `geometry/sdf.py`) for validation against measured thermal
data (TUM2TWIN, TBBR). See [`experiment-plan.md`](experiment-plan.md). Results here as
they land.
