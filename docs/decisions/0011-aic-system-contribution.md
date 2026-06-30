# 11. The contribution is the SYSTEM (Automation in Construction), not a residual mechanism

Date: 2026-06-30

## Status

Accepted — target **Automation in Construction**; ICLR dropped.

## Context

A three-agent adversarial novelty re-scan (2026-06-30) settled the novelty question ruthlessly:

- **Mechanism novelty: ~5/10 (will not survive a top-ML review).** Every leg of the proposed
  "boundary-layer uncertainty-aware residual" is 2025–26 prior art: the analytic-outer-prior +
  neural-operator inner correction is published (eFEONet **arXiv:2512.22006**, Dec 2025; PVD-ONet);
  the residual-on-physics-prior operator is published (Ma et al. **2606.03469**, Jun 2026); per-point
  operator UQ is published (REEF-GP 2606.17513, Evidential PINN); and the forward-UQ→inverse-likelihood
  hand-off is textbook covariance-inflation (review 2310.12046). The only fresh mechanistic twist —
  SDF-as-matched-asymptotic-stretched-coordinate on as-built interior interfaces — is incremental.
- **System novelty: ~8/10 (unoccupied).** No prior work joins the **scan→thermal-field** rail
  (geometry-conditioned operator) and the **inverse-thermography** rail (per-surface U + thermal-bridge
  conductance from IR) on **real as-built 3D building envelopes**, with uncertainty. The two
  literatures have never been joined; the building-envelope instantiation is empty.

## Decision

**Frame ThermoTwin-3D as a systems / construction-automation contribution for Automation in
Construction.** The headline is the end-to-end twin; the operator internals (forward recipe,
boundary-layer ablation, UQ) are method-section evidence, not novelty claims.

**The systems-novelty statement (defensible, 8/10 for AiC):**
> *The first geometry-resolved neural thermal twin of building envelopes that closes the loop from an
> as-built scan to a predicted thermal field to a differentiable inverse recovering per-surface
> U-values and localized thermal-bridge conductances from measured IR — demonstrating that a learned
> inverse resolves the ill-posedness pure optimization cannot, with quantified identifiability
> boundaries and uncertainty.*

We do NOT claim a new residual-learning mechanism. We claim the **join**, the **empirical findings**
(below), and the **building-domain instantiation**.

## Consequences — the evidence (what we have)

**Forward (the engine, honestly framed).** 14-model × 6-corpus gate (ADR 0010): the analytic 1-D
prior is near-optimal on real clear-wall envelopes; learned operators earn their keep in proportion
to bridge severity. The delta-prior recipe is the engine, not the contribution.

**Boundary-layer A/B ablation (`results/bl_*.json`, all 7 corpora) — an honest split:**
- *The SDF-keyed window is a tested-and-REJECTED hypothesis.* It does not measurably beat plain
  additive residual learning: bridge corr-relL2 Δ ≤ 0.011 on five corpora, a marginal +0.044 only
  on severe-bridge `hard`, and it *hurts* on axis-aligned `box` (−0.070); `delta_input ≈ bl`
  everywhere, so the window adds nothing beyond feeding interface-distance as a feature. This
  documents the Block-2 additive null as a finding: the boundary-layer *structure* does not earn
  its keep — consistent with its non-novelty.
- *The uncertainty head is the KEEPER.* The learned heteroscedastic UQ is well-calibrated
  (1σ coverage 0.69–0.72 ≈ ideal 0.68, err–σ correlation 0.64–0.80 on synthetic; conservative
  0.92–0.97 on real geometry), **dramatically beating the inverse twin's optimization-ensemble UQ**
  (1σ coverage 0.18–0.36). Calibrated per-point uncertainty + the reliability map is the
  non-refutable capability and the fix for the inverse's poorly-calibrated UQ.

**Inverse twin (the headline) — `results/inverse_{hard,realcg,realcg_lod3}.json`:**
- A **learned amortized inverse recovers the thermal-bridge field** from θ: bridge-localisation
  IoU **0.51** / recall 0.74 on severe `hard`; **precision 0.88–0.92** on real CityGML LoD2/LoD3
  geometry (high-confidence flags, recall 0.33–0.44 — the weak real bridges sit at the
  identifiability edge).
- **Ill-posedness, demonstrated and resolved:** pure optimization through the forward fits the
  observed field (data-fit 0.008–0.02) but recovers the *wrong* conductivity (logk rel-L2 > 1,
  i.e. worse than the clear-wall init — the inverse is non-unique); the learned amortized inverse
  resolves it (logk rel-L2 0.2–0.6). Hybrid (optimize-refine the learned solution) *degrades* it —
  the data-fit refinement pulls back toward the non-unique optimum.
- **U-value recovery is accurate** (U-MAE 0.001–0.04 W/m²K); on real geometry U≈U_clear so the
  inverse content is bridge localisation, not U.
- **Honest limitation:** the optimization-ensemble UQ is **poorly calibrated** (1σ coverage
  0.18–0.36 vs ideal 0.68; err–σ correlation 0.07–0.30) — motivating the learned heteroscedastic
  reliability head (BL+UQ, `boundary_layer.py`, jobs running) and an explicit calibration study.

**Cross-task real-data validation (corroboration):** Twin Houses (measured U), ThermoScenes
(measured °C field), TUM2TWIN-TIR (airborne IR), plus the new Dublin U-value asset (ISO-9869) —
the U-arm validators. Measured-data validation of the inverse is the key remaining experiment.

## Honest boundaries to pre-register (so a domain reviewer can't ambush us)

1. **GT credibility:** current bridges are simulated/painted; a credible AiC submission needs
   ISO-10211-validated bridge geometry (EUROKOBRA + THERM) for the headline bridge claim.
2. **Measured-data hand-off:** the inverse is so far validated on simulated GT + measured U; running
   the trained inverse on a measured 3-D field is the gap to close.
3. **Identifiability:** surface IR under-determines interior properties; U and Ψ (integrals) are the
   recoverable quantities, and the UQ must honestly reflect the residual non-uniqueness.

## Update 2026-06-30 — the three boundaries, addressed experimentally

Each pre-registered boundary above was turned into an experiment. The goal is not to make the
boundary disappear (two of them are real and permanent) but to *characterize* it, so a
building-physics reviewer meets a measured result instead of an unguarded assumption.

**1. GT credibility — closed (Exp 1).** Built an **independent general-boundary reference
conduction solver** (`src/thermotwin/physics/reference_solver.py`): a genuinely separate
implementation of the same finite-volume scheme, but with Dirichlet/Robin conditions on any
face(-region), per-patch flux, and the surface temperature factor `f_Rsi`. Credibility chain:
- reproduces the **analytic 1-D layered-wall U** with surface films (`tests/test_reference_solver.py`, 6/6);
- **passes ISO 10211 Case A.1** (the conduction-method validation case) against the closed-form
  Laplace solution: max error **0.010 K** away from the (0,8) corner singularity, vs the **0.1 K**
  ISO tolerance (`results/iso10211_validation.json`);
- **cross-validates the production solver to machine precision** on generated corpus cases
  (field rel-L2 max **3.2e-15**, U rel-err max **7.4e-14**; `results/solver_crosscheck.json`).

So the simulated bridge GT comes from an engine that is bug-free against an independent solver
that itself passes the thermal-bridge standard. ISO A.2–A.4 (material thermal-bridge details)
are deferred — they need the standard's exact 2-D/3-D figures.

**2. Measured-data hand-off — closed (Exp 2).** The differentiable inverse twin now runs
end-to-end on a **real calibrated 3-D thermal field** (`scripts/inverse_thermoscenes.py`): the
measured heat-loss residual is modeled as the graph-diffused footprint of a sparse, coherent
source field (`r ≈ Pᵐ s`, P = row-normalised kNN diffusion on the fused COLMAP facade points),
and the *same* regularised inverse (sparsity + kNN-TV + ensemble UQ) recovers the sources.
Validation is **relative and convergent**: the recovered map and an *independent*
local-statistics detector agree at a matched anomaly fraction — overlap IoU **0.09 / 0.19** on
BI-building / INR-building, i.e. **×4.3 / ×7.4 over a random-mask null** (mean ×5.8); ensemble
mask stability **~0.99**; data-fit rel-L2 0.72–0.76 (`results/inverse_thermoscenes/summary.json`).
Honest scope, stated in the outputs: relative source localisation, convergent (not held-out-label)
validation, **not** absolute U — the dataset ships no conductivity, materials or air temperatures.

**3. Identifiability — quantified (Exp 3).** `benchmark_inverse.py --identifiability` stores the
recovered conductivity field, its per-point ensemble spread, the true field, the through-wall
position and the distance to the nearest bridge per validation block, and adds an integration-scale
U decomposition, an **observation-masking sweep** (recover from surface-only / interior / full,
scored over the whole field — the clean lever, since axis-0 is through-wall), and a per-point
UQ-vs-error correlation; `scripts/analyze_identifiability.py` produces the figure +
`results/identifiability_hard.json`. The headline run (`hard`, severe bridges) makes the boundary
concrete:
- **Integrals recover, the full field does not.** Point-level recovered-field rel-L2 ≈ **1.16**
  (the optimization inverse lands *worse* than the clear-wall init — the per-point field is
  non-identifiable from θ), while the **U-MAE is 0.044–0.053 W/m²K (MAPE 11–14%)** and is stable
  across face-band and observation. The recoverable content is the integral, not the field.
- **Depth.** Per-point recovery error is **0.31 near the observed indoor surface vs 0.43 in the
  interior**, rising monotonically into the wall — you can read back what the surface flux pins,
  not the deep field.
- **What observing the interior buys.** Recovering from surface-only vs interior/full raises
  bridge-localisation IoU from **0.03 → 0.09–0.10** (≈3×) while U-MAE stays ~0.04 throughout. (This
  sweep uses the *optimization* inverse, which is non-unique — the *amortized* inverse reaches IoU
  0.51; the point here is the observation lever, not the absolute IoU.)
- **UQ tracks the non-uniqueness, but under-disperses.** Ensemble spread vs recovery error
  correlates positively (Spearman **0.49**), yet 1σ coverage is **0.38** (ideal 0.68) — the
  optimization-ensemble UQ is under-confident-calibrated exactly as flagged above, which is why the
  learned heteroscedastic reliability head is the fix.

Corroborated across corpora: the same pattern holds on real CityGML geometry (`realcg`,
`realcg_lod3`: near-clear-wall, U-MAE **0.001-0.005**, field rel-L2 $\sim$1.0) and on axis-aligned
`box` (simplest, hence the most field-recoverable: rel-L2 **0.46**, U-MAE 0.034). Integrals recover
everywhere; the full field never does.

Evidence: `results/inverse_{hard,realcg,realcg_lod3,box}.json` (identifiability block),
`results/identifiability_{hard,realcg,realcg_lod3,box}.json` + figures; per-point field dumps
`inverse_fields_*.npz` (regenerable, not committed). Headline run `sbatch
scripts/slurm/inverse.slurm hard`.

Net effect for AiC: the systems contribution now ships with an independently-validated GT engine,
a measured-data demonstration of the inverse, and a quantified identifiability envelope — the
three things a domain reviewer would otherwise press on.
