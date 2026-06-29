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
to bridge severity. The delta-prior recipe is the engine, not the contribution. Boundary-layer A/B
ablation (`results/bl_*.json`, running) tests whether the SDF-keyed window beats plain additive.

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
