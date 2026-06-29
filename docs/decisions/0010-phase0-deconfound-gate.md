# 10. Adversarial audit walks back the H1 verdict; a Phase-0 gate decides H1-rescue vs H2-pivot

Date: 2026-06-29

## Status

Accepted — **gate landed 2026-06-30: forward H1 is a null on real geometry → pivot to H2** (the
inverse twin), using the physics-prior-conditioned operator as a fast forward engine. Full verdict
at the end of this ADR.

## Context

After the full 14-model × 6-corpus matrix landed, the working narrative was "the delta-prior
recipe beats the analytic prior on real geometry, backbone-agnostically; `delta_pointnet2`
(0.17 M) is the lead." Before committing this to an ICLR submission, a five-angle adversarial
self-audit was run (read-only, evidence-cited against the committed code and result JSONs):

**Cleared (bank these — pre-empt in the paper).** No train/test **geometry leakage** (every real
corpus splits *by building* on a deterministically sorted list); no fitted-normalisation leakage
(fixed physical constants); the prior channel is the *analytic* 1-D solution, not GT-derived; the
bridge mask is applied identically to every model; the competitor backbones are faithful (real
`neuralop.GINO`, a verbatim Transolver port), not strawmen.

**Blockers.**
- **B1 — the trained operator is never validated on measured data.** All four "real validation"
  rungs are decoupled from the model: Twin Houses runs the analytic 1-D U *formula*; ThermoScenes
  is a k-NN smoother; TBBR/TIR are saliency heuristics. The network touches zero measured fields.
- **B2 — "beats the prior, backbone-agnostically" is not supported on real geometry.** On the real
  corpora 5 of 6 delta backbones have `correction_rel_l2 ≥ 1` (do not beat the zero-parameter
  prior); only `delta_pointnet2` does, by ~4%. On DOE `prior_only` wins field rel-L2 outright. The
  prior explains ~98% of the real-geometry field.
- **B3 — the core comparison is confounded.** "Delta" changes two things at once: the prior as an
  extra *input channel* (`feats` gains `theta1d`; data-only has it deleted) **and** a residual
  *target* (the model adds the prior back). `fno_voxel` (prior-as-input, full-field target) already
  recovers most of the gain, so "the residual recipe is decisive" is not isolated.
- **B4 — the GT is welded to the prior.** The prior is the FV solver's own per-column integral on
  the same conductivity field, so the residual is ≡ 0 off-bridge; the learning target is the
  hand-painted bridge prisms (`real_citygml_3d.py` widens them to 12–30% "so the benchmark can
  separate the operators").
- **B5 — U-MAE is vacuous on the real corpora.** `u_clear == u_true` ⇒ prior U-MAE = 0 ⇒ every
  learned model is strictly worse on exactly the datasets meant to validate the method.

**Majors.** "Real geometry" is synthetic per-surface *box* physics with the cloud rotated into
building coords; DOE never runs EnergyPlus; no 2-D/3-D field is checked against an independent
solver; tiny held-out building counts (realcg/LoD3 = 7, DOE = 4) on a single split whose seed-std
is init-noise not data-variance; one-size-fits-all LR handicaps the big transformers (confounding
"smallest wins"); no held-out test set; raw (not mean-removed) field rel-L2 flatters the prior;
the datasets are narrow/correlated (Munich + Amsterdam + idealised DOE, one climate); the
"physics enforced in the loss" thesis claim is empirically dead (`fno_physics` loses; the
PDE-residual loss is in no Block-2 run) — the real mechanism is analytic-prior + correction.

The deepest point: the prior explaining ~98% of the real-envelope field may be **physics, not a
bug** — real envelopes are mostly clear wall — in which case a forward neural operator cannot beat
the prior by much, ever, and chasing a forward-prediction H1 paper is fighting an opponent that is
already near-optimal.

## Decision

Run a cheap **Phase-0 decision gate** before any ICLR commitment, on the real-geometry corpus
(realcg, primary) and a synthetic contrast (hard). For four backbones spanning the families
(point / transformer / branch-trunk / GNO), add three leak-free ablation families to
`scripts/benchmark_block2.py` (existing kinds unchanged), plus a mean-removed metric and a measured
FV-solve time:

1. **`cond_<bb>`** — data-only architecture fed the full 4-ch features (incl `theta1d`) but
   predicting the FULL field (no residual add-back): prior as *input only*. Isolates B3.
2. **`delta_const_<bb>`** — the delta wrapper with `theta1d` and the added-back prior both replaced
   by a constant (train-mean θ): isolates the *physics* of the prior from "any prior + residual".
3. **`predict_mean`** — trivial constant floor.
4. **`field_rel_l2_fluct`** — mean-removed rel-L2, so the trivial through-wall gradient stops
   flattering the prior.
5. **`scripts/fem_speedup.py`** — times the FV (GT) solve, the missing denominator of the speedup
   claim.

**The fork.** On real geometry:
- **1 (`cond` ≪ `delta`) AND 2 (`delta_const` ≈ data-only) AND 3 (best `delta` beats `prior_only`)**
  → the forward result is real → proceed to the **H1-rescue** (Phase-1: independent-solver GT,
  measured-data model test, realistic coupled bridges, K-fold-over-buildings CV, per-family LR
  tuning, broader datasets) → aim ICLR.
- **1 ✓ 2 ✓ but 3 ✗** (recipe reconstructs the field but does not *beat* the prior on real
  envelopes) → the honest result is "a fast analytic+correction surrogate; the prior is near-optimal
  on real geometry" → **pivot to H2** (inverse twin), using the recipe as the engine. *Most likely.*
- **1 ✗ or 2 ✗** → the recipe framing collapses → **pivot to H2.**

Regardless of the fork: rewrite the dead "physics-in-the-loss" thesis claim, and wire the trained
operator into at least one measured-data test (B1).

## Consequences

- The committed `docs/baselines.md` verdict and `results/unified_eval.*` matrix are marked
  **superseded / under audit**; the earlier "delta beats physics, backbone-agnostically" framing is
  withdrawn pending the gate.
- 1-epoch DOE smoke already shows `cond_pointnet2` (0.41) ≈ data-only (0.52) ≫ `delta` (0.011) and
  `delta_const_pointnet2` (0.52) ≈ data-only — i.e. early signal that the residual *structure* and
  the *physics* of the prior both matter (favourable on B3, refuting "just prior-as-input"), while
  `delta` only *matches* `prior_only` on DOE (0.011 vs 0.0075) — the open question B2/Q3 decides.
- Apparatus committed at `5d0d725`; Phase-0 jobs `26617386` (realcg) / `26617387` (hard) +
  `fem_speedup` running; results land to `results/block2_*_phase0.{json,md}`.

### Interim results — lead-backbone read (pointnet2, seed 1337; full cross-backbone verdict pending)

The strongest backbone's full deconfounded quad is in on both corpora (the slower transolver/gino
blocks decide cross-backbone generalisation, ~6 h out). `correction_rel_l2` (↓, `prior_only` ≡ 1.0):

| variant | realcg (clear-wall, 4.5% bridge) | hard (severe sub-voxel bridge, 29%) |
|---|---|---|
| `predict_mean` (trivial floor) | 24.3 | 7.65 |
| `pointnet2` (data-only) | 12.65 | 1.15 |
| `delta_const_pointnet2` (constant prior) | 12.67 | 1.15 |
| `cond_pointnet2` (prior as input) | 0.916 | 0.371 |
| `delta_pointnet2` (recipe) | **0.881** | **0.360** |

Three findings, consistent across both corpora:
1. **Q3 — beats the prior: yes**, and *in proportion to bridge severity* — ~12% better than the
   zero-param prior on clear-wall realcg, ~64% better on severe-bridge hard. The operator earns its
   keep where the 1-D prior breaks.
2. **Q2 — the physics of the prior is essential: confirmed.** Replacing it with a constant
   (`delta_const`) collapses the model to data-only failure (12.67 ≈ 12.65; 1.15 ≈ 1.15). It is the
   physics, not "any prior + a network".
3. **Q1 — the residual *structure* is a minor refinement: confound real.** `cond` (prior as a plain
   input feature) ≈ `delta` (0.916 vs 0.881; 0.371 vs 0.360). The lever is *conditioning on the
   physics prior*; the residual add-back buys ~3–4%. (The 1-epoch DOE smoke that showed `cond` ≈
   data-only was an under-training artefact — at 300 epochs `cond` converges to ≈ `delta`.)

**Reframe this implies:** the defensible contribution is *"conditioning a geometry operator on a
closed-form physics prior lets it beat the prior on real geometry, scaling with bridge severity; the
physics is load-bearing and the residual form is a minor design choice"* — not "the residual recipe
is the magic."

**Speedup (`fem_speedup.json`):** FV solve = 59 ms (coarse) / 28 s (fine grid) vs ~2 ms model infer
⇒ ~30× to ~1.3×10⁴×; the 10³–10⁴× headline holds for fine-resolution GT.

**Open (decides the fork):** does this replicate on transolver/deeponet/gino, or is "beats the
prior" pointnet2-specific? The committed full-roster matrix hinted at the latter on real geometry —
the running blocks settle it.

## Verdict (2026-06-30) — forward H1 is a null on real geometry; pivot to H2

Both gate jobs completed (realcg 6h39m, hard 5h50m; 4 backbones × 4 variants + 2 controls × 3
seeds). `correction_rel_l2` (↓, `prior_only` ≡ 1.000) and the mean-removed `field_rel_l2_fluct`
(prior is the number to beat):

| | realcg corr | realcg fluct | hard corr | hard fluct |
|---|---|---|---|---|
| `prior_only` (zero-param) | 1.000 | 0.036 | 1.000 | 0.136 |
| `delta_pointnet2` | **0.894** | **0.034** | **0.358** | **0.051** |
| `delta_transolver` | 1.136 | 0.043 | 0.474 | 0.067 |
| `delta_deeponet` | 1.003 | 0.037 | 0.948 | 0.129 |
| `delta_gino` | 1.525 | 0.060 | 0.563 | 0.078 |

1. **On real geometry the operator does not beat the analytic prior in any robust, backbone-general
   way.** Only `delta_pointnet2` edges it (corr 0.894, ~11%; fluct 0.034 vs 0.036, ~6%); the other
   three backbones **tie or lose** to the zero-parameter prior (corr 1.14 / 1.00 / 1.53). On the
   honest mean-removed metric the prior is 0.036 and the *best* operator is 0.034 — a 0.2-point edge,
   from one backbone, on a 7-building single split whose error bars are init-noise. That is a null.
2. **The reason is physical, confirmed by the realcg↔hard contrast.** On `hard` (severe sub-voxel
   bridges) the operator wins decisively (corr 0.36–0.56, fluct 0.05–0.08 vs 0.136). The operator's
   value scales with bridge severity — and real envelopes are mostly clear wall, where the 1-D prior
   is near-optimal. Not a fixable bug; a property of the problem.
3. **Physics-prior-conditioning is real and essential** (`delta_const` collapses to data-only failure
   on realcg for every backbone) — but "essential to reconstruct the field" ≠ "beats the prior". The
   prior already reconstructs the field; conditioning lets the network *match* it, not exceed it.
4. **Speedup holds** (FV 59 ms coarse / 28 s fine vs ~2 ms infer ⇒ ~30×–1.3×10⁴×).

**Fork → pivot to H2 (the inverse twin).** A forward-prediction-beats-baselines paper is not
supported on real geometry. The honest, publishable assets that survive: (a) a well-characterised
fast forward *engine* (analytic prior + physics-prior-conditioned correction) for H2; (b) a clean
boundary result — "a closed-form 1-D prior is near-optimal for real envelope thermal fields; learned
operators earn their keep only in proportion to bridge severity" — as H2's motivation; (c) the
physics-prior-conditioning mechanism. H2 (calibrate against measured IR → per-surface U / bridges /
retrofit) is where geometry-resolution + measured data create the real, defensible contribution, and
is untouched by this audit. Cross-cutting items still required for H2: run the operator on measured
data (B1), and the corrected thesis framing (already done).
