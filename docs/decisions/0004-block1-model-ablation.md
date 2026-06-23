# 4. Block-1 model ablation — beating the data-only FNO on U-MAE

Date: 2026-06-24

## Status

Accepted

## Context

Block-1 predicts the dimensionless steady temperature field θ=(T−T_out)/(T_in−T_out)
on 2-D wall cross-sections (axis-0 through-wall, axis-1 along-wall) punctured by
thermal bridges — high-k inclusions through the insulation. Two metrics matter: the
field relative L2 and, the building-relevant one, the **U-value MAE** (W/m²K). The
data-only FNO (300 epochs, A100) set the bar — field rel-L2 **0.0144**, single-seed
U-MAE **0.0205** (ADR `0003`) — and is **5.7×** better than the geometry-blind 1-D
clear-wall baseline (U-MAE 0.1168). That is the number to beat.

The key diagnosis: U-value is a boundary functional, not a bulk one. With
`U = Σ_lo g_bnd·(1−θ_lo) / area`, U-MAE is governed **entirely by the through-wall
θ-gradient at the indoor (axis-0 lo) face**, not by the bulk field. The plain FNO
loses there twice:

1. **Spectral bias.** The truncated Fourier basis smears sharp features — exactly
   the high-k bridge edges whose near-boundary gradient sets the U-value. The
   reference FNO's stratified U-MAE confirms this: clear **0.0076** vs bridge
   **0.0293** — the bridges are where it bleeds.
2. **Non-periodic boundary.** The FFT assumes periodicity, but the axis-0 faces are
   Dirichlet/film (non-periodic); the implicit wraparound contaminates the very
   boundary the U-value is read from.

A corollary worth stating: **field rel-L2 ≠ U-MAE.** A model can be excellent in the
bulk and still misread the boundary gradient. So the sweep optimises U-MAE directly
and tracks rel-L2 only to ensure we do not trade it away.

## Decision

Run a controlled ablation (`scripts/ablate.py`, `scripts/slurm/ablate.slurm`) over a
fixed roster of countermeasures, each chosen to target one of the diagnosed
weaknesses, trained over seeds `{1337, 1, 2}` and evaluated at **native resolution**
on the 64-sample val set, stratified by thermal-bridge presence (15 clear / 49
bridged). A *robust win* requires the mean U-MAE to fall below the reference by more
than the pooled (RSS) seed σ.

| Variant | What it changes | Which weakness it targets |
|---|---|---|
| `fno` *(ref)* | data-only FNO | — (the bar) |
| `fno_padded` | `domain_padding=[0.25, 0.0]` on the through-wall axis | non-periodic FFT boundary |
| `fno_enriched` | analytic 1-D clear-wall θ as an **input channel** | hands the operator the boundary structure |
| `delta_fno` | predicts `θ = θ_prior + fno(x)` (prior as **additive residual**) | spectral bias + boundary, by construction on clear columns |
| `ufno` | parallel **local Conv2d** path summed with the spectral path per block | spectral bias (local high-freq capacity) |
| `fno_uloss` | + differentiable indoor-face U-value loss (gradient on row 0 of θ only) | directly penalises the boundary functional |
| `delta_fno_uloss` | delta head + U-value loss | both of the above |
| `fno_physics` | + PDE-residual loss, weight 0.1 (ADR `0003`) | physics-consistency at the boundary |

The two prior-based variants (`fno_enriched`, `delta_fno`) carry the verified
closed-form 1-D field (`physics/conduction.py`) as the `enriched` `feature_set`
channel from `data/dataset.py`; the prior already reproduces the cell-centred FV
field to machine precision wherever the 1-D assumption holds.

## Results

Mean ± std over 3 seeds, 300 epochs · batch 64 · A100, native-resolution eval on 64
val samples. Full numbers in `results/block1_ablations.{json,md}`.

| Variant | Field rel-L2 ↓ | U-MAE ↓ (W/m²K) | U-MAE clear | U-MAE bridge | Robust win |
|---|---|---|---|---|---|
| **delta_fno** | **0.0131 ± 0.0002** | **0.0105 ± 0.0009** | 0.0017 | 0.0133 | **yes (3.9σ)** |
| delta_fno_uloss | 0.0132 ± 0.0002 | 0.0111 ± 0.0005 | 0.0015 | 0.0141 | yes (3.8σ) |
| fno_enriched | 0.0139 ± 0.0004 | 0.0162 ± 0.0017 | 0.0058 | 0.0194 | yes (2.1σ) |
| ufno | 0.0136 ± 0.0002 | 0.0196 ± 0.0010 | 0.0032 | 0.0247 | yes (1.3σ) |
| fno_uloss | 0.0157 ± 0.0009 | 0.0200 ± 0.0009 | 0.0072 | 0.0240 | yes (1.2σ) |
| fno *(ref)* | 0.0147 ± 0.0004 | 0.0242 ± 0.0034 | 0.0076 | 0.0293 | — |
| fno_physics | 0.0144 ± 0.0003 | 0.0248 ± 0.0035 | 0.0084 | 0.0299 | no (−0.0006) |
| fno_padded | 0.0156 ± 0.0008 | 0.0256 ± 0.0064 | 0.0115 | 0.0299 | no (−0.0014) |

Reading of the table:

- **Five of eight variants robustly beat the reference.** The winner `delta_fno`
  cuts U-MAE **−56%** (0.0242 → 0.0105, 3.9× the pooled σ) and posts the **best
  field rel-L2 in the whole sweep** — it improves the primary metric without
  trading off the secondary.
- **The win is stratified where it matters.** `delta_fno` collapses *both* strata:
  clear 0.0076 → 0.0017 (−78%, near machine-level — the analytic prior nails
  bridge-free columns by construction) and bridge 0.0293 → 0.0133 (−55%), so the
  **largest absolute gain (Δ 0.0160) lands on the hard bridged cases**.
- **Prior-as-residual beats prior-as-input.** `fno_enriched` feeds the same prior
  only as an input channel and gets ~half the gain (clear 0.0058 vs 0.0017); adding
  it back to the output (`delta_fno`) is what frees the network to spend all its
  capacity on the localized lateral-spreading correction.
- **The U-loss is redundant once the prior pins the boundary.** `delta_fno_uloss`
  is statistically indistinguishable from `delta_fno` (0.56σ apart); the simpler
  head wins. The U-loss does help the prior-less FNO (`fno_uloss` 0.0200 < 0.0242)
  but mildly regresses field rel-L2.
- **The two pure boundary-treatment levers lose.** `fno_padded` and `fno_physics`,
  which target the FFT-periodicity story without supplying the clear-wall
  structure, sit within noise of the reference; padding even doubles clear-wall
  error and adds the highest seed variance.

## Decision outcome

**Adopt `delta_fno` as the Block-1 backbone.** A geometry/physics prior that hands
the operator the boundary structure (and lets the network learn only the residual
bridge correction) is what moves U-MAE; loss-only and architecture-only boundary
tweaks without that prior do not.

## Consequences

- **+** A robust, well-understood **−56% U-MAE** improvement on the building-relevant
  metric with the **best field rel-L2** in the sweep, at the same parameter count
  (~308k) and train time (~96 s/seed). Gated by `tests/test_variants.py`,
  `tests/test_input_channels.py`, `tests/test_building_loss.py`.
- **+** The mechanism is principled and portable: a verified analytic prior as an
  additive residual removes the exact FNO weakness that sets U-MAE. The
  `feature_set`/`build_input_channels` plumbing and the `delta`-head pattern carry
  forward to Block-2 GINO once a clear-wall prior on irregular geometry exists.
- **+** Honest negative on two theses: domain padding and the PDE-residual loss are
  *not* the levers for in-distribution U-MAE here. The physics residual remains a
  consistency rail for the data-scarce / OOD regimes (ADR `0003`), not an accuracy
  win on this corpus.
- **−** The prior assumes a 1-D layered clear-wall decomposition; it is exact on the
  synthetic Block-1 walls but its strength on real, irregular as-built geometry is
  unverified — Block-2 must confirm the residual-on-prior framing survives the
  geometry shift, and the prior must be regenerated per-surface from material layers
  rather than read from a synthetic channel.
- **−** `delta_fno` requires the `enriched` feature set (≥ 4 channels with the
  clear-wall channel present); it is not a drop-in for the bare 3-channel input.
- The data-only `fno` is kept in the roster as the standing reference for future
  sweeps; `cnn`/`unet` remain the no-operator controls.
