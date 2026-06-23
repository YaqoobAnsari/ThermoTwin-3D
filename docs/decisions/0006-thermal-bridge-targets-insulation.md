# 6. Thermal bridges target the insulation layer, not the thickest layer

Date: 2026-06-24

## Status

Accepted

## Context

The synthetic corpus places thermal bridges by puncturing a layer of the wall with
a more-conductive inclusion. `random_sample` originally chose that layer as the
**thickest** one (`argmax(thickness)`). For the `light_framed` assembly the thickest
layer *is* the mineral-wool insulation, so this was fine. But for the
`mass_insulated` assembly the thickest layer is the structural **concrete**
(k = 1.95), not the EPS insulation (k = 0.035). A low-conductivity bridge material
(timber, k = 0.13) placed in concrete is *less* conductive than what it replaces, so
it acts as a mild **thermal break** and slightly *lowers* the effective U-value —
the opposite of a thermal bridge.

The OOD foundation agent surfaced this empirically: ~11 % of `block1_train` samples
(28/256) had `u_value < u_clear`, by at most ~0.4 % in-distribution and up to ~2 %
in the deliberately bridge-heavy `ood_bridges` set.

## Decision

Target the **lowest-conductivity layer** (`argmin(conductivity)` — the insulation)
and restrict bridge materials to those **strictly more conductive than that layer**.
With the insulation as the host, every bridge material (timber/concrete/steel/
aluminium) is genuinely more conductive, so a bridge always *raises* U. Enforced by
`tests/test_synthetic_fem.py::test_bridges_are_genuine_and_raise_u`.

## Consequences

- **+** The corpus now matches the physical semantics of a "thermal bridge"
  everywhere; no thermal-break artefacts. Cleaner story for the paper.
- **−** The committed Block-1 corpora (`block1_train`/`block1_val`, the four `ood_*`
  sets) and the results computed on them (Exp 1.1–1.4, `results/block1_*.{json,md}`)
  predate this fix. **The conclusions are unaffected** — `delta_fno` wins by 2–10× on
  every split and every metric, far larger than the ≤2 % artefact — but the *final*
  paper numbers should be produced from a regenerated corpus. That regeneration is a
  cheap, seeded follow-up (no code changes beyond this fix); it is intentionally
  deferred so it can be batched with the next full benchmark pass rather than
  re-running everything now.
- The 3-D Block-2 corpus inherits the corrected targeting from the start.
