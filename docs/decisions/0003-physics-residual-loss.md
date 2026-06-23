# 3. Physics-informed steady-conduction residual loss

Date: 2026-06-24

## Status

Accepted

## Context

The Block-1 operator predicts the dimensionless steady temperature field
θ=(T−T_out)/(T_in−T_out) on layered walls punctured by thermal bridges. The
data-only FNO baseline (300 epochs, A100) already sets a strong bar:

- field relative L2 **0.0144**
- U-value MAE **0.0205 W/m²K** (U-MAPE 4.9%), **5.70×** better than the
  geometry-blind 1-D clear-wall baseline (U-MAE 0.1168 W/m²K).

The thesis frames ThermoTwin-3D as *physics-informed*: the heat equation should be
enforced in the loss, not just regressed implicitly from labels. Before scaling to
real geometry (Block 2, GINO on point clouds + SDF) we want the physics term in
place and measured on the controlled Block-1 corpus, where the discrete operator is
fully known and the GT field satisfies it to solver tolerance (~1e-9).

A pure PINN collocation residual (autograd derivatives of a continuous field) would
not match our **finite-volume** ground truth: GT is the exact solution of a discrete
FV linear system, so the physically faithful residual is the *same discrete
operator*, not a continuum Laplacian. We already control that operator
(`physics/steady_fv`, ADR `0002`).

## Decision

Add `losses/heat_residual.py`: a PyTorch, autograd-enabled re-evaluation of the
`steady_fv` cell-centred FV operator as a **residual** on the predicted field,
rather than a solve. It operates directly on θ — the steady equation `∇·(k∇T)=0` is
linear and homogeneous, so the affine map `T→θ` rescales every conductance by the
same `1/(T_in−T_out)` and the per-cell balance stays at zero; indoor air is θ=1
(axis-0 *lo* face, film `r_si`), outdoor air is θ=0 (axis-0 *hi* face, film `r_se`).

Per-cell residual (net conductive heat into the cell [W]):

```
R_cell = Σ_{existing neighbour faces} g_face · (θ_nb − θ_cell)
         + [cell on lo face] · g_bnd_lo · (1 − θ_cell)
         + [cell on hi face] · g_bnd_hi · (0 − θ_cell)
```

with internal face conductance `g = A_face / ((dx_i/2)/k_i + (dx_j/2)/k_j)` (exact
series of the two half-cell resistances; `A_face = dy` for an axis-0 face, the
through-wall cell size for an axis-1 face), boundary film conductance
`g_bnd = A_face / ((dx_face/2)/k_face + r_film)`, and adiabatic lateral edges (no
term where a neighbour is absent). `heat_residual_loss` returns `mean(R_cell²)` over
all cells and the batch, differentiable wrt θ.

Wiring:

- `configs/train/default.yaml` gains `physics_weight` (default `0.0`). When `>0`,
  the total loss is `data_loss + physics_weight · heat_residual_loss(θ̂, …)`.
- `data/dataset.py` `return_physics=True` ships the per-batch bundle
  `{k, dx0, dy, r_si, r_se}` on the *same resampled grid* as `x`/`y`, so the residual
  is evaluated on the network's actual output grid.
- `scripts/train.py` activates the physics path when `physics_weight>0`.
  `scripts/benchmark.py` adds an `fno_physics` roster entry — the same FNO
  architecture trained with `physics_weight=0.1`.

This keeps the data-only path byte-for-byte unchanged (weight 0 is a no-op) and
shares one solver/operator between GT generation, the residual loss, and the
machine-precision oracle test.

## Consequences

- **+** "Physics-consistent" means consistent with the *exact discrete operator that
  generated the labels*, not an approximate continuum stencil — no FV-vs-PINN
  discretisation mismatch. Gated by `tests/test_heat_residual.py` (residual ~0 on
  GT, gradient checks).
- **+** Pluggable: any wired model can be trained physics-informed via one config
  knob; ready to carry into Block-2 GINO once a differentiable residual on
  irregular geometry exists.
- **−** Measured Block-1 delta vs the data-only FNO is **marginal**: at weight 0.1,
  `fno_physics` ties on field rel-L2 (**0.0143** vs 0.0144) but is **not** better on
  the building-relevant metric (U-MAE **0.0218** vs **0.0205 W/m²K**; U-MAPE 5.1% vs
  4.9%). On this corpus the data signal already saturates field accuracy, so the
  residual is presently a *consistency regulariser / safety rail* rather than an
  accuracy win — its expected payoff is in the data-scarce and out-of-distribution
  regimes (real scans, sparse IR) of later blocks. Weight and schedule are left as
  tunable; 0.1 is a conservative default, not an optimum.
- **−** The residual is steady-state and structured-grid only; transient conduction
  and unstructured/as-built geometry remain future work (tracked with the
  "steady vs transient" open decision in `architecture.md`).
