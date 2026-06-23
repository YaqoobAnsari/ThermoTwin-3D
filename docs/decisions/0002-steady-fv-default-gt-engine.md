# 2. scipy finite-volume as the default steady-state GT / residual engine

Date: 2026-06-23

## Status

Accepted

## Context

We need a steady-state heat-conduction solver for two jobs: generating Block-1
ground-truth temperature/heat-flux fields, and providing the discrete operator the
PINN residual loss is validated against. `docs/architecture.md` names FEniCS as the
default ground-truth generator (with COMSOL as a higher-fidelity cross-check).

FEniCS/dolfinx is powerful but heavyweight on Spartan: the conda-forge build is
fragile (it is deliberately commented out in `env/requirements.txt`), it pulls a
large dependency tree, and it is overkill for the *controlled* flat-and-blocky
geometries that Block 1 starts from. We also want every learned number gated
against the closed-form 1-D oracle (`physics/conduction.py`), which demands a
solver whose discretisation provably reproduces EN ISO 6946 series resistance.

## Decision

Use a self-contained **cell-centred finite-volume solver in scipy**
(`physics/steady_fv.py`) as the *default* steady-state engine. Face conductivity
is the exact series combination of half-cell resistances (`A / (R_i + R_j)`),
which for piecewise-constant `k` reproduces `R = Σ d/λ` to machine precision at any
resolution. Grids may be non-uniform per axis so real layer thicknesses are exact.
The solver supports 1-D/2-D/3-D, Dirichlet and Robin (film) BCs on the through-axis,
adiabatic lateral faces.

FEniCS and COMSOL (the latter is available as a Spartan module) remain the
**higher-fidelity cross-checks** for transient behaviour, complex/curved geometry,
and convergence audits — not the default path.

## Consequences

- **+** No fragile dolfinx build; CPU-testable; zero-tolerance unit test against the
  analytic oracle (`tests/test_steady_fv.py`).
- **+** The same discrete operator backs both GT generation and the PINN residual,
  so "physics-consistent" means consistent with a solver we fully control.
- **−** Not a general unstructured-mesh FEM: curved/as-built envelope geometry will
  need either a voxelised SDF representation fed to this solver, or the FEniCS
  cross-check path. Revisit when Block-2 real geometry lands.
- Transient conduction + zone coupling are future work (see architecture "steady vs
  transient" open decision).
