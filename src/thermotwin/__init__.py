"""ThermoTwin-3D: a geometry-resolved, physics-informed thermal twin of building
envelopes learned from as-built scans.

Subpackages
-----------
geometry     Point-cloud / SDF / mesh ingestion and per-surface material layers.
physics      Transient heat-conduction operators, boundary conditions, FEM/E+ GT.
data         Dataset loaders (TUM2TWIN, TBBR, DOE Reference Buildings, synthetic FEM).
models       GINO backbone and neural-operator / PINN baselines.
losses       PDE-residual and physics-consistency losses.
calibration  Inverse problem: assimilate measured IR to infer envelope properties.
training     Training loops, optimizers, schedulers.
eval         Field + building-relevant metrics and speedup benchmarking.
viz          Heat-loss maps and field visualisation.
utils        Config, logging, seeding, IO helpers.
"""

__version__ = "0.0.1"
