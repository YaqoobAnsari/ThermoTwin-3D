"""Steady-state 1-D heat conduction through a multilayer building element.

This is the analytic backbone and first sanity check for ThermoTwin-3D. A planar
multilayer wall obeys Fourier's law in series, so its thermal transmittance (U-value)
is the inverse of the summed thermal resistances (EN ISO 6946). Every learned /
operator prediction over a flat assembly must reproduce these closed-form numbers,
so this module doubles as ground truth for ``eval/`` and as a unit-test oracle.

Units
-----
thickness     m
conductivity  W / (m K)
resistance    m^2 K / W
U-value       W / (m^2 K)
heat flux     W / m^2     (positive = indoor/warm side -> outdoor)
temperature   deg C       (any consistent linear scale works)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

__all__ = [
    "Layer",
    "SurfaceFilm",
    "WallResult",
    "total_resistance",
    "u_value",
    "steady_state_1d",
]


@dataclass(frozen=True)
class Layer:
    """A single homogeneous material layer of a building element."""

    name: str
    thickness_m: float
    conductivity_w_mk: float

    def __post_init__(self) -> None:
        if self.thickness_m <= 0:
            raise ValueError(
                f"layer {self.name!r}: thickness_m must be > 0, got {self.thickness_m}"
            )
        if self.conductivity_w_mk <= 0:
            raise ValueError(
                f"layer {self.name!r}: conductivity_w_mk must be > 0, got {self.conductivity_w_mk}"
            )

    @property
    def resistance(self) -> float:
        """Thermal resistance ``R = d / lambda``  [m^2 K / W]."""
        return self.thickness_m / self.conductivity_w_mk


@dataclass(frozen=True)
class SurfaceFilm:
    """Internal/external surface (film) resistances.

    Defaults follow EN ISO 6946 for horizontal heat flow:
    ``r_si = 0.13``, ``r_se = 0.04`` m^2 K / W.
    """

    r_si: float = 0.13
    r_se: float = 0.04


@dataclass(frozen=True)
class WallResult:
    """Result of a steady 1-D conduction solve through one assembly."""

    r_total: float  # m^2 K / W
    u_value: float  # W / (m^2 K)
    heat_flux: float  # W / m^2
    node_names: tuple[str, ...]
    node_temperatures: tuple[float, ...]  # deg C, indoor air -> outdoor air

    def summary(self) -> str:
        head = (
            f"R_total = {self.r_total:.4f} m^2K/W   "
            f"U = {self.u_value:.4f} W/m^2K   "
            f"q = {self.heat_flux:.3f} W/m^2"
        )
        rows = [
            f"  {name:<24s} {temp:8.3f} degC"
            for name, temp in zip(self.node_names, self.node_temperatures)
        ]
        return "\n".join([head, *rows])


def total_resistance(layers: Sequence[Layer], film: SurfaceFilm | None = None) -> float:
    """Total thermal resistance ``R_si + sum(d/lambda) + R_se``  [m^2 K / W]."""
    if not layers:
        raise ValueError("need at least one layer")
    film = film or SurfaceFilm()
    return film.r_si + sum(layer.resistance for layer in layers) + film.r_se


def u_value(layers: Sequence[Layer], film: SurfaceFilm | None = None) -> float:
    """Thermal transmittance ``U = 1 / R_total``  [W / (m^2 K)]."""
    return 1.0 / total_resistance(layers, film)


def steady_state_1d(
    layers: Sequence[Layer],
    t_indoor: float,
    t_outdoor: float,
    film: SurfaceFilm | None = None,
) -> WallResult:
    """Solve the 1-D steady conduction profile through a multilayer assembly.

    In series the same heat flux ``q`` passes through every resistance, so each
    temperature drop is ``q * R``. Returns the U-value, the flux, and the
    temperature at every node from indoor air through each material interface to
    outdoor air.
    """
    film = film or SurfaceFilm()
    r_total = total_resistance(layers, film)
    u = 1.0 / r_total
    q = u * (t_indoor - t_outdoor)

    resistances = [film.r_si, *(layer.resistance for layer in layers), film.r_se]

    names = ["indoor air", "internal surface"]
    for i in range(len(layers) - 1):
        names.append(f"{layers[i].name} | {layers[i + 1].name}")
    names += ["external surface", "outdoor air"]

    temps = [float(t_indoor)]
    for r in resistances:
        temps.append(temps[-1] - q * r)
    # temps[-1] equals t_outdoor up to floating-point error.

    return WallResult(
        r_total=r_total,
        u_value=u,
        heat_flux=q,
        node_names=tuple(names),
        node_temperatures=tuple(temps),
    )


if __name__ == "__main__":
    demo_wall = [
        Layer("plasterboard", 0.0125, 0.25),
        Layer("mineral wool", 0.120, 0.035),
        Layer("brick", 0.200, 0.77),
    ]
    print(steady_state_1d(demo_wall, t_indoor=20.0, t_outdoor=0.0).summary())
