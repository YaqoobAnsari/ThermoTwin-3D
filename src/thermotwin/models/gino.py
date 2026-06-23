"""GINO — geometry-conditioned neural operator for irregular as-built scans.

Block-1 proved that the winning recipe is *delta learning on an analytic 1-D
clear-wall prior*: ``delta_fno`` predicts only the correction the prior misses,
which is where lateral spreading at thermal bridges bends the field. Block-2 carries
that recipe off the regular grid and onto the irregular point clouds of real scans.

The Geometry-Informed Neural Operator (Li et al., 2023) is the natural backbone: an
input GNO graph-encodes per-point features from arbitrary coordinates onto a regular
latent grid, an FNO mixes globally on that grid, and an output GNO graph-decodes back
to *any* query coordinates. Geometry enters twice — as the input/output point
coordinates and as the signed-distance field fed on the latent grid — so the operator
is conditioned on the as-built shape rather than assuming a box.

This module exposes:

* :class:`GinoOperator` — a thin wrapper over ``neuralop.models.GINO`` that fixes the
  verified-good construction (nonlinear input transform so the per-point feature width
  is decoupled from the FNO latent width; the SDF wired in as ``latent_features``) and
  normalises the leading batch / geometry dims of every forward argument.
* :class:`DeltaGino` — the delta-learning wrapper. Its output is
  ``query_prior + GinoOperator(correction)``, mirroring ``DeltaFNO`` exactly: the
  analytic 1-D theta prior, evaluated at each *output query's* through-wall position,
  is supplied per-query by the dataset and added back after the network predicts only
  the residual.

Unlike the grid models, GINO's forward signature is point-cloud-shaped
``(input_geom, x, latent_queries, sdf, output_queries) -> (B, n_out, 1)`` and does not
fit the ``(B, C, H, W)`` grid contract, so the Block-2 runner calls
:func:`build_gino` / :func:`build_delta_gino` directly.
"""

from __future__ import annotations

from neuralop.models import GINO
from torch import Tensor, nn

__all__ = ["GinoOperator", "DeltaGino", "build_gino", "build_delta_gino"]


def _validate_modes(fno_n_modes: tuple[int, ...], latent_grid: int, gno_coord_dim: int) -> None:
    """Guard the two GINO invariants: 3-D modes, each strictly below ``G // 2``."""
    if len(fno_n_modes) != gno_coord_dim:
        raise ValueError(
            f"fno_n_modes must have length gno_coord_dim={gno_coord_dim}, got {fno_n_modes}."
        )
    limit = latent_grid // 2
    for axis, m in enumerate(fno_n_modes):
        if m >= limit:
            raise ValueError(
                f"fno_n_modes[{axis}]={m} must be < latent_grid//2={limit} "
                f"(latent_grid={latent_grid}); the FFT cannot retain modes at or above "
                "the Nyquist limit of the latent grid."
            )


class GinoOperator(nn.Module):
    """Thin wrapper over ``neuralop.models.GINO`` for scalar-theta prediction.

    Fixes the verified-good construction (``in_gno_transform_type='nonlinear'`` so the
    per-point feature width is free of the FNO latent width; the SDF carried as
    ``latent_features`` with ``latent_feature_channels=1``;
    ``gno_use_torch_scatter=False`` since ``torch_scatter`` is absent and GINO falls
    back silently) and normalises the leading dims expected by GINO's forward.

    Args:
        in_channels: per-input-point feature width (e.g. 4 for the Block-2 layout
            ``[logk_std, r_si, r_se, theta1d]``).
        out_channels: output channels (1 = theta).
        gno_coord_dim: coordinate dimensionality (3 for the 3-D scans).
        fno_in_channels: latent (FNO) channel width — free, decoupled from
            ``in_channels`` by the nonlinear input transform.
        fno_n_modes: retained Fourier modes per latent axis; each must be
            ``< latent_grid // 2``.
        fno_hidden_channels: width of the FNO spectral-conv hidden representation.
        fno_n_layers: number of Fourier layers.
        in_gno_radius: input-GNO neighbourhood radius, in normalised ``[0, 1]`` coords.
        out_gno_radius: output-GNO neighbourhood radius, in normalised ``[0, 1]`` coords.
        latent_grid: side length ``G`` of the cubic latent grid; used only to validate
            ``fno_n_modes`` (the grid itself is built by the caller / dataset).
        latent_feature_channels: channels of the latent-grid feature field (1 = SDF).
        in_gno_transform_type: input-GNO integral-transform type. ``'nonlinear'``
            decouples the feature width from the latent width — required when
            ``in_channels != fno_in_channels``.
        out_gno_transform_type: output-GNO integral-transform type.
        gno_use_open3d: use Open3D's fixed-radius search (present on CPU and GPU).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 1,
        gno_coord_dim: int = 3,
        fno_in_channels: int = 32,
        fno_n_modes: tuple[int, ...] = (8, 8, 8),
        fno_hidden_channels: int = 64,
        fno_n_layers: int = 4,
        in_gno_radius: float = 0.05,
        out_gno_radius: float = 0.05,
        latent_grid: int = 16,
        latent_feature_channels: int | None = 1,
        in_gno_transform_type: str = "nonlinear",
        out_gno_transform_type: str = "linear",
        gno_use_open3d: bool = True,
    ) -> None:
        super().__init__()
        fno_n_modes = tuple(int(m) for m in fno_n_modes)
        _validate_modes(fno_n_modes, int(latent_grid), int(gno_coord_dim))
        self.gno_coord_dim = int(gno_coord_dim)
        self.latent_grid = int(latent_grid)
        self.gino = GINO(
            in_channels=int(in_channels),
            out_channels=int(out_channels),
            gno_coord_dim=int(gno_coord_dim),
            in_gno_transform_type=in_gno_transform_type,
            out_gno_transform_type=out_gno_transform_type,
            fno_in_channels=int(fno_in_channels),
            fno_n_modes=fno_n_modes,
            fno_hidden_channels=int(fno_hidden_channels),
            fno_n_layers=int(fno_n_layers),
            in_gno_radius=float(in_gno_radius),
            out_gno_radius=float(out_gno_radius),
            latent_feature_channels=latent_feature_channels,
            gno_use_open3d=bool(gno_use_open3d),
            gno_use_torch_scatter=False,
        )

    @staticmethod
    def _ensure_leading_geom(coords: Tensor, name: str) -> Tensor:
        """Add the leading ``1`` geometry dim if a per-batch tensor is passed bare."""
        if coords.dim() == 2:
            return coords.unsqueeze(0)
        if coords.dim() == 3 and coords.shape[0] == 1:
            return coords
        raise ValueError(
            f"{name} must be (n, {coords.shape[-1]}) or (1, n, {coords.shape[-1]}), "
            f"got {tuple(coords.shape)}."
        )

    def forward(
        self,
        input_geom: Tensor,
        x: Tensor,
        latent_queries: Tensor,
        sdf: Tensor,
        output_queries: Tensor,
    ) -> Tensor:
        """Predict theta (or its correction) at ``output_queries``.

        Args:
            input_geom: input-point coordinates ``(n_in, 3)`` or ``(1, n_in, 3)`` in
                a shared ``[0, 1]^3`` box. The geometry is shared across the batch.
            x: per-input-point features ``(B, n_in, in_channels)``.
            latent_queries: latent-grid coordinates ``(1, G, G, G, 3)`` (or unbatched
                ``(G, G, G, 3)``) on ``[0, 1]^3``.
            sdf: signed-distance field on the latent grid, ``(B, G, G, G, 1)`` or
                ``(B, G, G, G)`` (a trailing channel of 1 is added).
            output_queries: query coordinates ``(n_out, 3)`` or ``(1, n_out, 3)`` on
                ``[0, 1]^3``. Gradients flow through these (PINN-ready).

        Returns:
            Tensor of shape ``(B, n_out, out_channels)``.
        """
        input_geom = self._ensure_leading_geom(input_geom, "input_geom")
        output_queries = self._ensure_leading_geom(output_queries, "output_queries")
        if latent_queries.dim() == self.gno_coord_dim + 1:
            latent_queries = latent_queries.unsqueeze(0)
        if latent_queries.dim() != self.gno_coord_dim + 2:
            raise ValueError(
                f"latent_queries must be (1, G, G, G, {self.gno_coord_dim}) "
                f"or (G, G, G, {self.gno_coord_dim}), got {tuple(latent_queries.shape)}."
            )
        if x.dim() != 3:
            raise ValueError(f"x must be (B, n_in, in_channels), got {tuple(x.shape)}.")
        if sdf.dim() == self.gno_coord_dim + 1:  # (B, G, G, G) -> add channel
            sdf = sdf.unsqueeze(-1)
        if sdf.dim() != self.gno_coord_dim + 2:
            raise ValueError(
                f"sdf must be (B, G, G, G, 1) or (B, G, G, G), got {tuple(sdf.shape)}."
            )
        return self.gino(
            input_geom=input_geom,
            latent_queries=latent_queries,
            output_queries=output_queries,
            x=x,
            latent_features=sdf,
        )


class DeltaGino(nn.Module):
    """GINO that predicts a correction added to the analytic 1-D theta prior.

    Mirrors :class:`~thermotwin.models.delta_fno.DeltaFNO` on irregular geometry: the
    smooth bulk is handled by the closed-form 1-D clear-wall prior evaluated at each
    output query's through-wall coordinate (supplied per-query by the dataset, exactly
    as ``theta1d`` is supplied per input point), and the operator spends its capacity
    only on the residual that lateral spreading at thermal bridges adds.

    The forward is ``query_prior + GinoOperator(...)``: the prior is *not* read from a
    feature channel (output queries need not coincide with input points), it is passed
    in directly, keeping the additive structure explicit and PINN-friendly.
    """

    def __init__(self, operator: GinoOperator) -> None:
        super().__init__()
        self.operator = operator

    def forward(
        self,
        input_geom: Tensor,
        x: Tensor,
        latent_queries: Tensor,
        sdf: Tensor,
        output_queries: Tensor,
        query_prior: Tensor,
    ) -> Tensor:
        """Return ``query_prior + correction`` at ``output_queries``.

        Args:
            input_geom, x, latent_queries, sdf, output_queries: as in
                :meth:`GinoOperator.forward`.
            query_prior: analytic 1-D theta prior at each output query, shape
                ``(B, n_out)``, ``(B, n_out, 1)``, or ``(n_out,)`` /
                ``(n_out, 1)`` (broadcast over the batch). Added to the correction.

        Returns:
            Tensor of shape ``(B, n_out, 1)``.
        """
        correction = self.operator(input_geom, x, latent_queries, sdf, output_queries)
        prior = query_prior
        if prior.dim() == 1:  # (n_out,)
            prior = prior.unsqueeze(0).unsqueeze(-1)
        elif prior.dim() == 2:  # (B, n_out) or (n_out, 1)
            if prior.shape[-1] == 1 and prior.shape[0] != correction.shape[0]:
                prior = prior.unsqueeze(0)  # (n_out, 1) -> (1, n_out, 1)
            else:
                prior = prior.unsqueeze(-1)  # (B, n_out) -> (B, n_out, 1)
        if prior.shape[-2:] != correction.shape[-2:]:
            raise ValueError(
                f"query_prior shape {tuple(query_prior.shape)} is incompatible with "
                f"correction shape {tuple(correction.shape)}."
            )
        return prior + correction


def build_gino(
    in_channels: int,
    out_channels: int = 1,
    gno_coord_dim: int = 3,
    fno_in_channels: int = 32,
    fno_n_modes: tuple[int, ...] = (8, 8, 8),
    fno_hidden_channels: int = 64,
    fno_n_layers: int = 4,
    in_gno_radius: float = 0.05,
    out_gno_radius: float = 0.05,
    latent_grid: int = 16,
    latent_feature_channels: int | None = 1,
    in_gno_transform_type: str = "nonlinear",
    out_gno_transform_type: str = "linear",
    gno_use_open3d: bool = True,
) -> GinoOperator:
    """Construct a :class:`GinoOperator`. See the class for argument semantics."""
    return GinoOperator(
        in_channels=in_channels,
        out_channels=out_channels,
        gno_coord_dim=gno_coord_dim,
        fno_in_channels=fno_in_channels,
        fno_n_modes=fno_n_modes,
        fno_hidden_channels=fno_hidden_channels,
        fno_n_layers=fno_n_layers,
        in_gno_radius=in_gno_radius,
        out_gno_radius=out_gno_radius,
        latent_grid=latent_grid,
        latent_feature_channels=latent_feature_channels,
        in_gno_transform_type=in_gno_transform_type,
        out_gno_transform_type=out_gno_transform_type,
        gno_use_open3d=gno_use_open3d,
    )


def build_delta_gino(
    in_channels: int,
    out_channels: int = 1,
    gno_coord_dim: int = 3,
    fno_in_channels: int = 32,
    fno_n_modes: tuple[int, ...] = (8, 8, 8),
    fno_hidden_channels: int = 64,
    fno_n_layers: int = 4,
    in_gno_radius: float = 0.05,
    out_gno_radius: float = 0.05,
    latent_grid: int = 16,
    latent_feature_channels: int | None = 1,
    in_gno_transform_type: str = "nonlinear",
    out_gno_transform_type: str = "linear",
    gno_use_open3d: bool = True,
) -> DeltaGino:
    """Construct a :class:`DeltaGino` wrapping a fresh :class:`GinoOperator`.

    The operator predicts only the correction ``theta - query_prior``; the prior is
    added back in :meth:`DeltaGino.forward` from the per-query tensor the dataset
    supplies. ``out_channels`` must be 1 for the scalar-theta additive structure.
    """
    if out_channels != 1:
        raise ValueError(
            f"delta_gino predicts scalar theta; out_channels must be 1, got {out_channels}."
        )
    operator = build_gino(
        in_channels=in_channels,
        out_channels=out_channels,
        gno_coord_dim=gno_coord_dim,
        fno_in_channels=fno_in_channels,
        fno_n_modes=fno_n_modes,
        fno_hidden_channels=fno_hidden_channels,
        fno_n_layers=fno_n_layers,
        in_gno_radius=in_gno_radius,
        out_gno_radius=out_gno_radius,
        latent_grid=latent_grid,
        latent_feature_channels=latent_feature_channels,
        in_gno_transform_type=in_gno_transform_type,
        out_gno_transform_type=out_gno_transform_type,
        gno_use_open3d=gno_use_open3d,
    )
    return DeltaGino(operator)
