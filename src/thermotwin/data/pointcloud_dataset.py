"""Torch dataset over the Block-2 3-D point-cloud corpus (``data/processed/block2_*``).

Block-2 carries the Block-1 recipe — a geometry-conditioned operator predicting a
**correction on an analytic 1-D clear-wall prior** — onto the irregular point clouds
that real as-built scans produce. The corpus (see
:mod:`thermotwin.data.synthetic_3d`) stores, per solved 3-D wall block, a GINO sample:
``points`` ``(N, 3)`` in the unit cube, per-point features
``[logk_std, r_si, r_se, theta1d]``, the target dimensionless temperature ``theta``
``(N,)``, the per-point 1-D ``prior`` ``(N,)``, a signed-distance field ``sdf``
``(G, G, G)`` on a regular latent grid, and scalars (``u_value``, ``u_clear`` …).

This module turns that into the exact tensors each Block-2 model consumes:

* the GINO operators (``gino`` / ``delta_gino``) want an *irregular* cloud —
  ``input_geom`` ``(n, 3)``, per-point ``feats`` ``(n, F)``, the latent ``sdf``
  ``(G, G, G)``, ``output_queries`` ``(n, 3)`` (taken equal to the input points for
  v1), the per-query 1-D ``prior`` ``(n,)`` (for ``delta_gino``'s additive structure)
  and the scalar ``u_value``;
* the reference grid baseline (``fno_voxel``) wants a *dense* voxel field on a fixed
  ``G^3`` grid, reconstructed once at load time from the scattered cloud (linear
  interpolation, nearest-neighbour fill at the hull) — see
  :func:`voxelise_sample`.

The :class:`PointCloudDataset` returns the raw point-cloud sample (with the latent
grid + queries it needs); :func:`collate_pointcloud` keeps the GINO leading-``1``
geometry/grid convention while batching ``feats`` / ``queries`` / ``prior`` over
samples whose geometry happens to be co-sized (``points`` are independent per block,
so geometry is *not* truly shared — the safe default is ``batch_size=1`` per sample;
the collate also supports stacking equal-``n`` samples for throughput). The
``gino_feats`` view drops the prior channel (data-only ``gino``); ``feats`` keeps it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

__all__ = [
    "PointCloudDataset",
    "collate_pointcloud",
    "latent_grid_coords",
    "voxelise_sample",
    "PRIOR_CHANNEL",
    "FEATURE_DIM",
]

# Per-point feature layout is ``[logk_std, r_si, r_se, theta1d]`` (synthetic_3d).
FEATURE_DIM = 4
# Index of the analytic 1-D prior channel inside ``feats`` — dropped for data-only
# ``gino`` (which must not see the prior), kept for ``delta_gino`` provenance.
PRIOR_CHANNEL = 3


def latent_grid_coords(grid: int) -> torch.Tensor:
    """Cell-centred ``(G, G, G, 3)`` meshgrid on the unit cube ``[0, 1]^3``.

    Matches the latent frame the SDF is built on in
    :func:`thermotwin.data.synthetic_3d.box_sdf_grid` (cell centres
    ``(i + 0.5) / G``), so the latent queries and the SDF live on the same grid.
    """
    c = (torch.arange(grid, dtype=torch.float32) + 0.5) / grid
    gx, gy, gz = torch.meshgrid(c, c, c, indexing="ij")
    return torch.stack([gx, gy, gz], dim=-1)  # (G, G, G, 3)


def voxelise_sample(
    points: np.ndarray,
    values: np.ndarray,
    grid: int,
) -> np.ndarray:
    """Reconstruct a dense ``G^3`` (or ``G^3 × C``) field from a scattered cloud.

    The Block-2 corpus stores only the sampled points, not the dense voxel field; the
    grid baseline (``fno_voxel``) needs a full field, so we interpolate the cloud onto
    a fixed cell-centred ``G^3`` grid (linear inside the convex hull, nearest-neighbour
    fill at the boundary where linear returns NaN). A fixed ``G`` makes every sample
    the same size, so the voxel baseline batches.

    Args:
        points: ``(N, 3)`` coordinates in ``[0, 1]^3``.
        values: ``(N,)`` or ``(N, C)`` per-point values to interpolate.
        grid: target grid resolution ``G``.

    Returns:
        ``(G, G, G)`` if ``values`` is 1-D, else ``(G, G, G, C)``.
    """
    from scipy.interpolate import griddata

    points = np.asarray(points, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    c = (np.arange(grid) + 0.5) / grid
    gx, gy, gz = np.meshgrid(c, c, c, indexing="ij")
    query = np.stack([gx, gy, gz], axis=-1).reshape(-1, 3)

    lin = griddata(points, values, query, method="linear")
    near = griddata(points, values, query, method="nearest")
    if lin.ndim == 1:
        filled = np.where(np.isnan(lin), near, lin)
        return filled.reshape(grid, grid, grid).astype(np.float32)
    nan_rows = np.isnan(lin).any(axis=1)
    lin[nan_rows] = near[nan_rows]
    return lin.reshape(grid, grid, grid, values.shape[1]).astype(np.float32)


class PointCloudDataset(Dataset):
    """3-D point-cloud corpus for the Block-2 geometry-conditioned operators.

    Each item is a dict of tensors:

    * ``sample_index`` scalar — the sample's stable dataset index (keys the GINO
      neighbour-graph cache, whose geometry is fixed per sample across epochs);
    * ``input_geom`` ``(n, 3)`` — input-point coordinates in ``[0, 1]^3``;
    * ``feats`` ``(n, F)`` — per-point features ``[logk_std, r_si, r_se, theta1d]``;
    * ``gino_feats`` ``(n, F-1)`` — ``feats`` with the prior channel dropped (the
      data-only ``gino`` view);
    * ``sdf`` ``(G, G, G)`` — the latent-grid signed-distance field;
    * ``latent_queries`` ``(G, G, G, 3)`` — the latent-grid coordinates;
    * ``output_queries`` ``(n, 3)`` — query coordinates (``== input_geom`` for v1);
    * ``theta`` ``(n,)`` — the target dimensionless temperature;
    * ``prior`` ``(n,)`` — the per-query analytic 1-D prior (for ``delta_gino``);
    * ``u_value`` scalar — the effective U-value [W/(m²K)];
    * ``u_clear`` scalar — the 1-D clear-wall U-value baseline;
    * ``grid_shape`` ``(3,)`` — the native FV grid shape (for the voxel baseline);
    * ``points`` ``(n, 3)``, ``raw_feats`` ``(n, F)`` — the unmodified cloud (so the
      voxel baseline can rebuild a dense field without re-reading the npz).

    With ``voxelise=True`` (used by the ``fno_voxel`` baseline) each item also carries
    ``voxel_feats`` ``(F, G, G, G)`` and ``voxel_theta`` ``(G, G, G)`` — the cloud
    reconstructed on a fixed ``voxel_grid`` so the dense FNO can train and the field
    relative-L2 is measured on the same support as the GINO models (the points).

    Args:
        root: corpus directory holding ``manifest.json`` + ``sample_*.npz``.
        latent_grid: latent-grid resolution ``G`` for the SDF / latent queries. If
            ``None`` (default) it is taken from each sample's stored SDF.
        voxelise: build the dense voxel field/features (slower load; only the grid
            baseline needs it).
        voxel_grid: fixed resolution for the voxel reconstruction.
        cache_in_memory: decode every ``.npz`` once and hold the assembled item dicts in
            RAM. The Block-2 corpus is a few MB, so this lifts the per-sample
            ``np.load`` (the profile's third-largest bucket, ~1.8 ms/step) off the
            training hot path with negligible memory cost. Items are returned by
            reference; consumers must not mutate them in place (the runners only read /
            ``.to(device)``-copy, which is safe). Default off to preserve the lazy path.
    """

    def __init__(
        self,
        root: str | Path,
        latent_grid: int | None = None,
        voxelise: bool = False,
        voxel_grid: int = 16,
        cache_in_memory: bool = False,
    ) -> None:
        self.root = Path(root)
        manifest = json.loads((self.root / "manifest.json").read_text())
        self.files = [self.root / row["file"] for row in manifest["samples"]]
        self.latent_grid = latent_grid
        self.voxelise = bool(voxelise)
        self.voxel_grid = int(voxel_grid)
        self.cache_in_memory = bool(cache_in_memory)
        self._latent_cache: dict[int, torch.Tensor] = {}
        self._item_cache: dict[int, dict[str, torch.Tensor]] = {}

    def __len__(self) -> int:
        return len(self.files)

    def _latent_queries(self, grid: int) -> torch.Tensor:
        if grid not in self._latent_cache:
            self._latent_cache[grid] = latent_grid_coords(grid)
        return self._latent_cache[grid]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if self.cache_in_memory:
            cached = self._item_cache.get(idx)
            if cached is not None:
                return cached
        item = self._load_item(idx)
        if self.cache_in_memory:
            self._item_cache[idx] = item
        return item

    def _load_item(self, idx: int) -> dict[str, torch.Tensor]:
        d = np.load(self.files[idx], allow_pickle=True)
        points = d["points"].astype(np.float32)
        feats = d["feats"].astype(np.float32)
        theta = d["theta"].astype(np.float32)
        prior = d["prior"].astype(np.float32)
        sdf = d["sdf"].astype(np.float32)
        grid = int(self.latent_grid or sdf.shape[0])

        gino_feats = np.delete(feats, PRIOR_CHANNEL, axis=1)
        item: dict[str, torch.Tensor] = {
            # Stable per-sample id, used to key the GINO neighbour-graph cache (the
            # geometry is fixed per sample across epochs, so the CRS graph is too).
            "sample_index": torch.tensor(int(idx), dtype=torch.int64),
            "input_geom": torch.from_numpy(points),
            "feats": torch.from_numpy(feats),
            "gino_feats": torch.from_numpy(np.ascontiguousarray(gino_feats)),
            "sdf": torch.from_numpy(sdf),
            "latent_queries": self._latent_queries(grid),
            "output_queries": torch.from_numpy(points.copy()),
            "theta": torch.from_numpy(theta),
            "prior": torch.from_numpy(prior),
            "points": torch.from_numpy(points.copy()),
            "raw_feats": torch.from_numpy(feats.copy()),
            "u_value": torch.tensor(float(d["u_value"]), dtype=torch.float32),
            "u_clear": torch.tensor(float(d["u_clear"]), dtype=torch.float32),
            "r_si": torch.tensor(float(d["r_si"]), dtype=torch.float32),
            "r_se": torch.tensor(float(d["r_se"]), dtype=torch.float32),
            "grid_shape": torch.from_numpy(d["grid_shape"].astype(np.int64)),
        }
        if self.voxelise:
            voxel_feats = voxelise_sample(points, feats, self.voxel_grid)  # (G,G,G,F)
            voxel_theta = voxelise_sample(points, theta, self.voxel_grid)  # (G,G,G)
            item["voxel_feats"] = torch.from_numpy(
                np.ascontiguousarray(np.moveaxis(voxel_feats, -1, 0))  # (F,G,G,G)
            )
            item["voxel_theta"] = torch.from_numpy(voxel_theta)  # (G,G,G)
        return item


def collate_pointcloud(batch: list[dict]) -> dict:
    """Collate point-cloud items, honouring the GINO leading-``1`` geometry convention.

    GINO shares one geometry across the feature batch (leading dim ``1``); two blocks
    from this corpus have *different* point clouds, so the safe and default behaviour
    is ``batch_size == 1`` — the single item is returned with the leading geometry dim
    added (``input_geom`` / ``output_queries`` → ``(1, n, 3)``, ``latent_queries`` →
    ``(1, G, G, G, 3)``) and a batch dim on ``feats`` / ``sdf`` / ``prior`` →
    ``(1, …)``. For throughput, equal-``n`` items are stacked along the feature batch
    *only if* their geometries coincide; otherwise we fall back to per-sample by
    raising, so the caller loops sample-by-sample. Scalars are stacked to ``(B,)``.
    """
    if len(batch) == 1:
        b = batch[0]
        return {
            "sample_index": b["sample_index"].reshape(1),  # (1,) stable cache key
            "input_geom": b["input_geom"].unsqueeze(0),  # (1, n, 3)
            "feats": b["feats"].unsqueeze(0),  # (1, n, F)
            "gino_feats": b["gino_feats"].unsqueeze(0),  # (1, n, F-1)
            "sdf": b["sdf"].unsqueeze(0),  # (1, G, G, G)
            "latent_queries": b["latent_queries"].unsqueeze(0),  # (1, G, G, G, 3)
            "output_queries": b["output_queries"].unsqueeze(0),  # (1, n, 3)
            "theta": b["theta"].unsqueeze(0),  # (1, n)
            "prior": b["prior"].unsqueeze(0),  # (1, n)
            "points": b["points"].unsqueeze(0),
            "raw_feats": b["raw_feats"].unsqueeze(0),
            "u_value": b["u_value"].reshape(1),
            "u_clear": b["u_clear"].reshape(1),
            "r_si": b["r_si"].reshape(1),
            "r_se": b["r_se"].reshape(1),
            "grid_shape": b["grid_shape"].unsqueeze(0),
            **(
                {
                    "voxel_feats": b["voxel_feats"].unsqueeze(0),
                    "voxel_theta": b["voxel_theta"].unsqueeze(0),
                }
                if "voxel_feats" in b
                else {}
            ),
        }

    geom = batch[0]["input_geom"]
    same_geom = all(
        b["input_geom"].shape == geom.shape and torch.equal(b["input_geom"], geom) for b in batch
    )
    if not same_geom:
        raise ValueError(
            "collate_pointcloud cannot batch blocks with distinct geometry under the "
            "GINO shared-geometry convention; iterate with batch_size=1 (the default), "
            "or pre-group equal-geometry samples."
        )
    return {
        "sample_index": torch.stack([b["sample_index"] for b in batch]),
        "input_geom": geom.unsqueeze(0),
        "feats": torch.stack([b["feats"] for b in batch], dim=0),
        "gino_feats": torch.stack([b["gino_feats"] for b in batch], dim=0),
        "sdf": torch.stack([b["sdf"] for b in batch], dim=0),
        "latent_queries": batch[0]["latent_queries"].unsqueeze(0),
        "output_queries": batch[0]["output_queries"].unsqueeze(0),
        "theta": torch.stack([b["theta"] for b in batch], dim=0),
        "prior": torch.stack([b["prior"] for b in batch], dim=0),
        "points": torch.stack([b["points"] for b in batch], dim=0),
        "raw_feats": torch.stack([b["raw_feats"] for b in batch], dim=0),
        "u_value": torch.stack([b["u_value"] for b in batch]),
        "u_clear": torch.stack([b["u_clear"] for b in batch]),
        "r_si": torch.stack([b["r_si"] for b in batch]),
        "r_se": torch.stack([b["r_se"] for b in batch]),
        "grid_shape": torch.stack([b["grid_shape"] for b in batch], dim=0),
    }
