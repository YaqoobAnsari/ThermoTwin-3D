"""Signed-distance representation of an envelope, for the operator's latent grid.

GINO conditions its Fourier layers on a **signed-distance function (SDF)** sampled
on a regular latent grid: it is the smooth, resolution-independent encoding of "where
is the solid" that lets the operator reason about geometry it never meshed. This
module triangulates an :class:`~thermotwin.geometry.envelope.Envelope` into a mesh
and samples its SDF, on arbitrary query points or a regular grid.

Convention (standard for SDFs, and the negative of trimesh's): ``sdf < 0`` **inside**
the solid, ``sdf > 0`` outside, ``≈ 0`` on the surface. A reliable *sign* needs a
watertight mesh — real building envelopes are closed, so their assembled mesh is
watertight; open patches (e.g. a single test wall) give correct distances but only
heuristic signs.
"""

from __future__ import annotations

import numpy as np
import trimesh

from .envelope import Envelope
from .pointcloud import triangulate_polygon

__all__ = ["envelope_to_mesh", "signed_distance", "sdf_grid"]


def envelope_to_mesh(
    envelope: Envelope, mode: str = "shell", repair: bool = True
) -> trimesh.Trimesh:
    """Triangulate an envelope's surfaces into a single mesh.

    Args:
        mode: which surfaces to mesh — ``"shell"`` (closed outer boundary:
            outdoors + ground; the default, needed for a reliable SDF sign),
            ``"exterior"`` (outdoors only), or ``"all"``.
        repair: merge coincident vertices, fill holes, and fix winding/normals so
            the assembled shell is as close to watertight as the source allows.
    """
    if mode == "shell":
        surfaces = envelope.shell_surfaces()
    elif mode == "exterior":
        surfaces = envelope.exterior_opaque_surfaces()
    elif mode == "all":
        surfaces = list(envelope.surfaces)
    else:
        raise ValueError(f"mode must be shell/exterior/all, got {mode!r}")
    if not surfaces:
        raise ValueError("no surfaces to mesh")

    verts: list[np.ndarray] = []
    faces: list[np.ndarray] = []
    offset = 0
    for s in surfaces:
        tris = triangulate_polygon(s.vertices)  # (T, 3, 3)
        v = tris.reshape(-1, 3)
        f = np.arange(len(v)).reshape(-1, 3)
        verts.append(v)
        faces.append(f + offset)
        offset += len(v)

    mesh = trimesh.Trimesh(
        vertices=np.concatenate(verts), faces=np.concatenate(faces), process=True
    )
    if repair:
        # Merge coincident vertices from adjacent surfaces, then close small gaps
        # and make winding/normals consistent — the steps that recover watertightness.
        mesh.merge_vertices()
        trimesh.repair.fill_holes(mesh)
        trimesh.repair.fix_winding(mesh)
        trimesh.repair.fix_normals(mesh)
    return mesh


def signed_distance(mesh: trimesh.Trimesh, query: np.ndarray) -> np.ndarray:
    """Signed distance from ``query`` points to ``mesh`` (``< 0`` inside)."""
    query = np.asarray(query, dtype=float).reshape(-1, 3)
    # trimesh returns + inside / - outside; negate for the standard SDF convention.
    return -np.asarray(trimesh.proximity.signed_distance(mesh, query))


def sdf_grid(
    mesh: trimesh.Trimesh,
    resolution: int | tuple[int, int, int] = 32,
    padding: float = 0.1,
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Sample the SDF on a regular grid spanning the mesh bounds (+ padding).

    Args:
        mesh: the envelope mesh.
        resolution: cells per axis (int → same for all three axes).
        padding: fraction of each axis extent to pad the bounding box by.

    Returns:
        ``(sdf, (xs, ys, zs))`` where ``sdf`` has shape ``resolution`` and the
        tuple holds the per-axis grid coordinates.
    """
    res = (resolution,) * 3 if isinstance(resolution, int) else tuple(resolution)
    lo, hi = mesh.bounds
    span = hi - lo
    lo = lo - padding * span
    hi = hi + padding * span
    xs, ys, zs = (np.linspace(lo[a], hi[a], res[a]) for a in range(3))
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
    sdf = signed_distance(mesh, pts).reshape(res)
    return sdf, (xs, ys, zs)
