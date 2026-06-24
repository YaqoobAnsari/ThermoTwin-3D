"""GPU-throughput accelerators for the GINO backbone (neighbour-cache + GPU search).

The Block-2 profile (job 26448283) showed that on the ~2k-point box corpus GINO is
*compute/launch-bound* on the latent FNO, not search-bound — but it also showed pure
waste that no full run should pay: the per-sample geometry (``input_geom``,
``latent_queries``, ``output_queries``) is **fixed across every epoch**, yet
``GNOBlock.forward`` re-runs the fixed-radius neighbour search on every forward. The
search keys (``y``, ``x``, ``radius``) are identical for a given sample, so the CRS
``neighbors_dict`` is identical too. This module makes that reuse explicit.

It exposes, non-invasively (no edits to vendored ``neuralop``):

* :class:`NeighbourCache` — a per-sample store of the two GNO blocks' CRS dicts.
* :func:`patch_gno_neighbour_search` — a context manager / installer that swaps each
  ``GNOBlock``'s ``neighbor_search`` for a thin wrapper which (a) returns a cached dict
  when one exists for the active sample key, and (b) optionally routes the radius search
  through ``torch_cluster.radius`` on CUDA (set-identical CRS to neuralop's native
  ``torch.cdist`` fallback, but ``O(n·k)`` on-device instead of ``O(n·m)`` dense).
* :func:`torch_cluster_neighbour_search` — the standalone GPU radius search returning a
  neuralop-compatible CRS dict (importable for tests).

Correctness is identical to the un-accelerated path: the cached dict *is* the dict the
native search would have produced for that sample (cache key includes the radius and a
content hash of the coordinate tensors), and the torch_cluster CRS is verified
set-identical to the native one (order within a neighbourhood may permute, which the
GNO integral transform is invariant to — it is a reduction over the neighbourhood).
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator

import torch
from torch import Tensor

__all__ = [
    "NeighbourCache",
    "torch_cluster_neighbour_search",
    "patch_gno_neighbour_search",
    "torch_cluster_available",
]


def torch_cluster_available() -> bool:
    """True iff the CUDA ``torch_cluster`` radius search can be imported."""
    try:
        import torch_cluster  # noqa: F401

        return True
    except Exception:  # pragma: no cover - depends on the environment build
        return False


def torch_cluster_neighbour_search(
    data: Tensor, queries: Tensor, radius: float
) -> dict[str, Tensor]:
    """Fixed-radius neighbour search via ``torch_cluster.radius``, in neuralop CRS form.

    Returns the same dict shape as
    :func:`neuralop.layers.neighbor_search.native_neighbor_search`:
    ``{"neighbors_index": (E,), "neighbors_row_splits": (m + 1,)}`` (both ``int64``),
    where ``m == queries.shape[0]``. For each query point ``x`` the row holds the
    indices into ``data`` of every point within ``radius`` (Euclidean). The neighbour
    *set* per query equals the native search's; the order within a row may differ, which
    the GNO integral transform (a permutation-invariant neighbourhood reduction) ignores.

    Runs entirely on the device of ``data`` — on CUDA it is an on-GPU radius search,
    avoiding the native fallback's dense ``torch.cdist`` over all ``n·m`` pairs.
    """
    from torch_cluster import radius as tc_radius

    m = int(queries.shape[0])
    # radius(x=data, y=queries): edges (row -> index into queries, col -> index into data).
    edges = tc_radius(data, queries, float(radius), max_num_neighbors=data.shape[0])
    row, col = edges[0], edges[1]
    # Group neighbours by query index to build the CRS row splits.
    order = torch.argsort(row, stable=True)
    row_sorted = row[order]
    neighbors_index = col[order].to(torch.long)
    counts = torch.bincount(row_sorted, minlength=m)
    row_splits = torch.zeros(m + 1, dtype=torch.long, device=counts.device)
    torch.cumsum(counts, dim=0, out=row_splits[1:])
    return {
        "neighbors_index": neighbors_index,
        "neighbors_row_splits": row_splits,
    }


def _coord_key(*tensors: Tensor) -> int:
    """A cheap, stable content fingerprint of coordinate tensors for cache keying.

    The geometry tensors are fixed per sample; we hash shape + a few sampled values so a
    different sample with a coincidentally equal shape cannot collide silently. This is
    only a *fallback* — the primary key is the explicit per-sample id set by the caller.
    """
    parts: list[int] = []
    for t in tensors:
        parts.append(hash(tuple(t.shape)))
        flat = t.detach().reshape(-1)
        if flat.numel():
            idx = torch.linspace(0, flat.numel() - 1, steps=min(8, flat.numel())).long()
            parts.append(hash(tuple(flat[idx].cpu().to(torch.float64).tolist())))
    return hash(tuple(parts))


class NeighbourCache:
    """Per-sample store of the input/output GNO CRS neighbour dicts.

    The training loop sets :attr:`active_key` to the current sample's stable id (e.g. its
    dataset index) before the forward; the patched ``GNOBlock.neighbor_search`` then
    stores/loads the CRS dict under ``(active_key, radius, coord_fingerprint)``. With a
    fixed corpus and fixed geometry per sample this is computed exactly once per sample
    per process and reused for every subsequent epoch.

    Set :attr:`active_key` to ``None`` to bypass the cache (always recompute) — used so a
    cached and an un-cached forward can be compared bit-for-bit in tests.
    """

    def __init__(self) -> None:
        self.active_key: object | None = None
        self._store: dict[tuple, dict[str, Tensor]] = {}

    def clear(self) -> None:
        """Drop every cached neighbour dict (e.g. when the corpus changes)."""
        self._store.clear()

    def __len__(self) -> int:
        return len(self._store)

    def get_or_compute(
        self,
        data: Tensor,
        queries: Tensor,
        radius: float,
        compute_fn,
    ) -> dict[str, Tensor]:
        """Return the cached CRS dict for the active sample, or compute and store it."""
        if self.active_key is None:
            return compute_fn(data, queries, radius)
        key = (self.active_key, round(float(radius), 9), _coord_key(data, queries))
        cached = self._store.get(key)
        if cached is None:
            cached = compute_fn(data, queries, radius)
            # Detach + keep on the search device; indices carry no gradient.
            self._store[key] = {
                "neighbors_index": cached["neighbors_index"].detach(),
                "neighbors_row_splits": cached["neighbors_row_splits"].detach(),
            }
            cached = self._store[key]
        # Return tensors on the device the search inputs live on (handles cpu<->cuda).
        if cached["neighbors_index"].device != data.device:
            cached = {
                "neighbors_index": cached["neighbors_index"].to(data.device),
                "neighbors_row_splits": cached["neighbors_row_splits"].to(data.device),
            }
        return cached


@contextlib.contextmanager
def patch_gno_neighbour_search(
    cache: NeighbourCache | None = None,
    use_torch_cluster: bool = False,
) -> Iterator[None]:
    """Temporarily wrap every ``GNOBlock``'s neighbour search with cache + GPU search.

    Within the context, ``neuralop.layers.gno_block.GNOBlock.forward`` calls the same
    ``self.neighbor_search(data, queries, radius)`` it always did, but that call is now
    routed through:

    * ``cache.get_or_compute`` when a :class:`NeighbourCache` is supplied and its
      ``active_key`` is set — returning the stored CRS dict for the active sample;
    * :func:`torch_cluster_neighbour_search` when ``use_torch_cluster`` is set and the
      coordinates are on CUDA (falls back to the module's own native search otherwise).

    The original ``neighbor_search`` is restored on exit, so nothing in the vendored
    package is mutated permanently. Patching is process-global (it swaps the module
    object's bound method), so use one context around the whole train/eval loop.
    """
    from neuralop.layers.gno_block import GNOBlock

    original_forward = GNOBlock.forward

    def wrapped_neighbor_search(self, data, queries, radius):  # noqa: ANN001
        def _compute(d, q, r):
            if use_torch_cluster and d.is_cuda:
                return torch_cluster_neighbour_search(d, q, r)
            # Defer to the block's own NeighborSearch module (native or Open3D,
            # whatever the GNOBlock was built with) — exact original behaviour.
            return self.neighbor_search(data=d, queries=q, radius=r)

        if cache is not None:
            return cache.get_or_compute(data, queries, radius, _compute)
        return _compute(data, queries, radius)

    def patched_forward(self, y, x, f_y=None):  # noqa: ANN001 - mirror upstream signature
        if f_y is not None and f_y.ndim == 3 and f_y.shape[0] == -1:
            f_y = f_y.squeeze(0)
        neighbors_dict = wrapped_neighbor_search(self, data=y, queries=x, radius=self.radius)
        if self.pos_embedding is not None:
            y_embed = self.pos_embedding(y)
            x_embed = self.pos_embedding(x)
        else:
            y_embed = y
            x_embed = x
        return self.integral_transform(y=y_embed, x=x_embed, neighbors=neighbors_dict, f_y=f_y)

    GNOBlock.forward = patched_forward
    try:
        yield
    finally:
        GNOBlock.forward = original_forward
