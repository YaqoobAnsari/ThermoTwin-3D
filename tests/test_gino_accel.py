"""GINO GPU-throughput accelerators: neighbour cache + torch_cluster search parity.

These are CPU smokes (tiny ``G`` latent grid, a few hundred points) that lock the two
correctness guarantees the Block-2 throughput campaign relies on:

* the **neighbour cache** returns bit-for-bit the same field a legacy (un-cached)
  forward returns — caching the per-sample CRS graph is a pure-throughput change, not a
  numerical one — and gradients still flow through cached output queries (PINN-ready);
* the **torch_cluster** radius search produces a CRS dict whose neighbour *set* per query
  equals neuralop's native ``torch.cdist`` search, so swapping the on-GPU backend in
  cannot change which points integrate into each neighbourhood (order within a
  neighbourhood may permute, which the GNO reduction is invariant to).

The torch_cluster *device* path (``d.is_cuda``) is exercised only on GPU; here we test
the CRS builder directly on CPU tensors, which is what guarantees the GPU path's output.
"""

from __future__ import annotations

import torch

torch.manual_seed(1337)

from neuralop.layers.neighbor_search import native_neighbor_search  # noqa: E402

from thermotwin.models.gino import build_delta_gino, build_gino  # noqa: E402
from thermotwin.models.gino_accel import (  # noqa: E402
    NeighbourCache,
    torch_cluster_available,
    torch_cluster_neighbour_search,
)

G = 8  # latent grid side; modes must stay < G // 2 == 4
N_IN = 192
N_OUT = 128
FEAT = 4


def _make_inputs(batch: int = 1, *, requires_grad: bool = False):
    input_geom = torch.rand(1, N_IN, 3)
    x = torch.rand(batch, N_IN, FEAT)
    lin = torch.linspace(0.0, 1.0, G)
    gx, gy, gz = torch.meshgrid(lin, lin, lin, indexing="ij")
    latent_queries = torch.stack([gx, gy, gz], dim=-1).unsqueeze(0)
    sdf = torch.rand(batch, G, G, G, 1)
    output_queries = torch.rand(1, N_OUT, 3, requires_grad=requires_grad)
    return input_geom, x, latent_queries, sdf, output_queries


def _build(cache: bool, backend: str, seed: int = 0):
    torch.manual_seed(seed)
    return build_gino(
        in_channels=FEAT,
        out_channels=1,
        fno_in_channels=16,
        fno_n_modes=(3, 3, 3),
        fno_hidden_channels=16,
        fno_n_layers=2,
        in_gno_radius=0.2,
        out_gno_radius=0.2,
        latent_grid=G,
        cache_neighbours=cache,
        neighbour_search_backend=backend,
    )


def _crs_sets(idx: torch.Tensor, splits: torch.Tensor) -> list[set[int]]:
    return [set(idx[splits[j] : splits[j + 1]].tolist()) for j in range(len(splits) - 1)]


def test_cache_disabled_by_default():
    op = build_gino(in_channels=FEAT, fno_n_modes=(3, 3, 3), latent_grid=G)
    assert op.neighbour_cache is None
    # accelerate() with nothing to do leaves behaviour unchanged and still yields self.
    with op.accelerate() as same:
        assert same is op


def test_cached_forward_matches_uncached_bit_for_bit():
    """The neighbour cache is a throughput change only: identical field, to the bit."""
    ref_op = _build(cache=False, backend="native")
    inputs = _make_inputs()
    ref = ref_op(*inputs)

    cached_op = _build(cache=True, backend="native")  # same seed -> same weights
    with cached_op.accelerate():
        cached_op.set_sample_key(0)
        miss = cached_op(*inputs)  # cache miss: computes + stores
        hit = cached_op(*inputs)  # cache hit: reuses the stored CRS graph
    assert torch.equal(ref, miss)
    assert torch.equal(miss, hit)
    # Exactly two graphs cached for one sample: the input GNO and the output GNO.
    assert len(cached_op.neighbour_cache) == 2


def test_none_sample_key_bypasses_cache():
    op = _build(cache=True, backend="native")
    inputs = _make_inputs()
    with op.accelerate():
        op.set_sample_key(None)  # bypass: nothing is stored
        _ = op(*inputs)
    assert len(op.neighbour_cache) == 0


def test_distinct_keys_store_distinct_graphs():
    op = _build(cache=True, backend="native")
    a = _make_inputs()
    b = _make_inputs()
    with op.accelerate():
        op.set_sample_key("a")
        op(*a)
        op.set_sample_key("b")
        op(*b)
    assert len(op.neighbour_cache) == 4  # two GNOs x two samples
    op.clear_neighbour_cache()
    assert len(op.neighbour_cache) == 0


def test_gradient_flows_through_cached_output_queries():
    op = _build(cache=True, backend="native")
    inputs = _make_inputs(requires_grad=True)
    with op.accelerate():
        op.set_sample_key(0)
        out = op(*inputs)
        out.sum().backward()
    grad = inputs[-1].grad
    assert grad is not None
    assert torch.isfinite(grad).all()
    assert grad.abs().sum() > 0


def test_delta_gino_accelerate_is_additive():
    torch.manual_seed(0)
    dg = build_delta_gino(
        in_channels=FEAT,
        fno_in_channels=16,
        fno_n_modes=(3, 3, 3),
        fno_hidden_channels=16,
        fno_n_layers=2,
        in_gno_radius=0.2,
        out_gno_radius=0.2,
        latent_grid=G,
        cache_neighbours=True,
        neighbour_search_backend="native",
    )
    assert dg.neighbour_cache is not None
    inputs = _make_inputs()
    prior = torch.rand(1, N_OUT)
    with dg.accelerate():
        dg.set_sample_key(0)
        out = dg(*inputs, prior)
        correction = dg.operator(*inputs)
    assert torch.allclose(out, prior.unsqueeze(-1) + correction, atol=1e-6)


def test_torch_cluster_crs_matches_native_neighbour_sets():
    """The torch_cluster CRS neighbour sets equal the native search's, per query."""
    if not torch_cluster_available():  # pragma: no cover - env-dependent
        import pytest

        pytest.skip("torch_cluster not installed")
    torch.manual_seed(3)
    data = torch.rand(120, 3)
    queries = torch.rand(40, 3)
    radius = 0.3
    native = native_neighbor_search(data=data, queries=queries, radius=radius)
    tc = torch_cluster_neighbour_search(data, queries, radius)
    # Same number of query rows and same total edge count.
    assert tc["neighbors_row_splits"].shape == native["neighbors_row_splits"].shape
    assert tc["neighbors_index"].numel() == native["neighbors_index"].numel()
    # Same neighbour set for every query (order within a row may differ).
    native_sets = _crs_sets(native["neighbors_index"], native["neighbors_row_splits"])
    tc_sets = _crs_sets(tc["neighbors_index"], tc["neighbors_row_splits"])
    assert native_sets == tc_sets


def test_torch_cluster_handles_coincident_and_empty_neighbourhoods():
    """Self-inclusion (coincident x==y) and isolated queries match the native search."""
    if not torch_cluster_available():  # pragma: no cover - env-dependent
        import pytest

        pytest.skip("torch_cluster not installed")
    data = torch.tensor([[0.0, 0, 0], [0.01, 0, 0], [0.9, 0.9, 0.9]])
    queries = torch.tensor([[0.0, 0, 0], [0.5, 0.5, 0.5]])  # 2nd query is isolated
    radius = 0.05
    native = native_neighbor_search(data=data, queries=queries, radius=radius)
    tc = torch_cluster_neighbour_search(data, queries, radius)
    assert tc["neighbors_row_splits"].tolist() == native["neighbors_row_splits"].tolist()
    assert _crs_sets(tc["neighbors_index"], tc["neighbors_row_splits"]) == _crs_sets(
        native["neighbors_index"], native["neighbors_row_splits"]
    )


def test_neighbour_cache_unit_get_or_compute():
    """Bare NeighbourCache: computes on miss, reuses on hit, bypasses when key is None."""
    cache = NeighbourCache()
    data = torch.rand(50, 3)
    queries = torch.rand(12, 3)
    calls = {"n": 0}

    def compute(d, q, r):
        calls["n"] += 1
        return native_neighbor_search(data=d, queries=q, radius=r)

    # No active key -> always compute, nothing stored.
    cache.get_or_compute(data, queries, 0.3, compute)
    assert calls["n"] == 1 and len(cache) == 0
    # Active key -> compute once, then reuse.
    cache.active_key = 5
    first = cache.get_or_compute(data, queries, 0.3, compute)
    second = cache.get_or_compute(data, queries, 0.3, compute)
    assert calls["n"] == 2  # only one extra compute despite two calls
    assert first["neighbors_index"] is second["neighbors_index"]
    assert len(cache) == 1
