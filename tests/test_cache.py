"""Tests for Semantic Vector Cache functionality and latency."""

import time
import pytest
import numpy as np
from src.cache.memory_cache import MemorySemanticCache


@pytest.mark.asyncio
async def test_semantic_cache_hit_above_threshold():
    cache = MemorySemanticCache(default_ttl_seconds=300)

    # Base query vector
    base_vec = [1.0, 0.0, 0.0, 0.0]
    sources = [{"source": "spec.pdf", "chunk_id": "c1"}]

    await cache.set(
        query="What is the GPU thermal limit?",
        query_vector=base_vec,
        answer="The GPU thermal limit is 90C.",
        sources=sources,
    )

    # Query with 0.98 similarity vector
    near_vec = [0.99, 0.1, 0.0, 0.0]  # Cosine similarity > 0.99
    t0 = time.perf_counter()
    hit = await cache.get(
        query="What is the GPU thermal threshold?",
        query_vector=near_vec,
        similarity_threshold=0.92,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000

    assert hit is not None
    answer, returned_sources, score = hit
    assert answer == "The GPU thermal limit is 90C."
    assert score >= 0.92
    assert elapsed_ms < 25.0, f"Cache lookup took {elapsed_ms}ms, expected sub-25ms"


@pytest.mark.asyncio
async def test_semantic_cache_miss_below_threshold():
    cache = MemorySemanticCache(default_ttl_seconds=300)

    vec_a = [1.0, 0.0, 0.0, 0.0]
    await cache.set(
        query="How to drain Kubernetes node?",
        query_vector=vec_a,
        answer="Run kubectl drain.",
        sources=[],
    )

    # Orthogonal query vector (similarity = 0.0)
    vec_b = [0.0, 1.0, 0.0, 0.0]
    miss = await cache.get(
        query="What is the procurement spending limit?",
        query_vector=vec_b,
        similarity_threshold=0.92,
    )
    assert miss is None


@pytest.mark.asyncio
async def test_semantic_cache_invalidation():
    cache = MemorySemanticCache(default_ttl_seconds=300)
    v = [1.0, 0.0]

    await cache.set("Query A", v, "Answer A", [])
    await cache.set("Query B", v, "Answer B", [])

    # Invalidate Query A
    removed = await cache.invalidate("Query A")
    assert removed == 1

    # Invalidate all remaining
    total_removed = await cache.invalidate(None)
    assert total_removed == 1
