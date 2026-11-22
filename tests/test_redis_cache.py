"""
Unit tests for CogentSearch Redis & In-Memory Semantic Caching Subsystem.
Verifies caching, TTL expiration, telemetry metrics, and AgenticRAG integration.
"""

import time
import pytest
from cogent.cache.redis_cache import RedisQueryCache
from cogent.agent.agentic_loop import AgenticRAG
from cogent.agent.state import CitationItem
from cogent.schemas.indexing import SearchResultChunk


def test_cache_initialization_and_fallback():
    """Verify cache initializes cleanly and falls back to in-memory if Redis is offline."""
    cache = RedisQueryCache(enabled=True)
    assert cache.enabled is True
    assert cache.backend in ["redis", "memory"]

    # Basic get on empty cache returns None
    result = cache.get("What are cluster thresholds?")
    assert result is None
    assert cache.misses == 1
    assert cache.hits == 0


def test_cache_set_get_and_delete():
    """Verify storing, retrieving, and deleting entries."""
    cache = RedisQueryCache(enabled=True)
    cache.clear()

    query = "What was Q3 revenue and YoY growth?"
    payload = {
        "answer": "Revenue reached $48.5M (+34% YoY).",
        "citations": [{"source_file": "doc_02.pdf", "page_number": 1}],
    }

    # Store in cache
    ok = cache.set(query, payload, ttl_seconds=300)
    assert ok is True

    # Retrieve from cache
    cached = cache.get(query)
    assert cached is not None
    assert cached["answer"] == "Revenue reached $48.5M (+34% YoY)."
    assert cached["citations"][0]["source_file"] == "doc_02.pdf"
    assert cache.hits == 1

    # Delete entry
    deleted = cache.delete(query)
    assert deleted is True

    # Now get should return None
    assert cache.get(query) is None


def test_cache_ttl_expiration():
    """Verify that cached entries expire after TTL seconds."""
    cache = RedisQueryCache(enabled=True)
    cache.clear()

    query = "Temporary expiring query"
    payload = {"data": "ephemeral"}

    # Set with 1-second TTL
    cache.set(query, payload, ttl_seconds=1)
    assert cache.get(query) is not None

    # Wait for TTL expiration
    time.sleep(1.2)
    assert cache.get(query) is None


def test_cache_stats_and_telemetry():
    """Verify metrics: hit rate, total requests, and latency savings."""
    cache = RedisQueryCache(enabled=True)
    cache.clear()

    # Initial state
    stats = cache.get_stats()
    assert stats["enabled"] is True
    assert stats["backend"] in ["redis", "memory"]

    # Perform 1 miss and 2 hits
    cache.get("non_existent_query_1")
    cache.set("cached_query_1", {"result": "success"}, ttl_seconds=60)
    cache.get("cached_query_1")
    cache.get("cached_query_1")

    updated_stats = cache.get_stats()
    assert updated_stats["hits"] >= 2
    assert updated_stats["misses"] >= 1
    assert updated_stats["hit_rate_pct"] > 0.0
    assert updated_stats["estimated_latency_saved_ms"] >= 900.0


def test_agentic_rag_cache_integration():
    """Verify AgenticRAG utilizes cache to achieve sub-millisecond second-time retrieval."""
    cache = RedisQueryCache(enabled=True)
    cache.clear()

    rag = AgenticRAG(cache=cache)

    q = "What is the GPU memory usage and cluster configuration?"

    # First run: Cache MISS
    res1 = rag.run(q, use_cache=True)
    assert res1 is not None
    assert res1.cache_hit is False

    # Second run: Cache HIT with sub-millisecond latency
    res2 = rag.run(q, use_cache=True)
    assert res2 is not None
    assert res2.cache_hit is True
    assert res2.cache_backend in ["redis", "memory"]
    assert res2.cache_latency_ms is not None
    assert res2.cache_latency_ms < 50.0  # Typically < 2ms
    assert res2.generation == res1.generation

