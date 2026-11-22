"""
Redis Query Cache & Latency Optimization Subsystem.

Provides enterprise-grade query caching for sub-millisecond repeated lookups.
Features:
- Primary backend: Redis key-value store with TTL expiration.
- Resilient fallback: Thread-safe In-Memory dictionary cache with per-key TTL
  when Redis server is offline or unreachable (ensuring zero hard dependencies).
- Telemetry: Tracks hits, misses, latency saved, and hit-rate analytics.
"""

from __future__ import annotations
import json
import time
import hashlib
import threading
from typing import Dict, Any, Optional
from loguru import logger

from cogent.core.config import settings

# Attempt to import redis-py
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    redis = None
    REDIS_AVAILABLE = False


class _InMemoryCacheEntry:
    """Entry wrapper for in-memory cache with timestamp and TTL."""
    __slots__ = ("value", "expires_at")

    def __init__(self, value: Dict[str, Any], ttl_seconds: int):
        self.value = value
        self.expires_at = time.time() + ttl_seconds if ttl_seconds > 0 else float("inf")

    def is_expired(self) -> bool:
        return time.time() > self.expires_at


class RedisQueryCache:
    """
    Sub-millisecond semantic query cache with Redis backend and automatic
    thread-safe In-Memory fallback.
    """

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        db: Optional[int] = None,
        password: Optional[str] = None,
        enabled: Optional[bool] = None,
        default_ttl: Optional[int] = None,
    ):
        self.enabled = enabled if enabled is not None else settings.REDIS_ENABLED
        self.host = host or settings.REDIS_HOST
        self.port = port or settings.REDIS_PORT
        self.db = db if db is not None else settings.REDIS_DB
        self.password = password or settings.REDIS_PASSWORD or None
        self.default_ttl = default_ttl if default_ttl is not None else settings.CACHE_TTL_SECONDS

        self.backend: str = "disabled"
        self._redis_client: Optional[Any] = None
        self._memory_cache: Dict[str, _InMemoryCacheEntry] = {}
        self._lock = threading.Lock()

        # Telemetry metrics
        self.hits: int = 0
        self.misses: int = 0
        self.total_latency_saved_ms: float = 0.0

        if not self.enabled:
            logger.info("RedisQueryCache is explicitly disabled.")
            return

        # Attempt Redis connection
        self._init_backend()

    def _init_backend(self) -> None:
        """Attempts connection to Redis, falling back to In-Memory store."""
        if REDIS_AVAILABLE:
            try:
                client = redis.Redis(
                    host=self.host,
                    port=self.port,
                    db=self.db,
                    password=self.password,
                    socket_connect_timeout=0.8,
                    socket_timeout=0.8,
                    decode_responses=True,
                )
                # Verify ping
                client.ping()
                self._redis_client = client
                self.backend = "redis"
                logger.info(f"⚡ RedisQueryCache connected to Redis at {self.host}:{self.port} (db={self.db})")
                return
            except Exception as exc:
                logger.debug(f"Redis ping failed ({exc}). Falling back to In-Memory LRU cache.")

        # Graceful fallback to thread-safe memory store
        self.backend = "memory"
        logger.info("⚡ RedisQueryCache initialized with In-Memory fallback (Zero external dependencies).")

    def _generate_key(self, query: str, namespace: str = "query") -> str:
        """Produces a deterministic normalized SHA-256 hash key."""
        normalized = " ".join(query.strip().lower().split())
        query_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        return f"{settings.CACHE_NAMESPACE}:{namespace}:{query_hash}"

    def get(self, query: str, namespace: str = "query") -> Optional[Dict[str, Any]]:
        """
        Retrieves cached query response.
        Returns None on miss or expiration.
        """
        if not self.enabled or self.backend == "disabled":
            return None

        key = self._generate_key(query, namespace)
        start_time = time.perf_counter()

        try:
            if self.backend == "redis" and self._redis_client:
                raw_data = self._redis_client.get(key)
                if raw_data:
                    parsed = json.loads(raw_data)
                    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                    self.hits += 1
                    self.total_latency_saved_ms += 450.0  # Estimated avg LLM/Retrieval savings
                    logger.debug(f"⚡ Cache HIT (Redis) for key '{key[:28]}...' in {elapsed_ms:.2f}ms")
                    return parsed
            elif self.backend == "memory":
                with self._lock:
                    entry = self._memory_cache.get(key)
                    if entry:
                        if entry.is_expired():
                            del self._memory_cache[key]
                        else:
                            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                            self.hits += 1
                            self.total_latency_saved_ms += 450.0
                            logger.debug(f"⚡ Cache HIT (Memory) for key '{key[:28]}...' in {elapsed_ms:.2f}ms")
                            return entry.value
        except Exception as exc:
            logger.warning(f"Cache get exception: {exc}")

        self.misses += 1
        return None

    def set(
        self,
        query: str,
        value: Dict[str, Any],
        namespace: str = "query",
        ttl_seconds: Optional[int] = None,
    ) -> bool:
        """
        Stores response in cache with TTL.
        """
        if not self.enabled or self.backend == "disabled":
            return False

        key = self._generate_key(query, namespace)
        ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl

        try:
            serialized = json.dumps(value, default=str)

            if self.backend == "redis" and self._redis_client:
                self._redis_client.set(key, serialized, ex=ttl)
                return True
            elif self.backend == "memory":
                with self._lock:
                    # Enforce max memory size (LRU-like eviction if > 1000 keys)
                    if len(self._memory_cache) >= 1000:
                        oldest_key = next(iter(self._memory_cache))
                        del self._memory_cache[oldest_key]
                    self._memory_cache[key] = _InMemoryCacheEntry(value, ttl)
                return True
        except Exception as exc:
            logger.warning(f"Cache set exception: {exc}")
            return False

        return False

    def delete(self, query: str, namespace: str = "query") -> bool:
        """Deletes a cached query entry."""
        if not self.enabled or self.backend == "disabled":
            return False

        key = self._generate_key(query, namespace)
        try:
            if self.backend == "redis" and self._redis_client:
                return bool(self._redis_client.delete(key))
            elif self.backend == "memory":
                with self._lock:
                    if key in self._memory_cache:
                        del self._memory_cache[key]
                        return True
        except Exception as exc:
            logger.warning(f"Cache delete exception: {exc}")
        return False

    def clear(self, namespace: Optional[str] = None) -> int:
        """Flushes cache keys."""
        cleared_count = 0
        if not self.enabled or self.backend == "disabled":
            return 0

        try:
            if self.backend == "redis" and self._redis_client:
                pattern = f"{settings.CACHE_NAMESPACE}:{namespace}:*" if namespace else f"{settings.CACHE_NAMESPACE}:*"
                keys = self._redis_client.keys(pattern)
                if keys:
                    cleared_count = self._redis_client.delete(*keys)
            elif self.backend == "memory":
                with self._lock:
                    if namespace:
                        prefix = f"{settings.CACHE_NAMESPACE}:{namespace}:"
                        to_del = [k for k in self._memory_cache if k.startswith(prefix)]
                        for k in to_del:
                            del self._memory_cache[k]
                        cleared_count = len(to_del)
                    else:
                        cleared_count = len(self._memory_cache)
                        self._memory_cache.clear()
        except Exception as exc:
            logger.warning(f"Cache clear exception: {exc}")

        return cleared_count

    def get_stats(self) -> Dict[str, Any]:
        """Telemetry metrics for dashboards and health checks."""
        total_requests = self.hits + self.misses
        hit_rate = (self.hits / total_requests * 100.0) if total_requests > 0 else 0.0
        keys_count = 0

        try:
            if self.backend == "redis" and self._redis_client:
                keys_count = len(self._redis_client.keys(f"{settings.CACHE_NAMESPACE}:*"))
            elif self.backend == "memory":
                with self._lock:
                    keys_count = len(self._memory_cache)
        except Exception:
            keys_count = 0

        return {
            "backend": self.backend,
            "enabled": self.enabled,
            "hits": self.hits,
            "misses": self.misses,
            "total_requests": total_requests,
            "hit_rate_pct": round(hit_rate, 2),
            "keys_cached": keys_count,
            "estimated_latency_saved_ms": round(self.total_latency_saved_ms, 2),
        }

