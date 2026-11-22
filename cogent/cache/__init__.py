"""
CogentSearch Caching Subsystem.
Provides Redis-backed and In-Memory caching for sub-millisecond query retrieval.
"""

from cogent.cache.redis_cache import RedisQueryCache

__all__ = ["RedisQueryCache"]

