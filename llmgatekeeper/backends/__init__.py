"""Storage backend adapters for LLMGatekeeper."""

from llmgatekeeper.backends.base import (
    AsyncCacheBackend,
    CacheBackend,
    CacheEntry,
    SearchResult,
)
from llmgatekeeper.backends.factory import (
    create_async_redis_backend,
    create_redis_backend,
)
from llmgatekeeper.backends.redis_async import AsyncRedisBackend
from llmgatekeeper.backends.redis_search import RediSearchBackend
from llmgatekeeper.backends.redis_search_async import AsyncRediSearchBackend
from llmgatekeeper.backends.redis_simple import RedisSimpleBackend

__all__ = [
    "AsyncCacheBackend",
    "AsyncRediSearchBackend",
    "AsyncRedisBackend",
    "CacheBackend",
    "CacheEntry",
    "RediSearchBackend",
    "RedisSimpleBackend",
    "SearchResult",
    "create_async_redis_backend",
    "create_redis_backend",
]
