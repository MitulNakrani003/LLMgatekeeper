"""Storage backend adapters for LLMGatekeeper."""

from llmgatekeeper.backends.base import (
    CacheBackend,
    CacheEntry,
    SearchResult,
)
from llmgatekeeper.backends.factory import create_redis_backend
from llmgatekeeper.backends.redis_search import RediSearchBackend
from llmgatekeeper.backends.redis_simple import RedisSimpleBackend

__all__ = [
    "CacheBackend",
    "CacheEntry",
    "SearchResult",
    "RedisSimpleBackend",
    "RediSearchBackend",
    "create_redis_backend",
]
