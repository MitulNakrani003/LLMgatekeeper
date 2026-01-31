"""
LLMGatekeeper - Semantic caching library for LLM and RAG systems.

This library eliminates redundant LLM API calls and vector database queries
by using embedding-based similarity matching to recognize semantically
equivalent queries.

Example:
    >>> import redis
    >>> from llmgatekeeper import SemanticCache
    >>>
    >>> cache = SemanticCache(redis.Redis())
    >>> cache.set("What is Python?", "Python is a programming language.")
    >>> cache.get("Tell me about Python")  # Semantically similar
    'Python is a programming language.'

Async Example:
    >>> import redis.asyncio as redis
    >>> from llmgatekeeper import AsyncSemanticCache
    >>> from llmgatekeeper.backends.redis_async import AsyncRedisBackend
    >>>
    >>> async def main():
    ...     client = redis.Redis()
    ...     backend = AsyncRedisBackend(client)
    ...     cache = AsyncSemanticCache(backend)
    ...     await cache.set("What is Python?", "A programming language.")
    ...     result = await cache.get("What is Python?")
"""

from llmgatekeeper.cache import AsyncSemanticCache, CacheResult, SemanticCache
from llmgatekeeper.exceptions import (
    BackendError,
    CacheError,
    ConfigurationError,
    EmbeddingError,
)
from llmgatekeeper.logging import configure_logging, disable_logging

__version__ = "0.1.0"
__all__ = [
    "__version__",
    "AsyncSemanticCache",
    "BackendError",
    "CacheError",
    "CacheResult",
    "ConfigurationError",
    "EmbeddingError",
    "SemanticCache",
    "configure_logging",
    "disable_logging",
]
