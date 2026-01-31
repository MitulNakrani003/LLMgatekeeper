"""Factory for creating Redis backends with auto-detection."""

from typing import Optional

from redis import Redis

from llmgatekeeper.backends.base import CacheBackend
from llmgatekeeper.backends.redis_search import RediSearchBackend
from llmgatekeeper.backends.redis_simple import RedisSimpleBackend


def _is_redisearch_available(redis_client: Redis) -> bool:
    """Check if RediSearch module is loaded in Redis.

    Args:
        redis_client: Redis client instance to check.

    Returns:
        True if RediSearch module is available, False otherwise.
    """
    try:
        modules = redis_client.module_list()
        module_names = [m.get("name", m.get(b"name", b"")).lower() for m in modules]
        # Handle both bytes and string responses
        module_names = [
            name.decode() if isinstance(name, bytes) else name for name in module_names
        ]
        return "search" in module_names or "ft" in module_names
    except Exception:
        return False


def create_redis_backend(
    redis_client: Redis,
    namespace: str = "llmgk",
    vector_dimension: int = 384,
    force_simple: bool = False,
) -> CacheBackend:
    """Create a Redis backend with auto-detection of RediSearch availability.

    This factory function automatically detects if the RediSearch module is
    available in the provided Redis instance. If available, it returns a
    RediSearchBackend for efficient KNN vector search. Otherwise, it falls
    back to RedisSimpleBackend with brute-force similarity search.

    Args:
        redis_client: User's Redis client instance.
        namespace: Key prefix for all cache entries (default: "llmgk").
        vector_dimension: Dimension of embedding vectors (default: 384).
        force_simple: If True, always use RedisSimpleBackend even if
            RediSearch is available (default: False).

    Returns:
        A CacheBackend instance (either RediSearchBackend or RedisSimpleBackend).

    Example:
        >>> import redis
        >>> client = redis.Redis(host='localhost', port=6379)
        >>> backend = create_redis_backend(client)
        >>> # Automatically uses RediSearch if available
    """
    if force_simple:
        return RedisSimpleBackend(
            redis_client=redis_client,
            namespace=namespace,
        )

    if _is_redisearch_available(redis_client):
        return RediSearchBackend(
            redis_client=redis_client,
            namespace=namespace,
            vector_dimension=vector_dimension,
        )

    return RedisSimpleBackend(
        redis_client=redis_client,
        namespace=namespace,
    )
