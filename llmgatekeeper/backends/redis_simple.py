"""Redis Simple Mode Backend using hashes and brute-force similarity search."""

import json
from typing import Any, Dict, List, Optional

import numpy as np
from numpy.typing import NDArray
from redis import Redis

from llmgatekeeper.backends.base import CacheBackend, CacheEntry, SearchResult


class RedisSimpleBackend(CacheBackend):
    """Redis backend using hashes and brute-force similarity search.

    This backend is suitable for caches with fewer than ~10k entries.
    For larger caches, use RediSearchBackend for better performance.

    The user passes their own Redis instance, allowing them to manage
    connection pooling, authentication, SSL, and other configuration.
    """

    def __init__(
        self,
        redis_client: Redis,
        namespace: str = "llmgk",
        vector_dtype: str = "float32",
    ) -> None:
        """Initialize the Redis simple backend.

        Args:
            redis_client: User's Redis client instance.
            namespace: Key prefix for all cache entries.
            vector_dtype: Data type for vector serialization.
        """
        self._redis = redis_client
        self._namespace = namespace
        self._vector_dtype = vector_dtype
        self._keys_set = f"{namespace}:keys"

    def _make_key(self, key: str) -> str:
        """Create a namespaced Redis key."""
        return f"{self._namespace}:entry:{key}"

    def _serialize_vector(self, vector: NDArray[np.float32]) -> bytes:
        """Serialize a numpy vector to bytes."""
        return vector.astype(np.float32).tobytes()

    def _deserialize_vector(self, data: bytes) -> NDArray[np.float32]:
        """Deserialize bytes to a numpy vector."""
        return np.frombuffer(data, dtype=np.float32)

    def _cosine_similarity(
        self, v1: NDArray[np.float32], v2: NDArray[np.float32]
    ) -> float:
        """Compute cosine similarity between two vectors."""
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(v1, v2) / (norm1 * norm2))

    def store_vector(
        self,
        key: str,
        vector: NDArray[np.float32],
        metadata: Dict[str, Any],
        ttl: Optional[int] = None,
    ) -> None:
        """Store a vector with associated metadata.

        Args:
            key: Unique identifier for this cache entry.
            vector: The embedding vector to store.
            metadata: Arbitrary metadata to store with the vector.
            ttl: Optional time-to-live in seconds.
        """
        redis_key = self._make_key(key)
        vector_bytes = self._serialize_vector(vector)
        metadata_json = json.dumps(metadata)

        # Store as a hash with vector and metadata fields
        self._redis.hset(
            redis_key,
            mapping={
                "vector": vector_bytes,
                "metadata": metadata_json,
                "key": key,
            },
        )

        # Track the key in our set for iteration
        self._redis.sadd(self._keys_set, key)

        # Set TTL if specified
        if ttl is not None:
            self._redis.expire(redis_key, ttl)

    def search_similar(
        self,
        vector: NDArray[np.float32],
        threshold: float = 0.85,
        top_k: int = 1,
    ) -> List[SearchResult]:
        """Search for similar vectors using brute-force cosine similarity.

        Args:
            vector: The query vector to search for.
            threshold: Minimum similarity score (0-1) for results.
            top_k: Maximum number of results to return.

        Returns:
            List of SearchResult objects sorted by similarity (descending).
        """
        results: List[tuple[str, float, Dict[str, Any], NDArray[np.float32]]] = []

        # Get all keys from our tracking set
        keys = self._redis.smembers(self._keys_set)

        for key_bytes in keys:
            key = key_bytes.decode() if isinstance(key_bytes, bytes) else key_bytes
            entry = self.get_by_key(key)

            if entry is None:
                # Key expired or was deleted, remove from tracking set
                self._redis.srem(self._keys_set, key)
                continue

            similarity = self._cosine_similarity(vector, entry.vector)

            if similarity >= threshold:
                results.append((entry.key, similarity, entry.metadata, entry.vector))

        # Sort by similarity descending and take top_k
        results.sort(key=lambda x: x[1], reverse=True)
        results = results[:top_k]

        return [
            SearchResult(
                key=key,
                similarity=sim,
                metadata=meta,
                vector=vec,
            )
            for key, sim, meta, vec in results
        ]

    def delete(self, key: str) -> bool:
        """Delete a cache entry by key.

        Args:
            key: The key of the entry to delete.

        Returns:
            True if the entry was deleted, False if it didn't exist.
        """
        redis_key = self._make_key(key)
        deleted = self._redis.delete(redis_key)
        self._redis.srem(self._keys_set, key)
        return deleted > 0

    def get_by_key(self, key: str) -> Optional[CacheEntry]:
        """Retrieve a cache entry by its exact key.

        Args:
            key: The key to look up.

        Returns:
            The CacheEntry if found, None otherwise.
        """
        redis_key = self._make_key(key)
        data = self._redis.hgetall(redis_key)

        if not data:
            return None

        # Handle both bytes and string keys from Redis
        vector_data = data.get(b"vector") or data.get("vector")
        metadata_data = data.get(b"metadata") or data.get("metadata")

        if vector_data is None or metadata_data is None:
            return None

        vector = self._deserialize_vector(vector_data)
        metadata_str = (
            metadata_data.decode() if isinstance(metadata_data, bytes) else metadata_data
        )
        metadata = json.loads(metadata_str)

        return CacheEntry(
            key=key,
            vector=vector,
            metadata=metadata,
        )

    def clear(self) -> int:
        """Clear all entries from the cache.

        Returns:
            Number of entries deleted.
        """
        keys = self._redis.smembers(self._keys_set)
        count = 0

        for key_bytes in keys:
            key = key_bytes.decode() if isinstance(key_bytes, bytes) else key_bytes
            redis_key = self._make_key(key)
            if self._redis.delete(redis_key):
                count += 1

        self._redis.delete(self._keys_set)
        return count

    def count(self) -> int:
        """Return the number of entries in the cache.

        Returns:
            Number of cached entries.
        """
        return self._redis.scard(self._keys_set)
