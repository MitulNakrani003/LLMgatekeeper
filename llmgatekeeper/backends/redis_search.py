"""Redis RediSearch Backend using vector similarity search for scale."""

import json
from typing import Any, Dict, List, Optional

import numpy as np
from numpy.typing import NDArray
from redis import Redis
from redis.commands.search.field import TagField, TextField, VectorField
from redis.commands.search.index_definition import IndexDefinition, IndexType
from redis.commands.search.query import Query

from llmgatekeeper.backends.base import CacheBackend, CacheEntry, SearchResult
from llmgatekeeper.exceptions import BackendError, ConfigurationError


class RediSearchBackend(CacheBackend):
    """Redis backend using RediSearch vector similarity search.

    This backend is suitable for caches with many entries (10k+) where
    brute-force similarity search becomes too slow. It uses RediSearch's
    native KNN vector search capabilities.

    The user passes their own Redis instance, allowing them to manage
    connection pooling, authentication, SSL, and other configuration.
    """

    def __init__(
        self,
        redis_client: Redis,
        namespace: str = "llmgk",
        vector_dimension: int = 384,
        distance_metric: str = "COSINE",
        index_type: str = "HNSW",
    ) -> None:
        """Initialize the RediSearch backend.

        Args:
            redis_client: User's Redis client instance.
            namespace: Key prefix for all cache entries.
            vector_dimension: Dimension of the embedding vectors.
            distance_metric: Distance metric for similarity (COSINE, L2, IP).
            index_type: Index algorithm (HNSW or FLAT).

        Raises:
            RuntimeError: If RediSearch module is not available.
        """
        if distance_metric.upper() != "COSINE":
            raise ConfigurationError(
                f"Only COSINE distance is currently supported; got "
                f"distance_metric={distance_metric!r}. L2 and IP produce "
                f"similarity scores that aren't in [0, 1] under the current "
                f"conversion."
            )

        self._redis = redis_client
        self._namespace = namespace
        self._vector_dimension = vector_dimension
        self._distance_metric = distance_metric
        self._index_type = index_type
        self._index_name = f"{namespace}_idx"
        self._keys_set = f"{namespace}:keys"

        # Check if RediSearch is available
        if not self._is_redisearch_available():
            raise RuntimeError(
                "RediSearch module not available. "
                "Please install Redis Stack or enable the RediSearch module."
            )

        # Create index if it doesn't exist
        self._ensure_index_exists()

    def _is_redisearch_available(self) -> bool:
        """Check if RediSearch module is loaded in Redis."""
        try:
            modules = self._redis.module_list()
            module_names = [m.get("name", m.get(b"name", b"")).lower() for m in modules]
            # Handle both bytes and string responses
            module_names = [
                name.decode() if isinstance(name, bytes) else name
                for name in module_names
            ]
            return "search" in module_names or "ft" in module_names
        except Exception:
            return False

    def _ensure_index_exists(self) -> None:
        """Create the vector index if it doesn't exist; validate dim if it does."""
        try:
            info = self._redis.ft(self._index_name).info()
        except Exception:
            info = None

        if info is not None:
            existing_dim = self._extract_index_dimension(info)
            if existing_dim is not None and existing_dim != self._vector_dimension:
                raise ConfigurationError(
                    f"RediSearch index '{self._index_name}' was created with "
                    f"vector dimension {existing_dim}, but backend was constructed "
                    f"with vector_dimension={self._vector_dimension}. Drop the "
                    f"index or use a different namespace."
                )
            return

        schema = (
            TagField("$.key", as_name="key"),
            TextField("$.metadata", as_name="metadata"),
            VectorField(
                "$.vector",
                self._index_type,
                {
                    "TYPE": "FLOAT32",
                    "DIM": self._vector_dimension,
                    "DISTANCE_METRIC": self._distance_metric,
                },
                as_name="vector",
            ),
        )

        definition = IndexDefinition(
            prefix=[f"{self._namespace}:entry:"],
            index_type=IndexType.JSON,
        )

        self._redis.ft(self._index_name).create_index(
            schema,
            definition=definition,
        )

    @staticmethod
    def _extract_index_dimension(info: Any) -> Optional[int]:
        """Best-effort parse of an FT.INFO response to recover the vector DIM."""
        if not isinstance(info, dict):
            return None
        attributes = info.get("attributes")
        if not attributes:
            return None
        for attr in attributes:
            if not isinstance(attr, (list, tuple)):
                continue
            attr_map: Dict[str, Any] = {}
            for i in range(0, len(attr) - 1, 2):
                k, v = attr[i], attr[i + 1]
                if isinstance(k, bytes):
                    k = k.decode()
                if isinstance(v, bytes):
                    v = v.decode()
                attr_map[k] = v
            if str(attr_map.get("type", "")).upper() == "VECTOR":
                dim = attr_map.get("dim")
                if dim is not None:
                    try:
                        return int(dim)
                    except (TypeError, ValueError):
                        return None
        return None

    def _make_key(self, key: str) -> str:
        """Create a namespaced Redis key."""
        return f"{self._namespace}:entry:{key}"

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

        # Store as JSON document for RediSearch
        doc = {
            "key": key,
            "vector": vector.astype(np.float32).tolist(),
            "metadata": json.dumps(metadata),
        }

        self._redis.json().set(redis_key, "$", doc)

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
        """Search for similar vectors using KNN.

        Args:
            vector: The query vector to search for.
            threshold: Minimum similarity score (0-1) for results.
            top_k: Maximum number of results to return.

        Returns:
            List of SearchResult objects sorted by similarity (descending).
        """
        # Convert vector to bytes for query
        vector_bytes = vector.astype(np.float32).tobytes()

        # When the caller has a meaningful threshold, use a VECTOR_RANGE query
        # so we never miss valid matches because KNN's k was too small.
        # When threshold is 0 (caller wants the best match regardless), KNN is
        # cheaper and gives a stable best-first ordering.
        if threshold > 0.0:
            radius = max(0.0, 1.0 - threshold)
            query = (
                Query(
                    f"@vector:[VECTOR_RANGE $radius $query_vec]"
                    f"=>{{$YIELD_DISTANCE_AS: score}}"
                )
                .sort_by("score")
                .paging(0, top_k)
                .return_fields("key", "metadata", "score")
                .dialect(2)
            )
            query_params = {
                "query_vec": vector_bytes,
                "radius": str(radius),
            }
        else:
            k = min(top_k * 2, 100)
            query = (
                Query(f"*=>[KNN {k} @vector $query_vec AS score]")
                .sort_by("score")
                .return_fields("key", "metadata", "score")
                .dialect(2)
            )
            query_params = {"query_vec": vector_bytes}

        try:
            results = self._redis.ft(self._index_name).search(
                query, query_params=query_params
            )
        except Exception as e:
            if "no such index" in str(e).lower():
                return []
            raise BackendError(
                f"RediSearch query failed: {e}", original_error=e
            )

        search_results = []
        for doc in results.docs:
            # RediSearch returns distance, convert to similarity
            # For COSINE: similarity = 1 - distance
            distance = float(doc.score)
            similarity = 1.0 - distance

            if similarity >= threshold:
                # Parse metadata
                metadata_str = doc.metadata
                if isinstance(metadata_str, bytes):
                    metadata_str = metadata_str.decode()
                metadata = json.loads(metadata_str)

                # Get the key
                key = doc.key
                if isinstance(key, bytes):
                    key = key.decode()

                # Retrieve the full entry to get the vector
                entry = self.get_by_key(key)
                entry_vector = entry.vector if entry else None

                search_results.append(
                    SearchResult(
                        key=key,
                        similarity=similarity,
                        metadata=metadata,
                        vector=entry_vector,
                    )
                )

                if len(search_results) >= top_k:
                    break

        return search_results

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

        try:
            doc = self._redis.json().get(redis_key)
        except Exception:
            return None

        if not doc:
            return None

        vector = np.array(doc["vector"], dtype=np.float32)
        metadata_str = doc["metadata"]
        if isinstance(metadata_str, bytes):
            metadata_str = metadata_str.decode()
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

    def drop_index(self) -> None:
        """Drop the search index (useful for cleanup/reset)."""
        try:
            self._redis.ft(self._index_name).dropindex(delete_documents=False)
        except Exception:
            pass  # Index didn't exist
