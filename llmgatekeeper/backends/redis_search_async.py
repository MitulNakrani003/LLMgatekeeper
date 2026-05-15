"""Async RediSearch backend using redis.asyncio for KNN vector search."""

import json
from typing import Any, Dict, List, Optional

import numpy as np
from numpy.typing import NDArray
from redis.commands.search.field import TagField, TextField, VectorField
from redis.commands.search.index_definition import IndexDefinition, IndexType
from redis.commands.search.query import Query

from llmgatekeeper.backends.base import AsyncCacheBackend, CacheEntry, SearchResult
from llmgatekeeper.backends.redis_search import RediSearchBackend
from llmgatekeeper.exceptions import BackendError, ConfigurationError


class AsyncRediSearchBackend(AsyncCacheBackend):
    """Async backend using RediSearch KNN/range vector queries.

    Mirrors RediSearchBackend but performs all I/O over ``redis.asyncio``.
    Module availability and index existence are deferred to ``connect()``
    because constructor cannot perform awaitable checks.
    """

    def __init__(
        self,
        redis_client: Any,
        namespace: str = "llmgk",
        vector_dimension: int = 384,
        distance_metric: str = "COSINE",
        index_type: str = "HNSW",
    ) -> None:
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
        self._connected = False

    async def connect(self) -> None:
        """Verify module availability and ensure the index exists."""
        if not await self._is_redisearch_available():
            raise RuntimeError(
                "RediSearch module not available. "
                "Please install Redis Stack or enable the RediSearch module."
            )
        await self._ensure_index_exists()
        self._connected = True

    async def _is_redisearch_available(self) -> bool:
        try:
            modules = await self._redis.module_list()
            module_names = []
            for m in modules:
                if hasattr(m, "get"):
                    name = m.get("name", m.get(b"name", b""))
                else:
                    name = ""
                if isinstance(name, bytes):
                    name = name.decode()
                module_names.append(name.lower())
            return "search" in module_names or "ft" in module_names
        except Exception:
            return False

    async def _ensure_index_exists(self) -> None:
        try:
            info = await self._redis.ft(self._index_name).info()
        except Exception:
            info = None

        if info is not None:
            existing_dim = RediSearchBackend._extract_index_dimension(info)
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

        await self._redis.ft(self._index_name).create_index(
            schema, definition=definition
        )

    def _make_key(self, key: str) -> str:
        return f"{self._namespace}:entry:{key}"

    async def store_vector(
        self,
        key: str,
        vector: NDArray[np.float32],
        metadata: Dict[str, Any],
        ttl: Optional[int] = None,
    ) -> None:
        redis_key = self._make_key(key)
        doc = {
            "key": key,
            "vector": vector.astype(np.float32).tolist(),
            "metadata": json.dumps(metadata),
        }
        await self._redis.json().set(redis_key, "$", doc)
        await self._redis.sadd(self._keys_set, key)
        if ttl is not None:
            await self._redis.expire(redis_key, ttl)

    async def search_similar(
        self,
        vector: NDArray[np.float32],
        threshold: float = 0.85,
        top_k: int = 1,
    ) -> List[SearchResult]:
        vector_bytes = vector.astype(np.float32).tobytes()

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
            query_params = {"query_vec": vector_bytes, "radius": str(radius)}
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
            results = await self._redis.ft(self._index_name).search(
                query, query_params=query_params
            )
        except Exception as e:
            if "no such index" in str(e).lower():
                return []
            raise BackendError(
                f"RediSearch query failed: {e}", original_error=e
            )

        search_results: List[SearchResult] = []
        for doc in results.docs:
            distance = float(doc.score)
            similarity = 1.0 - distance
            if similarity < threshold:
                continue
            metadata_str = doc.metadata
            if isinstance(metadata_str, bytes):
                metadata_str = metadata_str.decode()
            metadata = json.loads(metadata_str)
            key = doc.key
            if isinstance(key, bytes):
                key = key.decode()
            search_results.append(
                SearchResult(
                    key=key,
                    similarity=similarity,
                    metadata=metadata,
                    vector=None,
                )
            )
            if len(search_results) >= top_k:
                break
        return search_results

    async def delete(self, key: str) -> bool:
        redis_key = self._make_key(key)
        deleted = await self._redis.delete(redis_key)
        await self._redis.srem(self._keys_set, key)
        return deleted > 0

    async def get_by_key(self, key: str) -> Optional[CacheEntry]:
        redis_key = self._make_key(key)
        try:
            doc = await self._redis.json().get(redis_key)
        except Exception:
            return None
        if not doc:
            return None
        vector = np.array(doc["vector"], dtype=np.float32)
        metadata_str = doc["metadata"]
        if isinstance(metadata_str, bytes):
            metadata_str = metadata_str.decode()
        metadata = json.loads(metadata_str)
        return CacheEntry(key=key, vector=vector, metadata=metadata)

    async def clear(self) -> int:
        keys = await self._redis.smembers(self._keys_set)
        count = 0
        for key_bytes in keys:
            key = key_bytes.decode() if isinstance(key_bytes, bytes) else key_bytes
            redis_key = self._make_key(key)
            if await self._redis.delete(redis_key):
                count += 1
        await self._redis.delete(self._keys_set)
        return count

    async def count(self) -> int:
        return await self._redis.scard(self._keys_set)

    async def drop_index(self) -> None:
        try:
            await self._redis.ft(self._index_name).dropindex(
                delete_documents=False
            )
        except Exception:
            pass
