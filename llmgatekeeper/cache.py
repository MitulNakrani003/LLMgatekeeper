"""SemanticCache - Main user-facing class for semantic caching.

This module provides the SemanticCache class, which is the primary interface
for storing and retrieving cached responses based on semantic similarity.
It also provides AsyncSemanticCache for async/await usage.
"""

import asyncio
import hashlib
import time
import uuid
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from llmgatekeeper.analytics import CacheAnalytics, CacheStats
from llmgatekeeper.backends.base import AsyncCacheBackend, CacheBackend, SearchResult
from llmgatekeeper.backends.factory import create_redis_backend
from llmgatekeeper.embeddings.base import EmbeddingProvider
from llmgatekeeper.embeddings.sentence_transformer import SentenceTransformerProvider
from llmgatekeeper.exceptions import BackendError, EmbeddingError
from llmgatekeeper.logging import get_logger
from llmgatekeeper.similarity.confidence import (
    ConfidenceClassifier,
    ConfidenceLevel,
    get_model_classifier,
)
from llmgatekeeper.similarity.retriever import (
    RetrievalResult,
    SimilarityRetriever,
)

# Module logger
_logger = get_logger(__name__)


@dataclass
class CacheResult:
    """Result from a cache lookup including metadata.

    Attributes:
        response: The cached response string.
        similarity: Similarity score between query and cached entry.
        confidence: Confidence level of the match.
        key: The cache key of the matched entry.
        metadata: Additional metadata stored with the entry.
    """

    response: str
    similarity: float
    confidence: ConfidenceLevel
    key: str
    metadata: Dict[str, Any]

    def __str__(self) -> str:
        """Return the response string."""
        return self.response


class SemanticCache:
    """Semantic cache for LLM responses using embedding similarity.

    SemanticCache provides intelligent caching that recognizes semantically
    equivalent queries (e.g., "What's the weather?" and "Tell me the weather")
    rather than requiring exact string matches.

    Example:
        >>> import redis
        >>> from llmgatekeeper import SemanticCache
        >>>
        >>> # Simple initialization with Redis client
        >>> cache = SemanticCache(redis.Redis())
        >>>
        >>> # Store a response
        >>> cache.set("What is Python?", "Python is a programming language.")
        >>>
        >>> # Retrieve with exact or similar query
        >>> cache.get("What is Python?")
        'Python is a programming language.'
        >>> cache.get("Tell me about Python")  # Semantically similar
        'Python is a programming language.'
        >>>
        >>> # Configure threshold for stricter matching
        >>> cache = SemanticCache(redis.Redis(), threshold=0.95)

    Args:
        redis_client: A Redis client instance. The cache does not manage
            the connection lifecycle - that's the user's responsibility.
        embedding_provider: Optional custom embedding provider. Defaults to
            SentenceTransformerProvider with all-MiniLM-L6-v2.
        threshold: Minimum similarity score for a cache hit. Default 0.85.
        default_ttl: Default time-to-live in seconds for cache entries.
            If None (default), entries never expire unless TTL is specified
            per-entry in set().
        backend: Optional custom cache backend. If not provided, creates
            one from the redis_client using auto-detection.
        model_name: Name of embedding model for confidence threshold tuning.
            If embedding_provider is SentenceTransformerProvider, this is
            auto-detected.

    Note:
        The cache uses cosine similarity by default. A threshold of 0.85
        typically works well for most use cases, but you may need to tune
        this based on your specific queries and embedding model.
    """

    def __init__(
        self,
        redis_client: Any,
        *,
        embedding_provider: Optional[EmbeddingProvider] = None,
        threshold: float = 0.85,
        default_ttl: Optional[int] = None,
        backend: Optional[CacheBackend] = None,
        model_name: Optional[str] = None,
        namespace: str = "default",
        enable_analytics: bool = False,
    ) -> None:
        """Initialize the semantic cache.

        Args:
            redis_client: Redis client instance for storage.
            embedding_provider: Custom embedding provider. Defaults to
                SentenceTransformerProvider.
            threshold: Minimum similarity for cache hits. Default 0.85.
            default_ttl: Default TTL in seconds. None means no expiration.
            backend: Custom cache backend. Auto-detected if not provided.
            model_name: Embedding model name for confidence tuning.
            namespace: Namespace for cache isolation. Default "default".
            enable_analytics: Enable analytics tracking. Default False.
        """
        # Validate threshold
        if not (0.0 <= threshold <= 1.0):
            raise ValueError(f"threshold must be in [0, 1], got {threshold}")

        # Set up embedding provider
        if embedding_provider is not None:
            self._embedding_provider = embedding_provider
            self._model_name = model_name
        else:
            self._embedding_provider = SentenceTransformerProvider()
            self._model_name = model_name or "all-MiniLM-L6-v2"

        # Set up backend
        if backend is not None:
            self._backend = backend
        else:
            self._backend = create_redis_backend(
                redis_client,
                namespace=namespace,
                vector_dimension=self._embedding_provider.dimension,
            )

        # Set up confidence classifier
        self._classifier = get_model_classifier(self._model_name)

        # Set up retriever
        self._retriever = SimilarityRetriever(
            backend=self._backend,
            top_k=1,
            threshold=threshold,
            confidence_classifier=self._classifier,
        )

        self._threshold = threshold
        self._default_ttl = default_ttl
        self._namespace = namespace
        self._redis_client = redis_client

        # Set up analytics
        self._enable_analytics = enable_analytics
        self._analytics: Optional[CacheAnalytics] = (
            CacheAnalytics() if enable_analytics else None
        )

    @property
    def threshold(self) -> float:
        """Return the similarity threshold for cache hits."""
        return self._threshold

    @threshold.setter
    def threshold(self, value: float) -> None:
        """Set the similarity threshold."""
        if not (0.0 <= value <= 1.0):
            raise ValueError(f"threshold must be in [0, 1], got {value}")
        self._threshold = value
        self._retriever.threshold = value

    @property
    def namespace(self) -> str:
        """Return the cache namespace."""
        return self._namespace

    @property
    def embedding_provider(self) -> EmbeddingProvider:
        """Return the embedding provider."""
        return self._embedding_provider

    @property
    def default_ttl(self) -> Optional[int]:
        """Return the default TTL for cache entries."""
        return self._default_ttl

    @default_ttl.setter
    def default_ttl(self, value: Optional[int]) -> None:
        """Set the default TTL for cache entries."""
        if value is not None and value < 0:
            raise ValueError(f"default_ttl must be non-negative, got {value}")
        self._default_ttl = value

    def _generate_key(self, query: str) -> str:
        """Generate a unique cache key for a query.

        Uses MD5 hash of the query text for consistent key generation.
        """
        hash_digest = hashlib.md5(query.encode("utf-8")).hexdigest()
        return f"llmgk:{self._namespace}:{hash_digest}"

    def _embed(self, text: str) -> NDArray[np.float32]:
        """Embed a text string using the configured provider.

        Raises:
            EmbeddingError: If embedding generation fails.
        """
        try:
            return self._embedding_provider.embed(text)
        except Exception as e:
            _logger.error(f"Embedding failed for text: {text[:50]}...", error=str(e))
            raise EmbeddingError(f"Failed to generate embedding: {e}", original_error=e)

    def set(
        self,
        query: str,
        response: str,
        *,
        metadata: Optional[Dict[str, Any]] = None,
        ttl: Optional[int] = None,
    ) -> str:
        """Store a query-response pair in the cache.

        Args:
            query: The query text to cache.
            response: The response to store.
            metadata: Optional metadata to store with the entry.
            ttl: Optional time-to-live in seconds. If not specified,
                uses the cache's default_ttl. Pass 0 to explicitly
                disable TTL for this entry when default_ttl is set.

        Returns:
            The cache key for the stored entry.

        Raises:
            EmbeddingError: If embedding generation fails.
            BackendError: If storing to the backend fails.

        Example:
            >>> cache.set("What is Python?", "Python is a programming language.")
            'llmgk:default:abc123...'
            >>> cache.set(
            ...     "What is Python?",
            ...     "Python is a programming language.",
            ...     metadata={"model": "gpt-4", "tokens": 50},
            ...     ttl=3600
            ... )
        """
        _logger.debug(f"Setting cache entry", query=query[:50], ttl=ttl)

        # Generate embedding for the query
        embedding = self._embed(query)

        # Generate cache key
        key = self._generate_key(query)

        # Prepare metadata
        entry_metadata = metadata.copy() if metadata else {}
        entry_metadata["query"] = query
        entry_metadata["response"] = response

        # Determine effective TTL:
        # - If ttl is explicitly provided (including 0), use it
        # - Otherwise, use the default_ttl
        # - ttl=0 means no expiration (overrides default_ttl)
        if ttl is not None:
            effective_ttl = ttl if ttl > 0 else None
        else:
            effective_ttl = self._default_ttl

        # Store in backend
        try:
            self._backend.store_vector(
                key=key,
                vector=embedding,
                metadata=entry_metadata,
                ttl=effective_ttl,
            )
        except Exception as e:
            _logger.error(f"Backend store failed", key=key, error=str(e))
            raise BackendError(f"Failed to store cache entry: {e}", original_error=e)

        _logger.debug(f"Cache entry stored", key=key)
        return key

    def get(
        self,
        query: str,
        *,
        include_metadata: bool = False,
        threshold: Optional[float] = None,
    ) -> Optional[Union[str, CacheResult]]:
        """Retrieve a cached response for a query.

        Searches for semantically similar cached queries and returns the
        response if a match is found above the threshold.

        Args:
            query: The query to look up.
            include_metadata: If True, returns CacheResult with full details.
                If False (default), returns just the response string.
            threshold: Override the default threshold for this query.

        Returns:
            If include_metadata is False: The cached response string, or None.
            If include_metadata is True: CacheResult with full details, or None.

        Raises:
            EmbeddingError: If embedding generation fails.
            BackendError: If searching the backend fails.

        Example:
            >>> # Simple lookup
            >>> response = cache.get("What is Python?")
            >>> if response:
            ...     print(response)
            'Python is a programming language.'

            >>> # With metadata
            >>> result = cache.get("What is Python?", include_metadata=True)
            >>> if result:
            ...     print(f"Response: {result.response}")
            ...     print(f"Confidence: {result.confidence}")
            ...     print(f"Similarity: {result.similarity:.2f}")
        """
        _logger.debug(f"Getting cache entry", query=query[:50])
        start_time = time.perf_counter()

        # Generate embedding for the query
        embedding = self._embed(query)

        # Search for similar entries
        effective_threshold = threshold if threshold is not None else self._threshold
        try:
            response = self._retriever.find_similar(
                query_vector=embedding,
                top_k=1,
                threshold=effective_threshold,
            )
        except Exception as e:
            _logger.error(f"Backend search failed", query=query[:50], error=str(e))
            raise BackendError(f"Failed to search cache: {e}", original_error=e)

        # Calculate latency
        latency_ms = (time.perf_counter() - start_time) * 1000

        # Check if we have a match
        if not response.results:
            _logger.debug(f"Cache miss", query=query[:50], latency_ms=f"{latency_ms:.2f}")
            # Record miss with analytics
            if self._analytics is not None:
                # Check for near-miss by searching without threshold
                near_miss_response = self._retriever.find_similar(
                    query_vector=embedding,
                    top_k=1,
                    threshold=0.0,
                )
                closest_similarity = (
                    near_miss_response.results[0].similarity
                    if near_miss_response.results
                    else None
                )
                self._analytics.record_miss(
                    latency_ms=latency_ms,
                    query=query,
                    closest_similarity=closest_similarity,
                    threshold=effective_threshold,
                )
            return None

        result = response.results[0]
        cached_response = result.metadata.get("response")

        if cached_response is None:
            _logger.debug(f"Cache miss (no response in metadata)", query=query[:50])
            # Record miss
            if self._analytics is not None:
                self._analytics.record_miss(
                    latency_ms=latency_ms,
                    query=query,
                    closest_similarity=result.similarity,
                    threshold=effective_threshold,
                )
            return None

        # Record hit
        _logger.debug(
            f"Cache hit",
            query=query[:50],
            similarity=f"{result.similarity:.3f}",
            latency_ms=f"{latency_ms:.2f}",
        )
        if self._analytics is not None:
            self._analytics.record_hit(latency_ms=latency_ms, query=query)

        if include_metadata:
            return CacheResult(
                response=cached_response,
                similarity=result.similarity,
                confidence=result.confidence,
                key=result.key,
                metadata={
                    k: v
                    for k, v in result.metadata.items()
                    if k not in ("query", "response")
                },
            )

        return cached_response

    def get_similar(
        self,
        query: str,
        *,
        top_k: int = 5,
        threshold: Optional[float] = None,
    ) -> List[CacheResult]:
        """Retrieve multiple similar cached responses.

        Useful for ensemble/voting scenarios or exploring similar cached content.

        Args:
            query: The query to look up.
            top_k: Maximum number of results to return.
            threshold: Override the default threshold for this query.

        Returns:
            List of CacheResult objects sorted by similarity (descending).

        Example:
            >>> results = cache.get_similar("What is Python?", top_k=3)
            >>> for result in results:
            ...     print(f"{result.similarity:.2f}: {result.response[:50]}...")
        """
        embedding = self._embed(query)
        effective_threshold = threshold if threshold is not None else self._threshold

        response = self._retriever.find_similar(
            query_vector=embedding,
            top_k=top_k,
            threshold=effective_threshold,
        )

        results = []
        for r in response.results:
            cached_response = r.metadata.get("response")
            if cached_response is not None:
                results.append(
                    CacheResult(
                        response=cached_response,
                        similarity=r.similarity,
                        confidence=r.confidence,
                        key=r.key,
                        metadata={
                            k: v
                            for k, v in r.metadata.items()
                            if k not in ("query", "response")
                        },
                    )
                )

        return results

    def delete(self, query: str) -> bool:
        """Delete a cached entry by query.

        Args:
            query: The query whose cache entry should be deleted.

        Returns:
            True if an entry was deleted, False if not found.

        Example:
            >>> cache.set("What is Python?", "A programming language.")
            >>> cache.delete("What is Python?")
            True
        """
        key = self._generate_key(query)
        return self._backend.delete(key)

    def delete_by_key(self, key: str) -> bool:
        """Delete a cached entry by its key.

        Args:
            key: The cache key to delete.

        Returns:
            True if an entry was deleted, False if not found.
        """
        return self._backend.delete(key)

    def exists(self, query: str) -> bool:
        """Check if a query has an exact cached entry.

        Note: This checks for an exact query match, not semantic similarity.

        Args:
            query: The query to check.

        Returns:
            True if an exact cache entry exists.
        """
        key = self._generate_key(query)
        return self._backend.get_by_key(key) is not None

    def clear(self) -> int:
        """Clear all entries from the cache.

        Returns:
            Number of entries deleted.
        """
        return self._backend.clear()

    def count(self) -> int:
        """Return the number of entries in the cache.

        Returns:
            Number of cached entries.
        """
        return self._backend.count()

    def stats(self) -> Optional[CacheStats]:
        """Get cache performance statistics.

        Returns statistics about cache hit/miss rates, latency percentiles,
        and frequent queries. Only available when enable_analytics=True.

        Returns:
            CacheStats object with metrics, or None if analytics disabled.

        Example:
            >>> cache = SemanticCache(redis_client, enable_analytics=True)
            >>> cache.set("query", "response")
            >>> cache.get("query")  # Hit
            >>> cache.get("other")  # Miss
            >>> stats = cache.stats()
            >>> print(f"Hit rate: {stats.hit_rate:.2%}")
        """
        if self._analytics is None:
            return None
        return self._analytics.get_stats()

    def reset_stats(self) -> None:
        """Reset analytics statistics.

        Clears all accumulated metrics. Only has effect when analytics enabled.
        """
        if self._analytics is not None:
            self._analytics.reset()

    @property
    def analytics_enabled(self) -> bool:
        """Return whether analytics tracking is enabled."""
        return self._enable_analytics

    def warm(
        self,
        pairs: Sequence[Tuple[str, str]],
        *,
        metadata: Optional[Dict[str, Any]] = None,
        ttl: Optional[int] = None,
        batch_size: int = 100,
        on_progress: Optional[Callable[[int, int], None]] = None,
    ) -> int:
        """Bulk-load query-response pairs into the cache.

        Efficiently loads multiple entries using batch embedding for better
        performance during deployment or cache warming scenarios.

        Args:
            pairs: Sequence of (query, response) tuples to cache.
            metadata: Optional metadata to apply to all entries.
            ttl: Optional TTL in seconds. Uses default_ttl if not specified.
            batch_size: Number of queries to embed in a single batch.
                Default 100.
            on_progress: Optional callback called after each batch with
                (completed_count, total_count).

        Returns:
            Number of entries successfully stored.

        Example:
            >>> pairs = [
            ...     ("What is Python?", "A programming language."),
            ...     ("What is Java?", "Another programming language."),
            ... ]
            >>> cache.warm(pairs)
            2
            >>> cache.warm(pairs, on_progress=lambda done, total: print(f"{done}/{total}"))
        """
        if not pairs:
            return 0

        total = len(pairs)
        stored = 0

        # Determine effective TTL
        if ttl is not None:
            effective_ttl = ttl if ttl > 0 else None
        else:
            effective_ttl = self._default_ttl

        # Process in batches
        for batch_start in range(0, total, batch_size):
            batch_end = min(batch_start + batch_size, total)
            batch_pairs = pairs[batch_start:batch_end]

            # Extract queries for batch embedding
            queries = [q for q, _ in batch_pairs]

            # Use batch embedding for efficiency
            embeddings = self._embedding_provider.embed_batch(queries)

            # Store each entry
            for i, (query, response) in enumerate(batch_pairs):
                embedding = embeddings[i]
                key = self._generate_key(query)

                # Prepare metadata
                entry_metadata = metadata.copy() if metadata else {}
                entry_metadata["query"] = query
                entry_metadata["response"] = response

                # Store in backend
                self._backend.store_vector(
                    key=key,
                    vector=embedding,
                    metadata=entry_metadata,
                    ttl=effective_ttl,
                )
                stored += 1

            # Report progress after each batch
            if on_progress is not None:
                on_progress(stored, total)

        return stored

    def __repr__(self) -> str:
        """Return a string representation of the cache."""
        return (
            f"SemanticCache(namespace={self._namespace!r}, "
            f"threshold={self._threshold}, "
            f"model={self._model_name!r})"
        )


class AsyncSemanticCache:
    """Async semantic cache for LLM responses using embedding similarity.

    AsyncSemanticCache provides the same semantic caching capabilities as
    SemanticCache but with async/await support for non-blocking operations.

    Example:
        >>> import redis.asyncio as redis
        >>> from llmgatekeeper import AsyncSemanticCache
        >>> from llmgatekeeper.backends.redis_async import AsyncRedisBackend
        >>>
        >>> async def main():
        ...     client = redis.Redis()
        ...     backend = AsyncRedisBackend(client)
        ...     cache = AsyncSemanticCache(backend)
        ...
        ...     await cache.set("What is Python?", "Python is a programming language.")
        ...     result = await cache.get("What is Python?")
        ...     print(result)
        >>>
        >>> asyncio.run(main())

    Args:
        backend: An AsyncCacheBackend instance for storage.
        embedding_provider: Optional custom embedding provider. Defaults to
            SentenceTransformerProvider with all-MiniLM-L6-v2.
        threshold: Minimum similarity score for a cache hit. Default 0.85.
        default_ttl: Default time-to-live in seconds for cache entries.
        model_name: Name of embedding model for confidence threshold tuning.
        namespace: Namespace for cache key generation. Default "default".
        enable_analytics: Enable analytics tracking. Default False.

    Note:
        The async cache uses the embedding provider's async methods (aembed)
        when available. If the provider only has sync methods, they will be
        called directly (which may block the event loop briefly).
    """

    def __init__(
        self,
        backend: AsyncCacheBackend,
        *,
        embedding_provider: Optional[EmbeddingProvider] = None,
        threshold: float = 0.85,
        default_ttl: Optional[int] = None,
        model_name: Optional[str] = None,
        namespace: str = "default",
        enable_analytics: bool = False,
    ) -> None:
        """Initialize the async semantic cache.

        Args:
            backend: Async cache backend for storage.
            embedding_provider: Custom embedding provider. Defaults to
                SentenceTransformerProvider.
            threshold: Minimum similarity for cache hits. Default 0.85.
            default_ttl: Default TTL in seconds. None means no expiration.
            model_name: Embedding model name for confidence tuning.
            namespace: Namespace for cache isolation. Default "default".
            enable_analytics: Enable analytics tracking. Default False.
        """
        # Validate threshold
        if not (0.0 <= threshold <= 1.0):
            raise ValueError(f"threshold must be in [0, 1], got {threshold}")

        # Set up embedding provider
        if embedding_provider is not None:
            self._embedding_provider = embedding_provider
            self._model_name = model_name
        else:
            self._embedding_provider = SentenceTransformerProvider()
            self._model_name = model_name or "all-MiniLM-L6-v2"

        self._backend = backend

        # Set up confidence classifier
        self._classifier = get_model_classifier(self._model_name)

        self._threshold = threshold
        self._default_ttl = default_ttl
        self._namespace = namespace

        # Set up analytics
        self._enable_analytics = enable_analytics
        self._analytics: Optional[CacheAnalytics] = (
            CacheAnalytics() if enable_analytics else None
        )

    @property
    def threshold(self) -> float:
        """Return the similarity threshold for cache hits."""
        return self._threshold

    @threshold.setter
    def threshold(self, value: float) -> None:
        """Set the similarity threshold."""
        if not (0.0 <= value <= 1.0):
            raise ValueError(f"threshold must be in [0, 1], got {value}")
        self._threshold = value

    @property
    def namespace(self) -> str:
        """Return the cache namespace."""
        return self._namespace

    @property
    def embedding_provider(self) -> EmbeddingProvider:
        """Return the embedding provider."""
        return self._embedding_provider

    @property
    def default_ttl(self) -> Optional[int]:
        """Return the default TTL for cache entries."""
        return self._default_ttl

    @default_ttl.setter
    def default_ttl(self, value: Optional[int]) -> None:
        """Set the default TTL for cache entries."""
        if value is not None and value < 0:
            raise ValueError(f"default_ttl must be non-negative, got {value}")
        self._default_ttl = value

    def _generate_key(self, query: str) -> str:
        """Generate a unique cache key for a query."""
        hash_digest = hashlib.md5(query.encode("utf-8")).hexdigest()
        return f"llmgk:{self._namespace}:{hash_digest}"

    async def _embed(self, text: str) -> NDArray[np.float32]:
        """Embed a text string using the configured provider."""
        return await self._embedding_provider.aembed(text)

    async def _embed_batch(self, texts: List[str]) -> List[NDArray[np.float32]]:
        """Embed multiple text strings using the configured provider."""
        return await self._embedding_provider.aembed_batch(texts)

    async def set(
        self,
        query: str,
        response: str,
        *,
        metadata: Optional[Dict[str, Any]] = None,
        ttl: Optional[int] = None,
    ) -> str:
        """Store a query-response pair in the cache asynchronously.

        Args:
            query: The query text to cache.
            response: The response to store.
            metadata: Optional metadata to store with the entry.
            ttl: Optional time-to-live in seconds. If not specified,
                uses the cache's default_ttl. Pass 0 to explicitly
                disable TTL for this entry when default_ttl is set.

        Returns:
            The cache key for the stored entry.

        Example:
            >>> await cache.set("What is Python?", "Python is a programming language.")
        """
        # Generate embedding for the query
        embedding = await self._embed(query)

        # Generate cache key
        key = self._generate_key(query)

        # Prepare metadata
        entry_metadata = metadata.copy() if metadata else {}
        entry_metadata["query"] = query
        entry_metadata["response"] = response

        # Determine effective TTL
        if ttl is not None:
            effective_ttl = ttl if ttl > 0 else None
        else:
            effective_ttl = self._default_ttl

        # Store in backend
        await self._backend.store_vector(
            key=key,
            vector=embedding,
            metadata=entry_metadata,
            ttl=effective_ttl,
        )

        return key

    async def get(
        self,
        query: str,
        *,
        include_metadata: bool = False,
        threshold: Optional[float] = None,
    ) -> Optional[Union[str, CacheResult]]:
        """Retrieve a cached response for a query asynchronously.

        Args:
            query: The query to look up.
            include_metadata: If True, returns CacheResult with full details.
            threshold: Override the default threshold for this query.

        Returns:
            If include_metadata is False: The cached response string, or None.
            If include_metadata is True: CacheResult with full details, or None.

        Example:
            >>> response = await cache.get("What is Python?")
            >>> if response:
            ...     print(response)
        """
        start_time = time.perf_counter()

        # Generate embedding for the query
        embedding = await self._embed(query)

        # Search for similar entries
        effective_threshold = threshold if threshold is not None else self._threshold
        results = await self._backend.search_similar(
            vector=embedding,
            threshold=effective_threshold,
            top_k=1,
        )

        # Calculate latency
        latency_ms = (time.perf_counter() - start_time) * 1000

        # Check if we have a match
        if not results:
            # Record miss with analytics
            if self._analytics is not None:
                # Check for near-miss by searching without threshold
                near_miss_results = await self._backend.search_similar(
                    vector=embedding,
                    threshold=0.0,
                    top_k=1,
                )
                closest_similarity = (
                    near_miss_results[0].similarity
                    if near_miss_results
                    else None
                )
                self._analytics.record_miss(
                    latency_ms=latency_ms,
                    query=query,
                    closest_similarity=closest_similarity,
                    threshold=effective_threshold,
                )
            return None

        result = results[0]
        cached_response = result.metadata.get("response")

        if cached_response is None:
            if self._analytics is not None:
                self._analytics.record_miss(
                    latency_ms=latency_ms,
                    query=query,
                    closest_similarity=result.similarity,
                    threshold=effective_threshold,
                )
            return None

        # Record hit
        if self._analytics is not None:
            self._analytics.record_hit(latency_ms=latency_ms, query=query)

        if include_metadata:
            confidence = self._classifier.classify(result.similarity)
            return CacheResult(
                response=cached_response,
                similarity=result.similarity,
                confidence=confidence,
                key=result.key,
                metadata={
                    k: v
                    for k, v in result.metadata.items()
                    if k not in ("query", "response")
                },
            )

        return cached_response

    async def get_similar(
        self,
        query: str,
        *,
        top_k: int = 5,
        threshold: Optional[float] = None,
    ) -> List[CacheResult]:
        """Retrieve multiple similar cached responses asynchronously.

        Args:
            query: The query to look up.
            top_k: Maximum number of results to return.
            threshold: Override the default threshold for this query.

        Returns:
            List of CacheResult objects sorted by similarity (descending).
        """
        embedding = await self._embed(query)
        effective_threshold = threshold if threshold is not None else self._threshold

        results = await self._backend.search_similar(
            vector=embedding,
            threshold=effective_threshold,
            top_k=top_k,
        )

        cache_results = []
        for r in results:
            cached_response = r.metadata.get("response")
            if cached_response is not None:
                confidence = self._classifier.classify(r.similarity)
                cache_results.append(
                    CacheResult(
                        response=cached_response,
                        similarity=r.similarity,
                        confidence=confidence,
                        key=r.key,
                        metadata={
                            k: v
                            for k, v in r.metadata.items()
                            if k not in ("query", "response")
                        },
                    )
                )

        return cache_results

    async def delete(self, query: str) -> bool:
        """Delete a cached entry by query asynchronously.

        Args:
            query: The query whose cache entry should be deleted.

        Returns:
            True if an entry was deleted, False if not found.
        """
        key = self._generate_key(query)
        return await self._backend.delete(key)

    async def delete_by_key(self, key: str) -> bool:
        """Delete a cached entry by its key asynchronously.

        Args:
            key: The cache key to delete.

        Returns:
            True if an entry was deleted, False if not found.
        """
        return await self._backend.delete(key)

    async def exists(self, query: str) -> bool:
        """Check if a query has an exact cached entry asynchronously.

        Args:
            query: The query to check.

        Returns:
            True if an exact cache entry exists.
        """
        key = self._generate_key(query)
        return await self._backend.get_by_key(key) is not None

    async def clear(self) -> int:
        """Clear all entries from the cache asynchronously.

        Returns:
            Number of entries deleted.
        """
        return await self._backend.clear()

    async def count(self) -> int:
        """Return the number of entries in the cache asynchronously.

        Returns:
            Number of cached entries.
        """
        return await self._backend.count()

    def stats(self) -> Optional[CacheStats]:
        """Get cache performance statistics.

        Returns:
            CacheStats object with metrics, or None if analytics disabled.
        """
        if self._analytics is None:
            return None
        return self._analytics.get_stats()

    def reset_stats(self) -> None:
        """Reset analytics statistics."""
        if self._analytics is not None:
            self._analytics.reset()

    @property
    def analytics_enabled(self) -> bool:
        """Return whether analytics tracking is enabled."""
        return self._enable_analytics

    async def warm(
        self,
        pairs: Sequence[Tuple[str, str]],
        *,
        metadata: Optional[Dict[str, Any]] = None,
        ttl: Optional[int] = None,
        batch_size: int = 100,
        on_progress: Optional[Callable[[int, int], None]] = None,
    ) -> int:
        """Bulk-load query-response pairs into the cache asynchronously.

        Args:
            pairs: Sequence of (query, response) tuples to cache.
            metadata: Optional metadata to apply to all entries.
            ttl: Optional TTL in seconds. Uses default_ttl if not specified.
            batch_size: Number of queries to embed in a single batch.
            on_progress: Optional callback called after each batch.

        Returns:
            Number of entries successfully stored.

        Example:
            >>> pairs = [
            ...     ("What is Python?", "A programming language."),
            ...     ("What is Java?", "Another programming language."),
            ... ]
            >>> await cache.warm(pairs)
        """
        if not pairs:
            return 0

        total = len(pairs)
        stored = 0

        # Determine effective TTL
        if ttl is not None:
            effective_ttl = ttl if ttl > 0 else None
        else:
            effective_ttl = self._default_ttl

        # Process in batches
        for batch_start in range(0, total, batch_size):
            batch_end = min(batch_start + batch_size, total)
            batch_pairs = pairs[batch_start:batch_end]

            # Extract queries for batch embedding
            queries = [q for q, _ in batch_pairs]

            # Use async batch embedding for efficiency
            embeddings = await self._embed_batch(queries)

            # Store each entry
            for i, (query, response) in enumerate(batch_pairs):
                embedding = embeddings[i]
                key = self._generate_key(query)

                # Prepare metadata
                entry_metadata = metadata.copy() if metadata else {}
                entry_metadata["query"] = query
                entry_metadata["response"] = response

                # Store in backend
                await self._backend.store_vector(
                    key=key,
                    vector=embedding,
                    metadata=entry_metadata,
                    ttl=effective_ttl,
                )
                stored += 1

            # Report progress after each batch
            if on_progress is not None:
                on_progress(stored, total)

        return stored

    def __repr__(self) -> str:
        """Return a string representation of the cache."""
        return (
            f"AsyncSemanticCache(namespace={self._namespace!r}, "
            f"threshold={self._threshold}, "
            f"model={self._model_name!r})"
        )
