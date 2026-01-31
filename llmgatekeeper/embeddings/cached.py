"""Cached embedding provider that wraps any EmbeddingProvider with caching."""

import hashlib
from collections import OrderedDict
from typing import List, Optional

import numpy as np
from numpy.typing import NDArray
from redis import Redis

from llmgatekeeper.embeddings.base import EmbeddingProvider


class CachedEmbeddingProvider(EmbeddingProvider):
    """Embedding provider wrapper that caches embeddings.

    This provider wraps any EmbeddingProvider and caches the results
    to avoid re-computing embeddings for identical strings. It uses
    an in-memory LRU cache with optional Redis-backed persistence.

    Example:
        >>> from llmgatekeeper.embeddings import SentenceTransformerProvider
        >>> base_provider = SentenceTransformerProvider()
        >>> cached = CachedEmbeddingProvider(base_provider, max_size=1000)
        >>> embedding1 = cached.embed("Hello world")  # Computes embedding
        >>> embedding2 = cached.embed("Hello world")  # Returns cached

        >>> # With Redis persistence
        >>> import redis
        >>> r = redis.Redis()
        >>> cached = CachedEmbeddingProvider(base_provider, redis_client=r)
    """

    def __init__(
        self,
        provider: EmbeddingProvider,
        max_size: int = 10000,
        redis_client: Optional[Redis] = None,
        redis_prefix: str = "llmgk:emb:",
        redis_ttl: Optional[int] = None,
    ) -> None:
        """Initialize the cached embedding provider.

        Args:
            provider: The underlying embedding provider to wrap.
            max_size: Maximum number of entries in the in-memory LRU cache.
                Default is 10000.
            redis_client: Optional Redis client for persistent caching.
                If provided, embeddings will be stored in Redis as well.
            redis_prefix: Key prefix for Redis storage. Default is "llmgk:emb:".
            redis_ttl: Optional TTL in seconds for Redis entries.
                If None, entries don't expire.
        """
        self._provider = provider
        self._max_size = max_size
        self._redis = redis_client
        self._redis_prefix = redis_prefix
        self._redis_ttl = redis_ttl

        # LRU cache using OrderedDict
        self._cache: OrderedDict[str, NDArray[np.float32]] = OrderedDict()

    @property
    def dimension(self) -> int:
        """Return the dimension of the embedding vectors.

        Returns:
            The number of dimensions in the output embedding vectors.
        """
        return self._provider.dimension

    @property
    def provider(self) -> EmbeddingProvider:
        """Return the underlying embedding provider.

        Returns:
            The wrapped embedding provider.
        """
        return self._provider

    def _hash_text(self, text: str) -> str:
        """Create a hash key for the text.

        Args:
            text: The text to hash.

        Returns:
            A hex string hash of the text.
        """
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _get_from_cache(self, text: str) -> Optional[NDArray[np.float32]]:
        """Try to get embedding from cache (memory first, then Redis).

        Args:
            text: The text to look up.

        Returns:
            The cached embedding if found, None otherwise.
        """
        key = self._hash_text(text)

        # Check in-memory cache first
        if key in self._cache:
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            return self._cache[key]

        # Check Redis if available
        if self._redis is not None:
            redis_key = f"{self._redis_prefix}{key}"
            data = self._redis.get(redis_key)
            if data is not None:
                embedding = np.frombuffer(data, dtype=np.float32)
                # Add to in-memory cache
                self._put_in_memory_cache(key, embedding)
                return embedding

        return None

    def _put_in_memory_cache(
        self, key: str, embedding: NDArray[np.float32]
    ) -> None:
        """Add embedding to in-memory LRU cache.

        Args:
            key: The cache key (hash of text).
            embedding: The embedding to cache.
        """
        # If key already exists, move to end
        if key in self._cache:
            self._cache.move_to_end(key)
            return

        # Evict oldest if at capacity
        while len(self._cache) >= self._max_size:
            self._cache.popitem(last=False)

        self._cache[key] = embedding

    def _put_in_cache(self, text: str, embedding: NDArray[np.float32]) -> None:
        """Add embedding to all cache layers.

        Args:
            text: The original text.
            embedding: The embedding to cache.
        """
        key = self._hash_text(text)

        # Add to in-memory cache
        self._put_in_memory_cache(key, embedding)

        # Add to Redis if available
        if self._redis is not None:
            redis_key = f"{self._redis_prefix}{key}"
            data = embedding.tobytes()
            if self._redis_ttl is not None:
                self._redis.setex(redis_key, self._redis_ttl, data)
            else:
                self._redis.set(redis_key, data)

    def embed(self, text: str) -> NDArray[np.float32]:
        """Embed a single text string into a vector, using cache if available.

        Args:
            text: The text to embed.

        Returns:
            A numpy array of shape (dimension,) containing the embedding.
        """
        # Try cache first
        cached = self._get_from_cache(text)
        if cached is not None:
            return cached

        # Compute embedding
        embedding = self._provider.embed(text)

        # Store in cache
        self._put_in_cache(text, embedding)

        return embedding

    def embed_batch(self, texts: List[str]) -> List[NDArray[np.float32]]:
        """Embed multiple text strings, using cache where available.

        This method checks the cache for each text and only computes
        embeddings for texts not found in the cache.

        Args:
            texts: List of texts to embed.

        Returns:
            List of numpy arrays, each of shape (dimension,).
        """
        if not texts:
            return []

        results: List[Optional[NDArray[np.float32]]] = [None] * len(texts)
        texts_to_embed: List[tuple[int, str]] = []

        # Check cache for each text
        for i, text in enumerate(texts):
            cached = self._get_from_cache(text)
            if cached is not None:
                results[i] = cached
            else:
                texts_to_embed.append((i, text))

        # Compute embeddings for cache misses
        if texts_to_embed:
            indices, uncached_texts = zip(*texts_to_embed)
            new_embeddings = self._provider.embed_batch(list(uncached_texts))

            for idx, text, embedding in zip(indices, uncached_texts, new_embeddings):
                results[idx] = embedding
                self._put_in_cache(text, embedding)

        return results  # type: ignore

    async def aembed(self, text: str) -> NDArray[np.float32]:
        """Asynchronously embed a single text string, using cache if available.

        Args:
            text: The text to embed.

        Returns:
            A numpy array of shape (dimension,) containing the embedding.
        """
        # Try cache first (sync operation, fast)
        cached = self._get_from_cache(text)
        if cached is not None:
            return cached

        # Compute embedding asynchronously
        embedding = await self._provider.aembed(text)

        # Store in cache (sync operation, fast)
        self._put_in_cache(text, embedding)

        return embedding

    async def aembed_batch(self, texts: List[str]) -> List[NDArray[np.float32]]:
        """Asynchronously embed multiple text strings, using cache where available.

        Args:
            texts: List of texts to embed.

        Returns:
            List of numpy arrays, each of shape (dimension,).
        """
        if not texts:
            return []

        results: List[Optional[NDArray[np.float32]]] = [None] * len(texts)
        texts_to_embed: List[tuple[int, str]] = []

        # Check cache for each text (sync, fast)
        for i, text in enumerate(texts):
            cached = self._get_from_cache(text)
            if cached is not None:
                results[i] = cached
            else:
                texts_to_embed.append((i, text))

        # Compute embeddings for cache misses
        if texts_to_embed:
            indices, uncached_texts = zip(*texts_to_embed)
            new_embeddings = await self._provider.aembed_batch(list(uncached_texts))

            for idx, text, embedding in zip(indices, uncached_texts, new_embeddings):
                results[idx] = embedding
                self._put_in_cache(text, embedding)

        return results  # type: ignore

    def clear_cache(self) -> None:
        """Clear the in-memory cache.

        Note: This does not clear the Redis cache. Use clear_all() for that.
        """
        self._cache.clear()

    def clear_all(self) -> None:
        """Clear both in-memory and Redis caches.

        Warning: This will delete all keys matching the redis_prefix pattern.
        """
        self._cache.clear()

        if self._redis is not None:
            # Find and delete all keys with the prefix
            pattern = f"{self._redis_prefix}*"
            cursor = 0
            while True:
                cursor, keys = self._redis.scan(cursor, match=pattern, count=100)
                if keys:
                    self._redis.delete(*keys)
                if cursor == 0:
                    break

    def cache_size(self) -> int:
        """Return the number of entries in the in-memory cache.

        Returns:
            The number of cached embeddings.
        """
        return len(self._cache)
