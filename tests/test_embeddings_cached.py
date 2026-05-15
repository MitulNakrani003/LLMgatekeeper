"""Tests for the CachedEmbeddingProvider."""

from typing import List
from unittest.mock import MagicMock

import numpy as np
import pytest
from numpy.typing import NDArray

from llmgatekeeper.embeddings.base import EmbeddingProvider
from llmgatekeeper.embeddings.cached import CachedEmbeddingProvider


class MockEmbeddingProvider(EmbeddingProvider):
    """Mock embedding provider for testing."""

    def __init__(self, dim: int = 384):
        self._dimension = dim
        self.embed_call_count = 0
        self.embed_batch_call_count = 0

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, text: str) -> NDArray[np.float32]:
        self.embed_call_count += 1
        # Return deterministic embedding based on text hash
        np.random.seed(hash(text) % (2**32))
        return np.random.rand(self._dimension).astype(np.float32)

    def embed_batch(self, texts: List[str]) -> List[NDArray[np.float32]]:
        self.embed_batch_call_count += 1
        return [self.embed(text) for text in texts]


class TestCachedEmbeddingProviderInit:
    """Tests for CachedEmbeddingProvider initialization."""

    def test_wraps_provider(self):
        """Cached provider wraps underlying provider."""
        mock_provider = MockEmbeddingProvider()
        cached = CachedEmbeddingProvider(mock_provider)
        assert cached.provider is mock_provider

    def test_dimension_from_underlying_provider(self):
        """Dimension comes from underlying provider."""
        mock_provider = MockEmbeddingProvider(dim=512)
        cached = CachedEmbeddingProvider(mock_provider)
        assert cached.dimension == 512

    def test_default_max_size(self):
        """Default max_size is 10000."""
        mock_provider = MockEmbeddingProvider()
        cached = CachedEmbeddingProvider(mock_provider)
        assert cached._max_size == 10000

    def test_custom_max_size(self):
        """Can specify custom max_size."""
        mock_provider = MockEmbeddingProvider()
        cached = CachedEmbeddingProvider(mock_provider, max_size=500)
        assert cached._max_size == 500


class TestCachedEmbeddingProviderEmbed:
    """Tests for embed method caching."""

    def test_caches_embeddings(self):
        """TC-3.4.1: Second call for same text doesn't call underlying provider."""
        mock_provider = MockEmbeddingProvider()
        cached = CachedEmbeddingProvider(mock_provider, max_size=100)

        cached.embed("test query")
        cached.embed("test query")

        assert mock_provider.embed_call_count == 1

    def test_different_texts_not_cached(self):
        """TC-3.4.2: Different texts call provider each time."""
        mock_provider = MockEmbeddingProvider()
        cached = CachedEmbeddingProvider(mock_provider, max_size=100)

        cached.embed("query 1")
        cached.embed("query 2")

        assert mock_provider.embed_call_count == 2

    def test_cached_result_matches_original(self):
        """Cached embedding matches originally computed embedding."""
        mock_provider = MockEmbeddingProvider()
        cached = CachedEmbeddingProvider(mock_provider)

        embedding1 = cached.embed("test")
        embedding2 = cached.embed("test")

        assert np.array_equal(embedding1, embedding2)

    def test_embed_returns_correct_shape(self):
        """Embed returns numpy array of correct shape."""
        mock_provider = MockEmbeddingProvider(dim=384)
        cached = CachedEmbeddingProvider(mock_provider)

        embedding = cached.embed("test")

        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (384,)
        assert embedding.dtype == np.float32


class TestCachedEmbeddingProviderLRU:
    """Tests for LRU cache eviction."""

    def test_lru_eviction(self):
        """TC-3.4.3: LRU eviction works when cache full."""
        mock_provider = MockEmbeddingProvider()
        cached = CachedEmbeddingProvider(mock_provider, max_size=2)

        cached.embed("a")  # Call 1
        cached.embed("b")  # Call 2
        cached.embed("c")  # Call 3, evicts "a"
        cached.embed("a")  # Call 4, should call provider again

        assert mock_provider.embed_call_count == 4

    def test_lru_keeps_recently_used(self):
        """LRU keeps recently accessed items."""
        mock_provider = MockEmbeddingProvider()
        cached = CachedEmbeddingProvider(mock_provider, max_size=2)

        cached.embed("a")  # Call 1
        cached.embed("b")  # Call 2
        cached.embed("a")  # Access "a" to make it recently used
        cached.embed("c")  # Call 3, should evict "b" (least recently used)
        cached.embed("a")  # Should still be cached

        # a: 2 calls (initial + after eviction check should be cached)
        # b: 1 call
        # c: 1 call
        assert mock_provider.embed_call_count == 3

    def test_cache_size_respects_limit(self):
        """Cache size never exceeds max_size."""
        mock_provider = MockEmbeddingProvider()
        cached = CachedEmbeddingProvider(mock_provider, max_size=5)

        for i in range(10):
            cached.embed(f"query_{i}")

        assert cached.cache_size() <= 5


class TestCachedEmbeddingProviderBatch:
    """Tests for embed_batch method caching."""

    def test_batch_uses_cache(self):
        """Batch embed uses cache for previously embedded texts."""
        mock_provider = MockEmbeddingProvider()
        cached = CachedEmbeddingProvider(mock_provider)

        # Pre-populate cache
        cached.embed("a")
        cached.embed("b")

        # Reset counters
        mock_provider.embed_call_count = 0
        mock_provider.embed_batch_call_count = 0

        # Batch with mix of cached and uncached
        cached.embed_batch(["a", "c", "b"])

        # Only "c" should require computation
        assert mock_provider.embed_batch_call_count == 1

    def test_batch_empty_list(self):
        """Batch embed handles empty list."""
        mock_provider = MockEmbeddingProvider()
        cached = CachedEmbeddingProvider(mock_provider)

        result = cached.embed_batch([])

        assert result == []

    def test_batch_all_cached(self):
        """Batch returns all from cache when all are cached."""
        mock_provider = MockEmbeddingProvider()
        cached = CachedEmbeddingProvider(mock_provider)

        # Pre-populate cache
        cached.embed("a")
        cached.embed("b")
        cached.embed("c")

        # Reset counters
        mock_provider.embed_batch_call_count = 0

        # All should be in cache
        cached.embed_batch(["a", "b", "c"])

        assert mock_provider.embed_batch_call_count == 0

    def test_batch_adds_to_cache(self):
        """Batch embed adds new results to cache."""
        mock_provider = MockEmbeddingProvider()
        cached = CachedEmbeddingProvider(mock_provider)

        cached.embed_batch(["x", "y", "z"])

        # Reset counter
        mock_provider.embed_call_count = 0

        # All should now be cached
        cached.embed("x")
        cached.embed("y")
        cached.embed("z")

        assert mock_provider.embed_call_count == 0


class TestCachedEmbeddingProviderRedis:
    """Tests for Redis-backed caching."""

    @pytest.fixture
    def mock_redis(self):
        """Create a mock Redis client."""
        redis = MagicMock()
        redis.get.return_value = None
        return redis

    def test_stores_in_redis(self, mock_redis):
        """Embeddings are stored in Redis when client provided."""
        mock_provider = MockEmbeddingProvider()
        cached = CachedEmbeddingProvider(
            mock_provider,
            redis_client=mock_redis,
            redis_prefix="test:",
        )

        cached.embed("test query")

        # Should have called set on Redis
        assert mock_redis.set.called or mock_redis.setex.called

    def test_retrieves_from_redis(self, mock_redis):
        """Embeddings are retrieved from Redis on cache miss."""
        mock_provider = MockEmbeddingProvider()

        # Simulate Redis having the embedding
        fake_embedding = np.zeros(384, dtype=np.float32)
        mock_redis.get.return_value = fake_embedding.tobytes()

        cached = CachedEmbeddingProvider(
            mock_provider,
            redis_client=mock_redis,
            redis_prefix="test:",
        )

        cached.embed("test query")

        # Should not have called the underlying provider
        assert mock_provider.embed_call_count == 0

    def test_redis_ttl(self, mock_redis):
        """Redis entries use TTL when specified."""
        mock_provider = MockEmbeddingProvider()
        cached = CachedEmbeddingProvider(
            mock_provider,
            redis_client=mock_redis,
            redis_prefix="test:",
            redis_ttl=3600,
        )

        cached.embed("test query")

        # Should have used setex with TTL
        mock_redis.setex.assert_called()
        call_args = mock_redis.setex.call_args
        assert call_args[0][1] == 3600  # TTL argument

    def test_no_redis_by_default(self):
        """No Redis operations when client not provided."""
        mock_provider = MockEmbeddingProvider()
        cached = CachedEmbeddingProvider(mock_provider)

        # Should work without Redis
        embedding = cached.embed("test")

        assert isinstance(embedding, np.ndarray)


class TestCachedEmbeddingProviderAsync:
    """Tests for async methods."""

    @pytest.mark.asyncio
    async def test_aembed_uses_cache(self):
        """Async embed uses cache."""
        mock_provider = MockEmbeddingProvider()
        cached = CachedEmbeddingProvider(mock_provider)

        await cached.aembed("test")
        await cached.aembed("test")

        # Underlying embed is called via aembed default implementation
        assert mock_provider.embed_call_count == 1

    @pytest.mark.asyncio
    async def test_aembed_batch_uses_cache(self):
        """Async batch embed uses cache."""
        mock_provider = MockEmbeddingProvider()
        cached = CachedEmbeddingProvider(mock_provider)

        # Pre-populate cache
        await cached.aembed("a")
        await cached.aembed("b")

        # Reset counter
        mock_provider.embed_call_count = 0

        # Batch with mix
        await cached.aembed_batch(["a", "c", "b"])

        # Only "c" should require computation
        assert mock_provider.embed_call_count == 1


class TestCachedEmbeddingProviderClearCache:
    """Tests for cache clearing methods."""

    def test_clear_cache(self):
        """clear_cache empties in-memory cache."""
        mock_provider = MockEmbeddingProvider()
        cached = CachedEmbeddingProvider(mock_provider)

        cached.embed("a")
        cached.embed("b")
        assert cached.cache_size() == 2

        cached.clear_cache()

        assert cached.cache_size() == 0

    def test_clear_cache_requires_recompute(self):
        """After clear_cache, texts must be recomputed."""
        mock_provider = MockEmbeddingProvider()
        cached = CachedEmbeddingProvider(mock_provider)

        cached.embed("test")
        cached.clear_cache()
        cached.embed("test")

        assert mock_provider.embed_call_count == 2

    def test_clear_all_with_redis(self):
        """clear_all clears both memory and Redis."""
        mock_redis = MagicMock()
        mock_redis.get.return_value = None
        mock_redis.scan.return_value = (0, [b"key1", b"key2"])

        mock_provider = MockEmbeddingProvider()
        cached = CachedEmbeddingProvider(
            mock_provider,
            redis_client=mock_redis,
            redis_prefix="test:",
        )

        cached.embed("a")
        cached.clear_all()

        assert cached.cache_size() == 0
        mock_redis.delete.assert_called()


class FakeRedis:
    """In-process dict-backed Redis stand-in for cross-provider tests."""

    def __init__(self):
        self.store = {}

    def get(self, key):
        return self.store.get(key)

    def set(self, key, value):
        self.store[key] = value

    def setex(self, key, ttl, value):
        self.store[key] = value

    def scan(self, cursor, match=None, count=None):
        keys = list(self.store.keys())
        if match:
            prefix = match.rstrip("*")
            keys = [k for k in keys if isinstance(k, str) and k.startswith(prefix)]
        return (0, keys)

    def delete(self, *keys):
        for k in keys:
            self.store.pop(k, None)


class TestCachedEmbeddingProviderProviderIsolation:
    """Different providers sharing a Redis cache must not collide."""

    def test_redis_cache_isolates_by_dimension(self):
        """A 384-dim provider must not serve a 768-dim provider's cache miss."""
        redis = FakeRedis()

        provider_a = MockEmbeddingProvider(dim=384)
        provider_b = MockEmbeddingProvider(dim=768)

        cached_a = CachedEmbeddingProvider(
            provider_a, redis_client=redis, redis_prefix="emb:"
        )
        cached_b = CachedEmbeddingProvider(
            provider_b, redis_client=redis, redis_prefix="emb:"
        )

        emb_a = cached_a.embed("hello")
        emb_b = cached_b.embed("hello")

        assert emb_a.shape == (384,)
        assert emb_b.shape == (768,)
        assert provider_b.embed_call_count == 1

    def test_redis_cache_isolates_by_model_name(self):
        """Same dim, different model_name must produce independent cache entries."""
        redis = FakeRedis()

        class NamedProvider(MockEmbeddingProvider):
            def __init__(self, model_name, dim=384):
                super().__init__(dim=dim)
                self.model_name = model_name

            def embed(self, text):
                self.embed_call_count += 1
                np.random.seed((hash(self.model_name) ^ hash(text)) % (2**32))
                return np.random.rand(self._dimension).astype(np.float32)

        provider_a = NamedProvider("model-a")
        provider_b = NamedProvider("model-b")

        cached_a = CachedEmbeddingProvider(
            provider_a, redis_client=redis, redis_prefix="emb:"
        )
        cached_b = CachedEmbeddingProvider(
            provider_b, redis_client=redis, redis_prefix="emb:"
        )

        emb_a = cached_a.embed("hello")
        emb_b = cached_b.embed("hello")

        assert not np.array_equal(emb_a, emb_b)
        assert provider_b.embed_call_count == 1


class TestCachedEmbeddingProviderConcurrency:
    """Concurrent embed calls must not crash the LRU."""

    def test_concurrent_embed_smoke(self):
        import concurrent.futures

        provider = MockEmbeddingProvider()
        cached = CachedEmbeddingProvider(provider, max_size=10)

        def worker(i):
            for j in range(50):
                cached.embed(f"text_{(i * 50 + j) % 20}")

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
            futures = [ex.submit(worker, i) for i in range(8)]
            for f in futures:
                f.result()

        assert cached.cache_size() <= 10


class TestCachedEmbeddingProviderEdgeCases:
    """Edge case tests."""

    def test_empty_string(self):
        """Handles empty string."""
        mock_provider = MockEmbeddingProvider()
        cached = CachedEmbeddingProvider(mock_provider)

        embedding = cached.embed("")

        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (384,)

    def test_unicode_text(self):
        """Handles unicode text correctly."""
        mock_provider = MockEmbeddingProvider()
        cached = CachedEmbeddingProvider(mock_provider)

        cached.embed("Hello 世界 🌍")
        cached.embed("Hello 世界 🌍")

        assert mock_provider.embed_call_count == 1

    def test_very_long_text(self):
        """Handles very long text."""
        mock_provider = MockEmbeddingProvider()
        cached = CachedEmbeddingProvider(mock_provider)

        long_text = "a" * 100000
        cached.embed(long_text)
        cached.embed(long_text)

        assert mock_provider.embed_call_count == 1
