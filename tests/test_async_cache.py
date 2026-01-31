"""Tests for the async semantic cache functionality."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from llmgatekeeper.backends.base import AsyncCacheBackend, SearchResult
from llmgatekeeper.cache import AsyncSemanticCache, CacheResult
from llmgatekeeper.similarity.confidence import ConfidenceLevel


@pytest.fixture
def mock_async_backend():
    """Create a mock async backend."""
    backend = AsyncMock(spec=AsyncCacheBackend)
    backend.store_vector = AsyncMock()
    backend.search_similar = AsyncMock(return_value=[])
    backend.delete = AsyncMock(return_value=True)
    backend.get_by_key = AsyncMock(return_value=None)
    backend.clear = AsyncMock(return_value=0)
    backend.count = AsyncMock(return_value=0)
    return backend


@pytest.fixture
def mock_async_embedding_provider():
    """Create a mock async embedding provider."""
    provider = MagicMock()
    provider.dimension = 384

    # Sync methods
    def mock_embed(text):
        hash_val = hash(text) % 10000
        np.random.seed(hash_val)
        return np.random.rand(384).astype(np.float32)

    def mock_embed_batch(texts):
        return [mock_embed(text) for text in texts]

    provider.embed.side_effect = mock_embed
    provider.embed_batch.side_effect = mock_embed_batch

    # Async methods (default implementations wrap sync)
    async def mock_aembed(text):
        return mock_embed(text)

    async def mock_aembed_batch(texts):
        return mock_embed_batch(texts)

    provider.aembed = AsyncMock(side_effect=mock_aembed)
    provider.aembed_batch = AsyncMock(side_effect=mock_aembed_batch)
    return provider


@pytest.fixture
def async_cache(mock_async_backend, mock_async_embedding_provider):
    """Create an AsyncSemanticCache with mocks."""
    return AsyncSemanticCache(
        backend=mock_async_backend,
        embedding_provider=mock_async_embedding_provider,
    )


class TestAsyncSemanticCacheInit:
    """Tests for AsyncSemanticCache initialization."""

    def test_init_with_backend(self, mock_async_backend, mock_async_embedding_provider):
        """Can initialize with backend and embedding provider."""
        cache = AsyncSemanticCache(
            backend=mock_async_backend,
            embedding_provider=mock_async_embedding_provider,
        )
        assert cache.threshold == 0.85
        assert cache.namespace == "default"

    def test_init_with_custom_threshold(
        self, mock_async_backend, mock_async_embedding_provider
    ):
        """Can initialize with custom threshold."""
        cache = AsyncSemanticCache(
            backend=mock_async_backend,
            embedding_provider=mock_async_embedding_provider,
            threshold=0.95,
        )
        assert cache.threshold == 0.95

    def test_init_invalid_threshold_raises(
        self, mock_async_backend, mock_async_embedding_provider
    ):
        """Invalid threshold raises ValueError."""
        with pytest.raises(ValueError, match="threshold must be in"):
            AsyncSemanticCache(
                backend=mock_async_backend,
                embedding_provider=mock_async_embedding_provider,
                threshold=1.5,
            )

    def test_init_with_namespace(
        self, mock_async_backend, mock_async_embedding_provider
    ):
        """Can initialize with custom namespace."""
        cache = AsyncSemanticCache(
            backend=mock_async_backend,
            embedding_provider=mock_async_embedding_provider,
            namespace="my_tenant",
        )
        assert cache.namespace == "my_tenant"


class TestAsyncSemanticCacheSetGet:
    """Tests for async set and get operations."""

    @pytest.mark.asyncio
    async def test_async_set(self, async_cache, mock_async_backend):
        """TC-7.1.1: Async set stores query-response pair."""
        key = await async_cache.set("What is Python?", "A programming language.")

        assert key.startswith("llmgk:default:")
        mock_async_backend.store_vector.assert_called_once()

    @pytest.mark.asyncio
    async def test_async_get_hit(
        self, async_cache, mock_async_backend, mock_async_embedding_provider
    ):
        """TC-7.1.1: Async get returns cached response."""
        # Set up backend to return a match
        mock_async_backend.search_similar.return_value = [
            SearchResult(
                key="key1",
                similarity=0.95,
                metadata={"query": "What is Python?", "response": "A programming language."},
            )
        ]

        result = await async_cache.get("What is Python?")

        assert result == "A programming language."
        mock_async_embedding_provider.aembed.assert_called_once()

    @pytest.mark.asyncio
    async def test_async_get_miss(self, async_cache, mock_async_backend):
        """Async get returns None for no match."""
        mock_async_backend.search_similar.return_value = []

        result = await async_cache.get("Unknown query")

        assert result is None

    @pytest.mark.asyncio
    async def test_async_get_with_metadata(self, async_cache, mock_async_backend):
        """Async get with include_metadata returns CacheResult."""
        mock_async_backend.search_similar.return_value = [
            SearchResult(
                key="key1",
                similarity=0.95,
                metadata={
                    "query": "What is Python?",
                    "response": "A programming language.",
                    "model": "gpt-4",
                },
            )
        ]

        result = await async_cache.get("What is Python?", include_metadata=True)

        assert isinstance(result, CacheResult)
        assert result.response == "A programming language."
        assert result.similarity == 0.95
        assert result.metadata["model"] == "gpt-4"

    @pytest.mark.asyncio
    async def test_async_concurrent_gets(self, async_cache, mock_async_backend):
        """TC-7.1.2: Async concurrent operations work."""
        mock_async_backend.search_similar.return_value = [
            SearchResult(
                key="key1",
                similarity=0.95,
                metadata={"query": "q1", "response": "r1"},
            )
        ]

        # Concurrent gets
        results = await asyncio.gather(
            async_cache.get("q1"),
            async_cache.get("q2"),
            async_cache.get("q3"),
        )

        assert len(results) == 3
        # All should return the mocked response
        assert all(r == "r1" for r in results)


class TestAsyncSemanticCacheDelete:
    """Tests for async delete operations."""

    @pytest.mark.asyncio
    async def test_async_delete(self, async_cache, mock_async_backend):
        """Async delete removes entry."""
        mock_async_backend.delete.return_value = True

        result = await async_cache.delete("What is Python?")

        assert result is True
        mock_async_backend.delete.assert_called_once()

    @pytest.mark.asyncio
    async def test_async_delete_by_key(self, async_cache, mock_async_backend):
        """Async delete_by_key removes entry."""
        mock_async_backend.delete.return_value = True

        result = await async_cache.delete_by_key("llmgk:default:abc123")

        assert result is True


class TestAsyncSemanticCacheClear:
    """Tests for async clear and count operations."""

    @pytest.mark.asyncio
    async def test_async_clear(self, async_cache, mock_async_backend):
        """Async clear removes all entries."""
        mock_async_backend.clear.return_value = 5

        result = await async_cache.clear()

        assert result == 5
        mock_async_backend.clear.assert_called_once()

    @pytest.mark.asyncio
    async def test_async_count(self, async_cache, mock_async_backend):
        """Async count returns entry count."""
        mock_async_backend.count.return_value = 10

        result = await async_cache.count()

        assert result == 10


class TestAsyncSemanticCacheGetSimilar:
    """Tests for async get_similar operation."""

    @pytest.mark.asyncio
    async def test_async_get_similar(self, async_cache, mock_async_backend):
        """Async get_similar returns multiple results."""
        mock_async_backend.search_similar.return_value = [
            SearchResult(
                key="key1",
                similarity=0.95,
                metadata={"query": "q1", "response": "r1"},
            ),
            SearchResult(
                key="key2",
                similarity=0.90,
                metadata={"query": "q2", "response": "r2"},
            ),
        ]

        results = await async_cache.get_similar("query", top_k=3)

        assert len(results) == 2
        assert results[0].response == "r1"
        assert results[1].response == "r2"


class TestAsyncSemanticCacheWarm:
    """Tests for async warm operation."""

    @pytest.mark.asyncio
    async def test_async_warm(
        self, mock_async_backend, mock_async_embedding_provider
    ):
        """Async warm loads pairs."""
        cache = AsyncSemanticCache(
            backend=mock_async_backend,
            embedding_provider=mock_async_embedding_provider,
        )

        pairs = [
            ("What is Python?", "A programming language"),
            ("What is Java?", "Another programming language"),
        ]

        result = await cache.warm(pairs)

        assert result == 2
        assert mock_async_backend.store_vector.call_count == 2
        mock_async_embedding_provider.aembed_batch.assert_called_once()

    @pytest.mark.asyncio
    async def test_async_warm_empty(self, async_cache):
        """Async warm with empty list returns 0."""
        result = await async_cache.warm([])
        assert result == 0

    @pytest.mark.asyncio
    async def test_async_warm_progress_callback(
        self, mock_async_backend, mock_async_embedding_provider
    ):
        """Async warm calls progress callback."""
        cache = AsyncSemanticCache(
            backend=mock_async_backend,
            embedding_provider=mock_async_embedding_provider,
        )

        progress_calls = []
        pairs = [("q1", "r1"), ("q2", "r2")]

        await cache.warm(
            pairs,
            on_progress=lambda done, total: progress_calls.append((done, total)),
        )

        assert (2, 2) in progress_calls


class TestAsyncSemanticCacheAnalytics:
    """Tests for async cache analytics."""

    @pytest.mark.asyncio
    async def test_analytics_disabled_by_default(self, async_cache):
        """Analytics disabled by default."""
        assert async_cache.analytics_enabled is False
        assert async_cache.stats() is None

    @pytest.mark.asyncio
    async def test_analytics_enabled(
        self, mock_async_backend, mock_async_embedding_provider
    ):
        """Analytics can be enabled."""
        cache = AsyncSemanticCache(
            backend=mock_async_backend,
            embedding_provider=mock_async_embedding_provider,
            enable_analytics=True,
        )

        assert cache.analytics_enabled is True
        assert cache.stats() is not None

    @pytest.mark.asyncio
    async def test_analytics_tracks_hits(
        self, mock_async_backend, mock_async_embedding_provider
    ):
        """Analytics tracks cache hits."""
        cache = AsyncSemanticCache(
            backend=mock_async_backend,
            embedding_provider=mock_async_embedding_provider,
            enable_analytics=True,
        )

        mock_async_backend.search_similar.return_value = [
            SearchResult(
                key="key1",
                similarity=0.95,
                metadata={"query": "q", "response": "r"},
            )
        ]

        await cache.get("query")

        stats = cache.stats()
        assert stats.hits == 1
        assert stats.misses == 0

    @pytest.mark.asyncio
    async def test_analytics_tracks_misses(
        self, mock_async_backend, mock_async_embedding_provider
    ):
        """Analytics tracks cache misses."""
        cache = AsyncSemanticCache(
            backend=mock_async_backend,
            embedding_provider=mock_async_embedding_provider,
            enable_analytics=True,
        )

        mock_async_backend.search_similar.return_value = []

        await cache.get("unknown query")

        stats = cache.stats()
        assert stats.hits == 0
        assert stats.misses == 1

    @pytest.mark.asyncio
    async def test_reset_stats(
        self, mock_async_backend, mock_async_embedding_provider
    ):
        """Can reset analytics stats."""
        cache = AsyncSemanticCache(
            backend=mock_async_backend,
            embedding_provider=mock_async_embedding_provider,
            enable_analytics=True,
        )

        mock_async_backend.search_similar.return_value = [
            SearchResult(
                key="key1",
                similarity=0.95,
                metadata={"query": "q", "response": "r"},
            )
        ]

        await cache.get("query")
        assert cache.stats().hits == 1

        cache.reset_stats()
        assert cache.stats().hits == 0


class TestAsyncSemanticCacheRepr:
    """Tests for AsyncSemanticCache representation."""

    def test_repr(self, async_cache):
        """repr() returns informative string."""
        repr_str = repr(async_cache)
        assert "AsyncSemanticCache" in repr_str
        assert "default" in repr_str
        assert "0.85" in repr_str
