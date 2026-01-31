"""Tests for error handling in SemanticCache."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from llmgatekeeper.backends.base import SearchResult
from llmgatekeeper.cache import SemanticCache
from llmgatekeeper.exceptions import BackendError, EmbeddingError


@pytest.fixture
def mock_backend():
    """Create a mock backend."""
    backend = MagicMock()
    backend.search_similar.return_value = []
    backend.count.return_value = 0
    return backend


@pytest.fixture
def mock_embedding_provider():
    """Create a mock embedding provider."""
    provider = MagicMock()
    provider.dimension = 384
    provider.embed.return_value = np.random.rand(384).astype(np.float32)
    provider.embed_batch.return_value = [np.random.rand(384).astype(np.float32)]
    return provider


@pytest.fixture
def mock_redis():
    """Create a mock Redis client."""
    return MagicMock()


class TestEmbeddingErrorHandling:
    """Tests for embedding error handling."""

    def test_embedding_error_on_set(
        self, mock_redis, mock_backend, mock_embedding_provider
    ):
        """TC-7.2.2: Embedding failure raises EmbeddingError on set."""
        mock_embedding_provider.embed.side_effect = RuntimeError("Model failed to load")

        cache = SemanticCache(
            mock_redis,
            backend=mock_backend,
            embedding_provider=mock_embedding_provider,
        )

        with pytest.raises(EmbeddingError) as exc_info:
            cache.set("query", "response")

        assert "Failed to generate embedding" in str(exc_info.value)
        assert exc_info.value.original_error is not None
        assert isinstance(exc_info.value.original_error, RuntimeError)

    def test_embedding_error_on_get(
        self, mock_redis, mock_backend, mock_embedding_provider
    ):
        """Embedding failure raises EmbeddingError on get."""
        mock_embedding_provider.embed.side_effect = ValueError("Invalid input")

        cache = SemanticCache(
            mock_redis,
            backend=mock_backend,
            embedding_provider=mock_embedding_provider,
        )

        with pytest.raises(EmbeddingError) as exc_info:
            cache.get("query")

        assert exc_info.value.original_error is not None


class TestBackendErrorHandling:
    """Tests for backend error handling."""

    def test_backend_error_on_set(
        self, mock_redis, mock_backend, mock_embedding_provider
    ):
        """TC-7.2.1: Backend failure raises BackendError on set."""
        mock_backend.store_vector.side_effect = ConnectionRefusedError("Connection refused")

        cache = SemanticCache(
            mock_redis,
            backend=mock_backend,
            embedding_provider=mock_embedding_provider,
        )

        with pytest.raises(BackendError) as exc_info:
            cache.set("query", "response")

        assert "Failed to store cache entry" in str(exc_info.value)
        assert exc_info.value.original_error is not None

    def test_backend_error_on_get_search(
        self, mock_redis, mock_backend, mock_embedding_provider
    ):
        """Backend search failure raises BackendError on get."""
        # First let embedding succeed
        mock_embedding_provider.embed.return_value = np.random.rand(384).astype(np.float32)

        cache = SemanticCache(
            mock_redis,
            backend=mock_backend,
            embedding_provider=mock_embedding_provider,
        )

        # Need to mock the retriever's find_similar since that's what gets called
        cache._retriever.find_similar = MagicMock(
            side_effect=TimeoutError("Operation timed out")
        )

        with pytest.raises(BackendError) as exc_info:
            cache.get("query")

        assert "Failed to search cache" in str(exc_info.value)


class TestErrorHierarchy:
    """Tests for catching errors with the hierarchy."""

    def test_catch_embedding_error_as_cache_error(
        self, mock_redis, mock_backend, mock_embedding_provider
    ):
        """EmbeddingError can be caught as CacheError."""
        from llmgatekeeper.exceptions import CacheError

        mock_embedding_provider.embed.side_effect = RuntimeError("Model error")

        cache = SemanticCache(
            mock_redis,
            backend=mock_backend,
            embedding_provider=mock_embedding_provider,
        )

        with pytest.raises(CacheError):
            cache.set("query", "response")

    def test_catch_backend_error_as_cache_error(
        self, mock_redis, mock_backend, mock_embedding_provider
    ):
        """BackendError can be caught as CacheError."""
        from llmgatekeeper.exceptions import CacheError

        mock_backend.store_vector.side_effect = RuntimeError("Backend error")

        cache = SemanticCache(
            mock_redis,
            backend=mock_backend,
            embedding_provider=mock_embedding_provider,
        )

        with pytest.raises(CacheError):
            cache.set("query", "response")


class TestGracefulDegradation:
    """Tests for graceful degradation scenarios."""

    def test_operations_succeed_without_errors(
        self, mock_redis, mock_backend, mock_embedding_provider
    ):
        """Normal operations work without raising errors."""
        mock_backend.search_similar.return_value = [
            SearchResult(
                key="key",
                similarity=0.95,
                metadata={"query": "q", "response": "r"},
            )
        ]

        cache = SemanticCache(
            mock_redis,
            backend=mock_backend,
            embedding_provider=mock_embedding_provider,
        )

        # These should not raise
        cache.set("query", "response")
        result = cache.get("query")
        assert result == "r"
