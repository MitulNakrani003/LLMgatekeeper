"""Tests for the SemanticCache class."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from llmgatekeeper.backends.base import CacheEntry, SearchResult
from llmgatekeeper.cache import CacheResult, SemanticCache
from llmgatekeeper.similarity.confidence import ConfidenceLevel


@pytest.fixture
def mock_backend():
    """Create a mock cache backend."""
    backend = MagicMock()
    backend.search_similar.return_value = []
    backend.get_by_key.return_value = None
    backend.store_vector.return_value = None
    backend.delete.return_value = True
    backend.clear.return_value = 0
    backend.count.return_value = 0
    return backend


@pytest.fixture
def mock_embedding_provider():
    """Create a mock embedding provider."""
    provider = MagicMock()
    provider.dimension = 384

    # Return a consistent embedding based on input hash for reproducibility
    def mock_embed(text):
        # Use text hash to generate deterministic but unique embeddings
        hash_val = hash(text) % 10000
        np.random.seed(hash_val)
        return np.random.rand(384).astype(np.float32)

    def mock_embed_batch(texts):
        # Return embeddings for batch of texts
        return [mock_embed(text) for text in texts]

    provider.embed.side_effect = mock_embed
    provider.embed_batch.side_effect = mock_embed_batch
    return provider


@pytest.fixture
def mock_redis():
    """Create a mock Redis client."""
    redis = MagicMock()
    redis.module_list.return_value = []  # No RediSearch
    return redis


@pytest.fixture
def cache_with_mocks(mock_backend, mock_embedding_provider, mock_redis):
    """Create a SemanticCache with mocked dependencies."""
    return SemanticCache(
        redis_client=mock_redis,
        backend=mock_backend,
        embedding_provider=mock_embedding_provider,
    )


class TestCacheResult:
    """Tests for the CacheResult dataclass."""

    def test_create_result(self):
        """Can create a cache result."""
        result = CacheResult(
            response="Test response",
            similarity=0.95,
            confidence=ConfidenceLevel.HIGH,
            key="test_key",
            metadata={"model": "gpt-4"},
        )
        assert result.response == "Test response"
        assert result.similarity == 0.95
        assert result.confidence == ConfidenceLevel.HIGH
        assert result.key == "test_key"
        assert result.metadata == {"model": "gpt-4"}

    def test_str_returns_response(self):
        """String representation returns the response."""
        result = CacheResult(
            response="Test response",
            similarity=0.95,
            confidence=ConfidenceLevel.HIGH,
            key="key",
            metadata={},
        )
        assert str(result) == "Test response"


class TestSemanticCacheInit:
    """Tests for SemanticCache initialization."""

    def test_init_with_redis_client(self, mock_redis, mock_backend):
        """Can initialize with just a Redis client."""
        with patch(
            "llmgatekeeper.cache.create_redis_backend", return_value=mock_backend
        ):
            cache = SemanticCache(mock_redis)
            assert cache is not None
            assert cache.threshold == 0.85
            assert cache.namespace == "default"

    def test_init_with_custom_threshold(self, mock_redis, mock_backend):
        """Custom threshold is accepted."""
        with patch(
            "llmgatekeeper.cache.create_redis_backend", return_value=mock_backend
        ):
            cache = SemanticCache(mock_redis, threshold=0.95)
            assert cache.threshold == 0.95

    def test_init_with_custom_backend(
        self, mock_redis, mock_backend, mock_embedding_provider
    ):
        """Custom backend is used when provided."""
        cache = SemanticCache(
            mock_redis,
            backend=mock_backend,
            embedding_provider=mock_embedding_provider,
        )
        assert cache._backend is mock_backend

    def test_init_with_custom_embedding_provider(
        self, mock_redis, mock_backend, mock_embedding_provider
    ):
        """Custom embedding provider is used when provided."""
        cache = SemanticCache(
            mock_redis,
            backend=mock_backend,
            embedding_provider=mock_embedding_provider,
        )
        assert cache._embedding_provider is mock_embedding_provider

    def test_init_with_custom_namespace(self, mock_redis, mock_backend):
        """Custom namespace is accepted."""
        with patch(
            "llmgatekeeper.cache.create_redis_backend", return_value=mock_backend
        ):
            cache = SemanticCache(mock_redis, namespace="tenant_123")
            assert cache.namespace == "tenant_123"

    def test_invalid_threshold_negative(self, mock_redis, mock_backend):
        """Negative threshold raises ValueError."""
        with pytest.raises(ValueError, match="threshold must be in"):
            with patch(
                "llmgatekeeper.cache.create_redis_backend", return_value=mock_backend
            ):
                SemanticCache(mock_redis, threshold=-0.1)

    def test_invalid_threshold_above_one(self, mock_redis, mock_backend):
        """Threshold > 1 raises ValueError."""
        with pytest.raises(ValueError, match="threshold must be in"):
            with patch(
                "llmgatekeeper.cache.create_redis_backend", return_value=mock_backend
            ):
                SemanticCache(mock_redis, threshold=1.5)

    def test_one_line_init(self, mock_redis, mock_backend):
        """TC-5.1.5: One-line initialization works."""
        with patch(
            "llmgatekeeper.cache.create_redis_backend", return_value=mock_backend
        ):
            cache = SemanticCache(mock_redis)
            assert cache is not None

    def test_repr(self, cache_with_mocks):
        """Repr shows configuration."""
        repr_str = repr(cache_with_mocks)
        assert "namespace=" in repr_str
        assert "threshold=" in repr_str


class TestSemanticCacheThreshold:
    """Tests for threshold property."""

    def test_get_threshold(self, cache_with_mocks):
        """Can get threshold."""
        assert cache_with_mocks.threshold == 0.85

    def test_set_threshold(self, cache_with_mocks):
        """Can set threshold."""
        cache_with_mocks.threshold = 0.95
        assert cache_with_mocks.threshold == 0.95

    def test_set_invalid_threshold(self, cache_with_mocks):
        """Setting invalid threshold raises error."""
        with pytest.raises(ValueError):
            cache_with_mocks.threshold = 1.5


class TestSemanticCacheSet:
    """Tests for the set method."""

    def test_set_stores_entry(self, cache_with_mocks, mock_backend):
        """Set stores entry in backend."""
        cache_with_mocks.set("What is Python?", "A programming language.")
        mock_backend.store_vector.assert_called_once()

    def test_set_returns_key(self, cache_with_mocks):
        """Set returns the cache key."""
        key = cache_with_mocks.set("What is Python?", "A programming language.")
        assert key.startswith("llmgk:")
        assert "default" in key

    def test_set_with_metadata(self, cache_with_mocks, mock_backend):
        """Set stores metadata."""
        cache_with_mocks.set(
            "What is Python?",
            "A programming language.",
            metadata={"model": "gpt-4", "tokens": 50},
        )
        call_args = mock_backend.store_vector.call_args
        metadata = call_args.kwargs["metadata"]
        assert metadata["model"] == "gpt-4"
        assert metadata["tokens"] == 50
        assert metadata["query"] == "What is Python?"
        assert metadata["response"] == "A programming language."

    def test_set_with_ttl(self, cache_with_mocks, mock_backend):
        """Set passes TTL to backend."""
        cache_with_mocks.set("query", "response", ttl=3600)
        call_args = mock_backend.store_vector.call_args
        assert call_args.kwargs["ttl"] == 3600

    def test_set_generates_embedding(
        self, cache_with_mocks, mock_embedding_provider
    ):
        """Set generates embedding for query."""
        cache_with_mocks.set("What is Python?", "A programming language.")
        mock_embedding_provider.embed.assert_called_with("What is Python?")


class TestSemanticCacheGet:
    """Tests for the get method."""

    def test_get_returns_none_for_new(self, cache_with_mocks, mock_backend):
        """TC-5.1.2: Get returns None for unseen query."""
        mock_backend.search_similar.return_value = []
        result = cache_with_mocks.get("Never seen this query before")
        assert result is None

    def test_get_exact_match(self, cache_with_mocks, mock_backend):
        """TC-5.1.1: Get returns response for exact match."""
        mock_backend.search_similar.return_value = [
            SearchResult(
                key="llmgk:default:abc123",
                similarity=0.99,
                metadata={"query": "What is Python?", "response": "A programming language."},
            )
        ]
        result = cache_with_mocks.get("What is Python?")
        assert result == "A programming language."

    def test_get_semantic_match(self, cache_with_mocks, mock_backend):
        """TC-5.1.3: Get returns response for semantically similar query."""
        mock_backend.search_similar.return_value = [
            SearchResult(
                key="llmgk:default:abc123",
                similarity=0.92,
                metadata={"query": "What's the weather?", "response": "It's sunny."},
            )
        ]
        result = cache_with_mocks.get("Tell me the weather")
        assert result == "It's sunny."

    def test_get_below_threshold_returns_none(self, cache_with_mocks, mock_backend):
        """Get returns None when similarity below threshold."""
        # The backend respects threshold, so it returns empty when no match
        mock_backend.search_similar.return_value = []
        result = cache_with_mocks.get("Some query", threshold=0.99)
        assert result is None

    def test_get_with_include_metadata(self, cache_with_mocks, mock_backend):
        """Get returns CacheResult when include_metadata=True."""
        mock_backend.search_similar.return_value = [
            SearchResult(
                key="llmgk:default:abc123",
                similarity=0.95,
                metadata={
                    "query": "What is Python?",
                    "response": "A programming language.",
                    "model": "gpt-4",
                },
            )
        ]
        result = cache_with_mocks.get("What is Python?", include_metadata=True)
        assert isinstance(result, CacheResult)
        assert result.response == "A programming language."
        assert result.similarity == 0.95
        assert result.key == "llmgk:default:abc123"
        assert result.metadata == {"model": "gpt-4"}

    def test_get_metadata_excludes_query_response(
        self, cache_with_mocks, mock_backend
    ):
        """CacheResult metadata excludes query and response fields."""
        mock_backend.search_similar.return_value = [
            SearchResult(
                key="key",
                similarity=0.95,
                metadata={
                    "query": "q",
                    "response": "r",
                    "custom": "value",
                },
            )
        ]
        result = cache_with_mocks.get("q", include_metadata=True)
        assert "query" not in result.metadata
        assert "response" not in result.metadata
        assert result.metadata == {"custom": "value"}

    def test_get_with_threshold_override(self, cache_with_mocks, mock_backend):
        """Can override threshold per query."""
        mock_backend.search_similar.return_value = []
        cache_with_mocks.get("query", threshold=0.99)
        # Check that retriever was called with overridden threshold
        call_args = mock_backend.search_similar.call_args
        assert call_args.kwargs["threshold"] == 0.99

    def test_get_generates_embedding(
        self, cache_with_mocks, mock_embedding_provider
    ):
        """Get generates embedding for query."""
        cache_with_mocks.get("What is Python?")
        mock_embedding_provider.embed.assert_called_with("What is Python?")


class TestSemanticCacheGetSimilar:
    """Tests for the get_similar method."""

    def test_returns_multiple_results(self, cache_with_mocks, mock_backend):
        """Returns multiple similar results."""
        mock_backend.search_similar.return_value = [
            SearchResult(
                key="key1",
                similarity=0.98,
                metadata={"query": "q1", "response": "r1"},
            ),
            SearchResult(
                key="key2",
                similarity=0.92,
                metadata={"query": "q2", "response": "r2"},
            ),
            SearchResult(
                key="key3",
                similarity=0.88,
                metadata={"query": "q3", "response": "r3"},
            ),
        ]
        results = cache_with_mocks.get_similar("query", top_k=3)
        assert len(results) == 3
        assert all(isinstance(r, CacheResult) for r in results)

    def test_results_have_confidence(self, cache_with_mocks, mock_backend):
        """Each result has confidence level."""
        mock_backend.search_similar.return_value = [
            SearchResult(
                key="key1",
                similarity=0.98,
                metadata={"query": "q1", "response": "r1"},
            ),
        ]
        results = cache_with_mocks.get_similar("query", top_k=1)
        assert results[0].confidence in list(ConfidenceLevel)

    def test_respects_top_k(self, cache_with_mocks, mock_backend):
        """Respects top_k parameter."""
        cache_with_mocks.get_similar("query", top_k=10)
        call_args = mock_backend.search_similar.call_args
        assert call_args.kwargs["top_k"] == 10

    def test_empty_results(self, cache_with_mocks, mock_backend):
        """Returns empty list when no matches."""
        mock_backend.search_similar.return_value = []
        results = cache_with_mocks.get_similar("query")
        assert results == []


class TestSemanticCacheDelete:
    """Tests for delete methods."""

    def test_delete_by_query(self, cache_with_mocks, mock_backend):
        """Delete by query calls backend."""
        mock_backend.delete.return_value = True
        result = cache_with_mocks.delete("What is Python?")
        assert result is True
        mock_backend.delete.assert_called_once()

    def test_delete_by_key(self, cache_with_mocks, mock_backend):
        """Delete by key calls backend."""
        mock_backend.delete.return_value = True
        result = cache_with_mocks.delete_by_key("llmgk:default:abc123")
        assert result is True
        mock_backend.delete.assert_called_with("llmgk:default:abc123")

    def test_delete_returns_false_for_missing(self, cache_with_mocks, mock_backend):
        """Delete returns False when key not found."""
        mock_backend.delete.return_value = False
        result = cache_with_mocks.delete("nonexistent")
        assert result is False


class TestSemanticCacheExists:
    """Tests for the exists method."""

    def test_exists_returns_true(self, cache_with_mocks, mock_backend):
        """Exists returns True when entry exists."""
        mock_backend.get_by_key.return_value = CacheEntry(
            key="key",
            vector=np.zeros(384, dtype=np.float32),
            metadata={"response": "r"},
        )
        assert cache_with_mocks.exists("query") is True

    def test_exists_returns_false(self, cache_with_mocks, mock_backend):
        """Exists returns False when entry doesn't exist."""
        mock_backend.get_by_key.return_value = None
        assert cache_with_mocks.exists("query") is False


class TestSemanticCacheClearCount:
    """Tests for clear and count methods."""

    def test_clear(self, cache_with_mocks, mock_backend):
        """Clear calls backend clear."""
        mock_backend.clear.return_value = 5
        result = cache_with_mocks.clear()
        assert result == 5
        mock_backend.clear.assert_called_once()

    def test_count(self, cache_with_mocks, mock_backend):
        """Count calls backend count."""
        mock_backend.count.return_value = 10
        result = cache_with_mocks.count()
        assert result == 10
        mock_backend.count.assert_called_once()


class TestSemanticCacheKeyGeneration:
    """Tests for key generation."""

    def test_same_query_same_key(self, cache_with_mocks):
        """Same query generates same key."""
        key1 = cache_with_mocks._generate_key("What is Python?")
        key2 = cache_with_mocks._generate_key("What is Python?")
        assert key1 == key2

    def test_different_query_different_key(self, cache_with_mocks):
        """Different queries generate different keys."""
        key1 = cache_with_mocks._generate_key("What is Python?")
        key2 = cache_with_mocks._generate_key("What is Java?")
        assert key1 != key2

    def test_key_includes_namespace(self, cache_with_mocks):
        """Key includes namespace."""
        key = cache_with_mocks._generate_key("query")
        assert "default" in key

    def test_key_format(self, cache_with_mocks):
        """Key has expected format."""
        key = cache_with_mocks._generate_key("query")
        assert key.startswith("llmgk:")
        parts = key.split(":")
        assert len(parts) == 3


class TestSemanticCacheIntegration:
    """Integration-style tests for SemanticCache."""

    def test_set_and_get_workflow(self, cache_with_mocks, mock_backend):
        """Complete set and get workflow."""
        # Set up mock to return the stored entry on search
        stored_metadata = {}

        def capture_store(*args, **kwargs):
            stored_metadata.update(kwargs["metadata"])

        mock_backend.store_vector.side_effect = capture_store

        # Store entry
        cache_with_mocks.set(
            "What is Python?",
            "Python is a programming language.",
            metadata={"model": "gpt-4"},
        )

        # Configure mock to return this entry
        mock_backend.search_similar.return_value = [
            SearchResult(
                key="llmgk:default:abc123",
                similarity=0.99,
                metadata=stored_metadata,
            )
        ]

        # Retrieve entry
        result = cache_with_mocks.get("What is Python?", include_metadata=True)
        assert result.response == "Python is a programming language."
        assert result.metadata["model"] == "gpt-4"

    def test_dissimilar_no_match(self, cache_with_mocks, mock_backend):
        """TC-5.1.4: Dissimilar query doesn't match."""
        # Cache returns empty for dissimilar queries
        mock_backend.search_similar.return_value = []

        cache_with_mocks.set("What's the weather?", "It's sunny.")
        result = cache_with_mocks.get("How do I cook pasta?")
        assert result is None

    def test_multiple_entries_best_match(self, cache_with_mocks, mock_backend):
        """Returns best match from multiple entries."""
        mock_backend.search_similar.return_value = [
            SearchResult(
                key="key1",
                similarity=0.92,
                metadata={"query": "best match", "response": "best response"},
            )
        ]

        result = cache_with_mocks.get("query")
        assert result == "best response"

    def test_confidence_levels_assigned(self, cache_with_mocks, mock_backend):
        """Confidence levels are correctly assigned."""
        # High similarity = HIGH confidence
        mock_backend.search_similar.return_value = [
            SearchResult(
                key="key",
                similarity=0.98,
                metadata={"query": "q", "response": "r"},
            )
        ]
        result = cache_with_mocks.get("query", include_metadata=True)
        assert result.confidence == ConfidenceLevel.HIGH

        # Medium similarity = MEDIUM confidence
        mock_backend.search_similar.return_value = [
            SearchResult(
                key="key",
                similarity=0.88,
                metadata={"query": "q", "response": "r"},
            )
        ]
        result = cache_with_mocks.get("query", include_metadata=True)
        assert result.confidence == ConfidenceLevel.MEDIUM


class TestSemanticCacheWithDefaultProvider:
    """Tests that verify behavior with default SentenceTransformer provider."""

    def test_creates_default_provider(self, mock_redis, mock_backend):
        """Creates default SentenceTransformer provider when not specified."""
        with patch(
            "llmgatekeeper.cache.create_redis_backend", return_value=mock_backend
        ):
            with patch(
                "llmgatekeeper.cache.SentenceTransformerProvider"
            ) as mock_st:
                mock_st.return_value.dimension = 384
                cache = SemanticCache(mock_redis)
                mock_st.assert_called_once()

    def test_model_name_auto_detected(self, mock_redis, mock_backend):
        """Model name auto-detected for default provider."""
        with patch(
            "llmgatekeeper.cache.create_redis_backend", return_value=mock_backend
        ):
            with patch(
                "llmgatekeeper.cache.SentenceTransformerProvider"
            ) as mock_st:
                mock_st.return_value.dimension = 384
                cache = SemanticCache(mock_redis)
                assert cache._model_name == "all-MiniLM-L6-v2"


class TestSemanticCacheTTL:
    """Tests for TTL (time-to-live) functionality."""

    def test_default_ttl_none(self, cache_with_mocks):
        """Default TTL is None (no expiration)."""
        assert cache_with_mocks.default_ttl is None

    def test_init_with_default_ttl(
        self, mock_redis, mock_backend, mock_embedding_provider
    ):
        """Can initialize with default_ttl."""
        cache = SemanticCache(
            mock_redis,
            backend=mock_backend,
            embedding_provider=mock_embedding_provider,
            default_ttl=3600,
        )
        assert cache.default_ttl == 3600

    def test_set_default_ttl_property(self, cache_with_mocks):
        """Can set default_ttl via property."""
        cache_with_mocks.default_ttl = 7200
        assert cache_with_mocks.default_ttl == 7200

    def test_set_default_ttl_none(self, cache_with_mocks):
        """Can set default_ttl to None."""
        cache_with_mocks.default_ttl = 3600
        cache_with_mocks.default_ttl = None
        assert cache_with_mocks.default_ttl is None

    def test_set_default_ttl_invalid_negative(self, cache_with_mocks):
        """Setting negative default_ttl raises ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            cache_with_mocks.default_ttl = -1

    def test_set_uses_explicit_ttl(self, cache_with_mocks, mock_backend):
        """TC-5.2.1: Set with explicit TTL uses that TTL."""
        cache_with_mocks.set("query", "response", ttl=3600)
        call_args = mock_backend.store_vector.call_args
        assert call_args.kwargs["ttl"] == 3600

    def test_set_uses_default_ttl_when_not_specified(
        self, mock_redis, mock_backend, mock_embedding_provider
    ):
        """TC-5.2.2: Set uses default_ttl when TTL not specified."""
        cache = SemanticCache(
            mock_redis,
            backend=mock_backend,
            embedding_provider=mock_embedding_provider,
            default_ttl=60,
        )
        cache.set("query", "response")
        call_args = mock_backend.store_vector.call_args
        assert call_args.kwargs["ttl"] == 60

    def test_set_no_ttl_when_default_none(self, cache_with_mocks, mock_backend):
        """TC-5.2.3: No TTL when default_ttl is None."""
        cache_with_mocks.set("query", "response")
        call_args = mock_backend.store_vector.call_args
        assert call_args.kwargs["ttl"] is None

    def test_explicit_ttl_overrides_default(
        self, mock_redis, mock_backend, mock_embedding_provider
    ):
        """Explicit TTL overrides default_ttl."""
        cache = SemanticCache(
            mock_redis,
            backend=mock_backend,
            embedding_provider=mock_embedding_provider,
            default_ttl=60,
        )
        cache.set("query", "response", ttl=3600)
        call_args = mock_backend.store_vector.call_args
        assert call_args.kwargs["ttl"] == 3600

    def test_ttl_zero_disables_expiration(
        self, mock_redis, mock_backend, mock_embedding_provider
    ):
        """TTL=0 explicitly disables expiration even with default_ttl."""
        cache = SemanticCache(
            mock_redis,
            backend=mock_backend,
            embedding_provider=mock_embedding_provider,
            default_ttl=60,
        )
        cache.set("query", "response", ttl=0)
        call_args = mock_backend.store_vector.call_args
        assert call_args.kwargs["ttl"] is None

    def test_ttl_with_metadata(
        self, mock_redis, mock_backend, mock_embedding_provider
    ):
        """TTL works correctly with metadata."""
        cache = SemanticCache(
            mock_redis,
            backend=mock_backend,
            embedding_provider=mock_embedding_provider,
            default_ttl=120,
        )
        cache.set("query", "response", metadata={"model": "gpt-4"})
        call_args = mock_backend.store_vector.call_args
        assert call_args.kwargs["ttl"] == 120
        assert call_args.kwargs["metadata"]["model"] == "gpt-4"


class TestSemanticCacheMetadata:
    """Tests for metadata support (Task 5.3)."""

    def test_store_and_retrieve_metadata(
        self, mock_redis, mock_backend, mock_embedding_provider
    ):
        """TC-5.3.1: Store and retrieve metadata."""
        cache = SemanticCache(
            mock_redis,
            backend=mock_backend,
            embedding_provider=mock_embedding_provider,
        )
        # Store with metadata
        cache.set(
            "query",
            "response",
            metadata={"model": "gpt-4", "tokens": 150},
        )

        # Verify metadata was stored
        call_args = mock_backend.store_vector.call_args
        stored_metadata = call_args.kwargs["metadata"]
        assert stored_metadata["model"] == "gpt-4"
        assert stored_metadata["tokens"] == 150

        # Mock retrieval with the stored metadata
        mock_backend.search_similar.return_value = [
            SearchResult(
                key="key",
                similarity=0.95,
                metadata={
                    "query": "query",
                    "response": "response",
                    "model": "gpt-4",
                    "tokens": 150,
                },
            )
        ]

        # Retrieve with include_metadata=True
        result = cache.get("query", include_metadata=True)
        assert result.response == "response"
        assert result.metadata["model"] == "gpt-4"
        assert result.metadata["tokens"] == 150

    def test_backward_compatible_get(
        self, mock_redis, mock_backend, mock_embedding_provider
    ):
        """TC-5.3.2: Default get still returns just response string."""
        cache = SemanticCache(
            mock_redis,
            backend=mock_backend,
            embedding_provider=mock_embedding_provider,
        )

        # Store with metadata
        cache.set("query", "response", metadata={"foo": "bar"})

        # Mock retrieval
        mock_backend.search_similar.return_value = [
            SearchResult(
                key="key",
                similarity=0.95,
                metadata={
                    "query": "query",
                    "response": "response",
                    "foo": "bar",
                },
            )
        ]

        # Default get returns just the string
        result = cache.get("query")
        assert result == "response"  # String, not CacheResult
        assert isinstance(result, str)
        assert not isinstance(result, CacheResult)

    def test_metadata_none_by_default(self, cache_with_mocks, mock_backend):
        """Set works without metadata."""
        cache_with_mocks.set("query", "response")
        call_args = mock_backend.store_vector.call_args
        metadata = call_args.kwargs["metadata"]
        # Only query and response should be present
        assert metadata["query"] == "query"
        assert metadata["response"] == "response"
        assert len(metadata) == 2

    def test_metadata_preserved_through_round_trip(
        self, mock_redis, mock_backend, mock_embedding_provider
    ):
        """Metadata survives store and retrieve cycle."""
        cache = SemanticCache(
            mock_redis,
            backend=mock_backend,
            embedding_provider=mock_embedding_provider,
        )

        # Complex metadata
        complex_metadata = {
            "model": "gpt-4",
            "tokens": 150,
            "temperature": 0.7,
            "tags": ["python", "programming"],
        }
        cache.set("query", "response", metadata=complex_metadata)

        # Verify all metadata stored
        call_args = mock_backend.store_vector.call_args
        stored = call_args.kwargs["metadata"]
        assert stored["model"] == "gpt-4"
        assert stored["tokens"] == 150
        assert stored["temperature"] == 0.7
        assert stored["tags"] == ["python", "programming"]

    def test_get_similar_includes_metadata(
        self, mock_redis, mock_backend, mock_embedding_provider
    ):
        """get_similar returns CacheResults with metadata."""
        cache = SemanticCache(
            mock_redis,
            backend=mock_backend,
            embedding_provider=mock_embedding_provider,
        )

        # Mock multiple results with metadata
        mock_backend.search_similar.return_value = [
            SearchResult(
                key="key1",
                similarity=0.95,
                metadata={
                    "query": "q1",
                    "response": "r1",
                    "source": "docs",
                },
            ),
            SearchResult(
                key="key2",
                similarity=0.90,
                metadata={
                    "query": "q2",
                    "response": "r2",
                    "source": "faq",
                },
            ),
        ]

        results = cache.get_similar("query", top_k=2)
        assert len(results) == 2
        assert results[0].metadata["source"] == "docs"
        assert results[1].metadata["source"] == "faq"


class TestSemanticCacheNamespaceIsolation:
    """Tests for namespace/tenant isolation (Task 6.1)."""

    def test_namespace_isolation(
        self, mock_redis, mock_embedding_provider
    ):
        """TC-6.1.1: Different namespaces are isolated."""
        # Create separate backends for each namespace
        backend_a = MagicMock()
        backend_a.search_similar.return_value = []
        backend_b = MagicMock()
        backend_b.search_similar.return_value = []

        cache_a = SemanticCache(
            mock_redis,
            namespace="tenant_a",
            backend=backend_a,
            embedding_provider=mock_embedding_provider,
        )
        cache_b = SemanticCache(
            mock_redis,
            namespace="tenant_b",
            backend=backend_b,
            embedding_provider=mock_embedding_provider,
        )

        # Store in both caches
        cache_a.set("query", "response A")
        cache_b.set("query", "response B")

        # Verify keys are different due to namespace
        key_a = cache_a._generate_key("query")
        key_b = cache_b._generate_key("query")
        assert key_a != key_b
        assert "tenant_a" in key_a
        assert "tenant_b" in key_b

        # Mock retrieval for each cache
        backend_a.search_similar.return_value = [
            SearchResult(
                key=key_a,
                similarity=0.95,
                metadata={"query": "query", "response": "response A"},
            )
        ]
        backend_b.search_similar.return_value = [
            SearchResult(
                key=key_b,
                similarity=0.95,
                metadata={"query": "query", "response": "response B"},
            )
        ]

        # Each cache should return its own response
        assert cache_a.get("query") == "response A"
        assert cache_b.get("query") == "response B"

    def test_namespace_clear_isolation(
        self, mock_redis, mock_embedding_provider
    ):
        """TC-6.1.2: Clearing one namespace doesn't affect another."""
        backend_a = MagicMock()
        backend_a.clear.return_value = 1
        backend_b = MagicMock()
        backend_b.search_similar.return_value = []

        cache_a = SemanticCache(
            mock_redis,
            namespace="tenant_a",
            backend=backend_a,
            embedding_provider=mock_embedding_provider,
        )
        cache_b = SemanticCache(
            mock_redis,
            namespace="tenant_b",
            backend=backend_b,
            embedding_provider=mock_embedding_provider,
        )

        # Store in both
        cache_a.set("query", "response A")
        cache_b.set("query", "response B")

        # Clear only cache_a
        cache_a.clear()

        # Verify clear was called only on backend_a
        backend_a.clear.assert_called_once()
        backend_b.clear.assert_not_called()

        # Mock that cache_b still has data
        backend_b.search_similar.return_value = [
            SearchResult(
                key="key",
                similarity=0.95,
                metadata={"query": "query", "response": "response B"},
            )
        ]

        # cache_b should still work
        assert cache_b.get("query") == "response B"

    def test_default_namespace(self, cache_with_mocks, mock_backend):
        """TC-6.1.3: Default namespace works."""
        # Default namespace should be "default"
        assert cache_with_mocks.namespace == "default"

        # Store and verify key uses default namespace
        cache_with_mocks.set("query", "response")
        key = cache_with_mocks._generate_key("query")
        assert "default" in key

        # Mock retrieval
        mock_backend.search_similar.return_value = [
            SearchResult(
                key=key,
                similarity=0.95,
                metadata={"query": "query", "response": "response"},
            )
        ]

        # Should work correctly
        assert cache_with_mocks.get("query") == "response"

    def test_namespace_in_key_format(self, mock_redis, mock_backend, mock_embedding_provider):
        """Keys include namespace prefix."""
        cache = SemanticCache(
            mock_redis,
            namespace="my_tenant",
            backend=mock_backend,
            embedding_provider=mock_embedding_provider,
        )

        key = cache._generate_key("test query")
        assert key.startswith("llmgk:my_tenant:")

    def test_different_namespaces_same_query_different_keys(
        self, mock_redis, mock_embedding_provider
    ):
        """Same query in different namespaces produces different keys."""
        backend1 = MagicMock()
        backend2 = MagicMock()

        cache1 = SemanticCache(
            mock_redis,
            namespace="ns1",
            backend=backend1,
            embedding_provider=mock_embedding_provider,
        )
        cache2 = SemanticCache(
            mock_redis,
            namespace="ns2",
            backend=backend2,
            embedding_provider=mock_embedding_provider,
        )

        key1 = cache1._generate_key("same query")
        key2 = cache2._generate_key("same query")

        assert key1 != key2
        assert "ns1" in key1
        assert "ns2" in key2


class TestSemanticCacheWarm:
    """Tests for cache warming functionality (Task 6.2)."""

    def test_warm_loads_pairs(self, cache_with_mocks, mock_backend):
        """TC-6.2.1: Warm loads all pairs."""
        pairs = [
            ("What is Python?", "A programming language"),
            ("What is Java?", "Another programming language"),
            ("What is Rust?", "A systems programming language"),
        ]

        result = cache_with_mocks.warm(pairs)

        assert result == 3
        assert mock_backend.store_vector.call_count == 3

    def test_warm_uses_batch_embedding(
        self, mock_redis, mock_backend, mock_embedding_provider
    ):
        """TC-6.2.2: Warm uses batch embedding."""
        # Set up embed_batch to return proper embeddings
        mock_embedding_provider.embed_batch.return_value = [
            np.random.rand(384).astype(np.float32) for _ in range(3)
        ]

        cache = SemanticCache(
            mock_redis,
            backend=mock_backend,
            embedding_provider=mock_embedding_provider,
        )

        pairs = [("q1", "r1"), ("q2", "r2"), ("q3", "r3")]
        cache.warm(pairs)

        # Should use batch embedding, not individual embed calls
        mock_embedding_provider.embed_batch.assert_called_once()
        mock_embedding_provider.embed.assert_not_called()

    def test_warm_progress_callback(self, cache_with_mocks):
        """TC-6.2.3: Progress callback invoked."""
        progress_calls = []

        pairs = [("q1", "r1"), ("q2", "r2")]
        cache_with_mocks.warm(
            pairs,
            on_progress=lambda done, total: progress_calls.append((done, total)),
        )

        # Should have been called at least once with final values
        assert (2, 2) in progress_calls

    def test_warm_empty_pairs(self, cache_with_mocks, mock_backend):
        """Warm with empty list returns 0."""
        result = cache_with_mocks.warm([])
        assert result == 0
        mock_backend.store_vector.assert_not_called()

    def test_warm_with_metadata(self, cache_with_mocks, mock_backend):
        """Warm applies metadata to all entries."""
        pairs = [("q1", "r1"), ("q2", "r2")]
        cache_with_mocks.warm(pairs, metadata={"source": "training"})

        # Check both calls have the metadata
        for call in mock_backend.store_vector.call_args_list:
            assert call.kwargs["metadata"]["source"] == "training"

    def test_warm_with_ttl(self, cache_with_mocks, mock_backend):
        """Warm applies TTL to all entries."""
        pairs = [("q1", "r1"), ("q2", "r2")]
        cache_with_mocks.warm(pairs, ttl=3600)

        for call in mock_backend.store_vector.call_args_list:
            assert call.kwargs["ttl"] == 3600

    def test_warm_uses_default_ttl(
        self, mock_redis, mock_backend, mock_embedding_provider
    ):
        """Warm uses default_ttl when TTL not specified."""
        mock_embedding_provider.embed_batch.return_value = [
            np.random.rand(384).astype(np.float32) for _ in range(2)
        ]

        cache = SemanticCache(
            mock_redis,
            backend=mock_backend,
            embedding_provider=mock_embedding_provider,
            default_ttl=1800,
        )

        pairs = [("q1", "r1"), ("q2", "r2")]
        cache.warm(pairs)

        for call in mock_backend.store_vector.call_args_list:
            assert call.kwargs["ttl"] == 1800

    def test_warm_batch_size(self, mock_redis, mock_backend, mock_embedding_provider):
        """Warm respects batch_size parameter."""
        # Create enough embeddings for multiple batches
        mock_embedding_provider.embed_batch.return_value = [
            np.random.rand(384).astype(np.float32) for _ in range(2)
        ]

        cache = SemanticCache(
            mock_redis,
            backend=mock_backend,
            embedding_provider=mock_embedding_provider,
        )

        # 5 pairs with batch_size=2 should result in 3 batch calls
        pairs = [(f"q{i}", f"r{i}") for i in range(5)]

        # Need to reset and set up different return values for each batch
        call_count = [0]

        def batch_side_effect(queries):
            call_count[0] += 1
            return [np.random.rand(384).astype(np.float32) for _ in queries]

        mock_embedding_provider.embed_batch.side_effect = batch_side_effect

        cache.warm(pairs, batch_size=2)

        # 5 items / batch_size 2 = 3 batches (2, 2, 1)
        assert call_count[0] == 3

    def test_warm_progress_callback_multiple_batches(
        self, mock_redis, mock_backend, mock_embedding_provider
    ):
        """Progress callback called after each batch."""
        progress_calls = []

        def batch_side_effect(queries):
            return [np.random.rand(384).astype(np.float32) for _ in queries]

        mock_embedding_provider.embed_batch.side_effect = batch_side_effect

        cache = SemanticCache(
            mock_redis,
            backend=mock_backend,
            embedding_provider=mock_embedding_provider,
        )

        pairs = [(f"q{i}", f"r{i}") for i in range(5)]
        cache.warm(
            pairs,
            batch_size=2,
            on_progress=lambda done, total: progress_calls.append((done, total)),
        )

        # Should be called after each batch: (2, 5), (4, 5), (5, 5)
        assert len(progress_calls) == 3
        assert progress_calls[-1] == (5, 5)

    def test_warm_stores_correct_data(self, cache_with_mocks, mock_backend):
        """Warm stores query and response in metadata."""
        pairs = [("What is Python?", "A programming language")]
        cache_with_mocks.warm(pairs)

        call_args = mock_backend.store_vector.call_args
        metadata = call_args.kwargs["metadata"]
        assert metadata["query"] == "What is Python?"
        assert metadata["response"] == "A programming language"

    def test_warm_returns_count(self, cache_with_mocks):
        """Warm returns number of stored entries."""
        pairs = [(f"q{i}", f"r{i}") for i in range(10)]
        result = cache_with_mocks.warm(pairs)
        assert result == 10


class TestSemanticCacheAnalytics:
    """Tests for analytics integration (Task 6.3)."""

    def test_analytics_disabled_by_default(self, cache_with_mocks):
        """Analytics is disabled by default."""
        assert cache_with_mocks.analytics_enabled is False
        assert cache_with_mocks.stats() is None

    def test_analytics_enabled(
        self, mock_redis, mock_backend, mock_embedding_provider
    ):
        """Analytics can be enabled."""
        cache = SemanticCache(
            mock_redis,
            backend=mock_backend,
            embedding_provider=mock_embedding_provider,
            enable_analytics=True,
        )
        assert cache.analytics_enabled is True
        assert cache.stats() is not None

    def test_hit_rate_tracked(
        self, mock_redis, mock_backend, mock_embedding_provider
    ):
        """TC-6.3.1: Hit rate calculated correctly."""
        cache = SemanticCache(
            mock_redis,
            backend=mock_backend,
            embedding_provider=mock_embedding_provider,
            enable_analytics=True,
        )

        # Set up backend to return results for "query"
        mock_backend.search_similar.return_value = [
            SearchResult(
                key="key",
                similarity=0.95,
                metadata={"query": "query", "response": "response"},
            )
        ]

        cache.get("query")  # Hit
        cache.get("query")  # Hit

        # Set up backend to return no results for "other"
        mock_backend.search_similar.return_value = []
        cache.get("other")  # Miss

        stats = cache.stats()
        assert stats.hit_rate == pytest.approx(2 / 3)

    def test_latency_tracked(
        self, mock_redis, mock_backend, mock_embedding_provider
    ):
        """TC-6.3.2: Latency percentiles tracked."""
        cache = SemanticCache(
            mock_redis,
            backend=mock_backend,
            embedding_provider=mock_embedding_provider,
            enable_analytics=True,
        )

        mock_backend.search_similar.return_value = [
            SearchResult(
                key="key",
                similarity=0.95,
                metadata={"query": "query", "response": "response"},
            )
        ]

        # Make multiple queries
        for _ in range(100):
            cache.get("query")

        stats = cache.stats()
        assert stats.p50_latency_ms > 0
        assert stats.p95_latency_ms >= stats.p50_latency_ms

    def test_top_queries_tracked(
        self, mock_redis, mock_backend, mock_embedding_provider
    ):
        """TC-6.3.4: Most frequent queries tracked."""
        cache = SemanticCache(
            mock_redis,
            backend=mock_backend,
            embedding_provider=mock_embedding_provider,
            enable_analytics=True,
        )

        mock_backend.search_similar.return_value = [
            SearchResult(
                key="key",
                similarity=0.95,
                metadata={"query": "query", "response": "response"},
            )
        ]

        # Popular query
        for _ in range(10):
            cache.get("popular query")

        # Rare query
        cache.get("rare query")

        stats = cache.stats()
        assert stats.top_queries[0].query == "popular query"
        assert stats.top_queries[0].count == 10

    def test_reset_stats(
        self, mock_redis, mock_backend, mock_embedding_provider
    ):
        """Can reset analytics stats."""
        cache = SemanticCache(
            mock_redis,
            backend=mock_backend,
            embedding_provider=mock_embedding_provider,
            enable_analytics=True,
        )

        mock_backend.search_similar.return_value = [
            SearchResult(
                key="key",
                similarity=0.95,
                metadata={"query": "query", "response": "response"},
            )
        ]

        cache.get("query")
        stats = cache.stats()
        assert stats.hits == 1

        cache.reset_stats()
        stats = cache.stats()
        assert stats.hits == 0
        assert stats.misses == 0

    def test_reset_stats_when_disabled(self, cache_with_mocks):
        """Reset stats does nothing when analytics disabled."""
        # Should not raise
        cache_with_mocks.reset_stats()

    def test_miss_recorded(
        self, mock_redis, mock_backend, mock_embedding_provider
    ):
        """Misses are recorded in analytics."""
        cache = SemanticCache(
            mock_redis,
            backend=mock_backend,
            embedding_provider=mock_embedding_provider,
            enable_analytics=True,
        )

        mock_backend.search_similar.return_value = []

        cache.get("unknown query")

        stats = cache.stats()
        assert stats.misses == 1
        assert stats.hits == 0