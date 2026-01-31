"""Tests for the RediSearchBackend."""

from unittest.mock import MagicMock, PropertyMock

import numpy as np
import pytest

from llmgatekeeper.backends.redis_search import RediSearchBackend


class MockFTInfo:
    """Mock for FT.INFO response."""

    pass


class MockFTSearch:
    """Mock for FT.SEARCH results."""

    def __init__(self, docs):
        self.docs = docs
        self.total = len(docs)


class MockDoc:
    """Mock search result document."""

    def __init__(self, key, metadata, score, vector=None):
        self.key = key
        self.metadata = metadata
        self.score = score
        self.vector = vector


@pytest.fixture
def mock_redis_with_redisearch():
    """Create a mock Redis client with RediSearch module available."""
    mock_redis = MagicMock()

    # Mock module_list to return search module
    mock_redis.module_list.return_value = [{"name": "search", "ver": 20800}]

    # Mock FT commands
    mock_ft = MagicMock()
    mock_ft.info.return_value = MockFTInfo()
    mock_ft.create_index.return_value = True
    mock_redis.ft.return_value = mock_ft

    # Mock JSON commands
    mock_json = MagicMock()
    mock_redis.json.return_value = mock_json

    # Mock set operations
    mock_redis.sadd.return_value = 1
    mock_redis.srem.return_value = 1
    mock_redis.smembers.return_value = set()
    mock_redis.scard.return_value = 0
    mock_redis.delete.return_value = 1

    return mock_redis


@pytest.fixture
def mock_redis_without_redisearch():
    """Create a mock Redis client without RediSearch module."""
    mock_redis = MagicMock()
    mock_redis.module_list.return_value = []
    return mock_redis


class TestRediSearchBackendInit:
    """Tests for RediSearchBackend initialization."""

    def test_creates_index_on_init(self, mock_redis_with_redisearch):
        """TC-2.3.1: Creates vector index if not exists."""
        # Make info() raise exception to trigger index creation
        mock_ft = mock_redis_with_redisearch.ft.return_value
        mock_ft.info.side_effect = Exception("Index not found")

        backend = RediSearchBackend(mock_redis_with_redisearch)

        # Verify create_index was called
        mock_ft.create_index.assert_called_once()
        assert backend is not None

    def test_uses_existing_index(self, mock_redis_with_redisearch):
        """Does not recreate index if it exists."""
        backend = RediSearchBackend(mock_redis_with_redisearch)

        mock_ft = mock_redis_with_redisearch.ft.return_value
        # info() succeeded, so create_index should not be called
        mock_ft.create_index.assert_not_called()

    def test_missing_redisearch_raises_error(self, mock_redis_without_redisearch):
        """TC-2.3.3: Handles missing RediSearch module gracefully."""
        with pytest.raises(RuntimeError, match="RediSearch module not available"):
            RediSearchBackend(mock_redis_without_redisearch)

    def test_custom_vector_dimension(self, mock_redis_with_redisearch):
        """Accepts custom vector dimension."""
        mock_ft = mock_redis_with_redisearch.ft.return_value
        mock_ft.info.side_effect = Exception("Index not found")

        backend = RediSearchBackend(
            mock_redis_with_redisearch, vector_dimension=1536
        )
        assert backend._vector_dimension == 1536

    def test_custom_namespace(self, mock_redis_with_redisearch):
        """Uses custom namespace for index and keys."""
        backend = RediSearchBackend(
            mock_redis_with_redisearch, namespace="custom_ns"
        )
        assert backend._namespace == "custom_ns"
        assert backend._index_name == "custom_ns_idx"


class TestRediSearchBackendStoreAndGet:
    """Tests for store_vector and get_by_key."""

    def test_store_vector(self, mock_redis_with_redisearch):
        """Store creates JSON document with vector."""
        backend = RediSearchBackend(mock_redis_with_redisearch)
        vector = np.array([0.1, 0.2, 0.3], dtype=np.float32)

        backend.store_vector("key1", vector, {"response": "test"})

        mock_json = mock_redis_with_redisearch.json.return_value
        mock_json.set.assert_called_once()

        # Verify the call arguments
        call_args = mock_json.set.call_args
        redis_key = call_args[0][0]
        path = call_args[0][1]
        doc = call_args[0][2]

        assert redis_key == "llmgk:entry:key1"
        assert path == "$"
        assert doc["key"] == "key1"
        assert doc["vector"] == vector.tolist()
        assert '"response": "test"' in doc["metadata"]

    def test_get_by_key(self, mock_redis_with_redisearch):
        """Get retrieves stored document."""
        backend = RediSearchBackend(mock_redis_with_redisearch)

        # Mock the JSON get response
        mock_json = mock_redis_with_redisearch.json.return_value
        mock_json.get.return_value = {
            "key": "key1",
            "vector": [0.1, 0.2, 0.3],
            "metadata": '{"response": "test"}',
        }

        result = backend.get_by_key("key1")

        assert result is not None
        assert result.key == "key1"
        assert result.metadata["response"] == "test"
        assert np.allclose(result.vector, [0.1, 0.2, 0.3])

    def test_get_nonexistent_key_returns_none(self, mock_redis_with_redisearch):
        """Get returns None for keys that don't exist."""
        backend = RediSearchBackend(mock_redis_with_redisearch)

        mock_json = mock_redis_with_redisearch.json.return_value
        mock_json.get.return_value = None

        result = backend.get_by_key("nonexistent")
        assert result is None

    def test_store_with_ttl(self, mock_redis_with_redisearch):
        """Store sets TTL when specified."""
        backend = RediSearchBackend(mock_redis_with_redisearch)
        vector = np.array([0.1, 0.2, 0.3], dtype=np.float32)

        backend.store_vector("key1", vector, {}, ttl=60)

        mock_redis_with_redisearch.expire.assert_called_once_with(
            "llmgk:entry:key1", 60
        )


class TestRediSearchBackendSearch:
    """Tests for search_similar with KNN."""

    def test_knn_search_returns_results(self, mock_redis_with_redisearch):
        """TC-2.3.2: KNN search returns top-k results."""
        backend = RediSearchBackend(mock_redis_with_redisearch)

        # Mock search results
        mock_ft = mock_redis_with_redisearch.ft.return_value
        mock_docs = [
            MockDoc("key1", '{"r": "a"}', 0.05),  # distance 0.05 -> similarity 0.95
            MockDoc("key2", '{"r": "b"}', 0.10),  # distance 0.10 -> similarity 0.90
        ]
        mock_ft.search.return_value = MockFTSearch(mock_docs)

        # Mock get_by_key for vector retrieval
        mock_json = mock_redis_with_redisearch.json.return_value
        mock_json.get.return_value = {
            "key": "key1",
            "vector": [1.0, 0.0, 0.0],
            "metadata": '{"r": "a"}',
        }

        query_vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        results = backend.search_similar(query_vec, threshold=0.85, top_k=5)

        assert len(results) == 2
        assert results[0].key == "key1"
        assert results[0].similarity == pytest.approx(0.95)
        assert results[1].key == "key2"
        assert results[1].similarity == pytest.approx(0.90)

    def test_knn_search_respects_threshold(self, mock_redis_with_redisearch):
        """Results below threshold are filtered out."""
        backend = RediSearchBackend(mock_redis_with_redisearch)

        mock_ft = mock_redis_with_redisearch.ft.return_value
        mock_docs = [
            MockDoc("key1", '{"r": "a"}', 0.05),  # similarity 0.95
            MockDoc("key2", '{"r": "b"}', 0.20),  # similarity 0.80 - below threshold
        ]
        mock_ft.search.return_value = MockFTSearch(mock_docs)

        mock_json = mock_redis_with_redisearch.json.return_value
        mock_json.get.return_value = {
            "key": "key1",
            "vector": [1.0, 0.0, 0.0],
            "metadata": '{"r": "a"}',
        }

        query_vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        results = backend.search_similar(query_vec, threshold=0.90, top_k=5)

        assert len(results) == 1
        assert results[0].key == "key1"

    def test_knn_search_respects_top_k(self, mock_redis_with_redisearch):
        """Returns at most top_k results."""
        backend = RediSearchBackend(mock_redis_with_redisearch)

        mock_ft = mock_redis_with_redisearch.ft.return_value
        mock_docs = [
            MockDoc("key1", '{"r": "a"}', 0.01),
            MockDoc("key2", '{"r": "b"}', 0.02),
            MockDoc("key3", '{"r": "c"}', 0.03),
            MockDoc("key4", '{"r": "d"}', 0.04),
            MockDoc("key5", '{"r": "e"}', 0.05),
        ]
        mock_ft.search.return_value = MockFTSearch(mock_docs)

        mock_json = mock_redis_with_redisearch.json.return_value
        mock_json.get.return_value = {
            "key": "key1",
            "vector": [1.0, 0.0, 0.0],
            "metadata": '{"r": "a"}',
        }

        query_vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        results = backend.search_similar(query_vec, threshold=0.5, top_k=3)

        assert len(results) == 3

    def test_knn_search_empty_results(self, mock_redis_with_redisearch):
        """Returns empty list when no matches."""
        backend = RediSearchBackend(mock_redis_with_redisearch)

        mock_ft = mock_redis_with_redisearch.ft.return_value
        mock_ft.search.side_effect = Exception("No results")

        query_vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        results = backend.search_similar(query_vec, threshold=0.9, top_k=5)

        assert len(results) == 0


class TestRediSearchBackendDelete:
    """Tests for delete."""

    def test_delete_removes_entry(self, mock_redis_with_redisearch):
        """Delete removes JSON document and key from set."""
        backend = RediSearchBackend(mock_redis_with_redisearch)

        result = backend.delete("key1")

        mock_redis_with_redisearch.delete.assert_called_with("llmgk:entry:key1")
        mock_redis_with_redisearch.srem.assert_called_with("llmgk:keys", "key1")
        assert result is True

    def test_delete_returns_false_when_not_exists(self, mock_redis_with_redisearch):
        """Delete returns False when key didn't exist."""
        mock_redis_with_redisearch.delete.return_value = 0

        backend = RediSearchBackend(mock_redis_with_redisearch)
        result = backend.delete("nonexistent")

        assert result is False


class TestRediSearchBackendClearAndCount:
    """Tests for clear and count methods."""

    def test_count_returns_set_cardinality(self, mock_redis_with_redisearch):
        """Count returns the number of keys in tracking set."""
        mock_redis_with_redisearch.scard.return_value = 42

        backend = RediSearchBackend(mock_redis_with_redisearch)
        assert backend.count() == 42

    def test_clear_removes_all_entries(self, mock_redis_with_redisearch):
        """Clear removes all entries and returns count."""
        mock_redis_with_redisearch.smembers.return_value = {b"k1", b"k2", b"k3"}
        mock_redis_with_redisearch.delete.return_value = 1

        backend = RediSearchBackend(mock_redis_with_redisearch)
        deleted = backend.clear()

        assert deleted == 3
        # Verify delete was called for each key plus the keys set
        assert mock_redis_with_redisearch.delete.call_count == 4


class TestRediSearchBackendDropIndex:
    """Tests for drop_index method."""

    def test_drop_index(self, mock_redis_with_redisearch):
        """Drop index removes the search index."""
        backend = RediSearchBackend(mock_redis_with_redisearch)
        backend.drop_index()

        mock_ft = mock_redis_with_redisearch.ft.return_value
        mock_ft.dropindex.assert_called_once_with(delete_documents=False)

    def test_drop_index_handles_missing(self, mock_redis_with_redisearch):
        """Drop index handles case when index doesn't exist."""
        mock_ft = mock_redis_with_redisearch.ft.return_value
        mock_ft.dropindex.side_effect = Exception("Index not found")

        backend = RediSearchBackend(mock_redis_with_redisearch)
        # Should not raise
        backend.drop_index()
