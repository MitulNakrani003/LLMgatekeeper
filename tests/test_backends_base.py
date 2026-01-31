"""Tests for the CacheBackend abstract base class."""

import numpy as np
import pytest

from llmgatekeeper.backends.base import CacheBackend, CacheEntry, SearchResult


class TestCacheBackendAbstract:
    """Tests for CacheBackend ABC."""

    def test_cache_backend_is_abstract(self):
        """TC-2.1.1: CacheBackend cannot be instantiated directly."""
        with pytest.raises(TypeError):
            CacheBackend()

    def test_cache_backend_has_required_methods(self):
        """TC-2.1.2: All abstract methods are defined."""
        assert hasattr(CacheBackend, "store_vector")
        assert hasattr(CacheBackend, "search_similar")
        assert hasattr(CacheBackend, "delete")
        assert hasattr(CacheBackend, "get_by_key")

    def test_cache_backend_methods_are_abstract(self):
        """Verify all required methods are abstract."""
        abstract_methods = CacheBackend.__abstractmethods__
        assert "store_vector" in abstract_methods
        assert "search_similar" in abstract_methods
        assert "delete" in abstract_methods
        assert "get_by_key" in abstract_methods


class TestCacheEntry:
    """Tests for CacheEntry Pydantic model."""

    def test_cache_entry_creation(self):
        """CacheEntry can be created with valid data."""
        vector = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        entry = CacheEntry(
            key="test_key",
            vector=vector,
            metadata={"response": "test response"},
        )
        assert entry.key == "test_key"
        assert np.array_equal(entry.vector, vector)
        assert entry.metadata["response"] == "test response"

    def test_cache_entry_converts_vector_dtype(self):
        """CacheEntry converts vector to float32."""
        vector = np.array([0.1, 0.2, 0.3], dtype=np.float64)
        entry = CacheEntry(
            key="test_key",
            vector=vector,
            metadata={},
        )
        assert entry.vector.dtype == np.float32

    def test_cache_entry_converts_list_to_array(self):
        """CacheEntry converts list to numpy array."""
        entry = CacheEntry(
            key="test_key",
            vector=[0.1, 0.2, 0.3],
            metadata={},
        )
        assert isinstance(entry.vector, np.ndarray)
        assert entry.vector.dtype == np.float32


class TestSearchResult:
    """Tests for SearchResult Pydantic model."""

    def test_search_result_creation(self):
        """SearchResult can be created with valid data."""
        result = SearchResult(
            key="test_key",
            similarity=0.95,
            metadata={"response": "test"},
        )
        assert result.key == "test_key"
        assert result.similarity == 0.95
        assert result.metadata["response"] == "test"
        assert result.vector is None

    def test_search_result_with_vector(self):
        """SearchResult can include optional vector."""
        vector = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        result = SearchResult(
            key="test_key",
            similarity=0.95,
            metadata={},
            vector=vector,
        )
        assert np.array_equal(result.vector, vector)
