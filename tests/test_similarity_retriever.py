"""Tests for the similarity retriever module."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from llmgatekeeper.backends.base import SearchResult
from llmgatekeeper.similarity.confidence import ConfidenceClassifier, ConfidenceLevel
from llmgatekeeper.similarity.retriever import (
    RetrievalResponse,
    RetrievalResult,
    SimilarityRetriever,
)


@pytest.fixture
def mock_backend():
    """Create a mock cache backend."""
    backend = MagicMock()
    return backend


@pytest.fixture
def sample_search_results():
    """Create sample search results for testing."""
    return [
        SearchResult(key="key1", similarity=0.98, metadata={"response": "answer1"}),
        SearchResult(key="key2", similarity=0.90, metadata={"response": "answer2"}),
        SearchResult(key="key3", similarity=0.82, metadata={"response": "answer3"}),
        SearchResult(key="key4", similarity=0.70, metadata={"response": "answer4"}),
        SearchResult(key="key5", similarity=0.50, metadata={"response": "answer5"}),
    ]


@pytest.fixture
def query_vector():
    """Create a sample query vector."""
    return np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)


class TestRetrievalResult:
    """Tests for the RetrievalResult dataclass."""

    def test_create_result(self):
        """Can create a retrieval result."""
        result = RetrievalResult(
            key="test_key",
            similarity=0.95,
            confidence=ConfidenceLevel.HIGH,
            metadata={"response": "test"},
            rank=1,
        )
        assert result.key == "test_key"
        assert result.similarity == 0.95
        assert result.confidence == ConfidenceLevel.HIGH
        assert result.metadata == {"response": "test"}
        assert result.rank == 1

    def test_repr(self):
        """Repr shows key, similarity, confidence, and rank."""
        result = RetrievalResult(
            key="test",
            similarity=0.9567,
            confidence=ConfidenceLevel.MEDIUM,
            metadata={},
            rank=2,
        )
        repr_str = repr(result)
        assert "test" in repr_str
        assert "0.9567" in repr_str
        assert "medium" in repr_str.lower()  # ConfidenceLevel uses lowercase values
        assert "rank=2" in repr_str


class TestRetrievalResponse:
    """Tests for the RetrievalResponse dataclass."""

    def test_empty_response(self):
        """Empty response has correct properties."""
        response = RetrievalResponse(results=[])
        assert len(response) == 0
        assert response.best_match is None
        assert not response.has_high_confidence
        assert response.high_confidence_results == []
        assert response.above_threshold_results == []

    def test_len(self):
        """Length returns number of results."""
        results = [
            RetrievalResult("k1", 0.9, ConfidenceLevel.MEDIUM, {}, 1),
            RetrievalResult("k2", 0.8, ConfidenceLevel.LOW, {}, 2),
        ]
        response = RetrievalResponse(results=results)
        assert len(response) == 2

    def test_iteration(self):
        """Can iterate over results."""
        results = [
            RetrievalResult("k1", 0.9, ConfidenceLevel.MEDIUM, {}, 1),
            RetrievalResult("k2", 0.8, ConfidenceLevel.LOW, {}, 2),
        ]
        response = RetrievalResponse(results=results)
        keys = [r.key for r in response]
        assert keys == ["k1", "k2"]

    def test_indexing(self):
        """Can access results by index."""
        results = [
            RetrievalResult("k1", 0.9, ConfidenceLevel.MEDIUM, {}, 1),
            RetrievalResult("k2", 0.8, ConfidenceLevel.LOW, {}, 2),
        ]
        response = RetrievalResponse(results=results)
        assert response[0].key == "k1"
        assert response[1].key == "k2"

    def test_best_match(self):
        """best_match returns first result."""
        results = [
            RetrievalResult("k1", 0.98, ConfidenceLevel.HIGH, {}, 1),
            RetrievalResult("k2", 0.90, ConfidenceLevel.MEDIUM, {}, 2),
        ]
        response = RetrievalResponse(results=results)
        assert response.best_match.key == "k1"
        assert response.best_match.similarity == 0.98

    def test_has_high_confidence(self):
        """has_high_confidence returns True when HIGH result exists."""
        results_with_high = [
            RetrievalResult("k1", 0.98, ConfidenceLevel.HIGH, {}, 1),
            RetrievalResult("k2", 0.90, ConfidenceLevel.MEDIUM, {}, 2),
        ]
        results_without_high = [
            RetrievalResult("k1", 0.90, ConfidenceLevel.MEDIUM, {}, 1),
            RetrievalResult("k2", 0.80, ConfidenceLevel.LOW, {}, 2),
        ]
        assert RetrievalResponse(results=results_with_high).has_high_confidence is True
        assert (
            RetrievalResponse(results=results_without_high).has_high_confidence is False
        )

    def test_high_confidence_results(self):
        """high_confidence_results filters correctly."""
        results = [
            RetrievalResult("k1", 0.98, ConfidenceLevel.HIGH, {}, 1),
            RetrievalResult("k2", 0.96, ConfidenceLevel.HIGH, {}, 2),
            RetrievalResult("k3", 0.90, ConfidenceLevel.MEDIUM, {}, 3),
        ]
        response = RetrievalResponse(results=results)
        high = response.high_confidence_results
        assert len(high) == 2
        assert all(r.confidence == ConfidenceLevel.HIGH for r in high)

    def test_above_threshold_results(self):
        """above_threshold_results excludes NONE confidence."""
        results = [
            RetrievalResult("k1", 0.98, ConfidenceLevel.HIGH, {}, 1),
            RetrievalResult("k2", 0.80, ConfidenceLevel.LOW, {}, 2),
            RetrievalResult("k3", 0.50, ConfidenceLevel.NONE, {}, 3),
        ]
        response = RetrievalResponse(results=results)
        above = response.above_threshold_results
        assert len(above) == 2
        assert all(r.confidence != ConfidenceLevel.NONE for r in above)

    def test_query_vector_stored(self):
        """Query vector is stored when provided."""
        vec = np.array([1.0, 2.0], dtype=np.float32)
        response = RetrievalResponse(results=[], query_vector=vec)
        assert response.query_vector is not None
        np.testing.assert_array_equal(response.query_vector, vec)

    def test_total_candidates(self):
        """Total candidates is accessible."""
        response = RetrievalResponse(results=[], total_candidates=100)
        assert response.total_candidates == 100


class TestSimilarityRetrieverInit:
    """Tests for SimilarityRetriever initialization."""

    def test_default_init(self, mock_backend):
        """Default initialization works."""
        retriever = SimilarityRetriever(mock_backend)
        assert retriever.top_k == 5
        assert retriever.threshold == 0.0
        assert retriever.confidence_classifier is not None

    def test_custom_top_k(self, mock_backend):
        """Custom top_k is accepted."""
        retriever = SimilarityRetriever(mock_backend, top_k=10)
        assert retriever.top_k == 10

    def test_custom_threshold(self, mock_backend):
        """Custom threshold is accepted."""
        retriever = SimilarityRetriever(mock_backend, threshold=0.8)
        assert retriever.threshold == 0.8

    def test_custom_classifier(self, mock_backend):
        """Custom classifier is accepted."""
        classifier = ConfidenceClassifier(high=0.9, medium=0.8, low=0.7)
        retriever = SimilarityRetriever(
            mock_backend, confidence_classifier=classifier
        )
        assert retriever.confidence_classifier is classifier

    def test_invalid_top_k_zero(self, mock_backend):
        """top_k=0 raises ValueError."""
        with pytest.raises(ValueError, match="top_k must be >= 1"):
            SimilarityRetriever(mock_backend, top_k=0)

    def test_invalid_top_k_negative(self, mock_backend):
        """Negative top_k raises ValueError."""
        with pytest.raises(ValueError, match="top_k must be >= 1"):
            SimilarityRetriever(mock_backend, top_k=-5)

    def test_invalid_threshold_negative(self, mock_backend):
        """Negative threshold raises ValueError."""
        with pytest.raises(ValueError, match="threshold must be in"):
            SimilarityRetriever(mock_backend, threshold=-0.1)

    def test_invalid_threshold_above_one(self, mock_backend):
        """Threshold > 1 raises ValueError."""
        with pytest.raises(ValueError, match="threshold must be in"):
            SimilarityRetriever(mock_backend, threshold=1.5)

    def test_repr(self, mock_backend):
        """Repr shows configuration."""
        retriever = SimilarityRetriever(mock_backend, top_k=3, threshold=0.5)
        repr_str = repr(retriever)
        assert "top_k=3" in repr_str
        assert "threshold=0.5" in repr_str


class TestSimilarityRetrieverProperties:
    """Tests for SimilarityRetriever property setters."""

    def test_set_top_k(self, mock_backend):
        """Can update top_k via setter."""
        retriever = SimilarityRetriever(mock_backend, top_k=5)
        retriever.top_k = 10
        assert retriever.top_k == 10

    def test_set_top_k_invalid(self, mock_backend):
        """Setting invalid top_k raises ValueError."""
        retriever = SimilarityRetriever(mock_backend)
        with pytest.raises(ValueError):
            retriever.top_k = 0

    def test_set_threshold(self, mock_backend):
        """Can update threshold via setter."""
        retriever = SimilarityRetriever(mock_backend, threshold=0.5)
        retriever.threshold = 0.8
        assert retriever.threshold == 0.8

    def test_set_threshold_invalid(self, mock_backend):
        """Setting invalid threshold raises ValueError."""
        retriever = SimilarityRetriever(mock_backend)
        with pytest.raises(ValueError):
            retriever.threshold = 2.0


class TestSimilarityRetrieverFindSimilar:
    """Tests for the find_similar method."""

    def test_returns_top_k(self, mock_backend, sample_search_results, query_vector):
        """TC-4.3.1: Returns top-k results."""
        mock_backend.search_similar.return_value = sample_search_results[:3]
        retriever = SimilarityRetriever(mock_backend, top_k=3)
        results = retriever.find_similar(query_vector)
        assert len(results) <= 3

    def test_results_sorted_by_similarity(
        self, mock_backend, sample_search_results, query_vector
    ):
        """TC-4.3.2: Results sorted by similarity descending."""
        mock_backend.search_similar.return_value = sample_search_results
        retriever = SimilarityRetriever(mock_backend, top_k=5)
        results = retriever.find_similar(query_vector)
        similarities = [r.similarity for r in results]
        assert similarities == sorted(similarities, reverse=True)

    def test_each_result_includes_confidence(
        self, mock_backend, sample_search_results, query_vector
    ):
        """TC-4.3.3: Each result includes confidence level."""
        mock_backend.search_similar.return_value = sample_search_results
        retriever = SimilarityRetriever(mock_backend, top_k=5)
        results = retriever.find_similar(query_vector)
        assert all(hasattr(r, "confidence") for r in results)
        assert all(isinstance(r.confidence, ConfidenceLevel) for r in results)

    def test_confidence_classification_correct(
        self, mock_backend, sample_search_results, query_vector
    ):
        """Confidence levels are classified correctly."""
        mock_backend.search_similar.return_value = sample_search_results
        classifier = ConfidenceClassifier(high=0.95, medium=0.85, low=0.75)
        retriever = SimilarityRetriever(
            mock_backend, top_k=5, confidence_classifier=classifier
        )
        results = retriever.find_similar(query_vector)

        # key1: 0.98 -> HIGH, key2: 0.90 -> MEDIUM, key3: 0.82 -> LOW
        # key4: 0.70 -> NONE, key5: 0.50 -> NONE
        assert results[0].confidence == ConfidenceLevel.HIGH  # 0.98
        assert results[1].confidence == ConfidenceLevel.MEDIUM  # 0.90
        assert results[2].confidence == ConfidenceLevel.LOW  # 0.82
        assert results[3].confidence == ConfidenceLevel.NONE  # 0.70
        assert results[4].confidence == ConfidenceLevel.NONE  # 0.50

    def test_rank_assigned_correctly(
        self, mock_backend, sample_search_results, query_vector
    ):
        """Rank is assigned starting from 1."""
        mock_backend.search_similar.return_value = sample_search_results[:3]
        retriever = SimilarityRetriever(mock_backend, top_k=3)
        results = retriever.find_similar(query_vector)
        ranks = [r.rank for r in results]
        assert ranks == [1, 2, 3]

    def test_metadata_preserved(
        self, mock_backend, sample_search_results, query_vector
    ):
        """Metadata is preserved from search results."""
        mock_backend.search_similar.return_value = sample_search_results[:1]
        retriever = SimilarityRetriever(mock_backend, top_k=1)
        results = retriever.find_similar(query_vector)
        assert results[0].metadata == {"response": "answer1"}

    def test_override_top_k(self, mock_backend, sample_search_results, query_vector):
        """Can override top_k per query."""
        mock_backend.search_similar.return_value = sample_search_results[:2]
        retriever = SimilarityRetriever(mock_backend, top_k=5)
        retriever.find_similar(query_vector, top_k=2)
        mock_backend.search_similar.assert_called_with(
            vector=query_vector, threshold=0.0, top_k=2
        )

    def test_override_threshold(self, mock_backend, sample_search_results, query_vector):
        """Can override threshold per query."""
        mock_backend.search_similar.return_value = sample_search_results[:2]
        retriever = SimilarityRetriever(mock_backend, threshold=0.5)
        retriever.find_similar(query_vector, threshold=0.8)
        mock_backend.search_similar.assert_called_with(
            vector=query_vector, threshold=0.8, top_k=5
        )

    def test_include_query_vector(
        self, mock_backend, sample_search_results, query_vector
    ):
        """Query vector included when requested."""
        mock_backend.search_similar.return_value = []
        retriever = SimilarityRetriever(mock_backend)
        response = retriever.find_similar(query_vector, include_query_vector=True)
        assert response.query_vector is not None
        np.testing.assert_array_equal(response.query_vector, query_vector)

    def test_query_vector_excluded_by_default(
        self, mock_backend, sample_search_results, query_vector
    ):
        """Query vector excluded by default."""
        mock_backend.search_similar.return_value = []
        retriever = SimilarityRetriever(mock_backend)
        response = retriever.find_similar(query_vector)
        assert response.query_vector is None

    def test_empty_results(self, mock_backend, query_vector):
        """Handles empty results gracefully."""
        mock_backend.search_similar.return_value = []
        retriever = SimilarityRetriever(mock_backend)
        response = retriever.find_similar(query_vector)
        assert len(response) == 0
        assert response.best_match is None


class TestSimilarityRetrieverFindBestMatch:
    """Tests for the find_best_match method."""

    def test_returns_best_match_above_min_confidence(
        self, mock_backend, query_vector
    ):
        """Returns best match when confidence meets minimum."""
        mock_backend.search_similar.return_value = [
            SearchResult(key="k1", similarity=0.96, metadata={"r": "1"})
        ]
        retriever = SimilarityRetriever(mock_backend)
        result = retriever.find_best_match(query_vector, ConfidenceLevel.HIGH)
        assert result is not None
        assert result.key == "k1"

    def test_returns_none_below_min_confidence(self, mock_backend, query_vector):
        """Returns None when confidence below minimum."""
        mock_backend.search_similar.return_value = [
            SearchResult(key="k1", similarity=0.80, metadata={"r": "1"})
        ]
        classifier = ConfidenceClassifier(high=0.95, medium=0.85, low=0.75)
        retriever = SimilarityRetriever(
            mock_backend, confidence_classifier=classifier
        )
        result = retriever.find_best_match(query_vector, ConfidenceLevel.HIGH)
        assert result is None

    def test_returns_none_when_no_results(self, mock_backend, query_vector):
        """Returns None when no results found."""
        mock_backend.search_similar.return_value = []
        retriever = SimilarityRetriever(mock_backend)
        result = retriever.find_best_match(query_vector)
        assert result is None

    def test_default_min_confidence_is_low(self, mock_backend, query_vector):
        """Default minimum confidence is LOW."""
        mock_backend.search_similar.return_value = [
            SearchResult(key="k1", similarity=0.76, metadata={"r": "1"})
        ]
        classifier = ConfidenceClassifier(high=0.95, medium=0.85, low=0.75)
        retriever = SimilarityRetriever(
            mock_backend, confidence_classifier=classifier
        )
        result = retriever.find_best_match(query_vector)  # Default is LOW
        assert result is not None
        assert result.confidence == ConfidenceLevel.LOW

    def test_medium_confidence_matches(self, mock_backend, query_vector):
        """Medium confidence result matches MEDIUM minimum."""
        mock_backend.search_similar.return_value = [
            SearchResult(key="k1", similarity=0.90, metadata={"r": "1"})
        ]
        classifier = ConfidenceClassifier(high=0.95, medium=0.85, low=0.75)
        retriever = SimilarityRetriever(
            mock_backend, confidence_classifier=classifier
        )
        result = retriever.find_best_match(query_vector, ConfidenceLevel.MEDIUM)
        assert result is not None
        assert result.confidence == ConfidenceLevel.MEDIUM

    def test_high_confidence_matches_any_min(self, mock_backend, query_vector):
        """High confidence result matches any minimum."""
        mock_backend.search_similar.return_value = [
            SearchResult(key="k1", similarity=0.98, metadata={"r": "1"})
        ]
        retriever = SimilarityRetriever(mock_backend)

        # Should match for all minimum levels
        for min_level in [ConfidenceLevel.LOW, ConfidenceLevel.MEDIUM, ConfidenceLevel.HIGH]:
            result = retriever.find_best_match(query_vector, min_level)
            assert result is not None


class TestSimilarityRetrieverFindByConfidence:
    """Tests for the find_by_confidence method."""

    def test_filters_by_exact_confidence(
        self, mock_backend, sample_search_results, query_vector
    ):
        """Returns only results with exact confidence level."""
        mock_backend.search_similar.return_value = sample_search_results
        classifier = ConfidenceClassifier(high=0.95, medium=0.85, low=0.75)
        retriever = SimilarityRetriever(
            mock_backend, confidence_classifier=classifier
        )

        high_results = retriever.find_by_confidence(query_vector, ConfidenceLevel.HIGH)
        assert all(r.confidence == ConfidenceLevel.HIGH for r in high_results)
        assert len(high_results) == 1  # Only 0.98

    def test_finds_multiple_at_same_level(self, mock_backend, query_vector):
        """Finds multiple results at same confidence level."""
        mock_backend.search_similar.return_value = [
            SearchResult(key="k1", similarity=0.98, metadata={}),
            SearchResult(key="k2", similarity=0.96, metadata={}),
            SearchResult(key="k3", similarity=0.90, metadata={}),
        ]
        classifier = ConfidenceClassifier(high=0.95, medium=0.85, low=0.75)
        retriever = SimilarityRetriever(
            mock_backend, confidence_classifier=classifier
        )

        high_results = retriever.find_by_confidence(query_vector, ConfidenceLevel.HIGH)
        assert len(high_results) == 2

    def test_returns_empty_when_no_matches(self, mock_backend, query_vector):
        """Returns empty list when no results match confidence."""
        mock_backend.search_similar.return_value = [
            SearchResult(key="k1", similarity=0.80, metadata={}),
        ]
        classifier = ConfidenceClassifier(high=0.95, medium=0.85, low=0.75)
        retriever = SimilarityRetriever(
            mock_backend, confidence_classifier=classifier
        )

        high_results = retriever.find_by_confidence(query_vector, ConfidenceLevel.HIGH)
        assert len(high_results) == 0

    def test_respects_top_k_override(self, mock_backend, query_vector):
        """Respects top_k override parameter."""
        mock_backend.search_similar.return_value = [
            SearchResult(key="k1", similarity=0.98, metadata={}),
        ]
        retriever = SimilarityRetriever(mock_backend)
        retriever.find_by_confidence(query_vector, ConfidenceLevel.HIGH, top_k=10)
        mock_backend.search_similar.assert_called_with(
            vector=query_vector, threshold=0.0, top_k=10
        )


class TestSimilarityRetrieverIntegration:
    """Integration tests for SimilarityRetriever."""

    def test_complete_workflow(self, mock_backend, query_vector):
        """Complete retrieval workflow works."""
        # Setup mock with realistic data
        mock_backend.search_similar.return_value = [
            SearchResult(key="exact", similarity=0.99, metadata={"response": "A"}),
            SearchResult(key="close", similarity=0.92, metadata={"response": "B"}),
            SearchResult(key="medium", similarity=0.87, metadata={"response": "C"}),
            SearchResult(key="low", similarity=0.78, metadata={"response": "D"}),
            SearchResult(key="none", similarity=0.60, metadata={"response": "E"}),
        ]

        # Create retriever with model-specific classifier
        classifier = ConfidenceClassifier.for_model("all-MiniLM-L6-v2")
        retriever = SimilarityRetriever(
            mock_backend, top_k=5, confidence_classifier=classifier
        )

        # Find similar
        response = retriever.find_similar(query_vector)

        # Verify results
        assert len(response) == 5
        assert response.best_match.key == "exact"
        assert response.has_high_confidence

        # Check confidence levels assigned correctly
        # MiniLM thresholds: high=0.92, medium=0.85, low=0.75
        assert response[0].confidence == ConfidenceLevel.HIGH  # 0.99
        assert response[1].confidence == ConfidenceLevel.HIGH  # 0.92
        assert response[2].confidence == ConfidenceLevel.MEDIUM  # 0.87
        assert response[3].confidence == ConfidenceLevel.LOW  # 0.78
        assert response[4].confidence == ConfidenceLevel.NONE  # 0.60

    def test_best_match_workflow(self, mock_backend, query_vector):
        """Best match with confidence filter workflow."""
        mock_backend.search_similar.return_value = [
            SearchResult(key="k1", similarity=0.85, metadata={"response": "test"}),
        ]

        classifier = ConfidenceClassifier(high=0.95, medium=0.85, low=0.75)
        retriever = SimilarityRetriever(
            mock_backend, confidence_classifier=classifier
        )

        # Require HIGH confidence - should fail
        high_result = retriever.find_best_match(query_vector, ConfidenceLevel.HIGH)
        assert high_result is None

        # Require MEDIUM confidence - should succeed (0.85 is medium threshold)
        medium_result = retriever.find_best_match(query_vector, ConfidenceLevel.MEDIUM)
        assert medium_result is not None
        assert medium_result.metadata["response"] == "test"

    def test_retrieval_with_threshold_filtering(self, mock_backend, query_vector):
        """Retrieval respects threshold parameter."""
        mock_backend.search_similar.return_value = []
        retriever = SimilarityRetriever(mock_backend, threshold=0.9)
        retriever.find_similar(query_vector)

        mock_backend.search_similar.assert_called_with(
            vector=query_vector, threshold=0.9, top_k=5
        )
