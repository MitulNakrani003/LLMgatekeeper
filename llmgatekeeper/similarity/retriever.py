"""Multi-result retrieval for similarity search.

This module provides the SimilarityRetriever class for retrieving multiple
similar results with confidence levels, supporting ensemble/voting scenarios.
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from numpy.typing import NDArray

from llmgatekeeper.backends.base import CacheBackend, SearchResult
from llmgatekeeper.similarity.confidence import (
    ConfidenceClassifier,
    ConfidenceLevel,
    get_default_classifier,
)


@dataclass
class RetrievalResult:
    """A single retrieval result with confidence information.

    Attributes:
        key: Unique identifier of the cached entry.
        similarity: Similarity score (0-1 for most metrics).
        confidence: Confidence level classification.
        metadata: Arbitrary metadata stored with the entry.
        rank: Position in the result list (1-indexed).
    """

    key: str
    similarity: float
    confidence: ConfidenceLevel
    metadata: dict
    rank: int

    def __repr__(self) -> str:
        """Return a string representation of the result."""
        return (
            f"RetrievalResult(key={self.key!r}, similarity={self.similarity:.4f}, "
            f"confidence={self.confidence}, rank={self.rank})"
        )


@dataclass
class RetrievalResponse:
    """Response from a similarity retrieval operation.

    Attributes:
        results: List of retrieval results, sorted by similarity descending.
        query_vector: The query vector used for retrieval (optional).
        total_candidates: Total number of candidates considered (if available).
    """

    results: List[RetrievalResult]
    query_vector: Optional[NDArray[np.float32]] = None
    total_candidates: Optional[int] = None

    def __len__(self) -> int:
        """Return the number of results."""
        return len(self.results)

    def __iter__(self):
        """Iterate over results."""
        return iter(self.results)

    def __getitem__(self, index):
        """Get result by index."""
        return self.results[index]

    @property
    def best_match(self) -> Optional[RetrievalResult]:
        """Return the best matching result, or None if no results."""
        return self.results[0] if self.results else None

    @property
    def has_high_confidence(self) -> bool:
        """Check if any result has high confidence."""
        return any(r.confidence == ConfidenceLevel.HIGH for r in self.results)

    @property
    def high_confidence_results(self) -> List[RetrievalResult]:
        """Return only results with high confidence."""
        return [r for r in self.results if r.confidence == ConfidenceLevel.HIGH]

    @property
    def above_threshold_results(self) -> List[RetrievalResult]:
        """Return results that are considered matches (not NONE confidence)."""
        return [r for r in self.results if r.confidence != ConfidenceLevel.NONE]


class SimilarityRetriever:
    """Retrieves multiple similar results from a cache backend.

    The SimilarityRetriever wraps a cache backend and provides enhanced
    retrieval functionality including:
    - Top-k result retrieval
    - Confidence level classification for each result
    - Sorted results by similarity (descending)

    Example:
        >>> backend = RedisSimpleBackend(redis_client)
        >>> retriever = SimilarityRetriever(backend, top_k=5)
        >>> results = retriever.find_similar(query_embedding)
        >>> for result in results:
        ...     print(f"{result.key}: {result.similarity:.2f} ({result.confidence})")

        >>> # Using model-specific confidence thresholds
        >>> retriever = SimilarityRetriever(
        ...     backend,
        ...     top_k=3,
        ...     confidence_classifier=ConfidenceClassifier.for_model("all-MiniLM-L6-v2")
        ... )
    """

    def __init__(
        self,
        backend: CacheBackend,
        top_k: int = 5,
        threshold: float = 0.0,
        confidence_classifier: Optional[ConfidenceClassifier] = None,
    ) -> None:
        """Initialize the similarity retriever.

        Args:
            backend: The cache backend to retrieve from.
            top_k: Maximum number of results to return. Default 5.
            threshold: Minimum similarity threshold for results. Default 0.0
                (return all results up to top_k).
            confidence_classifier: Classifier for confidence levels. If None,
                uses the default classifier.

        Raises:
            ValueError: If top_k < 1 or threshold is out of range.
        """
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        if not (0.0 <= threshold <= 1.0):
            raise ValueError(f"threshold must be in [0, 1], got {threshold}")

        self._backend = backend
        self._top_k = top_k
        self._threshold = threshold
        self._classifier = confidence_classifier or get_default_classifier()

    @property
    def top_k(self) -> int:
        """Return the maximum number of results to return."""
        return self._top_k

    @top_k.setter
    def top_k(self, value: int) -> None:
        """Set the maximum number of results to return."""
        if value < 1:
            raise ValueError(f"top_k must be >= 1, got {value}")
        self._top_k = value

    @property
    def threshold(self) -> float:
        """Return the minimum similarity threshold."""
        return self._threshold

    @threshold.setter
    def threshold(self, value: float) -> None:
        """Set the minimum similarity threshold."""
        if not (0.0 <= value <= 1.0):
            raise ValueError(f"threshold must be in [0, 1], got {value}")
        self._threshold = value

    @property
    def confidence_classifier(self) -> ConfidenceClassifier:
        """Return the confidence classifier."""
        return self._classifier

    def find_similar(
        self,
        query_vector: NDArray[np.float32],
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
        include_query_vector: bool = False,
    ) -> RetrievalResponse:
        """Find similar vectors in the cache.

        Args:
            query_vector: The query embedding vector.
            top_k: Override the default top_k for this query.
            threshold: Override the default threshold for this query.
            include_query_vector: Whether to include the query vector in response.

        Returns:
            RetrievalResponse with sorted results and metadata.

        Example:
            >>> results = retriever.find_similar(embedding, top_k=3)
            >>> if results.has_high_confidence:
            ...     return results.best_match.metadata["response"]
        """
        effective_top_k = top_k if top_k is not None else self._top_k
        effective_threshold = threshold if threshold is not None else self._threshold

        # Query the backend
        search_results: List[SearchResult] = self._backend.search_similar(
            vector=query_vector,
            threshold=effective_threshold,
            top_k=effective_top_k,
        )

        # Convert to RetrievalResult with confidence classification
        retrieval_results = []
        for rank, sr in enumerate(search_results, start=1):
            confidence = self._classifier.classify(sr.similarity)
            retrieval_results.append(
                RetrievalResult(
                    key=sr.key,
                    similarity=sr.similarity,
                    confidence=confidence,
                    metadata=sr.metadata,
                    rank=rank,
                )
            )

        return RetrievalResponse(
            results=retrieval_results,
            query_vector=query_vector if include_query_vector else None,
            total_candidates=None,  # Backend doesn't expose this
        )

    def find_best_match(
        self,
        query_vector: NDArray[np.float32],
        min_confidence: ConfidenceLevel = ConfidenceLevel.LOW,
    ) -> Optional[RetrievalResult]:
        """Find the single best matching result above a confidence level.

        This is a convenience method for cases where you only need the top
        result and want to filter by confidence.

        Args:
            query_vector: The query embedding vector.
            min_confidence: Minimum confidence level required. Default LOW.

        Returns:
            The best matching result if it meets the confidence threshold,
            None otherwise.

        Example:
            >>> result = retriever.find_best_match(embedding, ConfidenceLevel.HIGH)
            >>> if result:
            ...     return result.metadata["response"]
        """
        response = self.find_similar(query_vector, top_k=1)

        if not response.results:
            return None

        best = response.results[0]

        # Check confidence meets minimum
        confidence_order = [
            ConfidenceLevel.NONE,
            ConfidenceLevel.LOW,
            ConfidenceLevel.MEDIUM,
            ConfidenceLevel.HIGH,
        ]
        if confidence_order.index(best.confidence) >= confidence_order.index(
            min_confidence
        ):
            return best

        return None

    def find_by_confidence(
        self,
        query_vector: NDArray[np.float32],
        confidence_level: ConfidenceLevel,
        top_k: Optional[int] = None,
    ) -> List[RetrievalResult]:
        """Find results with a specific confidence level.

        Args:
            query_vector: The query embedding vector.
            confidence_level: The exact confidence level to filter for.
            top_k: Maximum results to consider (before filtering).

        Returns:
            List of results with the specified confidence level.

        Example:
            >>> high_confidence = retriever.find_by_confidence(
            ...     embedding, ConfidenceLevel.HIGH
            ... )
        """
        response = self.find_similar(query_vector, top_k=top_k)
        return [r for r in response.results if r.confidence == confidence_level]

    def __repr__(self) -> str:
        """Return a string representation of the retriever."""
        return (
            f"SimilarityRetriever(top_k={self._top_k}, "
            f"threshold={self._threshold}, "
            f"classifier={self._classifier!r})"
        )
