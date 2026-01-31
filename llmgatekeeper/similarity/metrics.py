"""Similarity metrics for vector comparison.

This module provides various distance and similarity metrics for comparing
embedding vectors. All metrics are designed to return values in the range
[0, 1] where 1 indicates maximum similarity.
"""

from enum import Enum
from typing import Callable

import numpy as np
from numpy.typing import NDArray


class SimilarityMetric(Enum):
    """Enum for selecting similarity metrics.

    Each metric computes a similarity score between two vectors,
    with 1.0 indicating identical vectors and 0.0 indicating
    maximum dissimilarity.
    """

    COSINE = "cosine"
    DOT_PRODUCT = "dot_product"
    EUCLIDEAN = "euclidean"


def cosine_similarity(
    v1: NDArray[np.float32], v2: NDArray[np.float32]
) -> float:
    """Compute cosine similarity between two vectors.

    Cosine similarity measures the cosine of the angle between two vectors.
    It is commonly used for comparing text embeddings as it is insensitive
    to vector magnitude and focuses on direction.

    Args:
        v1: First vector.
        v2: Second vector.

    Returns:
        Similarity score in range [-1, 1], where 1 means identical direction,
        0 means orthogonal, and -1 means opposite direction.

    Example:
        >>> v1 = np.array([1.0, 0.0])
        >>> v2 = np.array([1.0, 0.0])
        >>> cosine_similarity(v1, v2)
        1.0
    """
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return float(np.dot(v1, v2) / (norm1 * norm2))


def dot_product_similarity(
    v1: NDArray[np.float32], v2: NDArray[np.float32]
) -> float:
    """Compute dot product between two vectors.

    The dot product is a simple measure of similarity that takes into
    account both direction and magnitude. For normalized vectors, this
    is equivalent to cosine similarity.

    Args:
        v1: First vector.
        v2: Second vector.

    Returns:
        The dot product of the two vectors. Unlike other metrics in this
        module, the output is not bounded to [0, 1] unless vectors are
        normalized.

    Example:
        >>> v1 = np.array([1.0, 2.0])
        >>> v2 = np.array([3.0, 4.0])
        >>> dot_product_similarity(v1, v2)
        11.0
    """
    return float(np.dot(v1, v2))


def euclidean_similarity(
    v1: NDArray[np.float32], v2: NDArray[np.float32]
) -> float:
    """Compute Euclidean similarity between two vectors.

    This converts Euclidean distance to a similarity score using the
    formula: similarity = 1 / (1 + distance). This maps distance to
    a similarity score in the range (0, 1], where 1 means identical
    vectors.

    Args:
        v1: First vector.
        v2: Second vector.

    Returns:
        Similarity score in range (0, 1], where 1 means identical vectors.

    Example:
        >>> v1 = np.array([0.0, 0.0])
        >>> v2 = np.array([0.0, 0.0])
        >>> euclidean_similarity(v1, v2)
        1.0
    """
    distance = float(np.linalg.norm(v1 - v2))
    return 1.0 / (1.0 + distance)


def euclidean_distance(
    v1: NDArray[np.float32], v2: NDArray[np.float32]
) -> float:
    """Compute Euclidean distance between two vectors.

    The Euclidean distance is the straight-line distance between two
    points in n-dimensional space.

    Args:
        v1: First vector.
        v2: Second vector.

    Returns:
        The Euclidean distance (L2 norm of the difference).

    Example:
        >>> v1 = np.array([0.0, 0.0])
        >>> v2 = np.array([3.0, 4.0])
        >>> euclidean_distance(v1, v2)
        5.0
    """
    return float(np.linalg.norm(v1 - v2))


def normalize_similarity(
    value: float,
    metric: SimilarityMetric,
    max_distance: float = 10.0,
) -> float:
    """Normalize a similarity value to the range [0, 1].

    Different metrics produce values in different ranges. This function
    normalizes them all to [0, 1] for consistent threshold comparison.

    Args:
        value: The raw similarity or distance value.
        metric: The metric that produced the value.
        max_distance: Maximum expected distance for Euclidean metric,
            used for normalization. Default is 10.0.

    Returns:
        Normalized similarity in range [0, 1].

    Example:
        >>> normalize_similarity(0.5, SimilarityMetric.COSINE)
        0.75
    """
    if metric == SimilarityMetric.COSINE:
        # Cosine is in [-1, 1], map to [0, 1]
        return (value + 1.0) / 2.0
    elif metric == SimilarityMetric.DOT_PRODUCT:
        # Dot product can be any value, we assume normalized vectors
        # For normalized vectors, dot product is same as cosine
        return max(0.0, min(1.0, (value + 1.0) / 2.0))
    elif metric == SimilarityMetric.EUCLIDEAN:
        # Already normalized by euclidean_similarity
        return value
    else:
        return value


def get_similarity_function(
    metric: SimilarityMetric,
) -> Callable[[NDArray[np.float32], NDArray[np.float32]], float]:
    """Get the similarity function for a given metric.

    Args:
        metric: The similarity metric to use.

    Returns:
        A function that computes similarity between two vectors.

    Raises:
        ValueError: If the metric is not recognized.

    Example:
        >>> func = get_similarity_function(SimilarityMetric.COSINE)
        >>> v1 = np.array([1.0, 0.0])
        >>> v2 = np.array([1.0, 0.0])
        >>> func(v1, v2)
        1.0
    """
    if metric == SimilarityMetric.COSINE:
        return cosine_similarity
    elif metric == SimilarityMetric.DOT_PRODUCT:
        return dot_product_similarity
    elif metric == SimilarityMetric.EUCLIDEAN:
        return euclidean_similarity
    else:
        raise ValueError(f"Unknown similarity metric: {metric}")


def compute_similarity(
    v1: NDArray[np.float32],
    v2: NDArray[np.float32],
    metric: SimilarityMetric = SimilarityMetric.COSINE,
) -> float:
    """Compute similarity between two vectors using the specified metric.

    This is a convenience function that combines metric selection and
    computation in a single call.

    Args:
        v1: First vector.
        v2: Second vector.
        metric: The similarity metric to use. Default is cosine similarity.

    Returns:
        Similarity score. The range depends on the metric:
        - COSINE: [-1, 1]
        - DOT_PRODUCT: unbounded (depends on vector magnitudes)
        - EUCLIDEAN: (0, 1]

    Example:
        >>> v1 = np.array([1.0, 0.0])
        >>> v2 = np.array([0.0, 1.0])
        >>> compute_similarity(v1, v2, SimilarityMetric.COSINE)
        0.0
    """
    func = get_similarity_function(metric)
    return func(v1, v2)


def batch_cosine_similarity(
    query: NDArray[np.float32],
    vectors: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Compute cosine similarity between a query and multiple vectors.

    This is an optimized batch operation that computes similarity between
    a single query vector and a matrix of vectors.

    Args:
        query: Query vector of shape (dim,).
        vectors: Matrix of vectors of shape (n, dim).

    Returns:
        Array of similarity scores of shape (n,).

    Example:
        >>> query = np.array([1.0, 0.0])
        >>> vectors = np.array([[1.0, 0.0], [0.0, 1.0], [0.707, 0.707]])
        >>> batch_cosine_similarity(query, vectors)
        array([1.0, 0.0, 0.707...], dtype=float32)
    """
    # Normalize query
    query_norm = np.linalg.norm(query)
    if query_norm == 0:
        return np.zeros(len(vectors), dtype=np.float32)
    normalized_query = query / query_norm

    # Normalize vectors
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    # Avoid division by zero
    norms = np.where(norms == 0, 1, norms)
    normalized_vectors = vectors / norms

    # Compute dot products
    similarities = np.dot(normalized_vectors, normalized_query)

    return similarities.astype(np.float32)
