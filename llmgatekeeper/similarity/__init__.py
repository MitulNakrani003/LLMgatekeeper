"""Similarity metrics and retrieval for LLMGatekeeper."""

from llmgatekeeper.similarity.confidence import (
    ConfidenceClassifier,
    ConfidenceLevel,
    get_default_classifier,
    get_model_classifier,
)
from llmgatekeeper.similarity.metrics import (
    SimilarityMetric,
    batch_cosine_similarity,
    compute_similarity,
    cosine_similarity,
    dot_product_similarity,
    euclidean_distance,
    euclidean_similarity,
    get_similarity_function,
    normalize_similarity,
)
from llmgatekeeper.similarity.retriever import (
    RetrievalResponse,
    RetrievalResult,
    SimilarityRetriever,
)

__all__ = [
    # Confidence
    "ConfidenceClassifier",
    "ConfidenceLevel",
    "get_default_classifier",
    "get_model_classifier",
    # Metrics
    "SimilarityMetric",
    "batch_cosine_similarity",
    "compute_similarity",
    "cosine_similarity",
    "dot_product_similarity",
    "euclidean_distance",
    "euclidean_similarity",
    "get_similarity_function",
    "normalize_similarity",
    # Retrieval
    "RetrievalResponse",
    "RetrievalResult",
    "SimilarityRetriever",
]
