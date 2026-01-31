"""Confidence level classification for similarity scores.

This module provides tools for classifying similarity scores into
confidence bands (HIGH, MEDIUM, LOW, NONE) based on configurable
thresholds. Different embedding models produce different score
distributions, so model-specific defaults are provided.
"""

from enum import Enum
from typing import Dict, Optional


class ConfidenceLevel(Enum):
    """Confidence levels for similarity match results.

    These levels indicate how confident we are that a similarity match
    represents a semantically equivalent query.

    Attributes:
        HIGH: Very confident match, safe to use cached response.
        MEDIUM: Moderately confident, may want human review for critical apps.
        LOW: Low confidence, use with caution.
        NONE: Below threshold, not a match.
    """

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NONE = "none"

    def __str__(self) -> str:
        """Return the string value of the confidence level."""
        return self.value


# Default thresholds for different embedding models.
# These are tuned based on empirical testing of each model's
# similarity score distributions.
MODEL_THRESHOLDS: Dict[str, Dict[str, float]] = {
    # Sentence Transformers models
    "all-MiniLM-L6-v2": {
        "high": 0.92,
        "medium": 0.85,
        "low": 0.75,
    },
    "all-mpnet-base-v2": {
        "high": 0.90,
        "medium": 0.82,
        "low": 0.72,
    },
    "paraphrase-MiniLM-L6-v2": {
        "high": 0.90,
        "medium": 0.82,
        "low": 0.72,
    },
    # OpenAI models
    "text-embedding-ada-002": {
        "high": 0.95,
        "medium": 0.90,
        "low": 0.85,
    },
    "text-embedding-3-small": {
        "high": 0.94,
        "medium": 0.88,
        "low": 0.82,
    },
    "text-embedding-3-large": {
        "high": 0.94,
        "medium": 0.88,
        "low": 0.82,
    },
}

# Default thresholds when model is not recognized
DEFAULT_THRESHOLDS = {
    "high": 0.95,
    "medium": 0.85,
    "low": 0.75,
}


class ConfidenceClassifier:
    """Classifies similarity scores into confidence levels.

    This classifier takes a similarity score and returns a confidence
    level based on configurable thresholds. It supports both custom
    thresholds and model-specific defaults.

    Example:
        >>> classifier = ConfidenceClassifier(high=0.95, medium=0.85, low=0.75)
        >>> classifier.classify(0.96)
        <ConfidenceLevel.HIGH: 'high'>
        >>> classifier.classify(0.90)
        <ConfidenceLevel.MEDIUM: 'medium'>
        >>> classifier.classify(0.50)
        <ConfidenceLevel.NONE: 'none'>

        >>> # Using model-specific defaults
        >>> classifier = ConfidenceClassifier.for_model("all-MiniLM-L6-v2")
        >>> classifier.classify(0.93)
        <ConfidenceLevel.HIGH: 'high'>
    """

    def __init__(
        self,
        high: float = 0.95,
        medium: float = 0.85,
        low: float = 0.75,
    ) -> None:
        """Initialize the confidence classifier with threshold values.

        Args:
            high: Minimum similarity for HIGH confidence. Default 0.95.
            medium: Minimum similarity for MEDIUM confidence. Default 0.85.
            low: Minimum similarity for LOW confidence. Default 0.75.

        Raises:
            ValueError: If thresholds are not in descending order or out of range.
        """
        # Validate thresholds
        if not (0 <= low <= medium <= high <= 1):
            raise ValueError(
                f"Thresholds must satisfy 0 <= low <= medium <= high <= 1. "
                f"Got: low={low}, medium={medium}, high={high}"
            )

        self._high = high
        self._medium = medium
        self._low = low

    @classmethod
    def for_model(cls, model_name: str) -> "ConfidenceClassifier":
        """Create a classifier with model-specific default thresholds.

        Different embedding models produce different similarity score
        distributions. This factory method returns a classifier tuned
        for the specified model.

        Args:
            model_name: Name of the embedding model.

        Returns:
            A ConfidenceClassifier with appropriate thresholds.

        Example:
            >>> classifier = ConfidenceClassifier.for_model("text-embedding-ada-002")
            >>> classifier.high_threshold
            0.95
        """
        thresholds = MODEL_THRESHOLDS.get(model_name, DEFAULT_THRESHOLDS)
        return cls(
            high=thresholds["high"],
            medium=thresholds["medium"],
            low=thresholds["low"],
        )

    @property
    def high_threshold(self) -> float:
        """Return the high confidence threshold."""
        return self._high

    @property
    def medium_threshold(self) -> float:
        """Return the medium confidence threshold."""
        return self._medium

    @property
    def low_threshold(self) -> float:
        """Return the low confidence threshold."""
        return self._low

    def classify(self, similarity: float) -> ConfidenceLevel:
        """Classify a similarity score into a confidence level.

        Args:
            similarity: Similarity score, typically in range [0, 1].

        Returns:
            The appropriate ConfidenceLevel for the score.

        Example:
            >>> classifier = ConfidenceClassifier(high=0.95, medium=0.85, low=0.75)
            >>> classifier.classify(0.96)
            <ConfidenceLevel.HIGH: 'high'>
        """
        if similarity >= self._high:
            return ConfidenceLevel.HIGH
        elif similarity >= self._medium:
            return ConfidenceLevel.MEDIUM
        elif similarity >= self._low:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.NONE

    def is_match(self, similarity: float) -> bool:
        """Check if a similarity score represents a valid match.

        A match is any score at or above the low threshold.

        Args:
            similarity: Similarity score to check.

        Returns:
            True if the score is at or above the low threshold.

        Example:
            >>> classifier = ConfidenceClassifier(high=0.95, medium=0.85, low=0.75)
            >>> classifier.is_match(0.80)
            True
            >>> classifier.is_match(0.50)
            False
        """
        return similarity >= self._low

    def get_thresholds(self) -> Dict[str, float]:
        """Return all thresholds as a dictionary.

        Returns:
            Dictionary with 'high', 'medium', and 'low' thresholds.

        Example:
            >>> classifier = ConfidenceClassifier(high=0.95, medium=0.85, low=0.75)
            >>> classifier.get_thresholds()
            {'high': 0.95, 'medium': 0.85, 'low': 0.75}
        """
        return {
            "high": self._high,
            "medium": self._medium,
            "low": self._low,
        }

    def __repr__(self) -> str:
        """Return a string representation of the classifier."""
        return (
            f"ConfidenceClassifier(high={self._high}, "
            f"medium={self._medium}, low={self._low})"
        )


def get_default_classifier() -> ConfidenceClassifier:
    """Get a classifier with default thresholds.

    Returns:
        A ConfidenceClassifier with default threshold values.
    """
    return ConfidenceClassifier()


def get_model_classifier(model_name: Optional[str] = None) -> ConfidenceClassifier:
    """Get a classifier for the specified model or default.

    Args:
        model_name: Optional model name. If None, returns default classifier.

    Returns:
        A ConfidenceClassifier with appropriate thresholds.
    """
    if model_name is None:
        return get_default_classifier()
    return ConfidenceClassifier.for_model(model_name)
