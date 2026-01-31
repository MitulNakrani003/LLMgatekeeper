"""Tests for the confidence classification module."""

import pytest

from llmgatekeeper.similarity.confidence import (
    DEFAULT_THRESHOLDS,
    MODEL_THRESHOLDS,
    ConfidenceClassifier,
    ConfidenceLevel,
    get_default_classifier,
    get_model_classifier,
)


class TestConfidenceLevel:
    """Tests for the ConfidenceLevel enum."""

    def test_all_levels_defined(self):
        """All expected confidence levels are defined."""
        assert ConfidenceLevel.HIGH.value == "high"
        assert ConfidenceLevel.MEDIUM.value == "medium"
        assert ConfidenceLevel.LOW.value == "low"
        assert ConfidenceLevel.NONE.value == "none"

    def test_str_representation(self):
        """String representation returns the value."""
        assert str(ConfidenceLevel.HIGH) == "high"
        assert str(ConfidenceLevel.NONE) == "none"

    def test_enum_comparison(self):
        """Enum values can be compared."""
        assert ConfidenceLevel.HIGH == ConfidenceLevel.HIGH
        assert ConfidenceLevel.HIGH != ConfidenceLevel.LOW


class TestConfidenceClassifierInit:
    """Tests for ConfidenceClassifier initialization."""

    def test_default_thresholds(self):
        """Default thresholds are set correctly."""
        classifier = ConfidenceClassifier()
        assert classifier.high_threshold == 0.95
        assert classifier.medium_threshold == 0.85
        assert classifier.low_threshold == 0.75

    def test_custom_thresholds(self):
        """Custom thresholds are accepted."""
        classifier = ConfidenceClassifier(high=0.90, medium=0.80, low=0.70)
        assert classifier.high_threshold == 0.90
        assert classifier.medium_threshold == 0.80
        assert classifier.low_threshold == 0.70

    def test_invalid_order_raises_error(self):
        """Invalid threshold order raises ValueError."""
        with pytest.raises(ValueError, match="Thresholds must satisfy"):
            ConfidenceClassifier(high=0.70, medium=0.80, low=0.90)

    def test_high_less_than_medium_raises_error(self):
        """High < medium raises ValueError."""
        with pytest.raises(ValueError):
            ConfidenceClassifier(high=0.80, medium=0.85, low=0.75)

    def test_medium_less_than_low_raises_error(self):
        """Medium < low raises ValueError."""
        with pytest.raises(ValueError):
            ConfidenceClassifier(high=0.95, medium=0.70, low=0.75)

    def test_negative_threshold_raises_error(self):
        """Negative threshold raises ValueError."""
        with pytest.raises(ValueError):
            ConfidenceClassifier(high=0.95, medium=0.85, low=-0.1)

    def test_threshold_above_one_raises_error(self):
        """Threshold > 1 raises ValueError."""
        with pytest.raises(ValueError):
            ConfidenceClassifier(high=1.5, medium=0.85, low=0.75)

    def test_equal_thresholds_allowed(self):
        """Equal thresholds are allowed."""
        classifier = ConfidenceClassifier(high=0.90, medium=0.90, low=0.90)
        assert classifier.high_threshold == 0.90

    def test_repr(self):
        """Repr shows thresholds."""
        classifier = ConfidenceClassifier(high=0.95, medium=0.85, low=0.75)
        repr_str = repr(classifier)
        assert "0.95" in repr_str
        assert "0.85" in repr_str
        assert "0.75" in repr_str


class TestConfidenceClassifierClassify:
    """Tests for the classify method."""

    def test_high_confidence(self):
        """TC-4.2.1: High confidence for similarity >= 0.95."""
        classifier = ConfidenceClassifier(high=0.95, medium=0.85, low=0.75)
        assert classifier.classify(0.96) == ConfidenceLevel.HIGH
        assert classifier.classify(0.95) == ConfidenceLevel.HIGH
        assert classifier.classify(1.0) == ConfidenceLevel.HIGH

    def test_medium_confidence(self):
        """TC-4.2.2: Medium confidence for 0.85 <= similarity < 0.95."""
        classifier = ConfidenceClassifier(high=0.95, medium=0.85, low=0.75)
        assert classifier.classify(0.90) == ConfidenceLevel.MEDIUM
        assert classifier.classify(0.85) == ConfidenceLevel.MEDIUM
        assert classifier.classify(0.94) == ConfidenceLevel.MEDIUM

    def test_low_confidence(self):
        """Low confidence for 0.75 <= similarity < 0.85."""
        classifier = ConfidenceClassifier(high=0.95, medium=0.85, low=0.75)
        assert classifier.classify(0.80) == ConfidenceLevel.LOW
        assert classifier.classify(0.75) == ConfidenceLevel.LOW
        assert classifier.classify(0.84) == ConfidenceLevel.LOW

    def test_no_confidence(self):
        """TC-4.2.3: None confidence below low threshold."""
        classifier = ConfidenceClassifier(high=0.95, medium=0.85, low=0.75)
        assert classifier.classify(0.50) == ConfidenceLevel.NONE
        assert classifier.classify(0.74) == ConfidenceLevel.NONE
        assert classifier.classify(0.0) == ConfidenceLevel.NONE

    def test_boundary_values(self):
        """Boundary values are classified correctly."""
        classifier = ConfidenceClassifier(high=0.95, medium=0.85, low=0.75)

        # Exact boundaries
        assert classifier.classify(0.95) == ConfidenceLevel.HIGH
        assert classifier.classify(0.85) == ConfidenceLevel.MEDIUM
        assert classifier.classify(0.75) == ConfidenceLevel.LOW

        # Just below boundaries
        assert classifier.classify(0.9499) == ConfidenceLevel.MEDIUM
        assert classifier.classify(0.8499) == ConfidenceLevel.LOW
        assert classifier.classify(0.7499) == ConfidenceLevel.NONE

    def test_negative_similarity(self):
        """Negative similarity returns NONE."""
        classifier = ConfidenceClassifier()
        assert classifier.classify(-0.5) == ConfidenceLevel.NONE


class TestConfidenceClassifierForModel:
    """Tests for the for_model factory method."""

    def test_model_specific_defaults(self):
        """TC-4.2.4: Default thresholds differ by model."""
        minilm_classifier = ConfidenceClassifier.for_model("all-MiniLM-L6-v2")
        openai_classifier = ConfidenceClassifier.for_model("text-embedding-ada-002")

        assert minilm_classifier.high_threshold != openai_classifier.high_threshold

    def test_minilm_thresholds(self):
        """MiniLM model has correct thresholds."""
        classifier = ConfidenceClassifier.for_model("all-MiniLM-L6-v2")
        assert classifier.high_threshold == 0.92
        assert classifier.medium_threshold == 0.85
        assert classifier.low_threshold == 0.75

    def test_openai_ada_thresholds(self):
        """OpenAI ada-002 model has correct thresholds."""
        classifier = ConfidenceClassifier.for_model("text-embedding-ada-002")
        assert classifier.high_threshold == 0.95
        assert classifier.medium_threshold == 0.90
        assert classifier.low_threshold == 0.85

    def test_openai_3_small_thresholds(self):
        """OpenAI 3-small model has correct thresholds."""
        classifier = ConfidenceClassifier.for_model("text-embedding-3-small")
        assert classifier.high_threshold == 0.94
        assert classifier.medium_threshold == 0.88
        assert classifier.low_threshold == 0.82

    def test_unknown_model_uses_defaults(self):
        """Unknown model uses default thresholds."""
        classifier = ConfidenceClassifier.for_model("some-unknown-model")
        assert classifier.high_threshold == DEFAULT_THRESHOLDS["high"]
        assert classifier.medium_threshold == DEFAULT_THRESHOLDS["medium"]
        assert classifier.low_threshold == DEFAULT_THRESHOLDS["low"]

    def test_all_known_models_have_thresholds(self):
        """All models in MODEL_THRESHOLDS have valid thresholds."""
        for model_name in MODEL_THRESHOLDS:
            classifier = ConfidenceClassifier.for_model(model_name)
            assert 0 <= classifier.low_threshold <= classifier.medium_threshold
            assert classifier.medium_threshold <= classifier.high_threshold <= 1


class TestConfidenceClassifierIsMatch:
    """Tests for the is_match method."""

    def test_above_low_is_match(self):
        """Similarity above low threshold is a match."""
        classifier = ConfidenceClassifier(high=0.95, medium=0.85, low=0.75)
        assert classifier.is_match(0.80) is True
        assert classifier.is_match(0.90) is True
        assert classifier.is_match(0.99) is True

    def test_at_low_is_match(self):
        """Similarity exactly at low threshold is a match."""
        classifier = ConfidenceClassifier(high=0.95, medium=0.85, low=0.75)
        assert classifier.is_match(0.75) is True

    def test_below_low_not_match(self):
        """Similarity below low threshold is not a match."""
        classifier = ConfidenceClassifier(high=0.95, medium=0.85, low=0.75)
        assert classifier.is_match(0.50) is False
        assert classifier.is_match(0.74) is False


class TestConfidenceClassifierGetThresholds:
    """Tests for the get_thresholds method."""

    def test_returns_all_thresholds(self):
        """Returns dictionary with all thresholds."""
        classifier = ConfidenceClassifier(high=0.95, medium=0.85, low=0.75)
        thresholds = classifier.get_thresholds()

        assert thresholds == {
            "high": 0.95,
            "medium": 0.85,
            "low": 0.75,
        }

    def test_returns_custom_thresholds(self):
        """Returns custom thresholds correctly."""
        classifier = ConfidenceClassifier(high=0.90, medium=0.80, low=0.70)
        thresholds = classifier.get_thresholds()

        assert thresholds["high"] == 0.90
        assert thresholds["medium"] == 0.80
        assert thresholds["low"] == 0.70


class TestHelperFunctions:
    """Tests for module-level helper functions."""

    def test_get_default_classifier(self):
        """get_default_classifier returns classifier with defaults."""
        classifier = get_default_classifier()
        assert classifier.high_threshold == 0.95
        assert classifier.medium_threshold == 0.85
        assert classifier.low_threshold == 0.75

    def test_get_model_classifier_with_model(self):
        """get_model_classifier with model name uses model thresholds."""
        classifier = get_model_classifier("all-MiniLM-L6-v2")
        assert classifier.high_threshold == 0.92

    def test_get_model_classifier_without_model(self):
        """get_model_classifier without model uses defaults."""
        classifier = get_model_classifier(None)
        assert classifier.high_threshold == 0.95


class TestModelThresholdsConstant:
    """Tests for the MODEL_THRESHOLDS constant."""

    def test_contains_sentence_transformer_models(self):
        """Contains Sentence Transformer models."""
        assert "all-MiniLM-L6-v2" in MODEL_THRESHOLDS
        assert "all-mpnet-base-v2" in MODEL_THRESHOLDS

    def test_contains_openai_models(self):
        """Contains OpenAI models."""
        assert "text-embedding-ada-002" in MODEL_THRESHOLDS
        assert "text-embedding-3-small" in MODEL_THRESHOLDS
        assert "text-embedding-3-large" in MODEL_THRESHOLDS

    def test_all_thresholds_valid(self):
        """All thresholds are valid (0-1 range, proper order)."""
        for model, thresholds in MODEL_THRESHOLDS.items():
            assert 0 <= thresholds["low"] <= 1, f"Invalid low for {model}"
            assert 0 <= thresholds["medium"] <= 1, f"Invalid medium for {model}"
            assert 0 <= thresholds["high"] <= 1, f"Invalid high for {model}"
            assert thresholds["low"] <= thresholds["medium"], f"low > medium for {model}"
            assert thresholds["medium"] <= thresholds["high"], f"medium > high for {model}"


class TestConfidenceClassifierIntegration:
    """Integration tests for confidence classification."""

    def test_classification_workflow(self):
        """Complete classification workflow works."""
        # Create classifier for a specific model
        classifier = ConfidenceClassifier.for_model("all-MiniLM-L6-v2")

        # Classify various similarities
        results = [
            (0.95, ConfidenceLevel.HIGH),
            (0.88, ConfidenceLevel.MEDIUM),
            (0.78, ConfidenceLevel.LOW),
            (0.50, ConfidenceLevel.NONE),
        ]

        for similarity, expected_level in results:
            assert classifier.classify(similarity) == expected_level

    def test_match_and_classify_consistency(self):
        """is_match and classify are consistent."""
        classifier = ConfidenceClassifier(high=0.95, medium=0.85, low=0.75)

        # Test various similarities
        for similarity in [0.99, 0.90, 0.80, 0.74, 0.50, 0.0]:
            level = classifier.classify(similarity)
            is_match = classifier.is_match(similarity)

            if level == ConfidenceLevel.NONE:
                assert is_match is False
            else:
                assert is_match is True
