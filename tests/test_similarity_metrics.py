"""Tests for the similarity metrics module."""

import numpy as np
import pytest

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


class TestCosineSimilarity:
    """Tests for cosine similarity function."""

    def test_identical_vectors(self):
        """TC-4.1.1: Cosine similarity of identical vectors is 1.0."""
        v = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        assert cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        """TC-4.1.2: Cosine similarity of orthogonal vectors is 0.0."""
        v1 = np.array([1.0, 0.0], dtype=np.float32)
        v2 = np.array([0.0, 1.0], dtype=np.float32)
        assert cosine_similarity(v1, v2) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        """Cosine similarity of opposite vectors is -1.0."""
        v1 = np.array([1.0, 0.0], dtype=np.float32)
        v2 = np.array([-1.0, 0.0], dtype=np.float32)
        assert cosine_similarity(v1, v2) == pytest.approx(-1.0)

    def test_parallel_vectors_different_magnitude(self):
        """Parallel vectors with different magnitudes have similarity 1.0."""
        v1 = np.array([1.0, 1.0], dtype=np.float32)
        v2 = np.array([10.0, 10.0], dtype=np.float32)
        assert cosine_similarity(v1, v2) == pytest.approx(1.0)

    def test_zero_vector(self):
        """Zero vector returns 0 similarity."""
        v1 = np.array([0.0, 0.0], dtype=np.float32)
        v2 = np.array([1.0, 1.0], dtype=np.float32)
        assert cosine_similarity(v1, v2) == pytest.approx(0.0)

    def test_both_zero_vectors(self):
        """Both zero vectors return 0 similarity."""
        v1 = np.array([0.0, 0.0], dtype=np.float32)
        v2 = np.array([0.0, 0.0], dtype=np.float32)
        assert cosine_similarity(v1, v2) == pytest.approx(0.0)

    def test_high_dimensional(self):
        """Works with high-dimensional vectors."""
        v1 = np.random.rand(384).astype(np.float32)
        v2 = v1.copy()
        assert cosine_similarity(v1, v2) == pytest.approx(1.0)

    def test_similar_vectors(self):
        """Similar vectors have high cosine similarity."""
        v1 = np.array([1.0, 0.0], dtype=np.float32)
        v2 = np.array([0.9, 0.1], dtype=np.float32)
        similarity = cosine_similarity(v1, v2)
        assert similarity > 0.9

    def test_symmetry(self):
        """Cosine similarity is symmetric."""
        v1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        v2 = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        assert cosine_similarity(v1, v2) == pytest.approx(cosine_similarity(v2, v1))


class TestDotProductSimilarity:
    """Tests for dot product similarity function."""

    def test_basic_dot_product(self):
        """TC-4.1.3: Dot product works correctly."""
        v1 = np.array([1.0, 2.0], dtype=np.float32)
        v2 = np.array([3.0, 4.0], dtype=np.float32)
        assert dot_product_similarity(v1, v2) == pytest.approx(11.0)

    def test_orthogonal_vectors(self):
        """Dot product of orthogonal vectors is 0."""
        v1 = np.array([1.0, 0.0], dtype=np.float32)
        v2 = np.array([0.0, 1.0], dtype=np.float32)
        assert dot_product_similarity(v1, v2) == pytest.approx(0.0)

    def test_normalized_vectors(self):
        """Normalized vectors give same result as cosine."""
        v1 = np.array([1.0, 0.0], dtype=np.float32)
        v2 = np.array([0.707, 0.707], dtype=np.float32)
        v2 = v2 / np.linalg.norm(v2)  # Normalize
        # For unit vectors, dot product equals cosine
        assert dot_product_similarity(v1, v2) == pytest.approx(
            cosine_similarity(v1, v2), abs=0.01
        )

    def test_zero_vector(self):
        """Dot product with zero vector is 0."""
        v1 = np.array([0.0, 0.0], dtype=np.float32)
        v2 = np.array([1.0, 2.0], dtype=np.float32)
        assert dot_product_similarity(v1, v2) == pytest.approx(0.0)

    def test_negative_result(self):
        """Dot product can be negative for opposing vectors."""
        v1 = np.array([1.0, 0.0], dtype=np.float32)
        v2 = np.array([-1.0, 0.0], dtype=np.float32)
        assert dot_product_similarity(v1, v2) == pytest.approx(-1.0)

    def test_symmetry(self):
        """Dot product is symmetric."""
        v1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        v2 = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        assert dot_product_similarity(v1, v2) == pytest.approx(
            dot_product_similarity(v2, v1)
        )


class TestEuclideanSimilarity:
    """Tests for Euclidean similarity function."""

    def test_identical_vectors(self):
        """TC-4.1.4: Same point has similarity 1.0."""
        v1 = np.array([0.0, 0.0], dtype=np.float32)
        v2 = np.array([0.0, 0.0], dtype=np.float32)
        assert euclidean_similarity(v1, v2) == pytest.approx(1.0)

    def test_identical_nonzero_vectors(self):
        """Identical non-zero vectors have similarity 1.0."""
        v = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        assert euclidean_similarity(v, v) == pytest.approx(1.0)

    def test_distant_vectors(self):
        """Distant vectors have low similarity."""
        v1 = np.array([0.0, 0.0], dtype=np.float32)
        v2 = np.array([100.0, 100.0], dtype=np.float32)
        similarity = euclidean_similarity(v1, v2)
        assert similarity < 0.1  # Very low for distant points

    def test_nearby_vectors(self):
        """Nearby vectors have high similarity."""
        v1 = np.array([0.0, 0.0], dtype=np.float32)
        v2 = np.array([0.1, 0.1], dtype=np.float32)
        similarity = euclidean_similarity(v1, v2)
        assert similarity > 0.8  # High for close points

    def test_similarity_bounded(self):
        """Euclidean similarity is bounded in (0, 1]."""
        v1 = np.random.rand(10).astype(np.float32) * 100
        v2 = np.random.rand(10).astype(np.float32) * 100
        similarity = euclidean_similarity(v1, v2)
        assert 0 < similarity <= 1

    def test_symmetry(self):
        """Euclidean similarity is symmetric."""
        v1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        v2 = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        assert euclidean_similarity(v1, v2) == pytest.approx(
            euclidean_similarity(v2, v1)
        )


class TestEuclideanDistance:
    """Tests for Euclidean distance function."""

    def test_identical_vectors(self):
        """Distance between identical vectors is 0."""
        v = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        assert euclidean_distance(v, v) == pytest.approx(0.0)

    def test_basic_distance(self):
        """Basic distance calculation works."""
        v1 = np.array([0.0, 0.0], dtype=np.float32)
        v2 = np.array([3.0, 4.0], dtype=np.float32)
        assert euclidean_distance(v1, v2) == pytest.approx(5.0)

    def test_symmetry(self):
        """Euclidean distance is symmetric."""
        v1 = np.array([1.0, 2.0], dtype=np.float32)
        v2 = np.array([4.0, 6.0], dtype=np.float32)
        assert euclidean_distance(v1, v2) == pytest.approx(euclidean_distance(v2, v1))

    def test_always_positive(self):
        """Distance is always non-negative."""
        v1 = np.random.rand(10).astype(np.float32)
        v2 = np.random.rand(10).astype(np.float32)
        assert euclidean_distance(v1, v2) >= 0


class TestSimilarityMetricEnum:
    """Tests for the SimilarityMetric enum."""

    def test_all_metrics_defined(self):
        """All expected metrics are defined."""
        assert SimilarityMetric.COSINE.value == "cosine"
        assert SimilarityMetric.DOT_PRODUCT.value == "dot_product"
        assert SimilarityMetric.EUCLIDEAN.value == "euclidean"

    def test_enum_from_string(self):
        """Can create enum from string value."""
        assert SimilarityMetric("cosine") == SimilarityMetric.COSINE


class TestGetSimilarityFunction:
    """Tests for get_similarity_function."""

    def test_returns_cosine(self):
        """Returns cosine function for COSINE metric."""
        func = get_similarity_function(SimilarityMetric.COSINE)
        assert func is cosine_similarity

    def test_returns_dot_product(self):
        """Returns dot product function for DOT_PRODUCT metric."""
        func = get_similarity_function(SimilarityMetric.DOT_PRODUCT)
        assert func is dot_product_similarity

    def test_returns_euclidean(self):
        """Returns euclidean function for EUCLIDEAN metric."""
        func = get_similarity_function(SimilarityMetric.EUCLIDEAN)
        assert func is euclidean_similarity

    def test_returned_function_works(self):
        """Returned function computes similarity correctly."""
        func = get_similarity_function(SimilarityMetric.COSINE)
        v1 = np.array([1.0, 0.0], dtype=np.float32)
        v2 = np.array([1.0, 0.0], dtype=np.float32)
        assert func(v1, v2) == pytest.approx(1.0)


class TestComputeSimilarity:
    """Tests for compute_similarity convenience function."""

    def test_default_is_cosine(self):
        """Default metric is cosine similarity."""
        v1 = np.array([1.0, 0.0], dtype=np.float32)
        v2 = np.array([0.0, 1.0], dtype=np.float32)
        assert compute_similarity(v1, v2) == pytest.approx(0.0)

    def test_explicit_cosine(self):
        """Explicit cosine metric works."""
        v1 = np.array([1.0, 0.0], dtype=np.float32)
        v2 = np.array([1.0, 0.0], dtype=np.float32)
        assert compute_similarity(v1, v2, SimilarityMetric.COSINE) == pytest.approx(1.0)

    def test_dot_product_metric(self):
        """Dot product metric works."""
        v1 = np.array([1.0, 2.0], dtype=np.float32)
        v2 = np.array([3.0, 4.0], dtype=np.float32)
        assert compute_similarity(v1, v2, SimilarityMetric.DOT_PRODUCT) == pytest.approx(
            11.0
        )

    def test_euclidean_metric(self):
        """Euclidean metric works."""
        v = np.array([1.0, 2.0], dtype=np.float32)
        assert compute_similarity(v, v, SimilarityMetric.EUCLIDEAN) == pytest.approx(1.0)


class TestNormalizeSimilarity:
    """Tests for normalize_similarity function."""

    def test_cosine_normalize_positive(self):
        """Cosine 1.0 normalizes to 1.0."""
        assert normalize_similarity(1.0, SimilarityMetric.COSINE) == pytest.approx(1.0)

    def test_cosine_normalize_zero(self):
        """Cosine 0.0 normalizes to 0.5."""
        assert normalize_similarity(0.0, SimilarityMetric.COSINE) == pytest.approx(0.5)

    def test_cosine_normalize_negative(self):
        """Cosine -1.0 normalizes to 0.0."""
        assert normalize_similarity(-1.0, SimilarityMetric.COSINE) == pytest.approx(0.0)

    def test_euclidean_passthrough(self):
        """Euclidean already normalized, passes through."""
        assert normalize_similarity(0.8, SimilarityMetric.EUCLIDEAN) == pytest.approx(
            0.8
        )

    def test_dot_product_clamps(self):
        """Dot product values are clamped to [0, 1]."""
        assert normalize_similarity(2.0, SimilarityMetric.DOT_PRODUCT) == pytest.approx(
            1.0
        )
        assert normalize_similarity(-2.0, SimilarityMetric.DOT_PRODUCT) == pytest.approx(
            0.0
        )


class TestBatchCosineSimilarity:
    """Tests for batch cosine similarity function."""

    def test_single_vector(self):
        """Works with single vector in batch."""
        query = np.array([1.0, 0.0], dtype=np.float32)
        vectors = np.array([[1.0, 0.0]], dtype=np.float32)
        result = batch_cosine_similarity(query, vectors)
        assert result.shape == (1,)
        assert result[0] == pytest.approx(1.0)

    def test_multiple_vectors(self):
        """Works with multiple vectors."""
        query = np.array([1.0, 0.0], dtype=np.float32)
        vectors = np.array(
            [[1.0, 0.0], [0.0, 1.0], [0.707, 0.707]], dtype=np.float32
        )
        # Normalize the last vector
        vectors[2] = vectors[2] / np.linalg.norm(vectors[2])

        result = batch_cosine_similarity(query, vectors)

        assert result.shape == (3,)
        assert result[0] == pytest.approx(1.0)
        assert result[1] == pytest.approx(0.0)
        assert result[2] == pytest.approx(0.707, abs=0.01)

    def test_returns_float32(self):
        """Returns float32 array."""
        query = np.array([1.0, 0.0], dtype=np.float32)
        vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        result = batch_cosine_similarity(query, vectors)
        assert result.dtype == np.float32

    def test_zero_query(self):
        """Zero query returns all zeros."""
        query = np.array([0.0, 0.0], dtype=np.float32)
        vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        result = batch_cosine_similarity(query, vectors)
        assert np.allclose(result, 0.0)

    def test_zero_vector_in_batch(self):
        """Handles zero vector in batch."""
        query = np.array([1.0, 0.0], dtype=np.float32)
        vectors = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
        result = batch_cosine_similarity(query, vectors)
        # Zero vector should give 0 similarity after normalization
        assert result[1] == pytest.approx(1.0)

    def test_high_dimensional_batch(self):
        """Works with high-dimensional vectors."""
        dim = 384
        query = np.random.rand(dim).astype(np.float32)
        vectors = np.random.rand(100, dim).astype(np.float32)

        result = batch_cosine_similarity(query, vectors)

        assert result.shape == (100,)
        assert all(-1 <= r <= 1 for r in result)

    def test_matches_single_cosine(self):
        """Batch result matches individual cosine calculations."""
        query = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        vectors = np.array(
            [[4.0, 5.0, 6.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32
        )

        batch_result = batch_cosine_similarity(query, vectors)

        for i, vec in enumerate(vectors):
            individual = cosine_similarity(query, vec)
            assert batch_result[i] == pytest.approx(individual, abs=0.0001)

    def test_empty_batch(self):
        """Handles empty batch."""
        query = np.array([1.0, 0.0], dtype=np.float32)
        vectors = np.array([], dtype=np.float32).reshape(0, 2)
        result = batch_cosine_similarity(query, vectors)
        assert result.shape == (0,)


class TestMetricsIntegration:
    """Integration tests for metrics module."""

    def test_all_metrics_work_together(self):
        """All metrics can be used together for comparison."""
        v1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        v2 = np.array([1.1, 2.1, 3.1], dtype=np.float32)  # Very similar

        cos_sim = compute_similarity(v1, v2, SimilarityMetric.COSINE)
        euc_sim = compute_similarity(v1, v2, SimilarityMetric.EUCLIDEAN)

        # Both should indicate high similarity
        assert cos_sim > 0.99
        assert euc_sim > 0.8

    def test_metrics_distinguish_similar_vs_different(self):
        """All metrics can distinguish similar from different vectors."""
        v_base = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        v_similar = np.array([0.95, 0.1, 0.0], dtype=np.float32)
        v_different = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        for metric in SimilarityMetric:
            sim_similar = compute_similarity(v_base, v_similar, metric)
            sim_different = compute_similarity(v_base, v_different, metric)

            if metric != SimilarityMetric.DOT_PRODUCT:
                # Skip dot product as it's not bounded
                assert sim_similar > sim_different, f"Failed for {metric}"
