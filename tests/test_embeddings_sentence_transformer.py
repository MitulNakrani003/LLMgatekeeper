"""Tests for the SentenceTransformerProvider."""

import numpy as np
import pytest

from llmgatekeeper.embeddings.sentence_transformer import SentenceTransformerProvider


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (norm1 * norm2))


@pytest.fixture(scope="module")
def provider():
    """Create a SentenceTransformerProvider for testing.

    Using module scope to avoid reloading the model for each test.
    """
    return SentenceTransformerProvider()


class TestSentenceTransformerProviderInit:
    """Tests for SentenceTransformerProvider initialization."""

    def test_default_model_is_minilm(self):
        """Default model is all-MiniLM-L6-v2."""
        provider = SentenceTransformerProvider()
        assert provider.model_name == "all-MiniLM-L6-v2"

    def test_custom_model_name(self):
        """Can specify a custom model name."""
        provider = SentenceTransformerProvider(model_name="all-mpnet-base-v2")
        assert provider.model_name == "all-mpnet-base-v2"

    def test_lazy_loading(self):
        """Model is not loaded until first embed call."""
        provider = SentenceTransformerProvider()
        assert provider._model is None


class TestSentenceTransformerProviderDimension:
    """Tests for dimension property."""

    def test_dimension_is_384(self, provider):
        """TC-3.2.1: Returns correct dimension."""
        assert provider.dimension == 384

    def test_dimension_before_loading(self):
        """Dimension is known even before model is loaded."""
        provider = SentenceTransformerProvider()
        assert provider.dimension == 384
        assert provider._model is None  # Still not loaded


class TestSentenceTransformerProviderEmbed:
    """Tests for embed method."""

    def test_embed_returns_correct_shape(self, provider):
        """TC-3.2.2: Embed returns numpy array of correct shape."""
        embedding = provider.embed("Hello world")

        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (384,)
        assert embedding.dtype == np.float32

    def test_embed_loads_model(self):
        """Embed call loads the model."""
        provider = SentenceTransformerProvider()
        assert provider._model is None

        provider.embed("test")
        assert provider._model is not None

    def test_embed_empty_string(self, provider):
        """Embed handles empty string."""
        embedding = provider.embed("")

        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (384,)

    def test_embed_long_text(self, provider):
        """Embed handles long text."""
        long_text = "This is a test. " * 100
        embedding = provider.embed(long_text)

        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (384,)

    def test_embed_is_normalized(self, provider):
        """Embeddings are normalized (unit length)."""
        embedding = provider.embed("Hello world")
        norm = np.linalg.norm(embedding)

        assert norm == pytest.approx(1.0, abs=0.01)


class TestSentenceTransformerProviderEmbedBatch:
    """Tests for embed_batch method."""

    def test_embed_batch(self, provider):
        """TC-3.2.3: Batch embed returns list of embeddings."""
        embeddings = provider.embed_batch(["Hello", "World", "Test"])

        assert len(embeddings) == 3
        assert all(isinstance(e, np.ndarray) for e in embeddings)
        assert all(e.shape == (384,) for e in embeddings)
        assert all(e.dtype == np.float32 for e in embeddings)

    def test_embed_batch_empty_list(self, provider):
        """Embed batch handles empty list."""
        embeddings = provider.embed_batch([])

        assert embeddings == []

    def test_embed_batch_single_item(self, provider):
        """Embed batch handles single item."""
        embeddings = provider.embed_batch(["single"])

        assert len(embeddings) == 1
        assert embeddings[0].shape == (384,)

    def test_embed_batch_consistency(self, provider):
        """Batch embed produces same results as individual embeds."""
        texts = ["Hello", "World"]

        individual = [provider.embed(t) for t in texts]
        batch = provider.embed_batch(texts)

        for ind, bat in zip(individual, batch):
            assert np.allclose(ind, bat, atol=1e-5)


class TestSentenceTransformerProviderSimilarity:
    """Tests for semantic similarity."""

    def test_similar_texts_have_high_similarity(self, provider):
        """TC-3.2.4: Similar texts have high cosine similarity."""
        e1 = provider.embed("What is the weather today?")
        e2 = provider.embed("Tell me today's weather")
        e3 = provider.embed("How to cook pasta")

        sim_12 = cosine_similarity(e1, e2)
        sim_13 = cosine_similarity(e1, e3)

        assert sim_12 > 0.8  # Similar queries
        assert sim_13 < 0.5  # Different topics

    def test_identical_texts_have_similarity_1(self, provider):
        """Identical texts have similarity of 1.0."""
        text = "This is a test sentence."
        e1 = provider.embed(text)
        e2 = provider.embed(text)

        sim = cosine_similarity(e1, e2)
        assert sim == pytest.approx(1.0, abs=0.001)

    def test_related_topics_have_moderate_similarity(self, provider):
        """Related but different topics have moderate similarity."""
        e1 = provider.embed("Python programming language")
        e2 = provider.embed("JavaScript programming language")
        e3 = provider.embed("Chocolate cake recipe")

        sim_12 = cosine_similarity(e1, e2)
        sim_13 = cosine_similarity(e1, e3)

        # Programming languages should be more similar to each other than unrelated topics
        assert sim_12 > sim_13
        assert sim_12 > 0.3  # At least some similarity between programming languages


class TestSentenceTransformerProviderAsync:
    """Tests for async methods."""

    @pytest.mark.asyncio
    async def test_aembed(self, provider):
        """Async embed returns same result as sync."""
        sync_result = provider.embed("test query")
        async_result = await provider.aembed("test query")

        assert np.array_equal(sync_result, async_result)

    @pytest.mark.asyncio
    async def test_aembed_batch(self, provider):
        """Async batch embed returns same result as sync."""
        texts = ["query 1", "query 2"]

        sync_results = provider.embed_batch(texts)
        async_results = await provider.aembed_batch(texts)

        assert len(sync_results) == len(async_results)
        for sync_r, async_r in zip(sync_results, async_results):
            assert np.array_equal(sync_r, async_r)
