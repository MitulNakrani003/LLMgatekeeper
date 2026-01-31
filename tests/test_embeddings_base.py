"""Tests for the EmbeddingProvider abstract base class."""

from typing import List

import numpy as np
import pytest
from numpy.typing import NDArray

from llmgatekeeper.embeddings.base import EmbeddingProvider


class TestEmbeddingProviderAbstract:
    """Tests for EmbeddingProvider ABC."""

    def test_embedding_provider_is_abstract(self):
        """TC-3.1.1: EmbeddingProvider cannot be instantiated directly."""
        with pytest.raises(TypeError):
            EmbeddingProvider()

    def test_has_required_methods(self):
        """TC-3.1.2: All required methods defined."""
        assert hasattr(EmbeddingProvider, "embed")
        assert hasattr(EmbeddingProvider, "embed_batch")
        assert hasattr(EmbeddingProvider, "dimension")

    def test_has_async_methods(self):
        """Async methods are defined."""
        assert hasattr(EmbeddingProvider, "aembed")
        assert hasattr(EmbeddingProvider, "aembed_batch")

    def test_required_methods_are_abstract(self):
        """Verify required methods are abstract."""
        abstract_methods = EmbeddingProvider.__abstractmethods__
        assert "embed" in abstract_methods
        assert "embed_batch" in abstract_methods
        assert "dimension" in abstract_methods

    def test_async_methods_are_not_abstract(self):
        """Async methods have default implementations."""
        abstract_methods = EmbeddingProvider.__abstractmethods__
        assert "aembed" not in abstract_methods
        assert "aembed_batch" not in abstract_methods


class ConcreteEmbeddingProvider(EmbeddingProvider):
    """Concrete implementation for testing."""

    def __init__(self, dim: int = 384):
        self._dimension = dim

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, text: str) -> NDArray[np.float32]:
        # Return a deterministic embedding based on text length
        np.random.seed(len(text))
        return np.random.rand(self._dimension).astype(np.float32)

    def embed_batch(self, texts: List[str]) -> List[NDArray[np.float32]]:
        return [self.embed(text) for text in texts]


class TestConcreteEmbeddingProvider:
    """Tests for a concrete implementation of EmbeddingProvider."""

    def test_concrete_provider_can_be_instantiated(self):
        """Concrete implementations can be instantiated."""
        provider = ConcreteEmbeddingProvider()
        assert provider is not None

    def test_dimension_property(self):
        """Dimension property returns correct value."""
        provider = ConcreteEmbeddingProvider(dim=512)
        assert provider.dimension == 512

    def test_embed_returns_correct_shape(self):
        """Embed returns array of correct shape."""
        provider = ConcreteEmbeddingProvider(dim=384)
        embedding = provider.embed("test text")

        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (384,)
        assert embedding.dtype == np.float32

    def test_embed_batch_returns_list(self):
        """Embed batch returns list of arrays."""
        provider = ConcreteEmbeddingProvider(dim=384)
        embeddings = provider.embed_batch(["text1", "text2", "text3"])

        assert isinstance(embeddings, list)
        assert len(embeddings) == 3
        assert all(isinstance(e, np.ndarray) for e in embeddings)
        assert all(e.shape == (384,) for e in embeddings)

    @pytest.mark.asyncio
    async def test_aembed_default_implementation(self):
        """Default aembed wraps sync embed."""
        provider = ConcreteEmbeddingProvider(dim=384)

        sync_result = provider.embed("test")
        async_result = await provider.aembed("test")

        assert np.array_equal(sync_result, async_result)

    @pytest.mark.asyncio
    async def test_aembed_batch_default_implementation(self):
        """Default aembed_batch wraps sync embed_batch."""
        provider = ConcreteEmbeddingProvider(dim=384)
        texts = ["a", "b", "c"]

        sync_results = provider.embed_batch(texts)
        async_results = await provider.aembed_batch(texts)

        assert len(sync_results) == len(async_results)
        for sync_r, async_r in zip(sync_results, async_results):
            assert np.array_equal(sync_r, async_r)


class TestEmbeddingProviderEdgeCases:
    """Edge case tests for EmbeddingProvider."""

    def test_empty_text(self):
        """Provider handles empty text."""
        provider = ConcreteEmbeddingProvider()
        embedding = provider.embed("")

        assert embedding.shape == (384,)

    def test_empty_batch(self):
        """Provider handles empty batch."""
        provider = ConcreteEmbeddingProvider()
        embeddings = provider.embed_batch([])

        assert embeddings == []

    def test_single_item_batch(self):
        """Provider handles single-item batch."""
        provider = ConcreteEmbeddingProvider()
        embeddings = provider.embed_batch(["single"])

        assert len(embeddings) == 1
        assert embeddings[0].shape == (384,)
