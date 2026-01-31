"""Tests for the OpenAIEmbeddingProvider."""

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from llmgatekeeper.embeddings.openai_provider import (
    DEFAULT_MODEL,
    MODEL_DIMENSIONS,
    OpenAIEmbeddingProvider,
)


class TestOpenAIEmbeddingProviderInit:
    """Tests for OpenAIEmbeddingProvider initialization."""

    def test_raises_without_api_key(self, monkeypatch):
        """TC-3.3.2: Raises error if API key not set."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(ValueError, match="API key"):
            OpenAIEmbeddingProvider()

    def test_accepts_api_key_parameter(self, monkeypatch):
        """Can pass API key as parameter."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        provider = OpenAIEmbeddingProvider(api_key="test-key")
        assert provider._api_key == "test-key"

    def test_uses_environment_variable(self, monkeypatch):
        """Uses OPENAI_API_KEY environment variable."""
        monkeypatch.setenv("OPENAI_API_KEY", "env-key")
        provider = OpenAIEmbeddingProvider()
        assert provider._api_key == "env-key"

    def test_parameter_overrides_env(self, monkeypatch):
        """API key parameter overrides environment variable."""
        monkeypatch.setenv("OPENAI_API_KEY", "env-key")
        provider = OpenAIEmbeddingProvider(api_key="param-key")
        assert provider._api_key == "param-key"

    def test_default_model_is_ada_002(self, monkeypatch):
        """Default model is text-embedding-ada-002."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        provider = OpenAIEmbeddingProvider()
        assert provider.model_name == "text-embedding-ada-002"

    def test_custom_model(self, monkeypatch):
        """Can specify a custom model."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        provider = OpenAIEmbeddingProvider(model="text-embedding-3-large")
        assert provider.model_name == "text-embedding-3-large"

    def test_lazy_loading(self, monkeypatch):
        """Client is not created until first embed call."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        provider = OpenAIEmbeddingProvider()
        assert provider._client is None
        assert provider._async_client is None


class TestOpenAIEmbeddingProviderDimension:
    """Tests for dimension property."""

    def test_dimension_ada_002(self, monkeypatch):
        """TC-3.3.1: Returns correct dimension for ada-002."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        provider = OpenAIEmbeddingProvider(model="text-embedding-ada-002")
        assert provider.dimension == 1536

    def test_dimension_3_small(self, monkeypatch):
        """Returns correct dimension for text-embedding-3-small."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        provider = OpenAIEmbeddingProvider(model="text-embedding-3-small")
        assert provider.dimension == 1536

    def test_dimension_3_large(self, monkeypatch):
        """Returns correct dimension for text-embedding-3-large."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        provider = OpenAIEmbeddingProvider(model="text-embedding-3-large")
        assert provider.dimension == 3072

    def test_dimension_unknown_model_defaults_to_1536(self, monkeypatch):
        """Unknown model defaults to 1536 dimensions."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        provider = OpenAIEmbeddingProvider(model="some-future-model")
        assert provider.dimension == 1536


class TestOpenAIEmbeddingProviderEmbed:
    """Tests for embed method with mocked API."""

    @pytest.fixture
    def mock_openai_response(self):
        """Create a mock OpenAI embedding response."""
        mock_data = MagicMock()
        mock_data.embedding = [0.1] * 1536
        mock_data.index = 0

        mock_response = MagicMock()
        mock_response.data = [mock_data]
        return mock_response

    @pytest.fixture
    def provider(self, monkeypatch):
        """Create provider with test API key."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        return OpenAIEmbeddingProvider()

    def test_embed_returns_correct_shape(self, provider, mock_openai_response):
        """Embed returns numpy array of correct shape."""
        with patch.object(provider, "_get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.embeddings.create.return_value = mock_openai_response
            mock_get_client.return_value = mock_client

            embedding = provider.embed("Hello world")

            assert isinstance(embedding, np.ndarray)
            assert embedding.shape == (1536,)
            assert embedding.dtype == np.float32

    def test_embed_calls_api_correctly(self, provider, mock_openai_response):
        """Embed calls OpenAI API with correct parameters."""
        with patch.object(provider, "_get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.embeddings.create.return_value = mock_openai_response
            mock_get_client.return_value = mock_client

            provider.embed("Test query")

            mock_client.embeddings.create.assert_called_once_with(
                model="text-embedding-ada-002",
                input="Test query",
            )


class TestOpenAIEmbeddingProviderEmbedBatch:
    """Tests for embed_batch method with mocked API."""

    @pytest.fixture
    def mock_openai_batch_response(self):
        """Create a mock OpenAI batch embedding response."""
        mock_data = []
        for i in range(3):
            item = MagicMock()
            item.embedding = [0.1 * (i + 1)] * 1536
            item.index = i
            mock_data.append(item)

        mock_response = MagicMock()
        mock_response.data = mock_data
        return mock_response

    @pytest.fixture
    def provider(self, monkeypatch):
        """Create provider with test API key."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        return OpenAIEmbeddingProvider()

    def test_embed_batch_returns_list(self, provider, mock_openai_batch_response):
        """Batch embed returns list of embeddings."""
        with patch.object(provider, "_get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.embeddings.create.return_value = mock_openai_batch_response
            mock_get_client.return_value = mock_client

            embeddings = provider.embed_batch(["Hello", "World", "Test"])

            assert len(embeddings) == 3
            assert all(isinstance(e, np.ndarray) for e in embeddings)
            assert all(e.shape == (1536,) for e in embeddings)
            assert all(e.dtype == np.float32 for e in embeddings)

    def test_embed_batch_empty_list(self, provider):
        """Embed batch handles empty list."""
        embeddings = provider.embed_batch([])
        assert embeddings == []

    def test_embed_batch_preserves_order(self, provider):
        """Batch embed preserves input order."""
        # Create response with shuffled indices
        mock_data = []
        for i, idx in enumerate([2, 0, 1]):  # Shuffled order
            item = MagicMock()
            item.embedding = [float(idx)] * 1536  # Value matches original index
            item.index = idx
            mock_data.append(item)

        mock_response = MagicMock()
        mock_response.data = mock_data

        with patch.object(provider, "_get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.embeddings.create.return_value = mock_response
            mock_get_client.return_value = mock_client

            embeddings = provider.embed_batch(["a", "b", "c"])

            # Results should be sorted by index, so values should be 0.0, 1.0, 2.0
            assert embeddings[0][0] == pytest.approx(0.0)
            assert embeddings[1][0] == pytest.approx(1.0)
            assert embeddings[2][0] == pytest.approx(2.0)


class TestOpenAIEmbeddingProviderAsync:
    """Tests for async methods with mocked API."""

    @pytest.fixture
    def provider(self, monkeypatch):
        """Create provider with test API key."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        return OpenAIEmbeddingProvider()

    @pytest.fixture
    def mock_async_response(self):
        """Create a mock async OpenAI embedding response."""
        mock_data = MagicMock()
        mock_data.embedding = [0.5] * 1536
        mock_data.index = 0

        mock_response = MagicMock()
        mock_response.data = [mock_data]
        return mock_response

    @pytest.mark.asyncio
    async def test_aembed_returns_correct_shape(self, provider, mock_async_response):
        """TC-3.3.3: Async embed works."""
        with patch.object(provider, "_get_async_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.embeddings.create = AsyncMock(return_value=mock_async_response)
            mock_get_client.return_value = mock_client

            embedding = await provider.aembed("Test query")

            assert isinstance(embedding, np.ndarray)
            assert embedding.shape == (1536,)
            assert embedding.dtype == np.float32

    @pytest.mark.asyncio
    async def test_aembed_batch_returns_list(self, provider):
        """Async batch embed returns list of embeddings."""
        mock_data = []
        for i in range(2):
            item = MagicMock()
            item.embedding = [0.1 * (i + 1)] * 1536
            item.index = i
            mock_data.append(item)

        mock_response = MagicMock()
        mock_response.data = mock_data

        with patch.object(provider, "_get_async_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.embeddings.create = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            embeddings = await provider.aembed_batch(["query 1", "query 2"])

            assert len(embeddings) == 2
            assert all(isinstance(e, np.ndarray) for e in embeddings)
            assert all(e.shape == (1536,) for e in embeddings)

    @pytest.mark.asyncio
    async def test_aembed_batch_empty_list(self, provider):
        """Async embed batch handles empty list."""
        embeddings = await provider.aembed_batch([])
        assert embeddings == []


class TestModelDimensions:
    """Tests for model dimension constants."""

    def test_all_known_models_have_dimensions(self):
        """All expected models are in the dimension mapping."""
        expected_models = [
            "text-embedding-ada-002",
            "text-embedding-3-small",
            "text-embedding-3-large",
        ]
        for model in expected_models:
            assert model in MODEL_DIMENSIONS

    def test_default_model_is_defined(self):
        """Default model constant is defined."""
        assert DEFAULT_MODEL == "text-embedding-ada-002"
        assert DEFAULT_MODEL in MODEL_DIMENSIONS
