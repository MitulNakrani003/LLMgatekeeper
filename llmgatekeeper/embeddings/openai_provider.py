"""OpenAI embedding provider using the OpenAI API."""

import os
from typing import List, Optional

import numpy as np
from numpy.typing import NDArray

from llmgatekeeper.embeddings.base import EmbeddingProvider

# Model dimension mapping for OpenAI models
MODEL_DIMENSIONS = {
    "text-embedding-ada-002": 1536,
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
}

DEFAULT_MODEL = "text-embedding-ada-002"


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """Embedding provider using OpenAI's embedding API.

    This provider uses the OpenAI API to generate embeddings. It supports
    various embedding models including text-embedding-ada-002 and the
    newer text-embedding-3 models.

    The API key can be provided directly or via the OPENAI_API_KEY
    environment variable.

    Example:
        >>> provider = OpenAIEmbeddingProvider()
        >>> embedding = provider.embed("Hello world")
        >>> embedding.shape
        (1536,)

        >>> # With custom model
        >>> provider = OpenAIEmbeddingProvider(model="text-embedding-3-large")
        >>> provider.dimension
        3072
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
    ) -> None:
        """Initialize the OpenAI embedding provider.

        Args:
            model: Name of the OpenAI embedding model to use.
                Default is "text-embedding-ada-002".
            api_key: OpenAI API key. If not provided, will look for
                OPENAI_API_KEY environment variable.

        Raises:
            ValueError: If API key is not provided and not found in environment.
        """
        self._model = model
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")

        if not self._api_key:
            raise ValueError(
                "OpenAI API key not provided. Set OPENAI_API_KEY environment "
                "variable or pass api_key parameter."
            )

        # Get dimension from known models or default
        self._dimension = MODEL_DIMENSIONS.get(model, 1536)

        # Lazy-loaded client
        self._client = None
        self._async_client = None

    def _get_client(self):
        """Get or create the OpenAI client."""
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(api_key=self._api_key)
        return self._client

    def _get_async_client(self):
        """Get or create the async OpenAI client."""
        if self._async_client is None:
            from openai import AsyncOpenAI

            self._async_client = AsyncOpenAI(api_key=self._api_key)
        return self._async_client

    @property
    def dimension(self) -> int:
        """Return the dimension of the embedding vectors.

        Returns:
            The number of dimensions in the output embedding vectors.
        """
        return self._dimension

    @property
    def model_name(self) -> str:
        """Return the name of the model being used.

        Returns:
            The model name string.
        """
        return self._model

    def embed(self, text: str) -> NDArray[np.float32]:
        """Embed a single text string into a vector.

        Args:
            text: The text to embed.

        Returns:
            A numpy array of shape (dimension,) containing the embedding.
        """
        client = self._get_client()
        response = client.embeddings.create(
            model=self._model,
            input=text,
        )
        embedding = response.data[0].embedding
        return np.array(embedding, dtype=np.float32)

    def embed_batch(self, texts: List[str]) -> List[NDArray[np.float32]]:
        """Embed multiple text strings into vectors.

        This method is more efficient than calling embed() multiple times
        as it batches the API call.

        Args:
            texts: List of texts to embed.

        Returns:
            List of numpy arrays, each of shape (dimension,).
        """
        if not texts:
            return []

        client = self._get_client()
        response = client.embeddings.create(
            model=self._model,
            input=texts,
        )

        # Sort by index to ensure order matches input
        sorted_data = sorted(response.data, key=lambda x: x.index)
        return [np.array(item.embedding, dtype=np.float32) for item in sorted_data]

    async def aembed(self, text: str) -> NDArray[np.float32]:
        """Asynchronously embed a single text string into a vector.

        Args:
            text: The text to embed.

        Returns:
            A numpy array of shape (dimension,) containing the embedding.
        """
        client = self._get_async_client()
        response = await client.embeddings.create(
            model=self._model,
            input=text,
        )
        embedding = response.data[0].embedding
        return np.array(embedding, dtype=np.float32)

    async def aembed_batch(self, texts: List[str]) -> List[NDArray[np.float32]]:
        """Asynchronously embed multiple text strings into vectors.

        Args:
            texts: List of texts to embed.

        Returns:
            List of numpy arrays, each of shape (dimension,).
        """
        if not texts:
            return []

        client = self._get_async_client()
        response = await client.embeddings.create(
            model=self._model,
            input=texts,
        )

        # Sort by index to ensure order matches input
        sorted_data = sorted(response.data, key=lambda x: x.index)
        return [np.array(item.embedding, dtype=np.float32) for item in sorted_data]
