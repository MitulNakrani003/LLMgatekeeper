"""SentenceTransformer embedding provider using local models."""

from typing import List, Optional

import numpy as np
from numpy.typing import NDArray

from llmgatekeeper.embeddings.base import EmbeddingProvider

# Model dimension mapping for common models
MODEL_DIMENSIONS = {
    "all-MiniLM-L6-v2": 384,
    "all-MiniLM-L12-v2": 384,
    "all-mpnet-base-v2": 768,
    "paraphrase-MiniLM-L6-v2": 384,
    "paraphrase-mpnet-base-v2": 768,
}

DEFAULT_MODEL = "all-MiniLM-L6-v2"


class SentenceTransformerProvider(EmbeddingProvider):
    """Embedding provider using SentenceTransformers library.

    This provider uses local transformer models for generating embeddings.
    The default model is all-MiniLM-L6-v2 which produces 384-dimensional
    embeddings and offers a good balance of speed and quality.

    The model is loaded lazily on the first embed call to avoid loading
    heavy models when the provider is instantiated but not used.

    Example:
        >>> provider = SentenceTransformerProvider()
        >>> embedding = provider.embed("Hello world")
        >>> embedding.shape
        (384,)
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: Optional[str] = None,
    ) -> None:
        """Initialize the SentenceTransformer provider.

        Args:
            model_name: Name of the SentenceTransformer model to use.
                Default is "all-MiniLM-L6-v2".
            device: Device to run the model on ('cpu', 'cuda', 'mps', etc.).
                If None, will auto-detect the best available device.
        """
        self._model_name = model_name
        self._device = device
        self._model = None  # Lazy loading

        # Get dimension from known models or default
        self._dimension = MODEL_DIMENSIONS.get(model_name, 384)

    def _load_model(self) -> None:
        """Load the SentenceTransformer model lazily."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                self._model_name,
                device=self._device,
            )
            # Update dimension from actual model if available
            self._dimension = self._model.get_sentence_embedding_dimension()

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
        return self._model_name

    def embed(self, text: str) -> NDArray[np.float32]:
        """Embed a single text string into a vector.

        Args:
            text: The text to embed.

        Returns:
            A numpy array of shape (dimension,) containing the embedding.
        """
        self._load_model()
        embedding = self._model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embedding.astype(np.float32)

    def embed_batch(self, texts: List[str]) -> List[NDArray[np.float32]]:
        """Embed multiple text strings into vectors.

        This method is more efficient than calling embed() multiple times
        as it batches the encoding operation.

        Args:
            texts: List of texts to embed.

        Returns:
            List of numpy arrays, each of shape (dimension,).
        """
        if not texts:
            return []

        self._load_model()
        embeddings = self._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            batch_size=32,
        )
        return [emb.astype(np.float32) for emb in embeddings]
