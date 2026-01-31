"""Abstract base class for embedding providers."""

from abc import ABC, abstractmethod
from typing import List

import numpy as np
from numpy.typing import NDArray


class EmbeddingProvider(ABC):
    """Abstract base class defining the embedding provider contract.

    All embedding providers must implement these methods to provide
    text-to-vector embedding capabilities. Providers can be synchronous
    or asynchronous, with async methods having default implementations
    that wrap the sync versions.
    """

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the dimension of the embedding vectors.

        Returns:
            The number of dimensions in the output embedding vectors.
        """
        pass

    @abstractmethod
    def embed(self, text: str) -> NDArray[np.float32]:
        """Embed a single text string into a vector.

        Args:
            text: The text to embed.

        Returns:
            A numpy array of shape (dimension,) containing the embedding.
        """
        pass

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[NDArray[np.float32]]:
        """Embed multiple text strings into vectors.

        This method may be more efficient than calling embed() multiple
        times, as it can batch API calls or utilize GPU parallelism.

        Args:
            texts: List of texts to embed.

        Returns:
            List of numpy arrays, each of shape (dimension,).
        """
        pass

    async def aembed(self, text: str) -> NDArray[np.float32]:
        """Asynchronously embed a single text string into a vector.

        Default implementation wraps the synchronous embed() method.
        Subclasses can override for true async support.

        Args:
            text: The text to embed.

        Returns:
            A numpy array of shape (dimension,) containing the embedding.
        """
        return self.embed(text)

    async def aembed_batch(self, texts: List[str]) -> List[NDArray[np.float32]]:
        """Asynchronously embed multiple text strings into vectors.

        Default implementation wraps the synchronous embed_batch() method.
        Subclasses can override for true async support.

        Args:
            texts: List of texts to embed.

        Returns:
            List of numpy arrays, each of shape (dimension,).
        """
        return self.embed_batch(texts)
