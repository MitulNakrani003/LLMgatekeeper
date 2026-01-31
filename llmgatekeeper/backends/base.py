"""Abstract base class for cache storage backends."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, field_validator


class CacheEntry(BaseModel):
    """Represents a cached entry with its vector and metadata."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    key: str
    vector: NDArray[np.float32]
    metadata: Dict[str, Any]

    @field_validator("vector", mode="before")
    @classmethod
    def convert_to_numpy(cls, v: Union[list, NDArray]) -> NDArray[np.float32]:
        """Convert input to numpy float32 array."""
        if isinstance(v, np.ndarray):
            return v.astype(np.float32) if v.dtype != np.float32 else v
        return np.array(v, dtype=np.float32)


class SearchResult(BaseModel):
    """Represents a search result with similarity score."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    key: str
    similarity: float
    metadata: Dict[str, Any]
    vector: Optional[NDArray[np.float32]] = None


class CacheBackend(ABC):
    """Abstract base class defining the storage backend contract.

    All cache backends must implement these methods to provide
    vector storage and similarity search capabilities.
    """

    @abstractmethod
    def store_vector(
        self,
        key: str,
        vector: NDArray[np.float32],
        metadata: Dict[str, Any],
        ttl: Optional[int] = None,
    ) -> None:
        """Store a vector with associated metadata.

        Args:
            key: Unique identifier for this cache entry.
            vector: The embedding vector to store.
            metadata: Arbitrary metadata to store with the vector.
            ttl: Optional time-to-live in seconds.
        """
        pass

    @abstractmethod
    def search_similar(
        self,
        vector: NDArray[np.float32],
        threshold: float = 0.85,
        top_k: int = 1,
    ) -> List[SearchResult]:
        """Search for similar vectors above a similarity threshold.

        Args:
            vector: The query vector to search for.
            threshold: Minimum similarity score (0-1) for results.
            top_k: Maximum number of results to return.

        Returns:
            List of SearchResult objects sorted by similarity (descending).
        """
        pass

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete a cache entry by key.

        Args:
            key: The key of the entry to delete.

        Returns:
            True if the entry was deleted, False if it didn't exist.
        """
        pass

    @abstractmethod
    def get_by_key(self, key: str) -> Optional[CacheEntry]:
        """Retrieve a cache entry by its exact key.

        Args:
            key: The key to look up.

        Returns:
            The CacheEntry if found, None otherwise.
        """
        pass

    def clear(self) -> int:
        """Clear all entries from the cache.

        Returns:
            Number of entries deleted.
        """
        raise NotImplementedError("clear() is not implemented for this backend")

    def count(self) -> int:
        """Return the number of entries in the cache.

        Returns:
            Number of cached entries.
        """
        raise NotImplementedError("count() is not implemented for this backend")


class AsyncCacheBackend(ABC):
    """Abstract base class for async storage backends.

    This class provides the async interface for cache backends that support
    non-blocking operations with asyncio.
    """

    @abstractmethod
    async def store_vector(
        self,
        key: str,
        vector: NDArray[np.float32],
        metadata: Dict[str, Any],
        ttl: Optional[int] = None,
    ) -> None:
        """Store a vector with associated metadata asynchronously.

        Args:
            key: Unique identifier for this cache entry.
            vector: The embedding vector to store.
            metadata: Arbitrary metadata to store with the vector.
            ttl: Optional time-to-live in seconds.
        """
        pass

    @abstractmethod
    async def search_similar(
        self,
        vector: NDArray[np.float32],
        threshold: float = 0.85,
        top_k: int = 1,
    ) -> List[SearchResult]:
        """Search for similar vectors asynchronously.

        Args:
            vector: The query vector to search for.
            threshold: Minimum similarity score (0-1) for results.
            top_k: Maximum number of results to return.

        Returns:
            List of SearchResult objects sorted by similarity (descending).
        """
        pass

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete a cache entry asynchronously.

        Args:
            key: The key of the entry to delete.

        Returns:
            True if the entry was deleted, False if it didn't exist.
        """
        pass

    @abstractmethod
    async def get_by_key(self, key: str) -> Optional[CacheEntry]:
        """Retrieve a cache entry by its exact key asynchronously.

        Args:
            key: The key to look up.

        Returns:
            The CacheEntry if found, None otherwise.
        """
        pass

    async def clear(self) -> int:
        """Clear all entries from the cache asynchronously.

        Returns:
            Number of entries deleted.
        """
        raise NotImplementedError("clear() is not implemented for this backend")

    async def count(self) -> int:
        """Return the number of entries in the cache asynchronously.

        Returns:
            Number of cached entries.
        """
        raise NotImplementedError("count() is not implemented for this backend")
