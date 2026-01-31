"""Custom exceptions for LLMGatekeeper.

This module provides a hierarchy of exceptions for error handling:
- CacheError: Base exception for all cache-related errors
- BackendError: Errors from the storage backend (e.g., Redis)
- EmbeddingError: Errors from the embedding provider
- ConfigurationError: Invalid configuration or parameters
"""


class CacheError(Exception):
    """Base exception for all LLMGatekeeper errors.

    All exceptions raised by LLMGatekeeper inherit from this class,
    allowing users to catch all library errors with a single handler.

    Example:
        >>> try:
        ...     cache.set("query", "response")
        ... except CacheError as e:
        ...     print(f"Cache operation failed: {e}")
    """

    pass


class BackendError(CacheError):
    """Exception raised when the storage backend encounters an error.

    This includes Redis connection failures, timeouts, and other
    backend-specific errors.

    Attributes:
        original_error: The underlying exception from the backend.

    Example:
        >>> try:
        ...     cache.set("query", "response")
        ... except BackendError as e:
        ...     print(f"Backend error: {e}")
        ...     print(f"Original: {e.original_error}")
    """

    def __init__(self, message: str, original_error: Exception = None) -> None:
        """Initialize BackendError.

        Args:
            message: Human-readable error description.
            original_error: The underlying exception that caused this error.
        """
        super().__init__(message)
        self.original_error = original_error


class EmbeddingError(CacheError):
    """Exception raised when embedding generation fails.

    This includes model loading failures, API errors for remote
    embedding providers, and invalid input errors.

    Attributes:
        original_error: The underlying exception from the provider.

    Example:
        >>> try:
        ...     cache.set("query", "response")
        ... except EmbeddingError as e:
        ...     print(f"Embedding failed: {e}")
    """

    def __init__(self, message: str, original_error: Exception = None) -> None:
        """Initialize EmbeddingError.

        Args:
            message: Human-readable error description.
            original_error: The underlying exception that caused this error.
        """
        super().__init__(message)
        self.original_error = original_error


class ConfigurationError(CacheError):
    """Exception raised for invalid configuration.

    This is raised when the cache is initialized with invalid
    parameters or when required dependencies are missing.

    Example:
        >>> try:
        ...     cache = SemanticCache(redis_client, threshold=2.0)
        ... except ConfigurationError as e:
        ...     print(f"Invalid config: {e}")
    """

    pass


class ConnectionError(BackendError):
    """Exception raised when unable to connect to the backend.

    This is a specific type of BackendError for connection failures,
    allowing more targeted error handling.

    Example:
        >>> try:
        ...     cache.set("query", "response")
        ... except ConnectionError as e:
        ...     print("Check your Redis connection")
    """

    pass


class TimeoutError(BackendError):
    """Exception raised when a backend operation times out.

    This is a specific type of BackendError for timeout scenarios.

    Example:
        >>> try:
        ...     cache.get("query")
        ... except TimeoutError as e:
        ...     print("Operation timed out, try again")
    """

    pass
