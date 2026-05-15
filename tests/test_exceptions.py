"""Tests for the exceptions module."""

import pytest

from llmgatekeeper.exceptions import (
    BackendConnectionError,
    BackendError,
    BackendTimeoutError,
    CacheError,
    ConfigurationError,
    EmbeddingError,
)


class TestCacheError:
    """Tests for the base CacheError exception."""

    def test_cache_error_is_exception(self):
        """CacheError inherits from Exception."""
        assert issubclass(CacheError, Exception)

    def test_cache_error_message(self):
        """CacheError stores message."""
        error = CacheError("Test error message")
        assert str(error) == "Test error message"


class TestBackendError:
    """Tests for the BackendError exception."""

    def test_backend_error_inherits_cache_error(self):
        """BackendError inherits from CacheError."""
        assert issubclass(BackendError, CacheError)

    def test_backend_error_message(self):
        """BackendError stores message."""
        error = BackendError("Redis connection failed")
        assert str(error) == "Redis connection failed"

    def test_backend_error_original_error(self):
        """TC-7.2.1: BackendError stores original error."""
        original = RuntimeError("Connection refused")
        error = BackendError("Redis connection failed", original_error=original)

        assert error.original_error is original
        assert str(error) == "Redis connection failed"

    def test_backend_error_without_original(self):
        """BackendError works without original error."""
        error = BackendError("Generic backend error")
        assert error.original_error is None

    def test_can_catch_as_cache_error(self):
        """BackendError can be caught as CacheError."""
        with pytest.raises(CacheError):
            raise BackendError("Test")


class TestEmbeddingError:
    """Tests for the EmbeddingError exception."""

    def test_embedding_error_inherits_cache_error(self):
        """EmbeddingError inherits from CacheError."""
        assert issubclass(EmbeddingError, CacheError)

    def test_embedding_error_message(self):
        """EmbeddingError stores message."""
        error = EmbeddingError("Model failed to load")
        assert str(error) == "Model failed to load"

    def test_embedding_error_original_error(self):
        """TC-7.2.2: EmbeddingError stores original error."""
        original = ValueError("Invalid input")
        error = EmbeddingError("Embedding generation failed", original_error=original)

        assert error.original_error is original

    def test_can_catch_as_cache_error(self):
        """EmbeddingError can be caught as CacheError."""
        with pytest.raises(CacheError):
            raise EmbeddingError("Test")


class TestConfigurationError:
    """Tests for the ConfigurationError exception."""

    def test_configuration_error_inherits_cache_error(self):
        """ConfigurationError inherits from CacheError."""
        assert issubclass(ConfigurationError, CacheError)

    def test_configuration_error_message(self):
        """ConfigurationError stores message."""
        error = ConfigurationError("Invalid threshold")
        assert str(error) == "Invalid threshold"


class TestBackendConnectionError:
    """Tests for the BackendConnectionError exception."""

    def test_connection_error_inherits_backend_error(self):
        """BackendConnectionError inherits from BackendError."""
        assert issubclass(BackendConnectionError, BackendError)

    def test_can_catch_as_cache_error(self):
        """BackendConnectionError can be caught as CacheError."""
        with pytest.raises(CacheError):
            raise BackendConnectionError("Test")

    def test_can_catch_as_backend_error(self):
        """BackendConnectionError can be caught as BackendError."""
        with pytest.raises(BackendError):
            raise BackendConnectionError("Test")


class TestBackendTimeoutError:
    """Tests for the BackendTimeoutError exception."""

    def test_timeout_error_inherits_backend_error(self):
        """BackendTimeoutError inherits from BackendError."""
        assert issubclass(BackendTimeoutError, BackendError)

    def test_can_catch_as_backend_error(self):
        """BackendTimeoutError can be caught as BackendError."""
        with pytest.raises(BackendError):
            raise BackendTimeoutError("Test")


class TestExceptionHierarchy:
    """Tests for the exception hierarchy."""

    def test_catching_all_library_errors(self):
        """All library errors can be caught with CacheError."""
        exceptions = [
            CacheError("base"),
            BackendError("backend"),
            EmbeddingError("embedding"),
            ConfigurationError("config"),
            BackendConnectionError("connection"),
            BackendTimeoutError("timeout"),
        ]

        for exc in exceptions:
            with pytest.raises(CacheError):
                raise exc

    def test_specific_catching(self):
        """Can catch specific exceptions."""
        # BackendError is not EmbeddingError
        with pytest.raises(BackendError):
            try:
                raise BackendError("test")
            except EmbeddingError:
                pytest.fail("Should not catch as EmbeddingError")
