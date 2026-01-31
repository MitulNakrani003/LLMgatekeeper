"""Tests for the exceptions module."""

import pytest

from llmgatekeeper.exceptions import (
    BackendError,
    CacheError,
    ConfigurationError,
    ConnectionError,
    EmbeddingError,
    TimeoutError,
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


class TestConnectionError:
    """Tests for the ConnectionError exception."""

    def test_connection_error_inherits_backend_error(self):
        """ConnectionError inherits from BackendError."""
        assert issubclass(ConnectionError, BackendError)

    def test_can_catch_as_cache_error(self):
        """ConnectionError can be caught as CacheError."""
        with pytest.raises(CacheError):
            raise ConnectionError("Test")

    def test_can_catch_as_backend_error(self):
        """ConnectionError can be caught as BackendError."""
        with pytest.raises(BackendError):
            raise ConnectionError("Test")


class TestTimeoutError:
    """Tests for the TimeoutError exception."""

    def test_timeout_error_inherits_backend_error(self):
        """TimeoutError inherits from BackendError."""
        assert issubclass(TimeoutError, BackendError)

    def test_can_catch_as_backend_error(self):
        """TimeoutError can be caught as BackendError."""
        with pytest.raises(BackendError):
            raise TimeoutError("Test")


class TestExceptionHierarchy:
    """Tests for the exception hierarchy."""

    def test_catching_all_library_errors(self):
        """All library errors can be caught with CacheError."""
        exceptions = [
            CacheError("base"),
            BackendError("backend"),
            EmbeddingError("embedding"),
            ConfigurationError("config"),
            ConnectionError("connection"),
            TimeoutError("timeout"),
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
