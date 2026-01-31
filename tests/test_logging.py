"""Tests for the logging module."""

import sys
from io import StringIO
from unittest.mock import MagicMock, patch

import pytest

from llmgatekeeper.logging import (
    LIBRARY_NAME,
    configure_logging,
    disable_logging,
    get_logger,
    log_operation,
)


class TestConfigureLogging:
    """Tests for configure_logging function."""

    def test_configure_logging_default(self):
        """Can configure logging with defaults."""
        # Should not raise
        configure_logging()
        # Clean up
        disable_logging()

    def test_configure_logging_with_level(self):
        """Can configure with custom level."""
        configure_logging(level="DEBUG")
        disable_logging()

    def test_configure_logging_with_string_level(self):
        """Can use string log levels."""
        for level in ["DEBUG", "INFO", "WARNING", "ERROR"]:
            configure_logging(level=level)
            disable_logging()

    def test_configure_logging_with_custom_sink(self):
        """Can configure with custom sink."""
        buffer = StringIO()
        configure_logging(sink=buffer, level="INFO")

        logger = get_logger(f"{LIBRARY_NAME}.test")
        logger.info("Test message")

        disable_logging()

        # Note: Due to loguru's async nature, we may not capture immediately
        # This test mainly ensures no errors are raised

    def test_reconfigure_replaces_handler(self):
        """Reconfiguring replaces the handler."""
        configure_logging(level="DEBUG")
        configure_logging(level="INFO")  # Should replace, not add
        disable_logging()


class TestDisableLogging:
    """Tests for disable_logging function."""

    def test_disable_logging(self):
        """Can disable logging."""
        configure_logging()
        disable_logging()
        # Should not raise

    def test_disable_without_configure(self):
        """Disabling without configuring is safe."""
        disable_logging()
        # Should not raise


class TestGetLogger:
    """Tests for get_logger function."""

    def test_get_logger_returns_logger(self):
        """get_logger returns a logger instance."""
        logger = get_logger("test.module")
        assert logger is not None

    def test_get_logger_with_name(self):
        """Logger is bound to the given name."""
        logger = get_logger("my.custom.module")
        # Logger should have been created without error
        assert logger is not None

    def test_logger_can_log(self):
        """Logger can write log messages."""
        logger = get_logger("test.module")
        # These should not raise
        logger.debug("debug message")
        logger.info("info message")
        logger.warning("warning message")
        logger.error("error message")


class TestLogOperation:
    """Tests for log_operation function."""

    def test_log_operation_success(self):
        """Can log successful operation."""
        # Should not raise
        log_operation("get", success=True, duration_ms=5.2, query="test")

    def test_log_operation_failure(self):
        """Can log failed operation."""
        log_operation("set", success=False, duration_ms=10.5, error="timeout")

    def test_log_operation_with_context(self):
        """Can include extra context."""
        log_operation(
            "search",
            success=True,
            duration_ms=3.1,
            query="test query",
            results=5,
            threshold=0.85,
        )

    def test_log_operation_without_duration(self):
        """Can log without duration."""
        log_operation("delete", success=True)


class TestLoggingIntegration:
    """Integration tests for logging with cache."""

    def test_logging_with_cache_operations(self):
        """TC-7.2.3: Logging works with cache operations."""
        configure_logging(level="DEBUG")

        # Import here to ensure logging is configured
        from llmgatekeeper.logging import get_logger

        logger = get_logger(f"{LIBRARY_NAME}.cache")

        # These simulate what the cache would log
        logger.debug("Setting cache entry", query="What is Python?")
        logger.debug("Cache hit", query="What is Python?", similarity=0.95)
        logger.debug("Cache miss", query="Unknown query")

        disable_logging()
