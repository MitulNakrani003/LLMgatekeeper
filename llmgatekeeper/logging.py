"""Logging configuration for LLMGatekeeper.

This module provides logging utilities using Loguru for structured
logging with configurable levels and formats.

The library uses a disabled-by-default approach to logging, allowing
users to opt-in when they need debugging information.

Example:
    >>> from llmgatekeeper.logging import configure_logging, get_logger
    >>>
    >>> # Enable logging at INFO level
    >>> configure_logging(level="INFO")
    >>>
    >>> # Or enable debug logging
    >>> configure_logging(level="DEBUG")
    >>>
    >>> # Get a logger for your module
    >>> logger = get_logger(__name__)
    >>> logger.info("Cache initialized")
"""

import sys
from typing import Optional, Union

from loguru import logger

# Library logger name
LIBRARY_NAME = "llmgatekeeper"

# Remove the default loguru handler to prevent unwanted output
# Users must explicitly call configure_logging() to enable logging
logger.remove()

# Default format for log messages
DEFAULT_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>"
)

# Simpler format for production
SIMPLE_FORMAT = "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}"

# Track if logging has been configured
_configured = False
_handler_id: Optional[int] = None


def configure_logging(
    level: Union[str, int] = "INFO",
    format: Optional[str] = None,
    sink=sys.stderr,
    colorize: bool = True,
    serialize: bool = False,
) -> None:
    """Configure library logging.

    Call this function to enable logging output from LLMGatekeeper.
    By default, the library does not emit any logs.

    Args:
        level: Minimum log level to display. Can be a string like "DEBUG",
            "INFO", "WARNING", "ERROR" or an integer. Default "INFO".
        format: Custom log format string. If None, uses the default format.
        sink: Where to write logs. Default is stderr.
        colorize: Whether to colorize output. Default True.
        serialize: If True, output logs as JSON. Default False.

    Example:
        >>> # Basic configuration
        >>> configure_logging()
        >>>
        >>> # Debug level with JSON output
        >>> configure_logging(level="DEBUG", serialize=True)
        >>>
        >>> # Custom format
        >>> configure_logging(format="{time} - {level} - {message}")
    """
    global _configured, _handler_id

    # Remove any existing handler
    if _handler_id is not None:
        logger.remove(_handler_id)

    # Use provided format or default
    log_format = format or (DEFAULT_FORMAT if not serialize else None)

    # Add the new handler
    _handler_id = logger.add(
        sink,
        level=level,
        format=log_format,
        colorize=colorize,
        serialize=serialize,
        filter=lambda record: record["name"].startswith(LIBRARY_NAME),
    )

    _configured = True


def disable_logging() -> None:
    """Disable all library logging.

    Call this to completely silence the library's log output.

    Example:
        >>> configure_logging(level="DEBUG")
        >>> # ... some operations ...
        >>> disable_logging()  # Silence logs
    """
    global _configured, _handler_id

    if _handler_id is not None:
        logger.remove(_handler_id)
        _handler_id = None

    _configured = False


def get_logger(name: str) -> "logger":
    """Get a logger instance for a module.

    Args:
        name: The module name, typically __name__.

    Returns:
        A Loguru logger instance bound to the given name.

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Cache hit", query="What is Python?", similarity=0.95)
    """
    return logger.bind(name=name)


# Default logger for the library
_default_logger = get_logger(LIBRARY_NAME)


def log_operation(
    operation: str,
    success: bool,
    duration_ms: Optional[float] = None,
    **context,
) -> None:
    """Log a cache operation with context.

    This is a convenience function for logging cache operations
    with consistent formatting.

    Args:
        operation: The operation name (e.g., "get", "set", "delete").
        success: Whether the operation succeeded.
        duration_ms: Operation duration in milliseconds.
        **context: Additional context to include in the log.

    Example:
        >>> log_operation("get", success=True, duration_ms=5.2, query="test")
    """
    level = "INFO" if success else "WARNING"

    message = f"Cache {operation}"
    if duration_ms is not None:
        message += f" ({duration_ms:.2f}ms)"

    if success:
        _default_logger.opt(depth=1).log(level, message, **context)
    else:
        _default_logger.opt(depth=1).log(level, f"{message} - failed", **context)
