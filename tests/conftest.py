"""Pytest configuration and shared fixtures."""

import pytest


@pytest.fixture
def sample_query() -> str:
    """Provide a sample query string for testing."""
    return "What is the weather today?"


@pytest.fixture
def sample_response() -> str:
    """Provide a sample response string for testing."""
    return "It's sunny and 72°F."
