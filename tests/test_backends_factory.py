"""Tests for the backend factory."""

from unittest.mock import MagicMock

import pytest

from llmgatekeeper.backends.factory import create_redis_backend
from llmgatekeeper.backends.redis_search import RediSearchBackend
from llmgatekeeper.backends.redis_simple import RedisSimpleBackend


@pytest.fixture
def mock_redis_with_redisearch():
    """Create a mock Redis client with RediSearch module available."""
    mock_redis = MagicMock()

    # Mock module_list to return search module
    mock_redis.module_list.return_value = [{"name": "search", "ver": 20800}]

    # Mock FT commands for RediSearchBackend initialization
    mock_ft = MagicMock()
    mock_ft.info.return_value = MagicMock()  # Index exists
    mock_redis.ft.return_value = mock_ft

    # Mock JSON commands
    mock_json = MagicMock()
    mock_redis.json.return_value = mock_json

    # Mock set operations
    mock_redis.sadd.return_value = 1
    mock_redis.srem.return_value = 1
    mock_redis.smembers.return_value = set()
    mock_redis.scard.return_value = 0
    mock_redis.delete.return_value = 1

    return mock_redis


@pytest.fixture
def mock_redis_without_redisearch():
    """Create a mock Redis client without RediSearch module."""
    mock_redis = MagicMock()
    mock_redis.module_list.return_value = []

    # Mock set operations for RedisSimpleBackend
    mock_redis.sadd.return_value = 1
    mock_redis.srem.return_value = 1
    mock_redis.smembers.return_value = set()
    mock_redis.scard.return_value = 0
    mock_redis.delete.return_value = 1
    mock_redis.hset.return_value = 1
    mock_redis.hgetall.return_value = {}

    return mock_redis


class TestCreateRedisBackend:
    """Tests for create_redis_backend factory function."""

    def test_factory_returns_redisearch_when_available(
        self, mock_redis_with_redisearch
    ):
        """TC-2.4.1: Returns RediSearch backend when module available."""
        backend = create_redis_backend(mock_redis_with_redisearch)

        assert isinstance(backend, RediSearchBackend)

    def test_factory_fallback_to_simple(self, mock_redis_without_redisearch):
        """TC-2.4.2: Falls back to simple backend when RediSearch unavailable."""
        backend = create_redis_backend(mock_redis_without_redisearch)

        assert isinstance(backend, RedisSimpleBackend)

    def test_factory_force_simple_mode(self, mock_redis_with_redisearch):
        """TC-2.4.3: Force simple mode with parameter."""
        backend = create_redis_backend(
            mock_redis_with_redisearch, force_simple=True
        )

        assert isinstance(backend, RedisSimpleBackend)

    def test_factory_passes_namespace(self, mock_redis_without_redisearch):
        """Factory passes namespace to backend."""
        backend = create_redis_backend(
            mock_redis_without_redisearch, namespace="custom_ns"
        )

        assert backend._namespace == "custom_ns"

    def test_factory_passes_vector_dimension_to_redisearch(
        self, mock_redis_with_redisearch
    ):
        """Factory passes vector_dimension to RediSearchBackend."""
        # Make info() raise to trigger index creation where we can check dimension
        mock_ft = mock_redis_with_redisearch.ft.return_value
        mock_ft.info.side_effect = Exception("Index not found")

        backend = create_redis_backend(
            mock_redis_with_redisearch, vector_dimension=1536
        )

        assert isinstance(backend, RediSearchBackend)
        assert backend._vector_dimension == 1536

    def test_factory_handles_module_list_exception(self):
        """Factory handles exception from module_list gracefully."""
        mock_redis = MagicMock()
        mock_redis.module_list.side_effect = Exception("Connection failed")

        # Should fall back to simple backend
        backend = create_redis_backend(mock_redis)

        assert isinstance(backend, RedisSimpleBackend)

    def test_factory_handles_bytes_module_names(self):
        """Factory handles module names as bytes."""
        mock_redis = MagicMock()
        mock_redis.module_list.return_value = [{b"name": b"search", b"ver": 20800}]

        # Mock FT commands
        mock_ft = MagicMock()
        mock_ft.info.return_value = MagicMock()
        mock_redis.ft.return_value = mock_ft

        backend = create_redis_backend(mock_redis)

        assert isinstance(backend, RediSearchBackend)

    def test_factory_detects_ft_module_name(self):
        """Factory detects 'ft' as RediSearch module name."""
        mock_redis = MagicMock()
        mock_redis.module_list.return_value = [{"name": "ft", "ver": 20800}]

        # Mock FT commands
        mock_ft = MagicMock()
        mock_ft.info.return_value = MagicMock()
        mock_redis.ft.return_value = mock_ft

        backend = create_redis_backend(mock_redis)

        assert isinstance(backend, RediSearchBackend)


class TestFactoryIntegration:
    """Integration-style tests for factory behavior."""

    def test_simple_backend_is_functional(self, mock_redis_without_redisearch):
        """Simple backend from factory has expected interface."""
        backend = create_redis_backend(mock_redis_without_redisearch)

        # Verify it has all required methods
        assert hasattr(backend, "store_vector")
        assert hasattr(backend, "search_similar")
        assert hasattr(backend, "delete")
        assert hasattr(backend, "get_by_key")
        assert hasattr(backend, "clear")
        assert hasattr(backend, "count")

    def test_redisearch_backend_is_functional(self, mock_redis_with_redisearch):
        """RediSearch backend from factory has expected interface."""
        backend = create_redis_backend(mock_redis_with_redisearch)

        # Verify it has all required methods
        assert hasattr(backend, "store_vector")
        assert hasattr(backend, "search_similar")
        assert hasattr(backend, "delete")
        assert hasattr(backend, "get_by_key")
        assert hasattr(backend, "clear")
        assert hasattr(backend, "count")
