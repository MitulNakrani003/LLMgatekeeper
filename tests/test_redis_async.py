"""Tests for the async Redis backend."""

from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from llmgatekeeper.backends.redis_async import AsyncRedisBackend


@pytest.fixture
def mock_async_redis():
    """Create a mock async Redis client."""
    redis = AsyncMock()
    redis.hset = AsyncMock()
    redis.hgetall = AsyncMock(return_value={})
    redis.sadd = AsyncMock()
    redis.srem = AsyncMock()
    redis.smembers = AsyncMock(return_value=set())
    redis.scard = AsyncMock(return_value=0)
    redis.delete = AsyncMock(return_value=1)
    redis.expire = AsyncMock()
    return redis


@pytest.fixture
def async_backend(mock_async_redis):
    """Create an AsyncRedisBackend with mock client."""
    return AsyncRedisBackend(mock_async_redis, namespace="test")


class TestAsyncRedisBackendInit:
    """Tests for AsyncRedisBackend initialization."""

    def test_init_with_defaults(self, mock_async_redis):
        """Can initialize with default namespace."""
        backend = AsyncRedisBackend(mock_async_redis)
        assert backend._namespace == "llmgk"

    def test_init_with_custom_namespace(self, mock_async_redis):
        """Can initialize with custom namespace."""
        backend = AsyncRedisBackend(mock_async_redis, namespace="my_cache")
        assert backend._namespace == "my_cache"


class TestAsyncRedisBackendStoreVector:
    """Tests for async store_vector."""

    @pytest.mark.asyncio
    async def test_store_vector(self, async_backend, mock_async_redis):
        """Store vector calls Redis correctly."""
        vector = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        metadata = {"response": "test"}

        await async_backend.store_vector("key1", vector, metadata)

        mock_async_redis.hset.assert_called_once()
        mock_async_redis.sadd.assert_called_once()

    @pytest.mark.asyncio
    async def test_store_vector_with_ttl(self, async_backend, mock_async_redis):
        """Store vector with TTL sets expiration."""
        vector = np.array([0.1, 0.2, 0.3], dtype=np.float32)

        await async_backend.store_vector("key1", vector, {}, ttl=3600)

        mock_async_redis.expire.assert_called_once()
        call_args = mock_async_redis.expire.call_args
        assert call_args[0][1] == 3600


class TestAsyncRedisBackendGetByKey:
    """Tests for async get_by_key."""

    @pytest.mark.asyncio
    async def test_get_by_key_found(self, async_backend, mock_async_redis):
        """get_by_key returns entry when found."""
        vector = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        mock_async_redis.hgetall.return_value = {
            b"vector": vector.tobytes(),
            b"metadata": '{"response": "test"}',
            b"key": "key1",
        }

        result = await async_backend.get_by_key("key1")

        assert result is not None
        assert result.key == "key1"
        np.testing.assert_array_almost_equal(result.vector, vector)

    @pytest.mark.asyncio
    async def test_get_by_key_not_found(self, async_backend, mock_async_redis):
        """get_by_key returns None when not found."""
        mock_async_redis.hgetall.return_value = {}

        result = await async_backend.get_by_key("nonexistent")

        assert result is None


class TestAsyncRedisBackendSearchSimilar:
    """Tests for async search_similar."""

    @pytest.mark.asyncio
    async def test_search_similar_no_entries(self, async_backend, mock_async_redis):
        """search_similar returns empty list when no entries."""
        mock_async_redis.smembers.return_value = set()

        vector = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        results = await async_backend.search_similar(vector)

        assert results == []

    @pytest.mark.asyncio
    async def test_search_similar_finds_match(self, async_backend, mock_async_redis):
        """search_similar finds matching vectors."""
        stored_vector = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        mock_async_redis.smembers.return_value = {b"key1"}
        mock_async_redis.hgetall.return_value = {
            b"vector": stored_vector.tobytes(),
            b"metadata": '{"response": "test"}',
            b"key": "key1",
        }

        query_vector = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        results = await async_backend.search_similar(query_vector, threshold=0.9)

        assert len(results) == 1
        assert results[0].key == "key1"
        assert results[0].similarity == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_search_similar_filters_by_threshold(
        self, async_backend, mock_async_redis
    ):
        """search_similar respects threshold."""
        stored_vector = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        mock_async_redis.smembers.return_value = {b"key1"}
        mock_async_redis.hgetall.return_value = {
            b"vector": stored_vector.tobytes(),
            b"metadata": '{"response": "test"}',
            b"key": "key1",
        }

        # Orthogonal vector should have 0 similarity
        query_vector = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        results = await async_backend.search_similar(query_vector, threshold=0.5)

        assert len(results) == 0


class TestAsyncRedisBackendDelete:
    """Tests for async delete."""

    @pytest.mark.asyncio
    async def test_delete_existing(self, async_backend, mock_async_redis):
        """delete returns True when entry existed."""
        mock_async_redis.delete.return_value = 1

        result = await async_backend.delete("key1")

        assert result is True
        mock_async_redis.delete.assert_called_once()
        mock_async_redis.srem.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, async_backend, mock_async_redis):
        """delete returns False when entry didn't exist."""
        mock_async_redis.delete.return_value = 0

        result = await async_backend.delete("nonexistent")

        assert result is False


class TestAsyncRedisBackendClear:
    """Tests for async clear."""

    @pytest.mark.asyncio
    async def test_clear_empty(self, async_backend, mock_async_redis):
        """clear returns 0 when cache is empty."""
        mock_async_redis.smembers.return_value = set()

        result = await async_backend.clear()

        assert result == 0

    @pytest.mark.asyncio
    async def test_clear_with_entries(self, async_backend, mock_async_redis):
        """clear removes all entries."""
        mock_async_redis.smembers.return_value = {b"key1", b"key2"}
        mock_async_redis.delete.return_value = 1

        result = await async_backend.clear()

        assert result == 2


class TestAsyncRedisBackendCount:
    """Tests for async count."""

    @pytest.mark.asyncio
    async def test_count(self, async_backend, mock_async_redis):
        """count returns number of entries."""
        mock_async_redis.scard.return_value = 5

        result = await async_backend.count()

        assert result == 5
        mock_async_redis.scard.assert_called_once()
