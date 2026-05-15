"""Tests for the AsyncRediSearchBackend."""

from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from redis.exceptions import ResponseError

from llmgatekeeper.exceptions import BackendError, ConfigurationError


class MockFTSearch:
    def __init__(self, docs):
        self.docs = docs
        self.total = len(docs)


class MockDoc:
    def __init__(self, key, metadata, score):
        self.key = key
        self.metadata = metadata
        self.score = score


@pytest.fixture
def mock_async_redis_with_redisearch():
    redis = AsyncMock()
    redis.module_list = AsyncMock(
        return_value=[{"name": "search", "ver": 20800}]
    )

    mock_ft = MagicMock()
    mock_ft.info = AsyncMock(return_value={})
    mock_ft.create_index = AsyncMock()
    mock_ft.search = AsyncMock(return_value=MockFTSearch([]))
    mock_ft.dropindex = AsyncMock()
    redis.ft = MagicMock(return_value=mock_ft)

    mock_json = MagicMock()
    mock_json.set = AsyncMock()
    mock_json.get = AsyncMock(return_value=None)
    redis.json = MagicMock(return_value=mock_json)

    redis.sadd = AsyncMock(return_value=1)
    redis.srem = AsyncMock(return_value=1)
    redis.smembers = AsyncMock(return_value=set())
    redis.scard = AsyncMock(return_value=0)
    redis.delete = AsyncMock(return_value=1)
    redis.expire = AsyncMock()
    redis.hget = AsyncMock(return_value=None)
    return redis


@pytest.mark.asyncio
async def test_async_redisearch_construction_requires_module():
    """If module_list is empty, construction should raise."""
    from llmgatekeeper.backends.redis_search_async import AsyncRediSearchBackend

    bad_redis = AsyncMock()
    bad_redis.module_list = AsyncMock(return_value=[])
    bad_redis.ft = MagicMock(return_value=AsyncMock())

    backend = AsyncRediSearchBackend(bad_redis)
    with pytest.raises(RuntimeError, match="RediSearch"):
        await backend.connect()


@pytest.mark.asyncio
async def test_async_redisearch_non_cosine_raises(
    mock_async_redis_with_redisearch,
):
    from llmgatekeeper.backends.redis_search_async import AsyncRediSearchBackend

    with pytest.raises(ConfigurationError, match="distance_metric"):
        AsyncRediSearchBackend(
            mock_async_redis_with_redisearch, distance_metric="L2"
        )


@pytest.mark.asyncio
async def test_async_redisearch_store_and_search(
    mock_async_redis_with_redisearch,
):
    from llmgatekeeper.backends.redis_search_async import AsyncRediSearchBackend

    backend = AsyncRediSearchBackend(
        mock_async_redis_with_redisearch, vector_dimension=3
    )
    await backend.connect()

    vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    await backend.store_vector("k1", vec, {"r": "a"})

    mock_async_redis_with_redisearch.json.return_value.set.assert_awaited_once()

    mock_ft = mock_async_redis_with_redisearch.ft.return_value
    mock_ft.search.return_value = MockFTSearch(
        [MockDoc("k1", '{"r": "a"}', 0.05)]
    )
    mock_async_redis_with_redisearch.json.return_value.get = AsyncMock(
        return_value={
            "key": "k1",
            "vector": [1.0, 0.0, 0.0],
            "metadata": '{"r": "a"}',
        }
    )

    query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    results = await backend.search_similar(query, threshold=0.9, top_k=5)

    assert len(results) == 1
    assert results[0].key == "k1"
    assert results[0].similarity == pytest.approx(0.95)


@pytest.mark.asyncio
async def test_async_redisearch_search_propagates_errors(
    mock_async_redis_with_redisearch,
):
    from llmgatekeeper.backends.redis_search_async import AsyncRediSearchBackend

    backend = AsyncRediSearchBackend(
        mock_async_redis_with_redisearch, vector_dimension=3
    )
    await backend.connect()

    mock_ft = mock_async_redis_with_redisearch.ft.return_value
    mock_ft.search.side_effect = ResponseError("connection lost")

    with pytest.raises(BackendError):
        await backend.search_similar(
            np.array([1.0, 0.0, 0.0], dtype=np.float32), threshold=0.9
        )


@pytest.mark.asyncio
async def test_async_factory_returns_redisearch_when_available(
    mock_async_redis_with_redisearch,
):
    """The async factory selects the RediSearch backend when the module is loaded."""
    from llmgatekeeper.backends.factory import create_async_redis_backend
    from llmgatekeeper.backends.redis_search_async import AsyncRediSearchBackend

    backend = await create_async_redis_backend(mock_async_redis_with_redisearch)
    assert isinstance(backend, AsyncRediSearchBackend)


@pytest.mark.asyncio
async def test_async_factory_falls_back_to_simple():
    """Without the search module, the async factory falls back to AsyncRedisBackend."""
    from llmgatekeeper.backends.factory import create_async_redis_backend
    from llmgatekeeper.backends.redis_async import AsyncRedisBackend

    plain = AsyncMock()
    plain.module_list = AsyncMock(return_value=[])
    backend = await create_async_redis_backend(plain)
    assert isinstance(backend, AsyncRedisBackend)


@pytest.mark.asyncio
async def test_async_redisearch_search_empty_index_returns_empty(
    mock_async_redis_with_redisearch,
):
    from llmgatekeeper.backends.redis_search_async import AsyncRediSearchBackend

    backend = AsyncRediSearchBackend(
        mock_async_redis_with_redisearch, vector_dimension=3
    )
    await backend.connect()

    mock_ft = mock_async_redis_with_redisearch.ft.return_value
    mock_ft.search.side_effect = ResponseError("no such index")

    results = await backend.search_similar(
        np.array([1.0, 0.0, 0.0], dtype=np.float32), threshold=0.9
    )
    assert results == []
