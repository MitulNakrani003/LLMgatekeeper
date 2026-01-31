"""Tests for the RedisSimpleBackend."""

import numpy as np
import pytest
from fakeredis import FakeRedis

from llmgatekeeper.backends.redis_simple import RedisSimpleBackend


@pytest.fixture
def fake_redis():
    """Create a fake Redis client for testing."""
    return FakeRedis(decode_responses=False)


@pytest.fixture
def redis_backend(fake_redis):
    """Create a RedisSimpleBackend with fake Redis."""
    return RedisSimpleBackend(fake_redis, namespace="test")


class TestRedisSimpleBackendStoreAndGet:
    """Tests for store_vector and get_by_key."""

    def test_store_and_get_by_key(self, redis_backend):
        """TC-2.2.1: Store and retrieve a vector by key."""
        vector = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        redis_backend.store_vector("key1", vector, {"response": "test"})
        result = redis_backend.get_by_key("key1")

        assert result is not None
        assert result.metadata["response"] == "test"
        assert np.allclose(result.vector, vector)

    def test_get_nonexistent_key_returns_none(self, redis_backend):
        """Get returns None for keys that don't exist."""
        result = redis_backend.get_by_key("nonexistent")
        assert result is None

    def test_store_overwrites_existing_key(self, redis_backend):
        """Storing to an existing key overwrites the value."""
        vector1 = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        vector2 = np.array([0.4, 0.5, 0.6], dtype=np.float32)

        redis_backend.store_vector("key1", vector1, {"v": 1})
        redis_backend.store_vector("key1", vector2, {"v": 2})

        result = redis_backend.get_by_key("key1")
        assert result.metadata["v"] == 2
        assert np.allclose(result.vector, vector2)


class TestRedisSimpleBackendSearch:
    """Tests for search_similar."""

    def test_search_similar_finds_match(self, redis_backend):
        """TC-2.2.2: Search returns similar vectors above threshold."""
        # Store vectors: key1 and key2 are similar to query, key3 is different
        redis_backend.store_vector(
            "key1", np.array([1.0, 0.0, 0.0], dtype=np.float32), {"r": "a"}
        )
        redis_backend.store_vector(
            "key2", np.array([0.99, 0.1, 0.0], dtype=np.float32), {"r": "b"}
        )
        redis_backend.store_vector(
            "key3", np.array([0.0, 1.0, 0.0], dtype=np.float32), {"r": "c"}
        )

        query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        results = redis_backend.search_similar(query, threshold=0.9, top_k=10)

        assert len(results) == 2  # key1 and key2
        keys = {r.key for r in results}
        assert "key1" in keys
        assert "key2" in keys
        assert "key3" not in keys

    def test_search_similar_respects_top_k(self, redis_backend):
        """Search returns at most top_k results."""
        # Store 5 similar vectors
        for i in range(5):
            vec = np.array([1.0, 0.1 * i, 0.0], dtype=np.float32)
            vec = vec / np.linalg.norm(vec)  # Normalize
            redis_backend.store_vector(f"key{i}", vec, {"i": i})

        query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        results = redis_backend.search_similar(query, threshold=0.5, top_k=3)

        assert len(results) <= 3

    def test_search_similar_sorted_by_similarity(self, redis_backend):
        """Results are sorted by similarity descending."""
        redis_backend.store_vector(
            "low", np.array([0.7, 0.7, 0.0], dtype=np.float32), {"sim": "low"}
        )
        redis_backend.store_vector(
            "high", np.array([1.0, 0.0, 0.0], dtype=np.float32), {"sim": "high"}
        )
        redis_backend.store_vector(
            "mid", np.array([0.9, 0.4, 0.0], dtype=np.float32), {"sim": "mid"}
        )

        query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        results = redis_backend.search_similar(query, threshold=0.5, top_k=10)

        # Check that similarities are in descending order
        similarities = [r.similarity for r in results]
        assert similarities == sorted(similarities, reverse=True)
        assert results[0].key == "high"

    def test_search_similar_returns_empty_when_no_match(self, redis_backend):
        """Search returns empty list when nothing matches threshold."""
        redis_backend.store_vector(
            "key1", np.array([1.0, 0.0, 0.0], dtype=np.float32), {"r": "a"}
        )

        # Query orthogonal to stored vector
        query = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        results = redis_backend.search_similar(query, threshold=0.9, top_k=10)

        assert len(results) == 0


class TestRedisSimpleBackendDelete:
    """Tests for delete."""

    def test_delete_removes_entry(self, redis_backend):
        """TC-2.2.3: Delete removes entry."""
        redis_backend.store_vector(
            "key1", np.array([0.1, 0.2], dtype=np.float32), {"r": "test"}
        )
        redis_backend.delete("key1")

        assert redis_backend.get_by_key("key1") is None

    def test_delete_returns_true_when_exists(self, redis_backend):
        """Delete returns True when key existed."""
        redis_backend.store_vector(
            "key1", np.array([0.1, 0.2], dtype=np.float32), {"r": "test"}
        )
        result = redis_backend.delete("key1")
        assert result is True

    def test_delete_returns_false_when_not_exists(self, redis_backend):
        """Delete returns False when key didn't exist."""
        result = redis_backend.delete("nonexistent")
        assert result is False

    def test_deleted_key_not_in_search(self, redis_backend):
        """Deleted entries don't appear in search results."""
        redis_backend.store_vector(
            "key1", np.array([1.0, 0.0, 0.0], dtype=np.float32), {"r": "a"}
        )
        redis_backend.store_vector(
            "key2", np.array([1.0, 0.0, 0.0], dtype=np.float32), {"r": "b"}
        )

        redis_backend.delete("key1")

        query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        results = redis_backend.search_similar(query, threshold=0.9, top_k=10)

        keys = {r.key for r in results}
        assert "key1" not in keys
        assert "key2" in keys


class TestRedisSimpleBackendInstance:
    """Tests for Redis instance handling."""

    def test_uses_existing_redis_instance(self):
        """TC-2.2.4: Uses user's existing Redis instance."""
        user_redis = FakeRedis()
        backend = RedisSimpleBackend(user_redis)
        assert backend._redis is user_redis

    def test_custom_namespace(self):
        """Custom namespace prefixes keys correctly."""
        user_redis = FakeRedis()
        backend = RedisSimpleBackend(user_redis, namespace="custom_ns")

        backend.store_vector(
            "key1", np.array([0.1, 0.2], dtype=np.float32), {"r": "test"}
        )

        # Check the key has the custom namespace prefix
        keys = list(user_redis.keys("custom_ns:*"))
        assert len(keys) > 0


class TestRedisSimpleBackendClearAndCount:
    """Tests for clear and count methods."""

    def test_count_returns_correct_number(self, redis_backend):
        """Count returns the number of stored entries."""
        assert redis_backend.count() == 0

        redis_backend.store_vector("k1", np.array([0.1], dtype=np.float32), {})
        assert redis_backend.count() == 1

        redis_backend.store_vector("k2", np.array([0.2], dtype=np.float32), {})
        assert redis_backend.count() == 2

    def test_clear_removes_all_entries(self, redis_backend):
        """Clear removes all entries and returns count."""
        redis_backend.store_vector("k1", np.array([0.1], dtype=np.float32), {})
        redis_backend.store_vector("k2", np.array([0.2], dtype=np.float32), {})
        redis_backend.store_vector("k3", np.array([0.3], dtype=np.float32), {})

        deleted = redis_backend.clear()

        assert deleted == 3
        assert redis_backend.count() == 0
        assert redis_backend.get_by_key("k1") is None
