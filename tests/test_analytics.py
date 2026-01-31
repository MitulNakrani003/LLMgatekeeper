"""Tests for the analytics module."""

import pytest

from llmgatekeeper.analytics import CacheAnalytics, CacheStats, NearMiss, QueryInfo


class TestQueryInfo:
    """Tests for the QueryInfo dataclass."""

    def test_create_query_info(self):
        """Can create a QueryInfo."""
        info = QueryInfo(query="test query")
        assert info.query == "test query"
        assert info.count == 1

    def test_query_info_with_count(self):
        """Can create QueryInfo with count."""
        info = QueryInfo(query="test", count=5)
        assert info.count == 5


class TestNearMiss:
    """Tests for the NearMiss dataclass."""

    def test_create_near_miss(self):
        """Can create a NearMiss."""
        nm = NearMiss(
            query="test query",
            closest_similarity=0.82,
            threshold=0.85,
        )
        assert nm.query == "test query"
        assert nm.closest_similarity == 0.82
        assert nm.threshold == 0.85


class TestCacheStats:
    """Tests for the CacheStats dataclass."""

    def test_create_cache_stats(self):
        """Can create CacheStats."""
        stats = CacheStats(
            hits=10,
            misses=5,
            hit_rate=10 / 15,
            total_queries=15,
            p50_latency_ms=5.0,
            p95_latency_ms=10.0,
            p99_latency_ms=15.0,
            avg_latency_ms=6.0,
            near_misses=[],
            top_queries=[],
        )
        assert stats.hits == 10
        assert stats.misses == 5
        assert stats.total_queries == 15


class TestCacheAnalytics:
    """Tests for the CacheAnalytics class."""

    def test_initial_state(self):
        """Analytics starts with zero counts."""
        analytics = CacheAnalytics()
        assert analytics.hits == 0
        assert analytics.misses == 0
        assert analytics.total_queries == 0
        assert analytics.hit_rate == 0.0

    def test_record_hit(self):
        """Recording a hit increments hit count."""
        analytics = CacheAnalytics()
        analytics.record_hit(latency_ms=5.0)
        assert analytics.hits == 1
        assert analytics.misses == 0
        assert analytics.total_queries == 1

    def test_record_miss(self):
        """Recording a miss increments miss count."""
        analytics = CacheAnalytics()
        analytics.record_miss(latency_ms=3.0)
        assert analytics.hits == 0
        assert analytics.misses == 1
        assert analytics.total_queries == 1

    def test_hit_rate_calculation(self):
        """TC-6.3.1: Hit rate calculated correctly."""
        analytics = CacheAnalytics()
        analytics.record_hit(latency_ms=5.0, query="query")
        analytics.record_hit(latency_ms=5.0, query="query")
        analytics.record_miss(latency_ms=5.0, query="other")

        assert analytics.hit_rate == pytest.approx(2 / 3)

    def test_latency_percentiles(self):
        """TC-6.3.2: Latency percentiles tracked."""
        analytics = CacheAnalytics()

        # Add 100 samples with known distribution
        for i in range(100):
            analytics.record_hit(latency_ms=float(i + 1), query="q")

        stats = analytics.get_stats()

        # p50 should be around 50
        assert stats.p50_latency_ms > 0
        assert 40 < stats.p50_latency_ms < 60

        # p95 should be around 95
        assert stats.p95_latency_ms >= stats.p50_latency_ms
        assert 90 < stats.p95_latency_ms < 100

        # p99 should be around 99
        assert stats.p99_latency_ms >= stats.p95_latency_ms

    def test_near_miss_tracking(self):
        """TC-6.3.3: Near-misses recorded."""
        analytics = CacheAnalytics(near_miss_threshold_gap=0.1)

        # Record a miss that's a near-miss (0.82 similarity with 0.85 threshold)
        analytics.record_miss(
            latency_ms=5.0,
            query="near miss query",
            closest_similarity=0.82,
            threshold=0.85,
        )

        stats = analytics.get_stats()
        assert len(stats.near_misses) == 1
        assert stats.near_misses[0].query == "near miss query"
        assert stats.near_misses[0].closest_similarity == 0.82

    def test_near_miss_not_recorded_if_too_far(self):
        """Near-miss not recorded if gap too large."""
        analytics = CacheAnalytics(near_miss_threshold_gap=0.1)

        # Record a miss that's too far (0.70 similarity with 0.85 threshold)
        analytics.record_miss(
            latency_ms=5.0,
            query="far miss query",
            closest_similarity=0.70,
            threshold=0.85,
        )

        stats = analytics.get_stats()
        assert len(stats.near_misses) == 0

    def test_top_queries_tracking(self):
        """TC-6.3.4: Most frequent queries tracked."""
        analytics = CacheAnalytics()

        # Record popular query many times
        for _ in range(10):
            analytics.record_hit(latency_ms=5.0, query="popular query")

        # Record rare query once
        analytics.record_miss(latency_ms=5.0, query="rare query")

        stats = analytics.get_stats()
        assert len(stats.top_queries) >= 2
        assert stats.top_queries[0].query == "popular query"
        assert stats.top_queries[0].count == 10

    def test_reset(self):
        """Reset clears all statistics."""
        analytics = CacheAnalytics()
        analytics.record_hit(latency_ms=5.0, query="q")
        analytics.record_miss(latency_ms=5.0, query="q")

        analytics.reset()

        assert analytics.hits == 0
        assert analytics.misses == 0
        assert analytics.total_queries == 0

    def test_average_latency(self):
        """Average latency calculated correctly."""
        analytics = CacheAnalytics()
        analytics.record_hit(latency_ms=10.0)
        analytics.record_hit(latency_ms=20.0)
        analytics.record_hit(latency_ms=30.0)

        stats = analytics.get_stats()
        assert stats.avg_latency_ms == pytest.approx(20.0)

    def test_max_latency_samples(self):
        """Old latency samples are discarded."""
        analytics = CacheAnalytics(max_latency_samples=10)

        # Add more samples than the limit
        for i in range(20):
            analytics.record_hit(latency_ms=float(i))

        # Should only have the last 10 samples (10-19)
        stats = analytics.get_stats()
        assert stats.avg_latency_ms >= 10.0  # Older samples should be gone

    def test_max_near_misses(self):
        """Old near-misses are discarded."""
        analytics = CacheAnalytics(max_near_misses=5, near_miss_threshold_gap=0.1)

        for i in range(10):
            analytics.record_miss(
                latency_ms=5.0,
                query=f"query{i}",
                closest_similarity=0.80,
                threshold=0.85,
            )

        stats = analytics.get_stats()
        assert len(stats.near_misses) == 5

    def test_query_tracking_without_query(self):
        """Query tracking works when query is None."""
        analytics = CacheAnalytics()
        analytics.record_hit(latency_ms=5.0)  # No query
        analytics.record_miss(latency_ms=5.0)  # No query

        stats = analytics.get_stats()
        assert len(stats.top_queries) == 0

    def test_empty_stats(self):
        """Stats work with no data."""
        analytics = CacheAnalytics()
        stats = analytics.get_stats()

        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.hit_rate == 0.0
        assert stats.p50_latency_ms == 0.0
        assert stats.avg_latency_ms == 0.0
        assert len(stats.near_misses) == 0
        assert len(stats.top_queries) == 0
