"""Analytics and observability for SemanticCache.

This module provides classes for tracking cache performance metrics including
hit/miss rates, latency percentiles, and near-miss tracking.
"""

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional


@dataclass
class QueryInfo:
    """Information about a query."""

    query: str
    count: int = 1
    last_accessed: float = field(default_factory=time.time)


@dataclass
class NearMiss:
    """Information about a near-miss (query that almost matched)."""

    query: str
    closest_similarity: float
    threshold: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class CacheStats:
    """Cache performance statistics.

    Attributes:
        hits: Total number of cache hits.
        misses: Total number of cache misses.
        hit_rate: Ratio of hits to total queries (0.0 to 1.0).
        total_queries: Total number of queries processed.
        p50_latency_ms: 50th percentile (median) latency in milliseconds.
        p95_latency_ms: 95th percentile latency in milliseconds.
        p99_latency_ms: 99th percentile latency in milliseconds.
        avg_latency_ms: Average latency in milliseconds.
        near_misses: List of recent near-miss queries.
        top_queries: List of most frequently accessed queries.
    """

    hits: int
    misses: int
    hit_rate: float
    total_queries: int
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    avg_latency_ms: float
    near_misses: List[NearMiss]
    top_queries: List[QueryInfo]


class CacheAnalytics:
    """Tracks cache performance metrics.

    This class collects and computes statistics about cache operations including
    hit/miss rates, latency distributions, near-misses, and query frequency.

    Args:
        max_latency_samples: Maximum number of latency samples to retain.
            Older samples are discarded. Default 10000.
        max_near_misses: Maximum number of near-misses to track. Default 100.
        max_top_queries: Maximum number of top queries to track. Default 100.
        near_miss_threshold_gap: Maximum gap below threshold to consider
            a near-miss. Default 0.1 (e.g., threshold=0.85, near-miss >= 0.75).

    Example:
        >>> analytics = CacheAnalytics()
        >>> analytics.record_hit(latency_ms=5.2, query="What is Python?")
        >>> analytics.record_miss(latency_ms=3.1, query="Unknown query")
        >>> stats = analytics.get_stats()
        >>> print(f"Hit rate: {stats.hit_rate:.2%}")
    """

    def __init__(
        self,
        max_latency_samples: int = 10000,
        max_near_misses: int = 100,
        max_top_queries: int = 100,
        near_miss_threshold_gap: float = 0.1,
    ) -> None:
        """Initialize analytics tracker."""
        self._hits = 0
        self._misses = 0
        self._latencies: Deque[float] = deque(maxlen=max_latency_samples)
        self._near_misses: Deque[NearMiss] = deque(maxlen=max_near_misses)
        self._query_counts: Dict[str, QueryInfo] = {}
        self._max_top_queries = max_top_queries
        self._near_miss_threshold_gap = near_miss_threshold_gap

    @property
    def hits(self) -> int:
        """Return total number of cache hits."""
        return self._hits

    @property
    def misses(self) -> int:
        """Return total number of cache misses."""
        return self._misses

    @property
    def total_queries(self) -> int:
        """Return total number of queries."""
        return self._hits + self._misses

    @property
    def hit_rate(self) -> float:
        """Return hit rate as ratio (0.0 to 1.0)."""
        total = self.total_queries
        if total == 0:
            return 0.0
        return self._hits / total

    def record_hit(
        self,
        latency_ms: float,
        query: Optional[str] = None,
    ) -> None:
        """Record a cache hit.

        Args:
            latency_ms: Operation latency in milliseconds.
            query: Optional query string for frequency tracking.
        """
        self._hits += 1
        self._latencies.append(latency_ms)
        if query is not None:
            self._record_query(query)

    def record_miss(
        self,
        latency_ms: float,
        query: Optional[str] = None,
        closest_similarity: Optional[float] = None,
        threshold: Optional[float] = None,
    ) -> None:
        """Record a cache miss.

        Args:
            latency_ms: Operation latency in milliseconds.
            query: Optional query string for frequency tracking.
            closest_similarity: Similarity of closest match (if any).
            threshold: Current threshold for cache hits.
        """
        self._misses += 1
        self._latencies.append(latency_ms)
        if query is not None:
            self._record_query(query)

        # Check for near-miss
        if (
            closest_similarity is not None
            and threshold is not None
            and query is not None
        ):
            gap = threshold - closest_similarity
            if 0 < gap <= self._near_miss_threshold_gap:
                self._near_misses.append(
                    NearMiss(
                        query=query,
                        closest_similarity=closest_similarity,
                        threshold=threshold,
                    )
                )

    def _record_query(self, query: str) -> None:
        """Record query access for frequency tracking."""
        if query in self._query_counts:
            self._query_counts[query].count += 1
            self._query_counts[query].last_accessed = time.time()
        else:
            self._query_counts[query] = QueryInfo(query=query)

    def _compute_percentile(self, percentile: float) -> float:
        """Compute a percentile from latency samples.

        Args:
            percentile: Percentile to compute (0-100).

        Returns:
            Latency at the given percentile, or 0.0 if no samples.
        """
        if not self._latencies:
            return 0.0

        sorted_latencies = sorted(self._latencies)
        n = len(sorted_latencies)
        index = (percentile / 100) * (n - 1)

        # Linear interpolation between adjacent values
        lower = int(index)
        upper = min(lower + 1, n - 1)
        weight = index - lower

        return sorted_latencies[lower] * (1 - weight) + sorted_latencies[upper] * weight

    def get_stats(self) -> CacheStats:
        """Get current cache statistics.

        Returns:
            CacheStats object with current metrics.
        """
        # Compute latency percentiles
        p50 = self._compute_percentile(50)
        p95 = self._compute_percentile(95)
        p99 = self._compute_percentile(99)
        avg = sum(self._latencies) / len(self._latencies) if self._latencies else 0.0

        # Get top queries sorted by count
        top_queries = sorted(
            self._query_counts.values(),
            key=lambda q: q.count,
            reverse=True,
        )[: self._max_top_queries]

        return CacheStats(
            hits=self._hits,
            misses=self._misses,
            hit_rate=self.hit_rate,
            total_queries=self.total_queries,
            p50_latency_ms=p50,
            p95_latency_ms=p95,
            p99_latency_ms=p99,
            avg_latency_ms=avg,
            near_misses=list(self._near_misses),
            top_queries=top_queries,
        )

    def reset(self) -> None:
        """Reset all statistics."""
        self._hits = 0
        self._misses = 0
        self._latencies.clear()
        self._near_misses.clear()
        self._query_counts.clear()
