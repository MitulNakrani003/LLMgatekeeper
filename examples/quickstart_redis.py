#!/usr/bin/env python3
"""
Complete usage reference for LLMGatekeeper.

Every public class, method, property, and utility in the package is demonstrated
here.  Sections that require an external service (OpenAI API, a hypothetical
Pinecone index) are clearly marked and wrapped so the script still runs end-to-end
against a plain Redis instance.

Prerequisites
-------------
1.  Install dependencies:
        pip install llmgatekeeper redis sentence-transformers numpy

2.  Start Redis (plain) OR Redis Stack (includes RediSearch):
        # plain Redis (simple backend, brute-force similarity)
        docker run -d --name redis-cache -p 6379:6379 redis:latest

        # Redis Stack (RediSearch backend, KNN vector search)
        docker run -d --name redis-stack -p 6379:6379 redis/redis-stack:latest

3.  Run this file:
        python quickstart_redis.py

4.  Cleanup:
        docker stop redis-cache && docker rm redis-cache
"""

import time

import numpy as np
import redis

# ---------------------------------------------------------------------------
# Public API imports – everything the package exports
# ---------------------------------------------------------------------------
from llmgatekeeper import (
    AsyncSemanticCache,
    BackendError,
    CacheError,
    CacheResult,
    ConfigurationError,
    EmbeddingError,
    SemanticCache,
    configure_logging,
    disable_logging,
)

# Sub-module imports used in deeper examples
from llmgatekeeper.backends.factory import create_redis_backend
from llmgatekeeper.backends.redis_simple import RedisSimpleBackend
from llmgatekeeper.embeddings.cached import CachedEmbeddingProvider
from llmgatekeeper.embeddings.sentence_transformer import SentenceTransformerProvider
from llmgatekeeper.exceptions import ConnectionError as CacheConnectionError
from llmgatekeeper.exceptions import TimeoutError as CacheTimeoutError
from llmgatekeeper.logging import get_logger, log_operation
from llmgatekeeper.similarity.confidence import (
    ConfidenceClassifier,
    ConfidenceLevel,
    get_model_classifier,
)
from llmgatekeeper.similarity.metrics import (
    SimilarityMetric,
    batch_cosine_similarity,
    compute_similarity,
    cosine_similarity,
    dot_product_similarity,
    euclidean_distance,
    euclidean_similarity,
    get_similarity_function,
    normalize_similarity,
)
from llmgatekeeper.similarity.retriever import SimilarityRetriever

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
REDIS_HOST = "localhost"
REDIS_PORT = 6379


def get_redis_client() -> redis.Redis:
    """Create and return a Redis client, verifying connectivity."""
    client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=False)
    client.ping()
    return client


# =============================================================================
# SECTION 1 – Basic Operations
#   set / get / delete / delete_by_key / exists / clear / count
# =============================================================================
def section_basic_operations():
    print("\n" + "=" * 70)
    print("SECTION 1 – Basic Operations")
    print("=" * 70)

    client = get_redis_client()
    cache = SemanticCache(client, namespace="s1_basic", threshold=0.80)
    cache.clear()  # wipe any leftovers from a previous run

    # --- set() returns the internal cache key --------------------------------
    key = cache.set("What is Python?", "Python is a high-level programming language.")
    print(f"set()  -> key = {key}")

    # --- get() – exact query returns the response string -----------------------
    response = cache.get("What is Python?")
    print(f"get()  exact  -> {response}")

    # --- get() – semantically similar query hits the same entry ---------------
    response = cache.get("Tell me about Python")
    print(f"get()  similar -> {response}")

    # --- get() – unrelated query returns None ----------------------------------
    response = cache.get("Describe the solar system")
    print(f"get()  miss   -> {response}")

    # --- exists() – checks for an exact-key match (NOT semantic) ---------------
    print(f"exists('What is Python?')       -> {cache.exists('What is Python?')}")
    print(f"exists('Tell me about Python')  -> {cache.exists('Tell me about Python')}")

    # --- delete() by query text ------------------------------------------------
    deleted = cache.delete("What is Python?")
    print(f"delete('What is Python?')       -> {deleted}")
    print(f"exists after delete             -> {cache.exists('What is Python?')}")

    # --- delete_by_key() using the raw key returned by set() -------------------
    key2 = cache.set("What is Java?", "Java is a compiled language.")
    deleted = cache.delete_by_key(key2)
    print(f"delete_by_key(key)              -> {deleted}")

    # --- count() / clear() -----------------------------------------------------
    cache.set("Q1", "A1")
    cache.set("Q2", "A2")
    print(f"count() before clear            -> {cache.count()}")
    removed = cache.clear()
    print(f"clear() removed                 -> {removed}")
    print(f"count() after clear             -> {cache.count()}")


# =============================================================================
# SECTION 2 – get() with include_metadata  →  CacheResult
#   CacheResult fields: response, similarity, confidence, key, metadata
# =============================================================================
def section_cache_result():
    print("\n" + "=" * 70)
    print("SECTION 2 – CacheResult (include_metadata=True)")
    print("=" * 70)

    client = get_redis_client()
    cache = SemanticCache(client, namespace="s2_result", threshold=0.80)
    cache.clear()

    cache.set(
        "Explain recursion",
        "Recursion is a technique where a function calls itself.",
        metadata={"model": "gpt-4", "tokens": 12, "category": "cs"},
    )

    # Passing include_metadata=True returns a CacheResult instead of a plain string
    result: CacheResult = cache.get("What is recursion?", include_metadata=True)

    if result:
        print(f"response    -> {result.response}")
        print(f"similarity  -> {result.similarity:.4f}")
        print(f"confidence  -> {result.confidence}")          # ConfidenceLevel enum
        print(f"key         -> {result.key}")
        # metadata contains only user-supplied fields; 'query' and 'response'
        # are stored internally but stripped from the returned metadata dict.
        print(f"metadata    -> {result.metadata}")
        # CacheResult.__str__() returns the response string directly
        print(f"str(result) -> {str(result)}")


# =============================================================================
# SECTION 3 – Semantic Matching & Threshold Behaviour
#   Demonstrates what hits and what misses at different thresholds,
#   and how to override threshold per-query.
# =============================================================================
def section_semantic_matching():
    print("\n" + "=" * 70)
    print("SECTION 3 – Semantic Matching & Per-Query Threshold Override")
    print("=" * 70)

    client = get_redis_client()
    cache = SemanticCache(client, namespace="s3_semantic", threshold=0.85)
    cache.clear()

    cache.set(
        "What is machine learning?",
        "Machine learning is a subset of AI where systems learn from data.",
    )

    queries = [
        "What is machine learning?",              # exact  -> always hits
        "Define machine learning",                # close  -> hits at 0.85
        "Explain machine learning to me",         # close  -> hits at 0.85
        "Tell me about artificial intelligence",  # further away
        "What is ML?",                            # abbreviation – very low similarity
    ]

    print("\n  Default threshold = 0.85")
    for q in queries:
        r = cache.get(q, include_metadata=True)
        sim = f"{r.similarity:.3f}" if r else "—"
        conf = str(r.confidence) if r else "—"
        print(f"    {q:<50} sim={sim:<8} conf={conf:<8} {'HIT' if r else 'MISS'}")

    # Per-query threshold override – relaxing to 0.70 picks up more matches
    print("\n  Per-query override threshold = 0.70")
    for q in queries:
        r = cache.get(q, include_metadata=True, threshold=0.70)
        sim = f"{r.similarity:.3f}" if r else "—"
        print(f"    {q:<50} sim={sim:<8} {'HIT' if r else 'MISS'}")


# =============================================================================
# SECTION 4 – get_similar()  →  multiple ranked results
# =============================================================================
def section_get_similar():
    print("\n" + "=" * 70)
    print("SECTION 4 – get_similar() – Top-K Retrieval")
    print("=" * 70)

    client = get_redis_client()
    cache = SemanticCache(client, namespace="s4_similar", threshold=0.65)
    cache.clear()

    entries = [
        ("What is Python used for?",         "Web dev, data science, automation.",      "web"),
        ("Python programming applications",  "ML, scripting, backend development.",     "ml"),
        ("Uses of the Python language",      "AI research, DevOps, scientific work.",   "ai"),
        ("Python in software development",   "APIs, testing, rapid prototyping.",       "apis"),
    ]
    for q, r, topic in entries:
        cache.set(q, r, metadata={"topic": topic})

    # get_similar returns a list of CacheResult sorted by similarity descending
    results = cache.get_similar("What can I do with Python?", top_k=3)
    print(f"  Top 3 matches for 'What can I do with Python?'")
    for i, res in enumerate(results, 1):
        print(f"    {i}. sim={res.similarity:.3f}  conf={res.confidence}  "
              f"topic={res.metadata.get('topic')}  response={res.response[:45]}…")

    # Threshold can be overridden here too
    results_strict = cache.get_similar("What can I do with Python?", top_k=5, threshold=0.80)
    print(f"\n  Same query, threshold=0.80 -> {len(results_strict)} result(s)")


# =============================================================================
# SECTION 5 – TTL (Time-To-Live)
#   default_ttl on the cache, per-entry ttl, and ttl=0 override
# =============================================================================
def section_ttl():
    print("\n" + "=" * 70)
    print("SECTION 5 – TTL (Time-To-Live)")
    print("=" * 70)

    client = get_redis_client()

    # default_ttl applies to every set() that doesn't specify its own ttl
    cache = SemanticCache(client, namespace="s5_ttl", threshold=0.80, default_ttl=120)
    cache.clear()

    # This entry inherits default_ttl=120s
    cache.set("default ttl query", "response A")
    print(f"default_ttl property -> {cache.default_ttl}")

    # Explicit ttl=5 overrides default_ttl for this entry only
    cache.set("short lived query", f"time is {time.strftime('%H:%M:%S')}", ttl=5)
    print(f"Stored short-lived entry with ttl=5")
    print(f"  Immediate get -> {cache.get('short lived query')}")

    print("  Waiting 6 seconds…")
    time.sleep(6)
    print(f"  After 6s get  -> {cache.get('short lived query')}")  # None – expired

    # ttl=0 means "no expiration" even when default_ttl is set
    cache.set("permanent query", "response C", ttl=0)
    print(f"  ttl=0 entry exists -> {cache.exists('permanent query')}")

    # default_ttl setter
    cache.default_ttl = 3600
    print(f"  Updated default_ttl -> {cache.default_ttl}")


# =============================================================================
# SECTION 6 – Cache Warming (Bulk Load)
#   warm() with batch_size and on_progress callback
# =============================================================================
def section_cache_warming():
    print("\n" + "=" * 70)
    print("SECTION 6 – Cache Warming (Bulk Load)")
    print("=" * 70)

    client = get_redis_client()
    cache = SemanticCache(client, namespace="s6_warm", threshold=0.80)
    cache.clear()

    faq = [
        ("How do I reset my password?",       "Settings > Security > Reset Password."),
        ("Where is my order history?",         "Account > Orders."),
        ("How do I contact support?",          "Email support@example.com or use chat."),
        ("What payment methods do you accept?", "Credit cards, PayPal, bank transfer."),
        ("How do I cancel my subscription?",   "Account > Subscription > Cancel."),
        ("What is your refund policy?",        "Full refund within 30 days."),
    ]

    progress_log = []

    def on_progress(completed: int, total: int):
        progress_log.append(f"{completed}/{total}")

    count = cache.warm(
        faq,
        metadata={"source": "faq", "version": "2.0"},
        ttl=7200,              # 2-hour TTL on every warmed entry
        batch_size=2,          # embed 2 queries at a time
        on_progress=on_progress,
    )

    print(f"  Warmed {count} entries.  Progress callbacks: {progress_log}")
    print(f"  count() -> {cache.count()}")
    print(f"  Semantic query -> {cache.get('How can I change my password?')}")


# =============================================================================
# SECTION 7 – Namespaces (Multi-Tenant Isolation)
# =============================================================================
def section_namespaces():
    print("\n" + "=" * 70)
    print("SECTION 7 – Namespaces (Multi-Tenant Isolation)")
    print("=" * 70)

    client = get_redis_client()

    cache_a = SemanticCache(client, namespace="tenant_a", threshold=0.80)
    cache_b = SemanticCache(client, namespace="tenant_b", threshold=0.80)
    cache_a.clear()
    cache_b.clear()

    # Same question, different answers per tenant
    cache_a.set("What are your hours?", "Tenant A: Mon-Fri 9am-5pm EST.")
    cache_b.set("What are your hours?", "Tenant B: Open 24/7.")

    print(f"  namespace property A -> {cache_a.namespace}")
    print(f"  namespace property B -> {cache_b.namespace}")
    print(f"  Tenant A get -> {cache_a.get('What are your hours?')}")
    print(f"  Tenant B get -> {cache_b.get('What are your hours?')}")

    # Searching tenant A never surfaces tenant B data
    print(f"  Tenant A count -> {cache_a.count()}")
    print(f"  Tenant B count -> {cache_b.count()}")


# =============================================================================
# SECTION 8 – Dynamic Threshold & Property Changes
#   threshold setter, repr
# =============================================================================
def section_dynamic_properties():
    print("\n" + "=" * 70)
    print("SECTION 8 – Dynamic Properties & repr")
    print("=" * 70)

    client = get_redis_client()
    cache = SemanticCache(client, namespace="s8_props", threshold=0.85)
    cache.clear()

    print(f"  repr  -> {repr(cache)}")
    print(f"  threshold before -> {cache.threshold}")

    # Adjust threshold at runtime – affects all subsequent get() calls
    cache.threshold = 0.70
    print(f"  threshold after  -> {cache.threshold}")

    # Verify it actually changes behaviour
    cache.set("What is deep learning?", "DL uses neural networks with many layers.")
    # At 0.70 this semantic match succeeds; at 0.95 it would not
    r = cache.get("Explain deep learning", include_metadata=True)
    print(f"  get at threshold=0.70 -> sim={r.similarity:.3f} HIT" if r else "  MISS")

    # embedding_provider property
    print(f"  embedding_provider  -> {type(cache.embedding_provider).__name__}")
    print(f"  analytics_enabled   -> {cache.analytics_enabled}")


# =============================================================================
# SECTION 9 – Analytics & Statistics
#   enable_analytics, stats(), reset_stats(), CacheStats, NearMiss, QueryInfo
# =============================================================================
def section_analytics():
    print("\n" + "=" * 70)
    print("SECTION 9 – Analytics & Statistics")
    print("=" * 70)

    client = get_redis_client()
    cache = SemanticCache(
        client,
        namespace="s9_analytics",
        threshold=0.80,
        enable_analytics=True,   # opt-in
    )
    cache.clear()

    print(f"  analytics_enabled -> {cache.analytics_enabled}")

    # Without analytics enabled, stats() returns None
    cache_no_analytics = SemanticCache(client, namespace="s9_no", threshold=0.80)
    print(f"  stats() without analytics -> {cache_no_analytics.stats()}")

    # Populate with data
    cache.set("What is JavaScript?", "JS is the language of the web.")
    cache.set("What is TypeScript?", "TS adds static typing to JS.")

    # Run a mix of hits and misses
    test_queries = [
        "What is JavaScript?",         # exact hit
        "Tell me about JavaScript",    # semantic hit
        "What is TypeScript?",         # exact hit
        "Explain TypeScript",          # semantic hit
        "What is Rust?",               # miss
        "What is Go?",                 # miss
        "How do I cook pasta?",        # miss
    ]
    for q in test_queries:
        cache.get(q)

    stats = cache.stats()
    print(f"\n  --- CacheStats ---")
    print(f"  total_queries   -> {stats.total_queries}")
    print(f"  hits            -> {stats.hits}")
    print(f"  misses          -> {stats.misses}")
    print(f"  hit_rate        -> {stats.hit_rate:.1%}")
    print(f"  avg_latency_ms  -> {stats.avg_latency_ms:.2f}")
    print(f"  p50_latency_ms  -> {stats.p50_latency_ms:.2f}")
    print(f"  p95_latency_ms  -> {stats.p95_latency_ms:.2f}")
    print(f"  p99_latency_ms  -> {stats.p99_latency_ms:.2f}")

    # NearMiss – queries that came close but fell below threshold
    if stats.near_misses:
        print(f"\n  --- Near Misses (came close but missed) ---")
        for nm in stats.near_misses:
            gap = nm.threshold - nm.closest_similarity
            print(f"    query='{nm.query}'  closest_sim={nm.closest_similarity:.3f}  "
                  f"threshold={nm.threshold}  gap={gap:.3f}")

    # QueryInfo – most frequently accessed queries
    if stats.top_queries:
        print(f"\n  --- Top Queries (by access count) ---")
        for qi in stats.top_queries[:4]:
            print(f"    '{qi.query}'  count={qi.count}")

    # reset_stats() wipes everything
    cache.reset_stats()
    stats_after = cache.stats()
    print(f"\n  After reset_stats(): total_queries={stats_after.total_queries}")


# =============================================================================
# SECTION 10 – Error Handling & Exception Hierarchy
#   CacheError, BackendError, EmbeddingError, ConfigurationError,
#   ConnectionError, TimeoutError  (all have .original_error where applicable)
# =============================================================================
def section_error_handling():
    print("\n" + "=" * 70)
    print("SECTION 10 – Error Handling & Exception Hierarchy")
    print("=" * 70)

    # --- Demonstrate exception hierarchy relationships -------------------------
    print("  Exception hierarchy:")
    print("    CacheError")
    print("      ├── BackendError          (.original_error)")
    print("      │     ├── ConnectionError")
    print("      │     └── TimeoutError")
    print("      ├── EmbeddingError        (.original_error)")
    print("      └── ConfigurationError")

    # --- Catching at different levels ------------------------------------------
    client = get_redis_client()
    cache = SemanticCache(client, namespace="s10_errors", threshold=0.80)

    try:
        cache.set("test query", "test response")
        result = cache.get("test query")
        print(f"\n  Normal operation -> {result}")

    except EmbeddingError as e:
        print(f"  EmbeddingError: {e}")
        print(f"    original_error: {e.original_error}")

    except BackendError as e:
        print(f"  BackendError: {e}")
        print(f"    original_error: {e.original_error}")

    except CacheError as e:
        # Catch-all for any LLMGatekeeper error
        print(f"  CacheError (catch-all): {e}")

    # --- Manually constructing exceptions (shows the .original_error field) ----
    original = ValueError("disk full")
    be = BackendError("Redis write failed", original_error=original)
    print(f"\n  Constructed BackendError: {be}")
    print(f"    .original_error -> {be.original_error}")

    ee = EmbeddingError("Model unavailable", original_error=RuntimeError("no GPU"))
    print(f"  Constructed EmbeddingError: {ee}")
    print(f"    .original_error -> {ee.original_error}")

    ce = ConfigurationError("threshold out of range")
    print(f"  Constructed ConfigurationError: {ce}")

    conn_err = CacheConnectionError("Redis unreachable", original_error=OSError("timeout"))
    print(f"  Constructed ConnectionError: {conn_err}  "
          f"(is BackendError: {isinstance(conn_err, BackendError)})")

    t_err = CacheTimeoutError("operation timed out", original_error=TimeoutError())
    print(f"  Constructed TimeoutError: {t_err}  "
          f"(is BackendError: {isinstance(t_err, BackendError)})")

    # --- Catching a bad threshold raises ValueError ----------------------------
    try:
        SemanticCache(client, namespace="bad", threshold=2.0)
    except ValueError as e:
        print(f"\n  Invalid threshold caught: {e}")


# =============================================================================
# SECTION 11 – Logging
#   configure_logging, disable_logging, get_logger, log_operation
# =============================================================================
def section_logging():
    print("\n" + "=" * 70)
    print("SECTION 11 – Logging")
    print("=" * 70)

    # Enable INFO-level logging – only messages from llmgatekeeper are shown
    configure_logging(level="INFO")
    print("  Logging enabled at INFO.  Running a set+get…")

    client = get_redis_client()
    cache = SemanticCache(client, namespace="s11_log", threshold=0.80)
    cache.clear()
    cache.set("logging test", "response for logging")
    cache.get("logging test")

    # Use the library's own logger helper
    logger = get_logger("my_app")
    logger.info("Application-level log from get_logger()")

    # log_operation() – convenience wrapper for structured cache-op logging
    log_operation("get", success=True, duration_ms=3.2, query="logging test")
    log_operation("set", success=False, duration_ms=12.1, query="failed entry")

    # Silence everything again so the rest of the examples stay clean
    disable_logging()
    print("  Logging disabled.")


# =============================================================================
# SECTION 12 – Backend Selection & Factory
#   create_redis_backend(), force_simple, RediSearchBackend.drop_index()
# =============================================================================
def section_backends():
    print("\n" + "=" * 70)
    print("SECTION 12 – Backend Selection & Factory")
    print("=" * 70)

    client = get_redis_client()

    # Auto-detect: uses RediSearch if available, otherwise falls back to simple
    backend_auto = create_redis_backend(client, namespace="s12_auto", vector_dimension=384)
    print(f"  Auto-detected backend -> {type(backend_auto).__name__}")

    # Force simple mode regardless of RediSearch availability
    backend_simple = create_redis_backend(
        client, namespace="s12_simple", vector_dimension=384, force_simple=True
    )
    print(f"  Forced simple backend -> {type(backend_simple).__name__}")

    # You can pass a pre-built backend directly to SemanticCache
    cache = SemanticCache(client, backend=backend_simple, namespace="s12_simple", threshold=0.80)
    cache.clear()
    cache.set("backend test", "works with explicit backend")
    print(f"  Explicit backend get  -> {cache.get('backend test')}")

    # RedisSimpleBackend constructed manually
    manual_simple = RedisSimpleBackend(client, namespace="s12_manual")
    cache2 = SemanticCache(client, backend=manual_simple, namespace="s12_manual", threshold=0.80)
    cache2.clear()
    cache2.set("manual backend", "constructed by hand")
    print(f"  Manual backend get    -> {cache2.get('manual backend')}")

    # If RediSearch is available, demonstrate drop_index()
    from llmgatekeeper.backends.redis_search import RediSearchBackend

    try:
        rs_backend = RediSearchBackend(client, namespace="s12_rs_drop", vector_dimension=384)
        rs_backend.drop_index()
        print(f"  RediSearchBackend.drop_index() called successfully")
    except RuntimeError:
        print(f"  RediSearch not available – drop_index() skipped")


# =============================================================================
# SECTION 13 – Embedding Providers
#   SentenceTransformerProvider (model swap, device, dimension)
#   CachedEmbeddingProvider (LRU memory cache + optional Redis persistence)
#   OpenAIEmbeddingProvider (example-only, requires OPENAI_API_KEY)
# =============================================================================
def section_embedding_providers():
    print("\n" + "=" * 70)
    print("SECTION 13 – Embedding Providers")
    print("=" * 70)

    # --- SentenceTransformerProvider -------------------------------------------
    # Default model: all-MiniLM-L6-v2  (384 dims, lazy-loaded)
    provider = SentenceTransformerProvider()
    print(f"  Default model   -> {provider.model_name}")
    print(f"  Dimension       -> {provider.dimension}")

    vec = provider.embed("Hello world")
    print(f"  embed() shape   -> {vec.shape}  dtype={vec.dtype}")

    vecs = provider.embed_batch(["Hello", "World", "Test"])
    print(f"  embed_batch(3)  -> {len(vecs)} vectors, each shape {vecs[0].shape}")

    # Swap to a different model (768-dim family)
    provider_mpnet = SentenceTransformerProvider(model_name="all-mpnet-base-v2")
    print(f"  mpnet model     -> {provider_mpnet.model_name}, dim={provider_mpnet.dimension}")

    # --- CachedEmbeddingProvider (in-memory LRU) --------------------------------
    cached_provider = CachedEmbeddingProvider(
        provider=provider,
        max_size=500,          # LRU cache holds up to 500 embeddings
    )
    print(f"\n  CachedEmbeddingProvider wrapping {type(cached_provider.provider).__name__}")
    print(f"  dimension       -> {cached_provider.dimension}")

    _ = cached_provider.embed("cache me please")
    print(f"  cache_size() after 1 embed -> {cached_provider.cache_size()}")
    _ = cached_provider.embed("cache me please")   # second call is a cache hit
    print(f"  cache_size() after repeat  -> {cached_provider.cache_size()}")  # still 1

    _ = cached_provider.embed_batch(["one", "two", "three"])
    print(f"  cache_size() after batch   -> {cached_provider.cache_size()}")  # 4

    cached_provider.clear_cache()  # clears only in-memory LRU
    print(f"  cache_size() after clear_cache() -> {cached_provider.cache_size()}")

    # --- CachedEmbeddingProvider with Redis persistence -------------------------
    client = get_redis_client()
    cached_redis = CachedEmbeddingProvider(
        provider=provider,
        max_size=200,
        redis_client=client,       # persist embeddings in Redis
        redis_prefix="llmgk:emb:",
        redis_ttl=3600,            # Redis entries expire after 1 hour
    )
    _ = cached_redis.embed("persisted embedding")
    print(f"  Redis-backed cache_size() -> {cached_redis.cache_size()}")
    cached_redis.clear_all()       # clears both memory and Redis keys
    print(f"  After clear_all() cache_size() -> {cached_redis.cache_size()}")

    # --- Use a cached provider with SemanticCache -------------------------------
    cache = SemanticCache(
        client,
        namespace="s13_cached",
        embedding_provider=cached_redis,
        threshold=0.80,
    )
    cache.clear()
    cache.set("cached embedding query", "response via cached provider")
    print(f"  SemanticCache with CachedEmbeddingProvider -> {cache.get('cached embedding query')}")

    # --- OpenAIEmbeddingProvider (example – NOT executed) -----------------------
    print("\n  --- OpenAIEmbeddingProvider (requires OPENAI_API_KEY) ---")
    print("  # from llmgatekeeper.embeddings.openai_provider import OpenAIEmbeddingProvider")
    print("  #")
    print("  # provider = OpenAIEmbeddingProvider(")
    print("  #     model='text-embedding-ada-002',  # 1536 dims")
    print("  #     api_key='sk-...',                # or set OPENAI_API_KEY env var")
    print("  # )")
    print("  # print(provider.dimension)           # 1536")
    print("  # vec = provider.embed('Hello')")
    print("  #")
    print("  # # Other available models:")
    print("  # #   text-embedding-3-small  -> 1536 dims")
    print("  # #   text-embedding-3-large  -> 3072 dims")
    print("  # #")
    print("  # # Async support is built-in:")
    print("  # # vec = await provider.aembed('Hello')")
    print("  # # vecs = await provider.aembed_batch(['a', 'b'])")
    print("  # #")
    print("  # # Use with SemanticCache exactly like any other provider:")
    print("  # # cache = SemanticCache(redis_client, embedding_provider=provider)")


# =============================================================================
# SECTION 14 – Similarity Metrics (standalone usage)
#   cosine_similarity, dot_product_similarity, euclidean_similarity,
#   euclidean_distance, compute_similarity, get_similarity_function,
#   normalize_similarity, batch_cosine_similarity, SimilarityMetric enum
# =============================================================================
def section_similarity_metrics():
    print("\n" + "=" * 70)
    print("SECTION 14 – Similarity Metrics")
    print("=" * 70)

    v1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    v2 = np.array([1.0, 1.0, 0.0], dtype=np.float32)
    v3 = np.array([0.0, 0.0, 1.0], dtype=np.float32)  # orthogonal to v1

    # Individual metric functions
    print(f"  cosine_similarity(v1, v2)      -> {cosine_similarity(v1, v2):.4f}")
    print(f"  cosine_similarity(v1, v3)      -> {cosine_similarity(v1, v3):.4f}")  # 0 – orthogonal
    print(f"  dot_product_similarity(v1, v2) -> {dot_product_similarity(v1, v2):.4f}")
    print(f"  euclidean_similarity(v1, v2)   -> {euclidean_similarity(v1, v2):.4f}")
    print(f"  euclidean_distance(v1, v2)     -> {euclidean_distance(v1, v2):.4f}")

    # SimilarityMetric enum
    print(f"\n  SimilarityMetric values: {[m.value for m in SimilarityMetric]}")

    # compute_similarity – convenience wrapper
    for metric in SimilarityMetric:
        score = compute_similarity(v1, v2, metric)
        print(f"  compute_similarity(v1, v2, {metric.value:<12}) -> {score:.4f}")

    # get_similarity_function – returns the callable for a metric
    fn = get_similarity_function(SimilarityMetric.COSINE)
    print(f"\n  get_similarity_function(COSINE)(v1, v2) -> {fn(v1, v2):.4f}")

    # normalize_similarity – maps raw scores to [0, 1]
    raw_cosine = cosine_similarity(v1, v3)  # 0.0
    print(f"  normalize_similarity(0.0, COSINE)      -> "
          f"{normalize_similarity(raw_cosine, SimilarityMetric.COSINE):.4f}")

    # batch_cosine_similarity – one query vs many vectors at once
    query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    matrix = np.array([
        [1.0, 0.0, 0.0],   # identical
        [0.0, 1.0, 0.0],   # orthogonal
        [0.7, 0.7, 0.0],   # partial
    ], dtype=np.float32)

    scores = batch_cosine_similarity(query, matrix)
    print(f"\n  batch_cosine_similarity (query vs 3 vectors):")
    for i, s in enumerate(scores):
        print(f"    vector[{i}] -> {s:.4f}")


# =============================================================================
# SECTION 15 – Confidence Classification
#   ConfidenceLevel enum, ConfidenceClassifier (custom + for_model),
#   classify(), is_match(), get_thresholds(), high/medium/low_threshold
# =============================================================================
def section_confidence():
    print("\n" + "=" * 70)
    print("SECTION 15 – Confidence Classification")
    print("=" * 70)

    # ConfidenceLevel enum
    print(f"  ConfidenceLevel values: {[c.value for c in ConfidenceLevel]}")

    # Custom classifier with explicit thresholds
    clf = ConfidenceClassifier(high=0.95, medium=0.85, low=0.75)
    print(f"\n  Custom classifier thresholds: {clf.get_thresholds()}")
    print(f"  high_threshold   -> {clf.high_threshold}")
    print(f"  medium_threshold -> {clf.medium_threshold}")
    print(f"  low_threshold    -> {clf.low_threshold}")

    # classify() maps a score to a level
    scores = [0.98, 0.90, 0.80, 0.60]
    for s in scores:
        level = clf.classify(s)
        print(f"  classify({s}) -> {level}")

    # is_match() – True if score >= low threshold
    print(f"\n  is_match(0.80) -> {clf.is_match(0.80)}")  # True (>= 0.75)
    print(f"  is_match(0.50) -> {clf.is_match(0.50)}")  # False

    # for_model() – pre-tuned thresholds for known embedding models
    models = [
        "all-MiniLM-L6-v2",
        "all-mpnet-base-v2",
        "text-embedding-ada-002",
        "text-embedding-3-large",
    ]
    print(f"\n  Model-specific classifiers:")
    for model in models:
        mc = ConfidenceClassifier.for_model(model)
        print(f"    {model:<35} -> {mc.get_thresholds()}")

    # get_model_classifier() convenience function
    clf2 = get_model_classifier("all-MiniLM-L6-v2")
    print(f"\n  get_model_classifier('all-MiniLM-L6-v2') -> {repr(clf2)}")
    # None returns default thresholds
    clf_default = get_model_classifier(None)
    print(f"  get_model_classifier(None)               -> {repr(clf_default)}")


# =============================================================================
# SECTION 16 – SimilarityRetriever (lower-level retrieval API)
#   find_similar(), find_best_match(), find_by_confidence()
#   RetrievalResponse: best_match, has_high_confidence,
#       high_confidence_results, above_threshold_results, len, iter, index
# =============================================================================
def section_retriever():
    print("\n" + "=" * 70)
    print("SECTION 16 – SimilarityRetriever")
    print("=" * 70)

    client = get_redis_client()
    provider = SentenceTransformerProvider()

    # Build a backend and seed it via a SemanticCache
    backend = create_redis_backend(client, namespace="s16_ret", vector_dimension=384)
    seed_cache = SemanticCache(client, backend=backend, namespace="s16_ret", threshold=0.0)
    seed_cache.clear()
    seed_cache.set("What is Python?",  "Python is a programming language.")
    seed_cache.set("What is Java?",    "Java is a compiled language.")
    seed_cache.set("What is Rust?",    "Rust focuses on memory safety.")

    # Build a retriever on the same backend
    clf = ConfidenceClassifier.for_model("all-MiniLM-L6-v2")
    retriever = SimilarityRetriever(
        backend=backend,
        top_k=5,
        threshold=0.0,                     # return everything, we filter manually
        confidence_classifier=clf,
    )
    print(f"  repr -> {repr(retriever)}")
    print(f"  top_k={retriever.top_k}  threshold={retriever.threshold}")

    # Embed a query manually to feed the retriever
    query_vec = provider.embed("Tell me about Python")

    # --- find_similar() → RetrievalResponse ------------------------------------
    response = retriever.find_similar(query_vec, top_k=3, threshold=0.60)
    print(f"\n  find_similar() returned {len(response)} result(s)")
    print(f"  best_match            -> key={response.best_match.key}, "
          f"sim={response.best_match.similarity:.3f}, conf={response.best_match.confidence}")
    print(f"  has_high_confidence   -> {response.has_high_confidence}")

    # Iterate over RetrievalResponse (supports len, iter, indexing)
    for rr in response:
        print(f"    rank={rr.rank}  sim={rr.similarity:.3f}  conf={rr.confidence}  "
              f"response={rr.metadata.get('response', '?')[:35]}…")

    # high_confidence_results / above_threshold_results
    print(f"  high_confidence_results count -> {len(response.high_confidence_results)}")
    print(f"  above_threshold_results count -> {len(response.above_threshold_results)}")

    # --- find_best_match() – returns single result filtered by min confidence ---
    best = retriever.find_best_match(query_vec, min_confidence=ConfidenceLevel.MEDIUM)
    if best:
        print(f"\n  find_best_match(min=MEDIUM) -> sim={best.similarity:.3f} conf={best.confidence}")
    else:
        print(f"\n  find_best_match(min=MEDIUM) -> None")

    best_high = retriever.find_best_match(query_vec, min_confidence=ConfidenceLevel.HIGH)
    print(f"  find_best_match(min=HIGH)   -> {'found' if best_high else 'None'}")

    # --- find_by_confidence() – all results at a specific confidence -----------
    low_results = retriever.find_by_confidence(query_vec, ConfidenceLevel.LOW, top_k=10)
    print(f"  find_by_confidence(LOW)     -> {len(low_results)} result(s)")

    # Setters
    retriever.top_k = 10
    retriever.threshold = 0.50
    print(f"  After setters: top_k={retriever.top_k}  threshold={retriever.threshold}")


# =============================================================================
# SECTION 17 – Async API
#   AsyncSemanticCache + AsyncRedisBackend
#   All methods mirror the sync API; stats()/reset_stats() stay synchronous.
# =============================================================================
async def section_async():
    print("\n" + "=" * 70)
    print("SECTION 17 – Async API")
    print("=" * 70)

    import redis.asyncio as aioredis

    from llmgatekeeper.backends.redis_async import AsyncRedisBackend

    aclient = aioredis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=False)

    backend = AsyncRedisBackend(aclient, namespace="s17_async")
    cache = AsyncSemanticCache(
        backend,
        namespace="s17_async",
        threshold=0.80,
        enable_analytics=True,
    )

    # --- clear / set / get (exact) ---------------------------------------------
    await cache.clear()
    key = await cache.set(
        "What is async programming?",
        "Async allows non-blocking concurrent execution.",
        metadata={"paradigm": "concurrent"},
    )
    print(f"  await set() -> key={key}")

    result = await cache.get("What is async programming?")
    print(f"  await get() exact -> {result}")

    # --- get (semantic) --------------------------------------------------------
    result = await cache.get("Explain asynchronous code")
    print(f"  await get() semantic -> {result}")

    # --- get with include_metadata ---------------------------------------------
    cr: CacheResult = await cache.get("What is async programming?", include_metadata=True)
    if cr:
        print(f"  CacheResult -> sim={cr.similarity:.3f} conf={cr.confidence} meta={cr.metadata}")

    # --- exists / delete / delete_by_key / count --------------------------------
    print(f"  await exists() -> {await cache.exists('What is async programming?')}")
    deleted = await cache.delete("What is async programming?")
    print(f"  await delete() -> {deleted}")
    print(f"  await count()  -> {await cache.count()}")

    # Re-add for further tests
    await cache.set("What is async programming?", "Async allows non-blocking execution.")

    # --- get_similar ------------------------------------------------------------
    await cache.set("What is concurrency?", "Concurrency handles multiple tasks at once.")
    await cache.set("What is parallelism?", "Parallelism runs tasks simultaneously.")
    similars = await cache.get_similar("Async and concurrent programming", top_k=3, threshold=0.60)
    print(f"  await get_similar(top_k=3) -> {len(similars)} result(s)")
    for s in similars:
        print(f"    sim={s.similarity:.3f} conf={s.confidence} response={s.response[:40]}…")

    # --- warm (bulk load) -------------------------------------------------------
    pairs = [
        ("What is a coroutine?",   "A coroutine is a suspendable function."),
        ("What is an event loop?", "The event loop runs async tasks."),
    ]
    count = await cache.warm(pairs, metadata={"source": "async_warm"}, batch_size=2)
    print(f"  await warm() -> {count} entries loaded")
    print(f"  await count() -> {await cache.count()}")

    # --- stats / reset_stats (synchronous methods on async cache) ---------------
    # These don't touch Redis so they are sync even on AsyncSemanticCache
    stats = cache.stats()
    if stats:
        print(f"  stats().total_queries -> {stats.total_queries}")
        print(f"  stats().hit_rate      -> {stats.hit_rate:.1%}")
    cache.reset_stats()
    print(f"  After reset_stats(), total_queries -> {cache.stats().total_queries}")

    # --- dynamic properties (same as sync) --------------------------------------
    print(f"  threshold   -> {cache.threshold}")
    cache.threshold = 0.75
    print(f"  threshold after set -> {cache.threshold}")
    print(f"  namespace   -> {cache.namespace}")
    print(f"  default_ttl -> {cache.default_ttl}")
    cache.default_ttl = 600
    print(f"  default_ttl after set -> {cache.default_ttl}")
    print(f"  repr -> {repr(cache)}")
    print(f"  analytics_enabled -> {cache.analytics_enabled}")

    # --- clear ----------------------------------------------------------------
    removed = await cache.clear()
    print(f"  await clear() -> removed {removed}")

    await aclient.aclose()


# =============================================================================
# SECTION 18 – Custom Backend (pattern example – not executed)
# =============================================================================
def section_custom_backend():
    print("\n" + "=" * 70)
    print("SECTION 18 – Custom Backend Pattern (example code, not executed)")
    print("=" * 70)

    print("""
  Implement CacheBackend (or AsyncCacheBackend) to plug in any vector store:

      from llmgatekeeper.backends.base import CacheBackend, SearchResult, CacheEntry

      class PineconeBackend(CacheBackend):
          def __init__(self, index):
              self._index = index

          def store_vector(self, key, vector, metadata, ttl=None):
              self._index.upsert([(key, vector.tolist(), metadata)])

          def search_similar(self, vector, threshold=0.85, top_k=1):
              res = self._index.query(vector=vector.tolist(), top_k=top_k)
              return [
                  SearchResult(key=m.id, similarity=m.score, metadata=m.metadata)
                  for m in res.matches if m.score >= threshold
              ]

          def delete(self, key):
              self._index.delete(ids=[key])
              return True

          def get_by_key(self, key):
              # fetch by ID and reconstruct CacheEntry
              ...

          def clear(self):
              ...

          def count(self):
              ...

      # Then wire it directly into SemanticCache:
      #   backend = PineconeBackend(pinecone_index)
      #   cache = SemanticCache(redis_client, backend=backend)
    """)


# =============================================================================
# Main – run all sections
# =============================================================================
def main():
    print("=" * 70)
    print("LLMGatekeeper – Complete Usage Reference")
    print("=" * 70)

    # Verify Redis connectivity once up-front
    try:
        client = get_redis_client()
        modules = client.module_list()
        has_rs = any(
            (m.get("name") or m.get(b"name", b"")).lower()
            in (b"search", b"ft", "search", "ft")
            for m in modules
        )
        print(f"  Redis connected at {REDIS_HOST}:{REDIS_PORT}")
        print(f"  RediSearch available: {has_rs}")
        client.close()
    except redis.ConnectionError:
        print(f"\n  ERROR: Cannot connect to Redis at {REDIS_HOST}:{REDIS_PORT}")
        print("  Start Redis first:")
        print("    docker run -d --name redis-cache -p 6379:6379 redis:latest")
        return

    # Synchronous sections
    section_basic_operations()
    section_cache_result()
    section_semantic_matching()
    section_get_similar()
    section_ttl()
    section_cache_warming()
    section_namespaces()
    section_dynamic_properties()
    section_analytics()
    section_error_handling()
    section_logging()
    section_backends()
    section_embedding_providers()
    section_similarity_metrics()
    section_confidence()
    section_retriever()

    # Async section
    import asyncio
    asyncio.run(section_async())

    # Pattern-only section (no Redis calls)
    section_custom_backend()

    print("\n" + "=" * 70)
    print("All sections completed.")
    print("=" * 70)


if __name__ == "__main__":
    main()
