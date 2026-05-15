# LLMGatekeeper Cache Module

## Core Value Proposition

### What Problem Does This Solve?

1. **Cost Reduction** - LLM API calls are expensive ($0.01-0.10+ per query). If 40% of queries are semantically similar to previous ones, you're wasting 40% of your budget on redundant computation.
2. **Latency Improvement** - LLM responses take 500ms-5s. A cache hit returns in <10ms. For user-facing applications, this transforms the experience.
3. **The "Semantic" Differentiation** - Traditional caching fails for LLMs because:
    - "What's the weather today?" and "Tell me today's weather" are identical in meaning but different strings
    - Exact-match caching misses these opportunities
    - Your library bridges this gap with embedding-based similarity matching
4. **Developer Experience** - Wrapping an existing Redis instance with semantic capabilities shouldn't require a PhD. One line to initialize, two methods to use.

## Definition

**Semantic Cache** is a drop-in Python package that eliminates redundant computation in RAG (Retrieval-Augmented Generation) systems by intelligently caching semantically similar queries. Unlike traditional caching that only matches exact strings, Semantic Cache uses embedding-based similarity to recognize that "Find documents about AI safety" and "Show me papers on safe artificial intelligence" are functionally identical queries. By wrapping your existing Redis server, it intercepts queries before they hit expensive vector databases and LLM APIs, returning cached results in <10ms instead of waiting 1-3+ seconds for the full RAG pipeline. This translates to 40-60% cost reduction on API calls and embedding generation, while dramatically improving user experience with near-instant response times.

The package is designed for maximum simplicity and compatibility—requiring just one line to initialize with your Redis instance and two methods (`get`, `set`) to integrate into any RAG framework. Whether you're using LangChain, LlamaIndex, Haystack, or a custom implementation, Semantic Cache plugs seamlessly into your existing codebase without architectural changes. It works with any vector database (Pinecone, Weaviate, Qdrant, ChromaDB, etc.) and can cache at multiple pipeline layers: before document retrieval to avoid vector searches, before LLM calls to skip generation, or both for complete optimization. Your company retains full control with queries cached on your own Redis infrastructure, ensuring data privacy while achieving production-grade performance improvements with minimal code modification.

```
┌─────────────────────────────────────────────────────────────────────┐
│                        SemanticCache                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐   │
│  │   Embedding  │    │  Similarity  │    │   Storage Backend    │   │
│  │    Engine    │    │    Engine    │    │      Adapter         │   │
│  └──────────────┘    └──────────────┘    └──────────────────────┘   │
│                                                                     │            │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Detailed Component Breakdown

### 1. Storage Backend Adapter Layer

**Why it matters:** Different teams have different infrastructure. Some have Redis, some have Pinecone for other purposes, some want zero dependencies for local dev.

**Architecture points:**

- Abstract `CacheBackend` and `AsyncCacheBackend` interfaces with methods: `store_vector()`, `search_similar()`, `delete()`, `get_by_key()`, `clear()`, `count()` (all `@abstractmethod`)
- Redis adapter operates in two modes:
    - **Simple mode** (`RedisSimpleBackend`/`AsyncRedisBackend`): Redis hashes + brute-force similarity, fetched in a single pipelined round trip (good for <10k entries)
    - **RediSearch mode** (`RediSearchBackend`/`AsyncRediSearchBackend`): RediSearch vector index using `VECTOR_RANGE` for threshold-filtered queries and KNN for unfiltered ones (scales to millions)
- `create_redis_backend` / `create_async_redis_backend` factories auto-detect the RediSearch module on the supplied client and pick the right backend
- Connection pooling handled by user's existing Redis instance (you just wrap it)
- Backends record the embedding dimension on first write under `{namespace}:meta`; mismatched stores or queries raise `BackendError` before any numpy operation can produce garbage
- RediSearch backends additionally validate an existing index's `DIM` on attach and reject non-COSINE distance metrics

**Key insight:** The user passes their `redis.Redis` (or `redis.asyncio.Redis`) instance, you don't create connections. This respects their existing connection pooling, auth, SSL config, etc.

### 2. Embedding Engine Layer

**Why it matters:** Lock-in to one embedding provider is dangerous. Some users can't send data externally.

**Architecture points:**

- Pluggable embedding providers via strategy pattern (`EmbeddingProvider` ABC)
- Default to a small, fast local model (`all-MiniLM-L6-v2`) for zero-config start
- Batch embedding support for bulk operations
- Embedding caching via `CachedEmbeddingProvider` — LRU in memory plus optional Redis persistence. Cache keys are fingerprinted with the wrapped provider's class + dimension + model name, so two providers can safely share the same Redis embedding cache without colliding on identical text. The in-memory LRU is guarded by a `threading.Lock`.
- Async support for non-blocking embedding calls. `SentenceTransformerProvider.aembed/aembed_batch` use `asyncio.to_thread` so the heavy local model encode never blocks the event loop. `OpenAIEmbeddingProvider` uses the native async OpenAI client.
- Dimension is reported via the provider's `dimension` property; backends use this to validate that stored and queried vectors share a shape.

**Key insight:** The embedding model choice affects your similarity threshold. A 0.96 threshold on OpenAI embeddings means something different than 0.96 on MiniLM. The library ships per-model defaults (`MODEL_THRESHOLDS` in `similarity/confidence.py`) and auto-applies them when the model name is recognised.

### 3. Similarity Engine Layer

**Why it matters:** Threshold tuning is the difference between useful and dangerous caching.

**Architecture points:**

- Cosine similarity is the canonical metric — `RediSearchBackend` only accepts `DISTANCE_METRIC=COSINE` and the simple backends use cosine in `_cosine_similarity`. Dot-product and Euclidean helpers exist in `similarity/metrics.py` as standalone functions but are not wired through `SemanticCache`.
- Threshold is the critical parameter — too high misses valid cache hits, too low returns wrong answers.
- `ConfidenceClassifier` exposes confidence bands (`HIGH` / `MEDIUM` / `LOW` / `NONE`) with model-specific defaults.
- `SimilarityRetriever` supports multi-result retrieval (top-K) for ensemble/voting scenarios.
- Hit/miss decision happens in `SemanticCache.get`, not in the backend: the backend is queried with `threshold=0` and the cache filters in Python. This guarantees one backend round trip per `get`, even when analytics is tracking near-misses.

### 4. Advanced Features

**a) Namespace/Tenant Isolation**

- Multi-tenant SaaS apps need cache isolation per customer
- `SemanticCache(namespace="tenant_123")` prefixes all keys
- Each namespace owns an independent `{namespace}:meta` (recorded dimension), `{namespace}:keys` (tracked keys), and — for the RediSearch backend — `{namespace}_idx` index
- Prevents data leakage between tenants

**b) Cache Warming**

- Bulk-load common queries during deployment
- `cache.warm([(query1, response1), ...], batch_size=100, on_progress=callback)`
- Uses `embed_batch` under the hood for throughput, with progress callbacks per batch

**c) Analytics/Observability**

- Hit rate tracking, latency percentiles (p50/p95/p99/avg)
- Most frequently hit queries (top-K by access count)
- Near-miss tracking (queries that almost matched but fell below threshold)
- Opt-in via `enable_analytics=True`; disabled by default with zero overhead
- One backend round trip per `get()` whether it hits or misses — analytics never doubles request volume

**d) Internal-vs-user metadata isolation**

- The cache reserves a single key (`_llmgk`) in stored metadata for its own bookkeeping (query and response text)
- User-supplied metadata, including keys named `query` or `response`, is stored verbatim alongside and round-trips unchanged
- Legacy entries (pre-rename) without `_llmgk` are still readable: `query`/`response` are looked up at the top level for backward compatibility

# Tech Stack: LLMGatekeeper Cache Module

The **LLMGatekeeper Cache Module** is built on a high-performance Python-based stack designed for speed and modularity. The core logic is implemented in **Python 3.9+**, utilizing **Pydantic** for schema validation and **Loguru** for observability metrics like hit rates and latency tracking. For the storage layer, the project centers on **Redis**, leveraging its ability to act as a standard hash store for simple caching or as a vector database via **RediSearch** for high-scale semantic search.



Semantic processing is handled by the **Sentence-Transformers** library, which runs the `all-MiniLM-L6-v2` model locally to provide zero-config, low-latency embeddings. This is paired with **NumPy** to drive the similarity engine. Cosine similarity is the canonical metric — RediSearch indexes are created with `DISTANCE_METRIC=COSINE`, and the simple backends use cosine in their brute-force search; standalone dot-product and Euclidean helpers live in `similarity/metrics.py` for ad-hoc use. The architecture is provider-agnostic, supporting optional integration with **OpenAI** for high-dimension embeddings through a pluggable adapter layer; alternative vector stores can be added by implementing the `CacheBackend` / `AsyncCacheBackend` interfaces.