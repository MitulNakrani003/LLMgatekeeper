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

- Abstract `CacheBackend` interface with methods: `store_vector()`, `search_similar()`, `delete()`, `get_by_key()`
- Redis adapter operates in two modes:
    - **Simple mode**: Uses Redis hashes + brute-force similarity (good for <10k entries)
    - **RediSearch mode**: Uses Redis vector similarity search (scales to millions)
- Adapter auto-detects if RediSearch module is available and upgrades automatically
- Connection pooling handled by user's existing Redis instance (you just wrap it)

**Key insight:** The user passes their `redis.Redis` instance, you don't create connections. This respects their existing connection pooling, auth, SSL config, etc.

### 2. Embedding Engine Layer

**Why it matters:** Lock-in to one embedding provider is dangerous. Some users can't send data externally.

**Architecture points:**

- Pluggable embedding providers via strategy pattern
- Default to a small, fast local model (e.g., `all-MiniLM-L6-v2`) for zero-config start
- Batch embedding support for bulk operations
- Embedding caching (yes, cache the cache operation) - store query→embedding mappings to avoid re-embedding identical strings
- Async support for non-blocking embedding calls
- Dimension normalization across providers (some return 384-dim, others 1536-dim)

**Key insight:** The embedding model choice affects your similarity threshold. A 0.96 threshold on OpenAI embeddings means something different than 0.96 on MiniLM. Consider providing calibration utilities or documenting recommended thresholds per model.

### 3. Similarity Engine Layer

**Why it matters:** Cosine similarity isn't always the right choice. Some embedding models are trained with dot product, euclidean distance captures different semantics.

**Architecture points:**

- Configurable similarity metric
- Threshold is the critical parameter - too high misses valid cache hits, too low returns wrong answers
- Support for "confidence bands": return `high_confidence`, `medium_confidence`, `low_confidence` hits
- Multi-result retrieval: sometimes you want top-3 similar cached responses for ensemble/voting

### 4. Advanced Features To Include

**a) Namespace/Tenant Isolation**

- Multi-tenant SaaS apps need cache isolation per customer
- `SemanticCache(namespace="tenant_123")` prefixes all keys
- Prevents data leakage between tenants

**b) Cache Warming**

- Bulk-load common queries during deployment
- `cache.warm([(query1, response1), (query2, response2), ...])`

**c) Analytics/Observability**

- Hit rate tracking
- Latency percentiles
- Most frequently hit queries
- Near-miss tracking (queries that almost matched but fell below threshold)

# Tech Stack: LLMGatekeeper Cache Module

The **LLMGatekeeper Cache Module** is built on a high-performance Python-based stack designed for speed and modularity. The core logic is implemented in **Python 3.9+**, utilizing **Pydantic** for schema validation and **Loguru** for observability metrics like hit rates and latency tracking. For the storage layer, the project centers on **Redis**, leveraging its ability to act as a standard hash store for simple caching or as a vector database via **RediSearch** for high-scale semantic search.



Semantic processing is handled by the **Sentence-Transformers** library, which runs the `all-MiniLM-L6-v2` model locally to provide zero-config, low-latency embeddings. This is paired with **NumPy** or **scikit-learn** to drive the similarity engine, allowing for configurable distance metrics—such as Cosine Similarity or Euclidean Distance—to define precision thresholds. Finally, the architecture is designed to be provider-agnostic, supporting optional integration with **OpenAI** for high-dimension embeddings and various vector databases like **Pinecone** or **Qdrant** through a pluggable adapter layer.