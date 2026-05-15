# LLMGatekeeper Architecture

This document provides a comprehensive overview of the LLMGatekeeper package architecture, explaining the design decisions, component interactions, and data flow.

## Table of Contents

1. [Overview](#overview)
2. [Core Concepts](#core-concepts)
3. [Package Structure](#package-structure)
4. [Component Architecture](#component-architecture)
5. [Data Flow](#data-flow)
6. [Class Diagrams](#class-diagrams)
7. [Extension Points](#extension-points)

---

## Overview

LLMGatekeeper is a semantic caching library designed to reduce redundant LLM API calls in RAG (Retrieval-Augmented Generation) systems. Unlike traditional caches that require exact key matches, LLMGatekeeper uses embedding-based similarity matching to recognize semantically equivalent queries.

### Key Features

- **Semantic Matching**: "What's the weather?" matches "Tell me today's weather"
- **Pluggable Backends**: Redis (simple or RediSearch), extensible to other stores
- **Pluggable Embeddings**: Sentence Transformers, OpenAI, custom providers
- **Confidence Classification**: HIGH/MEDIUM/LOW/NONE confidence bands
- **Analytics**: Hit/miss rates, latency percentiles, near-miss tracking
- **Async Support**: Full async/await API for non-blocking operations

---

## Core Concepts

### Semantic Caching vs Traditional Caching

```
Traditional Cache:
  "What is Python?" -> Response
  "What's Python?"  -> MISS (different string)

Semantic Cache:
  "What is Python?" -> Response
  "What's Python?"  -> HIT (semantically similar, similarity=0.95)
```

### Similarity Threshold

The `threshold` parameter (default 0.85) determines how similar a query must be to return a cached result:

- **0.95**: Very strict, nearly identical queries only
- **0.85**: Good balance for most use cases
- **0.75**: More lenient, allows broader matches

### Confidence Levels

Similarity scores are classified into confidence bands:

| Level  | Description                                    |
|--------|------------------------------------------------|
| HIGH   | Very confident match, safe to use              |
| MEDIUM | Moderately confident, review for critical apps |
| LOW    | Low confidence, use with caution               |
| NONE   | Below threshold, not a match                   |

---

## Package Structure

```
llmgatekeeper/
├── __init__.py              # Public API exports
├── cache.py                 # SemanticCache & AsyncSemanticCache
├── analytics.py             # Performance tracking & statistics
├── exceptions.py            # Custom exception hierarchy
├── logging.py               # Loguru-based logging utilities
│
├── backends/                # Storage backend implementations
│   ├── __init__.py
│   ├── base.py              # Abstract CacheBackend & AsyncCacheBackend
│   ├── factory.py           # Sync + async auto-detection factories
│   ├── redis_simple.py      # Brute-force Redis backend (<10k entries)
│   ├── redis_search.py      # RediSearch VECTOR_RANGE + KNN backend
│   ├── redis_async.py       # Async brute-force Redis backend
│   └── redis_search_async.py# Async RediSearch backend
│
├── embeddings/              # Embedding provider implementations
│   ├── __init__.py
│   ├── base.py              # Abstract EmbeddingProvider
│   ├── sentence_transformer.py  # Local Sentence Transformers (aembed via to_thread)
│   ├── openai_provider.py   # OpenAI API embeddings (native async)
│   └── cached.py            # Caching wrapper (LRU + Redis, fingerprinted)
│
└── similarity/              # Similarity matching & retrieval
    ├── __init__.py
    ├── metrics.py           # Cosine + standalone dot/euclidean helpers
    ├── confidence.py        # Confidence level classification
    └── retriever.py         # Multi-result retrieval with ranking
```

---

## Component Architecture

### Layer Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         User Application                            │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      SemanticCache / AsyncSemanticCache             │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  • set(query, response) - Store Q&A pair                    │    │
│  │  • get(query) - Retrieve by semantic similarity             │    │
│  │  • get_similar(query, top_k) - Get multiple matches         │    │
│  │  • warm(pairs) - Bulk load entries                          │    │
│  │  • stats() - Get analytics                                  │    │
│  └─────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
          │                    │                    │
          ▼                    ▼                    ▼
┌──────────────────┐  ┌─────────────────---┐  ┌──────────────────────┐
│ EmbeddingProvider│  │ SimilarityRetriever│  │  CacheAnalytics      │
│                  │  │                    │  │                      │
│ • embed(text)    │  │ • find_similar()   │  │ • record_hit/miss()  │
│ • embed_batch()  │  │ • find_best_match  │  │ • get_stats()        │
│ • aembed()       │  │                    │  │ • near-miss tracking │
└──────────────────┘  └─────────────────---┘  └──────────────────────┘
          │                    │
          │                    ▼
          │           ┌─────────────────────┐
          │           │ ConfidenceClassifier│
          │           │                     │
          │           │ • classify(score)   │
          │           │ • model thresholds  │
          │           └─────────────────────┘
          │                    │
          ▼                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         CacheBackend / AsyncCacheBackend            │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  • store_vector(key, vector, metadata, ttl)                 │    │
│  │  • search_similar(vector, threshold, top_k)                 │    │
│  │  • delete(key), get_by_key(key), clear(), count()           │    │
│  └─────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
          │                              │
          ▼                              ▼
┌─────────────────────────┐    ┌─────────────────────────────┐
│  RedisSimpleBackend     │    │   RediSearchBackend         │
│  AsyncRedisBackend      │    │   AsyncRediSearchBackend    │
│                         │    │                             │
│  Brute-force scan       │    │  VECTOR_RANGE (threshold>0) │
│  Pipelined hgetalls     │    │  KNN (threshold=0)          │
│  < 10k entries          │    │  Scales to millions         │
└─────────────────────────┘    └─────────────────────────────┘
          │                                 │
          └────────────────┬────────────────┘
                           ▼
                ┌──────────────────────┐
                │     Redis Server     │
                │                      │
                │  • Standard Redis    │
                │  • Redis Stack       │
                └──────────────────────┘
```

---

## Data Flow

### Set Operation (cache.set)

```
1. User calls cache.set("What is Python?", "A programming language.",
                        metadata={"source": "docs"})

2. SemanticCache._embed(query)
   └── EmbeddingProvider.embed("What is Python?")
       └── Returns: float32[384]  # Vector embedding

3. SemanticCache._generate_key(query)
   └── Returns: "llmgk:default:abc123..."  # MD5 hash-based key

4. _pack_metadata(user_metadata, query, response)
   └── Returns: {
         "source": "docs",                                # user data, top level
         "_llmgk": {"query": "...", "response": "..."}    # internal, isolated
       }

5. CacheBackend.store_vector(key, vector, metadata, ttl)
   │
   │   First write per namespace also records vector_dim in {namespace}:meta.
   │   Subsequent writes validate vector.shape[0] against the recorded dim
   │   and raise BackendError on mismatch.
   │
   ├── RedisSimpleBackend / AsyncRedisBackend:
   │   └── HSET key vector=<bytes> metadata=<json>  + SADD {namespace}:keys
   │
   └── RediSearchBackend / AsyncRediSearchBackend:
       └── JSON.SET key $ {key, vector, metadata}  + SADD {namespace}:keys
           (the index auto-picks up the new document)

6. Return cache key to user
```

### Get Operation (cache.get)

```
1. User calls cache.get("Tell me about Python")

2. SemanticCache._embed(query)
   └── EmbeddingProvider.embed("Tell me about Python")
       └── Returns: float32[384]

3. SimilarityRetriever.find_similar(vector, threshold=0.0, top_k=1)
   │
   │   The backend is always queried unfiltered so we get the closest match.
   │   This lets analytics record the near-miss similarity without a second
   │   round trip on a miss.
   │
   └── CacheBackend.search_similar(vector, threshold=0.0, top_k=1)
       │
       │   Backends validate query dim against {namespace}:meta first.
       │
       ├── RedisSimpleBackend / AsyncRedisBackend:
       │   └── SMEMBERS + pipelined HGETALLs, then cosine similarity in numpy
       │
       └── RediSearchBackend / AsyncRediSearchBackend (threshold>0 path):
           └── FT.SEARCH "@vector:[VECTOR_RANGE $radius $vec]
                          =>{$YIELD_DISTANCE_AS: score}"
               radius = 1 - threshold (cosine)

4. ConfidenceClassifier.classify(top.similarity)
   └── Returns: ConfidenceLevel.HIGH / MEDIUM / LOW / NONE

5. SemanticCache compares top.similarity to effective_threshold:
   ├── similarity >= threshold:
   │   └── _unpack_response(metadata)  # prefers _llmgk.response,
   │       │                             falls back to top-level for legacy entries
   │       └── Return string  (or CacheResult with _user_metadata(...) when
   │                            include_metadata=True)
   │
   └── similarity < threshold (miss):
       ├── Optional: analytics.record_miss(closest_similarity=top.similarity)
       └── Return None
```

---

## Class Diagrams

### Exception Hierarchy

```
CacheError (base)
├── BackendError                  # holds .original_error
│   ├── BackendConnectionError    # renamed; no longer shadows builtin
│   └── BackendTimeoutError       # renamed; no longer shadows builtin
├── EmbeddingError                # holds .original_error
└── ConfigurationError            # invalid construction args
```

### Backend Hierarchy

```
<<abstract>>
CacheBackend
├── store_vector(key, vector, metadata, ttl)            @abstractmethod
├── search_similar(vector, threshold, top_k)            @abstractmethod
├── delete(key) -> bool                                 @abstractmethod
├── get_by_key(key) -> Optional[CacheEntry]             @abstractmethod
├── clear() -> int                                      @abstractmethod
└── count() -> int                                      @abstractmethod
        │
        ├── RedisSimpleBackend
        │   └── Brute-force cosine, pipelined hgetalls
        │
        └── RediSearchBackend
            ├── VECTOR_RANGE query when threshold > 0
            ├── KNN query when threshold == 0
            └── Validates existing index DIM on attach;
                rejects non-COSINE distance_metric

<<abstract>>
AsyncCacheBackend
└── Same methods, awaitable
        │
        ├── AsyncRedisBackend
        │
        └── AsyncRediSearchBackend
            └── Same query strategy as sync version; call await
                backend.connect() once before use (module check
                + index ensure)
```

### Embedding Provider Hierarchy

```
<<abstract>>
EmbeddingProvider
├── dimension: int
├── embed(text) -> float32[dim]
├── embed_batch(texts) -> List[float32[dim]]
├── aembed(text) -> float32[dim]        # async
└── aembed_batch(texts) -> List[...]    # async
        │
        ├── SentenceTransformerProvider
        │   ├── Default: "all-MiniLM-L6-v2" (384 dims)
        │   └── aembed / aembed_batch run via asyncio.to_thread,
        │       so the model encode never blocks the event loop
        │
        ├── OpenAIEmbeddingProvider
        │   ├── "text-embedding-ada-002" (1536 dims)
        │   └── Native async via AsyncOpenAI client
        │
        └── CachedEmbeddingProvider
            ├── Wraps any provider with thread-safe LRU + optional Redis persistence
            └── Cache keys include a provider fingerprint
                (class + dimension + model_name) so multiple providers can
                share the same Redis emb cache without collisions
```

### Data Models

```
CacheEntry                    SearchResult
├── key: str                  ├── key: str
├── vector: float32[dim]      ├── similarity: float
└── metadata: Dict            ├── metadata: Dict
                              └── vector: Optional[...]

CacheResult                   RetrievalResult
├── response: str             ├── key: str
├── similarity: float         ├── similarity: float
├── confidence: ConfidenceLevel├── confidence: ConfidenceLevel
├── key: str                  ├── metadata: Dict
└── metadata: Dict            └── rank: int
```

---

## Extension Points

### Custom Backend

Implement `CacheBackend` to add support for other vector stores:

```python
from llmgatekeeper.backends.base import CacheBackend, SearchResult

class PineconeBackend(CacheBackend):
    def __init__(self, index):
        self._index = index

    def store_vector(self, key, vector, metadata, ttl=None):
        self._index.upsert([(key, vector.tolist(), metadata)])

    def search_similar(self, vector, threshold=0.85, top_k=1):
        results = self._index.query(vector.tolist(), top_k=top_k)
        return [
            SearchResult(key=r.id, similarity=r.score, metadata=r.metadata)
            for r in results.matches
            if r.score >= threshold
        ]
    # ... implement other methods
```

### Custom Embedding Provider

Implement `EmbeddingProvider` for custom embedding models:

```python
from llmgatekeeper.embeddings.base import EmbeddingProvider

class CohereEmbeddingProvider(EmbeddingProvider):
    def __init__(self, api_key, model="embed-english-v3.0"):
        self._client = cohere.Client(api_key)
        self._model = model

    @property
    def dimension(self) -> int:
        return 1024  # Cohere embed-english-v3.0 dimension

    def embed(self, text: str):
        response = self._client.embed(texts=[text], model=self._model)
        return np.array(response.embeddings[0], dtype=np.float32)

    def embed_batch(self, texts):
        response = self._client.embed(texts=texts, model=self._model)
        return [np.array(e, dtype=np.float32) for e in response.embeddings]
```

### Custom Confidence Thresholds

Adjust confidence bands for your specific model and use case:

```python
from llmgatekeeper.similarity.confidence import ConfidenceClassifier

# Custom thresholds for a specific domain
classifier = ConfidenceClassifier(
    high=0.98,    # Very strict for medical/legal
    medium=0.92,
    low=0.85,
)

cache = SemanticCache(
    redis_client,
    model_name="custom",  # Prevents auto-detection
)
cache._classifier = classifier  # Override classifier
```

### Custom Async Backend

```python
from llmgatekeeper.backends.base import AsyncCacheBackend, SearchResult

class AsyncPineconeBackend(AsyncCacheBackend):
    def __init__(self, index):
        self._index = index

    async def store_vector(self, key, vector, metadata, ttl=None):
        await self._index.upsert(...)

    async def search_similar(self, vector, threshold=0.85, top_k=1):
        results = await self._index.query(vector.tolist(), top_k=top_k)
        return [
            SearchResult(key=r.id, similarity=r.score, metadata=r.metadata)
            for r in results.matches if r.score >= threshold
        ]
    # ... implement delete, get_by_key, clear, count
```

---

## Configuration Reference

### SemanticCache Parameters

| Parameter           | Type                  | Default                | Description                          |
|---------------------|-----------------------|------------------------|--------------------------------------|
| `redis_client`      | `redis.Redis`         | Required               | User's Redis connection              |
| `embedding_provider`| `EmbeddingProvider`   | SentenceTransformer    | Embedding model to use               |
| `threshold`         | `float`               | `0.85`                 | Minimum similarity for cache hit     |
| `default_ttl`       | `Optional[int]`       | `None`                 | Default TTL in seconds               |
| `backend`           | `CacheBackend`        | Auto-detected          | Storage backend                      |
| `model_name`        | `Optional[str]`       | Auto-detected          | Model name for confidence tuning     |
| `namespace`         | `str`                 | `"default"`            | Cache isolation namespace            |
| `enable_analytics`  | `bool`                | `False`                | Enable performance tracking          |

### AsyncSemanticCache Parameters

`AsyncSemanticCache` takes a pre-built `AsyncCacheBackend` (typically obtained via `await create_async_redis_backend(client)`); everything else mirrors the sync cache:

| Parameter           | Type                    | Default                | Description                          |
|---------------------|-------------------------|------------------------|--------------------------------------|
| `backend`           | `AsyncCacheBackend`     | Required               | Async backend (simple or RediSearch) |
| `embedding_provider`| `EmbeddingProvider`     | SentenceTransformer    | Same providers work for async        |
| `threshold`         | `float`                 | `0.85`                 | Minimum similarity for cache hit     |
| `default_ttl`       | `Optional[int]`         | `None`                 | Default TTL in seconds               |
| `model_name`        | `Optional[str]`         | Auto-detected          | Model name for confidence tuning     |
| `namespace`         | `str`                   | `"default"`            | Cache isolation namespace            |
| `enable_analytics`  | `bool`                  | `False`                | Enable performance tracking          |

### Backend Factory Parameters

`create_redis_backend` (sync) and `create_async_redis_backend` (async) take the same parameters:

| Parameter          | Type                          | Default    | Description                           |
|--------------------|-------------------------------|------------|---------------------------------------|
| `redis_client`     | `redis.Redis` / `asyncio.Redis` | Required | Redis connection                      |
| `namespace`        | `str`                         | `"llmgk"`  | Key prefix                            |
| `vector_dimension` | `int`                         | `384`      | Embedding vector dimension            |
| `force_simple`     | `bool`                        | `False`    | Always use simple backend             |

The async factory is itself async (`await create_async_redis_backend(...)`) because module detection requires a round trip. When it picks `AsyncRediSearchBackend`, it also calls `await backend.connect()` for you so the index is ready before the first store.

---

## Performance Considerations

### Backend Selection

| Backend                                              | Use Case                    | Entry Limit | Search Path                          |
|------------------------------------------------------|-----------------------------|-------------|--------------------------------------|
| `RedisSimpleBackend` / `AsyncRedisBackend`           | Development, small caches   | < 10,000    | Pipelined hgetalls + numpy cosine    |
| `RediSearchBackend` / `AsyncRediSearchBackend`       | Production, large caches    | Millions    | `VECTOR_RANGE` (threshold>0) or KNN  |

The RediSearch path uses a range query when a threshold is in play so KNN's fixed `k` can't accidentally exclude valid matches; it falls back to oversampled KNN when threshold is 0 (and `SemanticCache.get` always passes 0 since it filters in Python).

### Embedding Caching

For high-traffic scenarios, wrap your embedding provider:

```python
from llmgatekeeper.embeddings.cached import CachedEmbeddingProvider
from llmgatekeeper.embeddings.sentence_transformer import SentenceTransformerProvider

cached_provider = CachedEmbeddingProvider(
    provider=SentenceTransformerProvider(),
    max_size=10000  # Cache up to 10k embeddings in memory
)

cache = SemanticCache(redis_client, embedding_provider=cached_provider)
```

### Batch Operations

Use `cache.warm()` for bulk loading:

```python
# Efficient: Uses batch embedding
cache.warm(pairs, batch_size=100)

# Inefficient: Individual embedding calls
for query, response in pairs:
    cache.set(query, response)
```

---

## Safety Rails

The library raises `BackendError` / `ConfigurationError` early when state is inconsistent, rather than letting numpy or RediSearch silently produce wrong results:

| Check                                                          | Where                                                          | Failure                                          |
|----------------------------------------------------------------|----------------------------------------------------------------|--------------------------------------------------|
| Recorded `vector_dim` matches incoming vector                  | `RedisSimpleBackend` / `AsyncRedisBackend` `store_vector`/`search_similar` | `BackendError("Vector dimension mismatch")`      |
| Existing RediSearch index `DIM` matches constructor argument   | `RediSearchBackend.__init__` (and async equivalent's `connect`) | `ConfigurationError("vector dimension {existing} ...")` |
| `distance_metric` is `"COSINE"`                                | `RediSearchBackend.__init__` (and async)                       | `ConfigurationError("Only COSINE ... supported")` |
| Non-empty index response means a real error, not "no results"  | `RediSearchBackend.search_similar` (and async)                 | `BackendError` re-raised with `.original_error`   |
| User metadata never overwritten by internal fields             | `_pack_metadata` / `_unpack_response` in `cache.py`            | n/a — values round-trip                          |
| Renamed exceptions (`BackendConnectionError`, `BackendTimeoutError`) | `llmgatekeeper/exceptions.py`                            | No more shadowing of Python builtins             |

## Logging & Debugging

Enable logging for troubleshooting:

```python
from llmgatekeeper import configure_logging

# Basic info logging
configure_logging(level="INFO")

# Detailed debug logging
configure_logging(level="DEBUG")

# JSON format for log aggregation
configure_logging(level="INFO", serialize=True)
```

Log output includes:
- Cache hits/misses with similarity scores
- Embedding times
- Backend operations
- Error details with stack traces
