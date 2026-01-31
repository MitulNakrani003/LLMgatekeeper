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
│   ├── factory.py           # Backend auto-detection factory
│   ├── redis_simple.py      # Brute-force Redis backend (<10k entries)
│   ├── redis_search.py      # RediSearch KNN backend (scalable)
│   └── redis_async.py       # Async Redis backend
│
├── embeddings/              # Embedding provider implementations
│   ├── __init__.py
│   ├── base.py              # Abstract EmbeddingProvider
│   ├── sentence_transformer.py  # Local Sentence Transformers
│   ├── openai_provider.py   # OpenAI API embeddings
│   └── cached.py            # Caching wrapper for providers
│
└── similarity/              # Similarity matching & retrieval
    ├── __init__.py
    ├── metrics.py           # Similarity metrics (cosine, dot, euclidean)
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
┌────────────────────┐        ┌────────────────────────┐
│  RedisSimpleBackend│        │   RediSearchBackend    │
│                    │        │                        │
│  Brute-force scan  │        │  KNN vector search     │
│  < 10k entries     │        │  Scales to millions    │
└────────────────────┘        └────────────────────────┘
          │                              │
          └──────────────┬───────────────┘
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
1. User calls cache.set("What is Python?", "A programming language.")

2. SemanticCache._embed(query)
   └── EmbeddingProvider.embed("What is Python?")
       └── Returns: float32[384]  # Vector embedding

3. SemanticCache._generate_key(query)
   └── Returns: "llmgk:default:abc123..."  # MD5 hash-based key

4. CacheBackend.store_vector(key, vector, metadata, ttl)
   │
   ├── RedisSimpleBackend:
   │   └── HSET key vector=<bytes> metadata=<json> [EX ttl]
   │
   └── RediSearchBackend:
       └── HSET + FT.ADD to index for KNN search

5. Return cache key to user
```

### Get Operation (cache.get)

```
1. User calls cache.get("Tell me about Python")

2. SemanticCache._embed(query)
   └── EmbeddingProvider.embed("Tell me about Python")
       └── Returns: float32[384]

3. SimilarityRetriever.find_similar(vector, threshold=0.85, top_k=1)
   │
   └── CacheBackend.search_similar(vector, threshold, top_k)
       │
       ├── RedisSimpleBackend:
       │   └── Scan all keys, compute cosine similarity, filter & sort
       │
       └── RediSearchBackend:
           └── FT.SEARCH idx "(@vec:[VECTOR_RANGE ...])" -> efficient KNN

4. ConfidenceClassifier.classify(similarity_score)
   └── Returns: ConfidenceLevel.HIGH/MEDIUM/LOW/NONE

5. If match found above threshold:
   └── Return cached response (or CacheResult with metadata)
   Else:
   └── Return None (cache miss)
```

---

## Class Diagrams

### Exception Hierarchy

```
CacheError (base)
├── BackendError
│   ├── ConnectionError
│   └── TimeoutError
├── EmbeddingError
└── ConfigurationError
```

### Backend Hierarchy

```
<<abstract>>
CacheBackend
├── store_vector(key, vector, metadata, ttl)
├── search_similar(vector, threshold, top_k) -> List[SearchResult]
├── delete(key) -> bool
├── get_by_key(key) -> Optional[CacheEntry]
├── clear() -> int
└── count() -> int
        │
        ├── RedisSimpleBackend
        │   └── Brute-force cosine similarity
        │
        └── RediSearchBackend
            └── KNN vector search via FT.SEARCH

<<abstract>>
AsyncCacheBackend
└── Same methods with async/await
        │
        └── AsyncRedisBackend
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
        │   └── Default: "all-MiniLM-L6-v2" (384 dims)
        │
        ├── OpenAIEmbeddingProvider
        │   └── "text-embedding-ada-002" (1536 dims)
        │
        └── CachedEmbeddingProvider
            └── Wraps any provider with LRU cache
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

### Backend Factory Parameters

| Parameter          | Type           | Default    | Description                           |
|--------------------|----------------|------------|---------------------------------------|
| `redis_client`     | `redis.Redis`  | Required   | Redis connection                      |
| `namespace`        | `str`          | `"llmgk"`  | Key prefix                            |
| `vector_dimension` | `int`          | `384`      | Embedding vector dimension            |
| `force_simple`     | `bool`         | `False`    | Always use simple backend             |

---

## Performance Considerations

### Backend Selection

| Backend           | Use Case                    | Entry Limit | Search Speed        |
|-------------------|-----------------------------|-------------|---------------------|
| RedisSimpleBackend| Development, small caches   | < 10,000    | O(n) linear scan    |
| RediSearchBackend | Production, large caches    | Millions    | O(log n) KNN        |

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
