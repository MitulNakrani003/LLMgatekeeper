<div align="center">

# LLMGatekeeper

**A semantic caching library for LLM and RAG systems.**
Eliminate redundant LLM API calls and vector database queries by recognising semantically equivalent queries — *not* just exact string matches.

[![PyPI version](https://img.shields.io/pypi/v/llmgatekeeper.svg?color=blue)](https://pypi.org/project/llmgatekeeper/)
[![Python versions](https://img.shields.io/pypi/pyversions/llmgatekeeper.svg)](https://pypi.org/project/llmgatekeeper/)
[![License: MIT](https://img.shields.io/pypi/l/llmgatekeeper.svg?color=green)](LICENSE)
[![Downloads](https://static.pepy.tech/badge/llmgatekeeper)](https://pepy.tech/project/llmgatekeeper)
[![Downloads/month](https://img.shields.io/pypi/dm/llmgatekeeper.svg)](https://pypistats.org/packages/llmgatekeeper)
[![Tests](https://img.shields.io/badge/tests-497%20passing-brightgreen.svg)](#testing)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

[**Documentation**](#documentation) ·
[**Quick Start**](#quick-start) ·
[**Architecture**](ARCHITECTURE.md) ·
[**Examples**](examples/quickstart_redis.py) ·
[**PyPI**](https://pypi.org/project/llmgatekeeper/) ·
[**Issues**](https://github.com/MitulNakrani003/LLMgatekeeper/issues)

</div>

---

LLMGatekeeper understands that *"What is Python?"* and *"Tell me about Python"* are asking the same thing — and returns the cached response instantly. Cache hits return in **<10ms** instead of the **500ms–5s** a typical LLM call takes, while cutting 40–60% off your API spend.

## Table of contents

- [Why semantic caching?](#why-semantic-caching)
- [Features](#features)
- [Installation](#installation)
- [Requirements](#requirements)
- [Quick start](#quick-start)
- [Usage guide](#usage-guide)
  - [Working with metadata](#working-with-metadata)
  - [Async usage](#async-usage)
  - [Multi-tenant isolation](#multi-tenant-isolation)
  - [TTL (time-to-live)](#ttl-time-to-live)
  - [Analytics](#analytics)
  - [Error handling](#error-handling)
- [Architecture](#architecture)
- [Configuration reference](#configuration-reference)
- [Testing](#testing)
- [Project structure](#project-structure)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)

## Why semantic caching?

| Pain point | Traditional cache | LLMGatekeeper |
|---|---|---|
| `"What's the weather?"` vs `"Tell me today's weather"` | Cache miss | Cache hit (similarity ≈ 0.93) |
| Cost per LLM call | $0.01–0.10+ — paid every time | Paid once per *meaning*, not per phrasing |
| Latency on cached responses | Full LLM round-trip (500ms–5s) | Embedding + Redis lookup (<10ms) |
| Storage layer | Pluggable, but you write the glue | Drop-in around your existing Redis |

## Features

- 🧠 **Semantic matching** — Embedding-based similarity catches paraphrases, synonyms, and rewording automatically
- ⚡ **Two-tier Redis backends** — Brute-force mode for small caches (<10k entries); auto-upgrades to RediSearch when available, using `VECTOR_RANGE` for threshold-filtered queries and KNN for unfiltered ones
- 🔌 **Pluggable embeddings** — Ships with `all-MiniLM-L6-v2` (Sentence-Transformers); swap in OpenAI embeddings or any custom provider
- 🎯 **Confidence levels** — Scores classified into `HIGH` / `MEDIUM` / `LOW` / `NONE` bands with per-model tuned thresholds
- 🏢 **Multi-tenant isolation** — Namespace prefixing keeps tenants' caches partitioned in a shared Redis instance
- 🔁 **Full async stack** — `AsyncSemanticCache` + `AsyncRedisBackend` + `AsyncRediSearchBackend`; `SentenceTransformerProvider.aembed` offloads to a worker thread so the event loop never blocks
- 📊 **Analytics & observability** — Hit/miss rates, latency percentiles (p50/p95/p99), near-miss tracking, top-query frequency — with one backend call per `get()` even when analytics is enabled
- ⏳ **TTL control** — Global default TTL with per-entry overrides; `ttl=0` disables expiry for specific entries
- 📦 **Bulk warming** — Load many entries at once with `warm()`, including batch-size control and progress callbacks
- 🛡️ **Safety rails** — Backends record the embedding dimension on first write and reject mismatched vectors; the RediSearch backend validates an existing index's `DIM` on attach so switching embedding models can't silently corrupt results

## Installation

```bash
pip install llmgatekeeper
```

With OpenAI embedding support:

```bash
pip install "llmgatekeeper[openai]"
```

For development (tests, linting, type-checking):

```bash
pip install "llmgatekeeper[dev]"
```

## Requirements

- **Python** 3.9+
- **Redis** 6.2+ (Redis Stack recommended for RediSearch vector search)

## Quick start

```python
import redis
from llmgatekeeper import SemanticCache

# Connect to your existing Redis instance — LLMGatekeeper doesn't manage the connection
client = redis.Redis(host="localhost", port=6379)

# Create a semantic cache (auto-detects RediSearch if available)
cache = SemanticCache(client)

# Store a response
cache.set("What is Python?", "Python is a high-level programming language.")

# Retrieve with a semantically similar query — no exact match needed
result = cache.get("Tell me about Python")
print(result)
# "Python is a high-level programming language."
```

## Usage guide

### Working with metadata

```python
cache.set(
    "What is Python?",
    "Python is a high-level programming language.",
    metadata={"source": "docs", "version": "3.12"},
)

result = cache.get("Explain Python", include_metadata=True)
if result:
    print(result.response)    # the cached response
    print(result.similarity)  # e.g. 0.91
    print(result.confidence)  # ConfidenceLevel.HIGH
    print(result.metadata)    # {"source": "docs", "version": "3.12"}
```

User metadata keys are preserved verbatim — including reserved-sounding names like `query` or `response`. The cache stores its own bookkeeping under a single namespaced key (`_llmgk`) so it never collides with what you pass in.

### Async usage

```python
import asyncio
import redis.asyncio as aioredis
from llmgatekeeper import AsyncSemanticCache
from llmgatekeeper.backends import create_async_redis_backend

async def main():
    client = aioredis.Redis(host="localhost", port=6379)

    # Auto-detects RediSearch on the async client and returns the right backend.
    backend = await create_async_redis_backend(client)
    cache = AsyncSemanticCache(backend)

    await cache.set("What is Python?", "A high-level programming language.")
    result = await cache.get("Tell me about Python")
    print(result)

asyncio.run(main())
```

For direct backend construction (skipping the factory):

```python
from llmgatekeeper.backends import AsyncRedisBackend, AsyncRediSearchBackend

# Plain Redis (brute-force similarity, <10k entries)
backend = AsyncRedisBackend(client)

# Redis Stack with RediSearch
backend = AsyncRediSearchBackend(client, vector_dimension=384)
await backend.connect()   # verifies the module and creates/validates the index
```

### Multi-tenant isolation

```python
tenant_a = SemanticCache(client, namespace="tenant_a")
tenant_b = SemanticCache(client, namespace="tenant_b")

tenant_a.set("Hello", "Hi from A")
tenant_b.set("Hello", "Hi from B")

print(tenant_a.get("Hello"))  # "Hi from A"
print(tenant_b.get("Hello"))  # "Hi from B"
```

Each namespace owns an independent `{namespace}:meta` (recorded dimension), `{namespace}:keys` (tracked keys), and — for the RediSearch backend — `{namespace}_idx` index.

### TTL (time-to-live)

```python
# Global default: all entries expire after 1 hour
cache = SemanticCache(client, default_ttl=3600)

# Per-entry override: this entry lives forever
cache.set("Permanent fact", "Water is H2O", ttl=0)

# Short-lived entry: expires in 60 seconds
cache.set("Breaking news", "...", ttl=60)
```

### Analytics

```python
cache = SemanticCache(client, enable_analytics=True)

# ... use the cache normally ...

stats = cache.stats()
print(f"Hit rate:      {stats.hit_rate:.1%}")
print(f"P95 latency:   {stats.p95_latency_ms:.2f} ms")
print(f"Near misses:   {len(stats.near_misses)}")
print(f"Top queries:   {[(q.query, q.count) for q in stats.top_queries[:3]]}")
```

Analytics is opt-in via `enable_analytics=True`. Each `get()` issues exactly one backend round trip whether it hits or misses — near-miss tracking reuses the same result.

### Error handling

```python
from llmgatekeeper.exceptions import (
    CacheError,              # base for everything below
    BackendError,            # storage failures (raised on dim mismatch too)
    BackendConnectionError,  # subclass of BackendError
    BackendTimeoutError,     # subclass of BackendError
    EmbeddingError,          # embedding model / API failures
    ConfigurationError,      # invalid construction args
)

try:
    cache.set("q", "r")
except BackendError as e:
    print("Storage failed:", e, "(original:", e.original_error, ")")
```

The library never shadows Python builtins — `BackendConnectionError` and `BackendTimeoutError` are explicitly prefixed.

## Architecture

```
┌────────────────────────────────────────────────────────┐
│        SemanticCache / AsyncSemanticCache API          │
│        (get / set / delete / warm / stats …)           │
├─────────────────┬──────────────────────────────────────┤
│ Embedding       │ Similarity Engine                    │
│ Engine          │ (cosine, fixed at the backend)       │
│ (pluggable      │ (confidence classification)          │
│  providers)     │ (multi-result retrieval)             │
├─────────────────┴──────────────────────────────────────┤
│              Storage Backend Adapter                   │
│  RedisSimpleBackend       │ RediSearchBackend          │
│  AsyncRedisBackend        │ AsyncRediSearchBackend     │
│  (brute-force, pipelined) │ (VECTOR_RANGE + KNN)       │
└────────────────────────────────────────────────────────┘
```

For a detailed breakdown of every layer, class hierarchy, data flow, safety rails, and extension points, see **[ARCHITECTURE.md](ARCHITECTURE.md)**. The design rationale and original scope live in **[CoreDefinition.md](CoreDefinition.md)**.

## Configuration reference

### `SemanticCache` parameters

| Parameter | Default | Description |
|---|---|---|
| `redis_client` | *required* | Your existing `redis.Redis` instance |
| `embedding_provider` | `SentenceTransformerProvider` | Provider used to generate query embeddings |
| `threshold` | `0.85` | Minimum cosine similarity for a cache hit |
| `default_ttl` | `None` | Seconds until entries expire (`None` = no expiry) |
| `namespace` | `"default"` | Key prefix for multi-tenant isolation |
| `backend` | auto-detected | Override the storage backend |
| `model_name` | auto-detected | Embedding model name used for confidence tuning |
| `enable_analytics` | `False` | Track hit/miss rate, latency percentiles, near-misses |

`AsyncSemanticCache` accepts the same parameters, but takes a pre-built `AsyncCacheBackend` (typically obtained via `await create_async_redis_backend(client)`) in place of `redis_client`.

### Similarity metric

Cosine is the only similarity metric currently supported; the RediSearch index is created with `DISTANCE_METRIC=COSINE`. `SimilarityMetric.DOT_PRODUCT` and `SimilarityMetric.EUCLIDEAN` exist as standalone helpers in `llmgatekeeper.similarity.metrics` but are not wired through `SemanticCache`.

## Testing

```bash
pip install "llmgatekeeper[dev]"
pytest
```

The suite contains **497 tests** covering all backends (sync + async, simple + RediSearch), embedding providers, similarity metrics, async paths, error handling, analytics, dimension validation, and edge cases.

Run a single file or test:

```bash
pytest tests/test_cache.py                            # one file
pytest tests/test_cache.py::TestSemanticCacheSet      # one class
pytest -k "async and not openai"                      # filter by name
pytest --cov=llmgatekeeper --cov-report=term-missing  # with coverage
```

## Project structure

```
llmgatekeeper/
├── cache.py                 # SemanticCache & AsyncSemanticCache
├── analytics.py             # Hit/miss rates, latency percentiles, near-misses
├── exceptions.py            # CacheError hierarchy
├── logging.py               # Loguru-based, opt-in via configure_logging()
├── backends/
│   ├── base.py              # Abstract CacheBackend / AsyncCacheBackend
│   ├── factory.py           # create_redis_backend / create_async_redis_backend
│   ├── redis_simple.py      # Brute-force + pipelined hgetalls
│   ├── redis_search.py      # RediSearch VECTOR_RANGE + KNN
│   ├── redis_async.py       # Async simple backend
│   └── redis_search_async.py# Async RediSearch backend
├── embeddings/
│   ├── base.py              # Abstract EmbeddingProvider
│   ├── sentence_transformer.py  # Local Sentence Transformers (aembed via to_thread)
│   ├── openai_provider.py   # OpenAI API (native async)
│   └── cached.py            # LRU + Redis, fingerprint-isolated
└── similarity/
    ├── metrics.py           # Cosine + standalone dot/euclidean helpers
    ├── confidence.py        # HIGH / MEDIUM / LOW / NONE bands
    └── retriever.py         # Multi-result retrieval with ranking
```

Full walkthrough of every public class and method: see **[examples/quickstart_redis.py](examples/quickstart_redis.py)**.

## Roadmap

- Additional vector store backends (Pinecone, Qdrant, Weaviate)
- Native async support in more embedding providers
- Per-namespace eviction policies (LRU / LFU on top of TTL)
- Optional dot-product / Euclidean wiring through `SemanticCache`

Have a request? **[Open an issue](https://github.com/MitulNakrani003/LLMgatekeeper/issues)**.

## Contributing

Contributions are welcome. Open an issue to discuss substantive changes, or send a PR for small fixes. Please run `pytest`, `black`, and `ruff` against your changes before submitting.

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Citation

If you use LLMGatekeeper in research or production, a star on **[GitHub](https://github.com/MitulNakrani003/LLMgatekeeper)** is appreciated.

```bibtex
@software{llmgatekeeper,
  title  = {LLMGatekeeper: Semantic caching for LLM and RAG systems},
  author = {Nakrani, Mitul},
  year   = {2026},
  url    = {https://github.com/MitulNakrani003/LLMgatekeeper}
}
```
