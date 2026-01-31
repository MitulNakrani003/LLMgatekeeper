# LLMGatekeeper

A semantic caching library for LLM and RAG systems. Eliminates redundant LLM API calls and vector database queries by recognising semantically equivalent queries using embedding-based similarity matching.

Instead of requiring exact string matches, LLMGatekeeper understands that *"What is Python?"* and *"Tell me about Python"* are asking the same thing — and returns the cached response instantly.

## Features

- **Semantic matching** — Embedding-based similarity search catches paraphrases, synonyms, and rewording automatically
- **Redis backends** — Simple hash-based mode for small datasets (<10k entries); auto-upgrades to RediSearch KNN vector search when available
- **Pluggable embeddings** — Ships with `all-MiniLM-L6-v2` (Sentence-Transformers) out of the box; swap in OpenAI embeddings or any custom provider
- **Confidence levels** — Scores are classified into HIGH / MEDIUM / LOW / NONE bands with per-model tuned thresholds
- **Multi-tenant isolation** — Namespace prefixing keeps separate tenants' caches partitioned in the same Redis instance
- **Async support** — Full `AsyncSemanticCache` API mirrors the synchronous interface
- **Analytics & observability** — Hit/miss rates, latency percentiles (p50/p95/p99), near-miss tracking, and top-query frequency
- **TTL control** — Global default TTL with per-entry overrides; `ttl=0` disables expiry for specific entries
- **Bulk warming** — Load many entries at once with `warm()`, including batch-size control and progress callbacks

## Installation

```bash
pip install llmgatekeeper
```

With OpenAI embedding support:

```bash
pip install llmgatekeeper[openai]
```

For development (tests, linting, type-checking):

```bash
pip install llmgatekeeper[dev]
```

## Requirements

- Python 3.9+
- Redis 6.2+ (Redis Stack recommended for RediSearch vector search)

## Quick Start

```python
import redis
from llmgatekeeper import SemanticCache

# Connect to your Redis instance
client = redis.Redis(host="localhost", port=6379)

# Create a semantic cache (auto-detects RediSearch if available)
cache = SemanticCache(client)

# Store a response
cache.set("What is Python?", "Python is a high-level programming language.")

# Retrieve with a semantically similar query — no exact match needed
result = cache.get("Tell me about Python")
print(result)  # "Python is a high-level programming language."
```

### With metadata

```python
cache.set(
    "What is Python?",
    "Python is a high-level programming language.",
    metadata={"source": "docs", "version": "3.12"},
)

result = cache.get("Explain Python")
if result:
    print(result.response)   # the cached response
    print(result.similarity) # e.g. 0.91
    print(result.metadata)   # {"source": "docs", "version": "3.12"}
```

### Async usage

```python
import asyncio
import redis.asyncio as aioredis
from llmgatekeeper import AsyncSemanticCache
from llmgatekeeper.backends.redis_async import AsyncRedisBackend

async def main():
    client = aioredis.Redis(host="localhost", port=6379)
    backend = AsyncRedisBackend(client)
    cache = AsyncSemanticCache(backend)

    await cache.set("What is Python?", "A high-level programming language.")
    result = await cache.get("Tell me about Python")
    print(result)

asyncio.run(main())
```

### Multi-tenant isolation

```python
tenant_a = SemanticCache(client, namespace="tenant_a")
tenant_b = SemanticCache(client, namespace="tenant_b")

tenant_a.set("Hello", "Hi from A")
tenant_b.set("Hello", "Hi from B")

print(tenant_a.get("Hello").response)  # "Hi from A"
print(tenant_b.get("Hello").response)  # "Hi from B"
```

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
stats = cache.stats()
print(f"Hit rate:      {stats.hit_rate:.1%}")
print(f"P95 latency:   {stats.p95_latency_ms:.2f} ms")
print(f"Near misses:   {len(stats.near_misses)}")
print(f"Top queries:   {[(q.query, q.count) for q in stats.top_queries[:3]]}")
```

## Architecture

The library is organised into three layers:

```
┌─────────────────────────────────────────────┐
│              SemanticCache API               │
│   (get / set / delete / warm / stats …)      │
├────────────────┬────────────────────────────┤
│ Embedding      │ Similarity Engine          │
│ Engine         │ (cosine / dot / euclidean) │
│ (pluggable     │ (confidence classification)│
│  providers)    │ (multi-result retrieval)   │
├────────────────┴────────────────────────────┤
│         Storage Backend Adapter             │
│  RedisSimpleBackend │ RediSearchBackend     │
│  (brute-force)      │ (KNN vector search)   │
└─────────────────────────────────────────────┘
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for a detailed breakdown of every layer, class hierarchy, and extension point.

## Configuration Reference

| Parameter | Default | Description |
|---|---|---|
| `threshold` | `0.85` | Minimum cosine similarity for a cache hit |
| `default_ttl` | `None` | Seconds until entries expire (`None` = no expiry) |
| `namespace` | `None` | Key prefix for multi-tenant isolation |
| `similarity_metric` | `"cosine"` | Similarity function (`cosine`, `dot`, `euclidean`) |
| `embedding_provider` | `SentenceTransformerProvider` | Provider used to generate query embeddings |

## Running Tests

```bash
pip install llmgatekeeper[dev]
pytest
```

The test suite contains 468 tests covering all backends, embedding providers, similarity metrics, async paths, error handling, analytics, and edge cases.

## Contributing

Contributions are welcome. Please open an issue or pull request on GitHub.

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
