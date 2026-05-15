# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LLMGatekeeper is a Python 3.9+ semantic caching library for LLM/RAG systems. It uses embedding-based similarity matching so paraphrases of the same question hit the cache. Public surface is small: `SemanticCache` and `AsyncSemanticCache` with `get` / `set` / `warm` / `stats` / `delete`.

Implementation is complete; the package builds (`dist/`), publishes (`llmgatekeeper.egg-info/`), and has 468 tests. `CoreDefinition.md` is the original design doc — current behavior may diverge; trust the code and `ARCHITECTURE.md` over it.

## Commands

Install for development (uses the optional `dev` extra in `pyproject.toml`):

```bash
pip install -e ".[dev]"          # core + dev tools
pip install -e ".[openai,dev]"   # also pull in OpenAI provider
```

Run the test suite (pytest config lives in `pyproject.toml`; `asyncio_mode = "auto"`, so async tests need no `@pytest.mark.asyncio`):

```bash
pytest                                          # full suite
pytest tests/test_cache.py                      # one file
pytest tests/test_cache.py::test_semantic_hit   # one test
pytest -k "async and not openai"                # filter by name
pytest --cov=llmgatekeeper --cov-report=term-missing
```

Lint / format / type-check:

```bash
black llmgatekeeper tests
ruff check llmgatekeeper tests
mypy llmgatekeeper                              # strict: disallow_untyped_defs = true
```

Build a distribution:

```bash
python -m build                                 # produces sdist + wheel in dist/
```

## Architecture

Three pluggable layers sit under `SemanticCache`. Read `ARCHITECTURE.md` for diagrams; the cross-cutting points worth knowing before editing:

**1. `cache.py` (orchestrator).** Both `SemanticCache` and `AsyncSemanticCache` live here. They own the embedding provider, the backend, the `SimilarityRetriever`, the `ConfidenceClassifier`, and (optionally) `CacheAnalytics`. Keys are MD5-hashed and namespaced (`llmgk:<namespace>:<hash>`); default `namespace` is `"default"`, not `None` (the README's "namespace isolation" examples use explicit names).

**2. `backends/` (storage).** `CacheBackend` (sync) and `AsyncCacheBackend` (async) are abstract. `backends/factory.py` auto-detects RediSearch on the user's Redis instance: if `FT._LIST` succeeds it returns `RediSearchBackend` (FT.SEARCH KNN, scales to millions); otherwise `RedisSimpleBackend` (brute-force cosine over Redis hashes, intended for <10k entries). `force_simple=True` overrides detection. Async path is separate (`redis_async.py`) and requires `redis.asyncio.Redis`, not the sync client — `AsyncSemanticCache` cannot wrap a sync backend.

**3. `embeddings/` (vectorizers).** `EmbeddingProvider` is the strategy interface (sync `embed`/`embed_batch` + async `aembed`/`aembed_batch` + `dimension` property). Default is `SentenceTransformerProvider` (`all-MiniLM-L6-v2`, 384 dims). `OpenAIEmbeddingProvider` (1536 dims) requires the `openai` extra. `CachedEmbeddingProvider` is a decorator that wraps any provider with an LRU; use it instead of subclassing for memoization.

**4. `similarity/` (matching).** `metrics.py` implements cosine/dot/euclidean. `confidence.py` maps a raw similarity score to `HIGH/MEDIUM/LOW/NONE` bands — thresholds are **model-specific** (a cosine of 0.96 on MiniLM and on OpenAI ada-002 mean different things), so `ConfidenceClassifier` auto-tunes from `model_name`. If you swap embedding providers, expect to retune `threshold` and confidence bands together. `retriever.py` does top-k ranking on top of `backend.search_similar`.

**`analytics.py`** is opt-in (`enable_analytics=True`); it tracks hit/miss rate, latency percentiles, near-misses, and top-query frequency.

**`exceptions.py`** defines the hierarchy: `CacheError` → `BackendError` (with `ConnectionError`/`TimeoutError`), `EmbeddingError`, `ConfigurationError`. Public API surfaces these in `llmgatekeeper/__init__.py`.

**`logging.py`** wraps Loguru. Logging is disabled by default; users opt in via `configure_logging(level="INFO" | "DEBUG", serialize=False)`. Library code should log through this module, not `import loguru` directly.

## Design Invariants

- **The user owns the Redis connection.** `SemanticCache(redis_client, ...)` takes an existing `redis.Redis` (or `redis.asyncio.Redis`) so users keep their connection pooling/auth/SSL. Never construct a client inside the library.
- **Sync and async stacks do not mix.** A `redis.Redis` cannot back `AsyncSemanticCache`, and vice versa. The factory and the `*Async*` classes enforce this — don't add silent fallbacks.
- **Embedding-model changes are breaking.** Different providers produce different dimensions (384 vs 1536). Indices and stored vectors are not portable across providers; a backend's `vector_dimension` is fixed at construction. When changing the default model, treat it as a major-version change.
- **Two-tier backend selection is intentional.** Don't collapse `RedisSimpleBackend` and `RediSearchBackend` into one — the simple backend exists so the library works on plain Redis without the Stack module.
