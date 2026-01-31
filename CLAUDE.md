# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LLMGatekeeper is a Python 3.9+ semantic caching library that eliminates redundant LLM API calls and vector database queries in RAG (Retrieval-Augmented Generation) systems. It uses embedding-based similarity matching to recognize semantically equivalent queries (e.g., "What's the weather today?" and "Tell me today's weather") rather than requiring exact string matches.

**Current Status:** Early design phase - architecture is documented in CoreDefinition.md but implementation has not started.

## Architecture

The library has three core layers:

1. **Storage Backend Adapter** - Abstract `CacheBackend` interface with Redis implementation
   - Simple mode: Redis hashes + brute-force similarity (<10k entries)
   - RediSearch mode: Vector similarity search (scales to millions)
   - Auto-detects RediSearch availability and upgrades automatically
   - Users pass their own `redis.Redis` instance (respects existing connection pooling, auth, SSL)

2. **Embedding Engine** - Pluggable embedding providers via strategy pattern
   - Default: `all-MiniLM-L6-v2` local model (Sentence-Transformers)
   - Supports batch embedding, async calls, and embedding caching
   - Handles dimension normalization across providers (384-dim vs 1536-dim)

3. **Similarity Engine** - Configurable similarity matching
   - Supports cosine, dot product, euclidean distance
   - Threshold-based matching with confidence bands (high/medium/low)
   - Multi-result retrieval for ensemble/voting scenarios

## Tech Stack

- **Core:** Python 3.9+
- **Validation:** Pydantic
- **Logging:** Loguru
- **Storage:** Redis (optional RediSearch module)
- **Embeddings:** Sentence-Transformers (`all-MiniLM-L6-v2`)
- **Similarity:** NumPy or scikit-learn
- **Optional:** OpenAI embeddings, Pinecone/Qdrant adapters

## Key Design Decisions

- Users pass their existing Redis instance rather than the library managing connections
- Embedding model choice affects similarity thresholds (0.96 on OpenAI differs from 0.96 on MiniLM)
- Multi-tenant support via namespace prefixing (`SemanticCache(namespace="tenant_123")`)
- Two primary methods: `get` and `set` for simple integration with any RAG framework
