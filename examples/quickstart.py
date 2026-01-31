#!/usr/bin/env python3
"""
Quickstart example for LLMGatekeeper.

This example demonstrates basic usage of the SemanticCache for
caching LLM responses based on semantic similarity.

Prerequisites:
    pip install llmgatekeeper redis

Running:
    1. Start Redis: redis-server
    2. Run this example: python quickstart.py
"""

from unittest.mock import MagicMock

# For the example to run without Redis, we use mocks
# In production, you would use: import redis

# Simulating the redis client for demonstration
mock_redis = MagicMock()
mock_redis.module_list.return_value = []  # Simulate no RediSearch

# Import from llmgatekeeper
from llmgatekeeper import SemanticCache, configure_logging, disable_logging

# Disable logging for clean output (enable for debugging)
disable_logging()
# configure_logging(level="DEBUG")


def main():
    """Demonstrate basic semantic cache usage."""
    print("LLMGatekeeper Quickstart Example")
    print("=" * 40)

    # In production, you would use:
    # import redis
    # cache = SemanticCache(redis.Redis())

    # For this demo, we'll create a cache with a mock backend
    # that simulates the behavior
    from llmgatekeeper.backends.base import SearchResult
    from llmgatekeeper.embeddings.sentence_transformer import SentenceTransformerProvider

    # Create embedding provider (uses real model for semantic similarity)
    embedding_provider = SentenceTransformerProvider()

    # Create mock backend that stores in memory
    stored_entries = {}

    def mock_store(key, vector, metadata, ttl=None):
        stored_entries[key] = {"vector": vector, "metadata": metadata}

    def mock_search(vector, threshold=0.85, top_k=1):
        import numpy as np

        results = []
        for key, entry in stored_entries.items():
            stored_vec = entry["vector"]
            # Calculate cosine similarity
            sim = float(np.dot(vector, stored_vec) / (
                np.linalg.norm(vector) * np.linalg.norm(stored_vec)
            ))
            if sim >= threshold:
                results.append(SearchResult(
                    key=key,
                    similarity=sim,
                    metadata=entry["metadata"],
                ))
        results.sort(key=lambda x: x.similarity, reverse=True)
        return results[:top_k]

    mock_backend = MagicMock()
    mock_backend.store_vector.side_effect = mock_store
    mock_backend.search_similar.side_effect = mock_search
    mock_backend.count.return_value = 0

    # Create the cache
    cache = SemanticCache(
        mock_redis,
        backend=mock_backend,
        embedding_provider=embedding_provider,
        threshold=0.85,  # Minimum similarity for a cache hit
    )

    # ===== Example 1: Basic Set and Get =====
    print("\n1. Basic Set and Get")
    print("-" * 40)

    # Store a query-response pair
    cache.set(
        "What is Python?",
        "Python is a high-level programming language known for its simplicity."
    )
    print("Stored: 'What is Python?' -> 'Python is a high-level...'")

    # Retrieve with the exact same query
    result = cache.get("What is Python?")
    print(f"Get 'What is Python?': {result[:50]}...")

    # ===== Example 2: Semantic Similarity =====
    print("\n2. Semantic Similarity")
    print("-" * 40)

    # Try a semantically similar query
    similar_result = cache.get("Tell me about Python")
    if similar_result:
        print(f"Get 'Tell me about Python': {similar_result[:50]}...")
    else:
        print("No match found (similarity below threshold)")

    # Another variation
    another_result = cache.get("Explain Python programming language")
    if another_result:
        print(f"Get 'Explain Python...': {another_result[:50]}...")
    else:
        print("No match found (similarity below threshold)")

    # ===== Example 3: Unrelated Query =====
    print("\n3. Unrelated Query (Cache Miss)")
    print("-" * 40)

    miss_result = cache.get("How do I cook pasta?")
    print(f"Get 'How do I cook pasta?': {miss_result}")

    # ===== Example 4: Get with Metadata =====
    print("\n4. Get with Metadata")
    print("-" * 40)

    # Store with metadata
    cache.set(
        "What is machine learning?",
        "Machine learning is a subset of AI that enables systems to learn from data.",
        metadata={"model": "gpt-4", "tokens": 50, "category": "AI"}
    )

    # Retrieve with metadata
    result_with_meta = cache.get("Explain machine learning", include_metadata=True)
    if result_with_meta:
        print(f"Response: {result_with_meta.response[:50]}...")
        print(f"Similarity: {result_with_meta.similarity:.3f}")
        print(f"Confidence: {result_with_meta.confidence}")
        print(f"Metadata: {result_with_meta.metadata}")

    print("\n" + "=" * 40)
    print("Quickstart complete!")


if __name__ == "__main__":
    main()
