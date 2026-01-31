"""Tests to verify all public classes and methods have docstrings."""

import inspect

import pytest


class TestDocstringsPresent:
    """TC-7.3.2: All public classes/methods have docstrings."""

    def test_semantic_cache_docstring(self):
        """SemanticCache class has docstring."""
        import llmgatekeeper

        assert llmgatekeeper.SemanticCache.__doc__ is not None
        assert len(llmgatekeeper.SemanticCache.__doc__) > 100

    def test_semantic_cache_get_docstring(self):
        """SemanticCache.get has docstring."""
        import llmgatekeeper

        assert llmgatekeeper.SemanticCache.get.__doc__ is not None
        assert "query" in llmgatekeeper.SemanticCache.get.__doc__.lower()

    def test_semantic_cache_set_docstring(self):
        """SemanticCache.set has docstring."""
        import llmgatekeeper

        assert llmgatekeeper.SemanticCache.set.__doc__ is not None
        assert "response" in llmgatekeeper.SemanticCache.set.__doc__.lower()

    def test_async_semantic_cache_docstring(self):
        """AsyncSemanticCache class has docstring."""
        import llmgatekeeper

        assert llmgatekeeper.AsyncSemanticCache.__doc__ is not None
        assert "async" in llmgatekeeper.AsyncSemanticCache.__doc__.lower()

    def test_cache_result_docstring(self):
        """CacheResult class has docstring."""
        import llmgatekeeper

        assert llmgatekeeper.CacheResult.__doc__ is not None

    def test_cache_error_docstring(self):
        """CacheError class has docstring."""
        import llmgatekeeper

        assert llmgatekeeper.CacheError.__doc__ is not None

    def test_backend_error_docstring(self):
        """BackendError class has docstring."""
        import llmgatekeeper

        assert llmgatekeeper.BackendError.__doc__ is not None

    def test_embedding_error_docstring(self):
        """EmbeddingError class has docstring."""
        import llmgatekeeper

        assert llmgatekeeper.EmbeddingError.__doc__ is not None

    def test_configure_logging_docstring(self):
        """configure_logging function has docstring."""
        import llmgatekeeper

        assert llmgatekeeper.configure_logging.__doc__ is not None

    def test_module_docstring(self):
        """Main module has docstring."""
        import llmgatekeeper

        assert llmgatekeeper.__doc__ is not None
        assert "semantic caching" in llmgatekeeper.__doc__.lower()


class TestPublicAPIDocumented:
    """Tests that all public API items are documented."""

    def test_all_exports_have_docstrings(self):
        """All items in __all__ have docstrings."""
        import llmgatekeeper

        for name in llmgatekeeper.__all__:
            if name.startswith("_"):
                continue
            obj = getattr(llmgatekeeper, name)
            if inspect.isclass(obj) or inspect.isfunction(obj):
                assert obj.__doc__ is not None, f"{name} missing docstring"

    def test_semantic_cache_public_methods_documented(self):
        """All public methods of SemanticCache are documented."""
        from llmgatekeeper import SemanticCache

        public_methods = [
            name
            for name in dir(SemanticCache)
            if not name.startswith("_") and callable(getattr(SemanticCache, name))
        ]

        for method_name in public_methods:
            method = getattr(SemanticCache, method_name)
            # Skip property objects
            if isinstance(inspect.getattr_static(SemanticCache, method_name), property):
                continue
            assert method.__doc__ is not None, f"SemanticCache.{method_name} missing docstring"
