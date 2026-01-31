"""Tests for package initialization and version."""

import llmgatekeeper


def test_version_exists():
    """TC-1.1.1: Package exposes version string."""
    assert hasattr(llmgatekeeper, "__version__")
    assert isinstance(llmgatekeeper.__version__, str)
    assert llmgatekeeper.__version__ == "0.1.0"


def test_package_imports():
    """TC-1.1.1: Package imports without errors."""
    import llmgatekeeper

    assert llmgatekeeper is not None
