import os
import pytest


def test_config_loads_defaults():
    """Config should provide sensible defaults for all optional fields."""
    from backend.src.config import Settings

    settings = Settings(OPENROUTER_API_KEY="test-key")
    assert settings.OPENROUTER_API_KEY == "test-key"
    assert settings.OPENROUTER_MODEL == "anthropic/claude-haiku-4-5-20251001"
    assert settings.EMBEDDING_MODEL == "all-MiniLM-L6-v2"
    assert settings.RERANKER_MODEL == "cross-encoder/ms-marco-MiniLM-L-6-v2"
    assert settings.QDRANT_HOST == "localhost"
    assert settings.QDRANT_PORT == 6333
    assert settings.QDRANT_COLLECTION == "intellidocs"
    assert settings.VECTOR_STORE_BACKEND == "qdrant"
    assert settings.CHUNK_SIZE == 512
    assert settings.CHUNK_OVERLAP == 50
    assert settings.TOP_K == 5
    assert settings.SIMILARITY_THRESHOLD == 0.3


def test_config_from_env(monkeypatch):
    """Config should read values from environment variables."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "env-key")
    monkeypatch.setenv("CHUNK_SIZE", "1024")
    monkeypatch.setenv("TOP_K", "10")

    from backend.src.config import Settings

    settings = Settings()
    assert settings.OPENROUTER_API_KEY == "env-key"
    assert settings.CHUNK_SIZE == 1024
    assert settings.TOP_K == 10


def test_config_validates_chunk_overlap():
    """Chunk overlap must be less than chunk size."""
    from backend.src.config import Settings

    with pytest.raises(ValueError):
        Settings(OPENROUTER_API_KEY="test", CHUNK_SIZE=100, CHUNK_OVERLAP=150)
