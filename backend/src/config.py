"""Application settings loaded from environment variables.

Uses pydantic-settings to read configuration from environment variables
and .env files, with sensible defaults for all optional fields.
"""

from typing import Literal

from pydantic import model_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables.

    All fields have sensible defaults except for API keys, which default
    to empty strings. Values can be overridden via environment variables
    or a .env file in the project root.
    """

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    # --- OpenRouter ---
    OPENROUTER_API_KEY: str = ""
    OPENROUTER_MODEL: str = "anthropic/claude-haiku-4-5-20251001"

    # --- Embeddings ---
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # --- Qdrant ---
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION: str = "intellidocs"

    # --- Vector store backend ---
    VECTOR_STORE_BACKEND: Literal["qdrant", "chroma"] = "qdrant"

    # --- Chunking ---
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50

    # --- Retrieval ---
    TOP_K: int = 5
    SIMILARITY_THRESHOLD: float = 0.3

    @model_validator(mode="after")
    def validate_chunk_overlap(self) -> "Settings":
        """Ensure chunk overlap is strictly less than chunk size."""
        if self.CHUNK_OVERLAP >= self.CHUNK_SIZE:
            raise ValueError(
                f"CHUNK_OVERLAP ({self.CHUNK_OVERLAP}) must be less than "
                f"CHUNK_SIZE ({self.CHUNK_SIZE})"
            )
        return self
