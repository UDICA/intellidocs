from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class SearchResult:
    """A single search result from the vector store."""

    content: str
    score: float
    metadata: dict[str, str | int | float] = field(default_factory=dict)


class VectorStoreBase(ABC):
    """Abstract interface for vector store backends."""

    @abstractmethod
    def initialize(self, dimension: int) -> None:
        """Create the collection/index if it doesn't exist."""

    @abstractmethod
    def upsert(
        self,
        ids: list[str],
        contents: list[str],
        dense_vectors: list[list[float]],
        sparse_vectors: list[dict] | None = None,
        metadatas: list[dict] | None = None,
    ) -> None:
        """Insert or update documents with their vectors."""

    @abstractmethod
    def search(
        self,
        dense_vector: list[float],
        sparse_vector: dict | None = None,
        top_k: int = 5,
        score_threshold: float = 0.0,
    ) -> list[SearchResult]:
        """Search for similar documents using dense and optionally sparse vectors."""

    @abstractmethod
    def delete(self, ids: list[str]) -> None:
        """Delete documents by their IDs."""

    @abstractmethod
    def delete_by_metadata(self, key: str, value: str) -> None:
        """Delete all documents matching a metadata field value."""

    @abstractmethod
    def count(self) -> int:
        """Return the number of documents in the store."""

    @abstractmethod
    def list_documents(self) -> list[dict]:
        """Return metadata for all unique source documents."""
