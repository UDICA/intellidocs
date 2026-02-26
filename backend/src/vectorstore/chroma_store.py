from __future__ import annotations

import logging

import chromadb

from backend.src.vectorstore.base import SearchResult, VectorStoreBase

logger = logging.getLogger(__name__)


class ChromaStore(VectorStoreBase):
    """ChromaDB in-memory vector store (fallback, dense-only).

    Provides a lightweight alternative to Qdrant that requires no external
    services.  Useful for quick demos, CI environments, and local development.
    Sparse vectors are accepted by the interface but silently ignored since
    ChromaDB only supports dense retrieval.
    """

    def __init__(self, collection_name: str = "intellidocs") -> None:
        self.client = chromadb.Client()
        self.collection_name = collection_name
        self._collection: chromadb.Collection | None = None

    def initialize(self, dimension: int) -> None:
        """Create or retrieve the ChromaDB collection.

        The ``dimension`` parameter is accepted for interface compatibility
        but ChromaDB infers dimensionality from the first inserted embeddings.
        """
        self._collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("ChromaDB collection '%s' ready", self.collection_name)

    @property
    def collection(self) -> chromadb.Collection:
        """Access the underlying collection, raising if not yet initialized."""
        if self._collection is None:
            raise RuntimeError("Call initialize() before using the store")
        return self._collection

    # ------------------------------------------------------------------
    # CRUD operations
    # ------------------------------------------------------------------

    def upsert(
        self,
        ids: list[str],
        contents: list[str],
        dense_vectors: list[list[float]],
        sparse_vectors: list[dict] | None = None,
        metadatas: list[dict] | None = None,
    ) -> None:
        """Insert or update documents with their dense embeddings.

        ``sparse_vectors`` is accepted for interface compatibility but
        ignored — ChromaDB does not support sparse retrieval.
        """
        self.collection.upsert(
            ids=ids,
            embeddings=dense_vectors,
            documents=contents,
            metadatas=metadatas,
        )

    def search(
        self,
        dense_vector: list[float],
        sparse_vector: dict | None = None,
        top_k: int = 5,
        score_threshold: float = 0.0,
    ) -> list[SearchResult]:
        """Query the collection by dense vector similarity.

        ChromaDB returns cosine *distance* (0 = identical), so we convert
        to a similarity score via ``1 - distance``.
        """
        results = self.collection.query(
            query_embeddings=[dense_vector],
            n_results=top_k,
        )
        search_results: list[SearchResult] = []
        if results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                distance = results["distances"][0][i] if results["distances"] else 0.0
                score = 1.0 - distance
                if score >= score_threshold:
                    metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                    search_results.append(
                        SearchResult(content=doc, score=score, metadata=metadata)
                    )
        return search_results

    def delete(self, ids: list[str]) -> None:
        """Delete documents by their IDs."""
        self.collection.delete(ids=ids)

    def delete_by_metadata(self, key: str, value: str) -> None:
        """Delete all documents whose metadata field ``key`` equals ``value``."""
        results = self.collection.get(where={key: value})
        if results["ids"]:
            self.collection.delete(ids=results["ids"])

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def count(self) -> int:
        """Return the number of documents in the collection."""
        return self.collection.count()

    def list_documents(self) -> list[dict]:
        """Return metadata for all unique source documents.

        De-duplicates by the ``source`` field so that each source file
        appears only once regardless of how many chunks it contains.
        """
        all_data = self.collection.get(include=["metadatas"])
        seen: dict[str, dict] = {}
        for meta in all_data["metadatas"] or []:
            source = meta.get("source", "unknown")
            if source not in seen:
                seen[source] = meta
        return list(seen.values())
