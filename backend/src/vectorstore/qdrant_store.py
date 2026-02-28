from __future__ import annotations

import logging
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    SparseIndexParams,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from backend.src.vectorstore.base import SearchResult, VectorStoreBase

logger = logging.getLogger(__name__)


class QdrantStore(VectorStoreBase):
    """Qdrant vector store with dense and sparse vector support.

    Supports hybrid retrieval by storing both dense embeddings (for semantic
    similarity) and sparse vectors (for keyword matching via BM25) in the
    same collection. When ``host`` is ``None`` an in-memory Qdrant instance
    is used, which is convenient for testing and lightweight demos.
    """

    def __init__(
        self,
        host: str | None = "localhost",
        port: int | None = 6333,
        collection_name: str = "intellidocs",
    ) -> None:
        if host is None:
            self.client = QdrantClient(location=":memory:")
        else:
            self.client = QdrantClient(host=host, port=port)
        self.collection_name = collection_name

    # ------------------------------------------------------------------
    # Collection lifecycle
    # ------------------------------------------------------------------

    def initialize(self, dimension: int) -> None:
        """Create collection with dense + sparse vector config.

        If the collection already exists the call is a no-op so that
        restarting the application doesn't destroy stored data.
        """
        collections = [c.name for c in self.client.get_collections().collections]
        if self.collection_name in collections:
            logger.info("Collection '%s' already exists", self.collection_name)
            return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config={
                "dense": VectorParams(size=dimension, distance=Distance.COSINE),
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(index=SparseIndexParams()),
            },
        )
        logger.info("Created collection '%s' (dim=%d)", self.collection_name, dimension)

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
        """Insert or update documents with their dense (and optionally sparse) vectors.

        Each document id is mapped to a deterministic UUID via ``uuid5`` so
        that repeated upserts for the same logical document id overwrite the
        previous point instead of creating duplicates.
        """
        points: list[PointStruct] = []
        for i, doc_id in enumerate(ids):
            payload: dict = {"content": contents[i]}
            if metadatas and i < len(metadatas):
                payload.update(metadatas[i])

            vectors: dict = {"dense": dense_vectors[i]}
            if sparse_vectors and i < len(sparse_vectors):
                sv = sparse_vectors[i]
                vectors["sparse"] = SparseVector(
                    indices=sv["indices"],
                    values=sv["values"],
                )

            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, doc_id))
            points.append(PointStruct(id=point_id, vector=vectors, payload=payload))

        self.client.upsert(collection_name=self.collection_name, points=points)

    def search(
        self,
        dense_vector: list[float],
        sparse_vector: dict | None = None,
        top_k: int = 5,
        score_threshold: float = 0.0,
    ) -> list[SearchResult]:
        """Return the most similar documents ranked by cosine similarity.

        Score filtering is intentionally NOT applied at this level because
        ``query_points`` uses raw distance scores whose range depends on the
        metric and Qdrant version.  The caller (retriever + reranker) is
        responsible for meaningful score-based filtering after re-ranking.
        """
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=dense_vector,
            using="dense",
            limit=top_k,
        )

        return [
            SearchResult(
                content=point.payload.get("content", ""),
                score=point.score,
                metadata={k: v for k, v in point.payload.items() if k != "content"},
            )
            for point in results.points
        ]

    def delete(self, ids: list[str]) -> None:
        """Delete documents by their logical IDs."""
        point_ids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, doc_id)) for doc_id in ids]
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=point_ids,
        )

    def delete_by_metadata(self, key: str, value: str) -> None:
        """Delete all points whose payload field ``key`` equals ``value``."""
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        self.client.delete(
            collection_name=self.collection_name,
            points_selector=Filter(
                must=[FieldCondition(key=key, match=MatchValue(value=value))]
            ),
        )

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def count(self) -> int:
        """Return the number of points in the collection."""
        info = self.client.get_collection(self.collection_name)
        return info.points_count

    def list_documents(self) -> list[dict]:
        """Return metadata for all unique source documents.

        Scrolls through every point and de-duplicates by the ``source``
        field so that each source file appears only once regardless of
        how many chunks it was split into.
        """
        records, _ = self.client.scroll(
            collection_name=self.collection_name,
            limit=10000,
            with_payload=True,
        )
        seen: dict[str, dict] = {}
        for r in records:
            source = r.payload.get("source", "unknown")
            if source not in seen:
                seen[source] = {k: v for k, v in r.payload.items() if k != "content"}
        return list(seen.values())
