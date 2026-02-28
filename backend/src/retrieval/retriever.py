from __future__ import annotations

import logging

from backend.src.embeddings.embedder import Embedder
from backend.src.retrieval.reranker import Reranker
from backend.src.vectorstore.base import SearchResult, VectorStoreBase

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Hybrid retrieval: dense + sparse search, then cross-encoder re-ranking.

    The retrieval pipeline follows a two-stage recall-then-rerank pattern:

    1. **Initial retrieval** — embed the query into dense and sparse vectors,
       then fetch ``top_k * initial_fetch_multiplier`` candidates from the
       vector store using hybrid search.  Over-fetching ensures the re-ranker
       has enough candidates to find the truly relevant ones.

    2. **Re-ranking** — score each (query, candidate) pair with a cross-encoder
       that reads both texts jointly, producing far more accurate relevance
       estimates than bi-encoder dot products alone.  The top-k results after
       re-ranking are returned.

    If no reranker is provided the pipeline falls back to returning the
    vector-store results directly, truncated to ``top_k``.

    Args:
        embedder: Embedding model for dense and sparse query representations.
        vector_store: Backend store implementing ``VectorStoreBase``.
        reranker: Optional cross-encoder reranker for precision improvement.
        initial_fetch_multiplier: Factor by which to over-fetch candidates
            before re-ranking.  A value of 3 means fetching 3x ``top_k``
            candidates from the store.
    """

    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStoreBase,
        reranker: Reranker | None = None,
        initial_fetch_multiplier: int = 3,
    ) -> None:
        self.embedder = embedder
        self.vector_store = vector_store
        self.reranker = reranker
        self.initial_fetch_multiplier = initial_fetch_multiplier

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.0,
    ) -> list[SearchResult]:
        """Retrieve relevant documents using hybrid search + re-ranking.

        Args:
            query: Natural language query string.
            top_k: Number of final results to return.
            score_threshold: Minimum similarity score for initial retrieval.

        Returns:
            List of ``SearchResult`` objects ranked by relevance.
        """
        dense_vector = self.embedder.embed([query])[0]
        sparse_vector = self.embedder.sparse_embed([query])[0]

        fetch_k = top_k * self.initial_fetch_multiplier
        results = self.vector_store.search(
            dense_vector=dense_vector,
            sparse_vector=sparse_vector,
            top_k=fetch_k,
        )

        if not results:
            return []

        if self.reranker:
            results = self.reranker.rerank(query=query, results=results, top_k=top_k)
        else:
            results = results[:top_k]

        if score_threshold > 0:
            filtered = [r for r in results if r.score >= score_threshold]
            results = filtered if filtered else results

        logger.info("Retrieved %d results for query: %s", len(results), query[:80])
        return results
