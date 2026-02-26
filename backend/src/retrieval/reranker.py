from __future__ import annotations

import logging

from backend.src.vectorstore.base import SearchResult

logger = logging.getLogger(__name__)


class Reranker:
    """Cross-encoder re-ranker for improving retrieval precision.

    Initial retrieval casts a wide net (high recall, lower precision).
    The cross-encoder then scores each (query, document) pair jointly,
    producing much more accurate relevance estimates than bi-encoder
    similarity alone.  This recall-then-rerank pattern is the standard
    approach for high-quality search pipelines.

    Args:
        model_name: HuggingFace model ID for a cross-encoder model.
            Defaults to ``cross-encoder/ms-marco-MiniLM-L-6-v2`` which
            offers a good speed/quality tradeoff.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name)
        self.model_name = model_name
        logger.info("Loaded reranker model: %s", model_name)

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int = 5,
    ) -> list[SearchResult]:
        """Re-rank search results using cross-encoder scores.

        Each result is paired with the query and scored by the cross-encoder.
        Results are sorted by descending score and truncated to ``top_k``.

        Args:
            query: The user query to score against.
            results: Candidate search results from initial retrieval.
            top_k: Maximum number of results to return after re-ranking.

        Returns:
            Re-ranked list of ``SearchResult`` objects with updated scores.
        """
        if not results:
            return []

        pairs = [(query, r.content) for r in results]
        scores = self.model.predict(pairs)

        scored = [
            SearchResult(
                content=r.content,
                score=float(s),
                metadata=r.metadata,
            )
            for r, s in zip(results, scores)
        ]
        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:top_k]
