import pytest

from backend.src.vectorstore.base import SearchResult


class TestReranker:
    def test_rerank_changes_order(self):
        from backend.src.retrieval.reranker import Reranker

        reranker = Reranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
        results = [
            SearchResult(content="Python is a programming language", score=0.8),
            SearchResult(content="The capital of France is Paris", score=0.9),
            SearchResult(content="Python was created by Guido van Rossum", score=0.7),
        ]
        reranked = reranker.rerank(query="Who created Python?", results=results, top_k=2)
        assert len(reranked) == 2
        assert "Python" in reranked[0].content

    def test_rerank_preserves_metadata(self):
        from backend.src.retrieval.reranker import Reranker

        reranker = Reranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
        results = [
            SearchResult(content="Test content", score=0.5, metadata={"source": "test.txt"}),
        ]
        reranked = reranker.rerank(query="test", results=results, top_k=1)
        assert reranked[0].metadata["source"] == "test.txt"
