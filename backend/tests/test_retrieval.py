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


class TestHybridRetriever:
    @pytest.fixture
    def retriever_with_data(self):
        from backend.src.embeddings.embedder import Embedder
        from backend.src.retrieval.reranker import Reranker
        from backend.src.retrieval.retriever import HybridRetriever
        from backend.src.vectorstore.qdrant_store import QdrantStore

        embedder = Embedder(model_name="all-MiniLM-L6-v2")
        store = QdrantStore(host=None, port=None, collection_name="test_retriever")
        store.initialize(dimension=embedder.dimension)
        reranker = Reranker()

        texts = [
            "Python is a high-level programming language.",
            "Machine learning uses algorithms to learn from data.",
            "The Eiffel Tower is located in Paris, France.",
        ]
        dense = embedder.embed(texts)
        sparse = embedder.sparse_embed(texts)
        store.upsert(
            ids=["d1", "d2", "d3"],
            contents=texts,
            dense_vectors=dense,
            sparse_vectors=sparse,
            metadatas=[{"source": f"doc{i}.txt"} for i in range(3)],
        )

        return HybridRetriever(
            embedder=embedder,
            vector_store=store,
            reranker=reranker,
        )

    def test_retrieve_returns_relevant_results(self, retriever_with_data):
        results = retriever_with_data.retrieve("What is Python?", top_k=2)
        assert len(results) > 0
        assert "Python" in results[0].content

    def test_retrieve_respects_top_k(self, retriever_with_data):
        results = retriever_with_data.retrieve("programming", top_k=1)
        assert len(results) == 1
