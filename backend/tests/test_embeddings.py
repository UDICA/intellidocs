import pytest


class TestEmbedder:
    def test_embed_single_text(self):
        from backend.src.embeddings.embedder import Embedder

        embedder = Embedder(model_name="all-MiniLM-L6-v2")
        vectors = embedder.embed(["Hello world"])
        assert len(vectors) == 1
        assert len(vectors[0]) == 384  # MiniLM output dimension

    def test_embed_multiple_texts(self):
        from backend.src.embeddings.embedder import Embedder

        embedder = Embedder(model_name="all-MiniLM-L6-v2")
        vectors = embedder.embed(["Hello", "World", "Test"])
        assert len(vectors) == 3
        assert all(len(v) == 384 for v in vectors)

    def test_embed_returns_normalized_vectors(self):
        import math
        from backend.src.embeddings.embedder import Embedder

        embedder = Embedder(model_name="all-MiniLM-L6-v2")
        vectors = embedder.embed(["Test normalization"])
        magnitude = math.sqrt(sum(x * x for x in vectors[0]))
        assert abs(magnitude - 1.0) < 0.01

    def test_sparse_embed(self):
        from backend.src.embeddings.embedder import Embedder

        embedder = Embedder(model_name="all-MiniLM-L6-v2")
        sparse = embedder.sparse_embed(["Machine learning is great"])
        assert len(sparse) == 1
        assert "indices" in sparse[0]
        assert "values" in sparse[0]
        assert len(sparse[0]["indices"]) == len(sparse[0]["values"])
        assert len(sparse[0]["indices"]) > 0
