import pytest


@pytest.fixture
def qdrant_store():
    """Create an in-memory Qdrant store for testing."""
    from backend.src.vectorstore.qdrant_store import QdrantStore

    store = QdrantStore(host=None, port=None, collection_name="test_collection")
    store.initialize(dimension=4)
    return store


class TestQdrantStore:
    def test_upsert_and_count(self, qdrant_store):
        qdrant_store.upsert(
            ids=["doc1", "doc2"],
            contents=["Hello world", "Goodbye world"],
            dense_vectors=[[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]],
            metadatas=[{"source": "a.txt"}, {"source": "b.txt"}],
        )
        assert qdrant_store.count() == 2

    def test_search_returns_ranked_results(self, qdrant_store):
        qdrant_store.upsert(
            ids=["doc1", "doc2", "doc3"],
            contents=["Alpha", "Beta", "Gamma"],
            dense_vectors=[
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.9, 0.1, 0.0, 0.0],
            ],
        )
        results = qdrant_store.search(
            dense_vector=[1.0, 0.0, 0.0, 0.0],
            top_k=2,
        )
        assert len(results) == 2
        assert results[0].content == "Alpha"
        assert results[0].score >= results[1].score

    def test_delete(self, qdrant_store):
        qdrant_store.upsert(
            ids=["doc1", "doc2"],
            contents=["A", "B"],
            dense_vectors=[[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]],
        )
        qdrant_store.delete(ids=["doc1"])
        assert qdrant_store.count() == 1

    def test_list_documents(self, qdrant_store):
        qdrant_store.upsert(
            ids=["doc1", "doc2"],
            contents=["A", "B"],
            dense_vectors=[[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]],
            metadatas=[{"source": "a.txt"}, {"source": "b.txt"}],
        )
        docs = qdrant_store.list_documents()
        sources = {d["source"] for d in docs}
        assert "a.txt" in sources
        assert "b.txt" in sources


@pytest.fixture
def chroma_store():
    """Create an in-memory ChromaDB store for testing."""
    from backend.src.vectorstore.chroma_store import ChromaStore

    store = ChromaStore(collection_name="test_chroma")
    store.initialize(dimension=4)
    return store


class TestChromaStore:
    def test_upsert_and_count(self, chroma_store):
        chroma_store.upsert(
            ids=["doc1", "doc2"],
            contents=["Hello world", "Goodbye world"],
            dense_vectors=[[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]],
            metadatas=[{"source": "a.txt"}, {"source": "b.txt"}],
        )
        assert chroma_store.count() == 2

    def test_search(self, chroma_store):
        chroma_store.upsert(
            ids=["doc1", "doc2"],
            contents=["Alpha", "Beta"],
            dense_vectors=[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
        )
        results = chroma_store.search(
            dense_vector=[1.0, 0.0, 0.0, 0.0],
            top_k=1,
        )
        assert len(results) == 1
        assert results[0].content == "Alpha"

    def test_delete(self, chroma_store):
        chroma_store.upsert(
            ids=["doc1", "doc2"],
            contents=["A", "B"],
            dense_vectors=[[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]],
        )
        chroma_store.delete(ids=["doc1"])
        assert chroma_store.count() == 1
