# backend/tests/test_integration.py
"""Integration tests for the full RAG pipeline exercised through the FastAPI endpoints.

These tests spin up a test FastAPI app backed by an in-memory Qdrant store,
upload documents via the API, and verify the entire retrieve-and-generate flow.
The OpenRouter LLM is always mocked so that no real HTTP requests leave the
test process.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from backend.src.api import deps
from backend.src.config import Settings
from backend.src.embeddings.embedder import Embedder
from backend.src.generation.llm_client import OpenRouterClient
from backend.src.generation.rag_chain import RAGChain
from backend.src.ingestion.processor import IngestionProcessor
from backend.src.retrieval.reranker import Reranker
from backend.src.retrieval.retriever import HybridRetriever
from backend.src.vectorstore.qdrant_store import QdrantStore


@pytest.fixture()
def test_app():
    """Create a test FastAPI app with in-memory vector store.

    Clears all ``lru_cache`` singletons so every test gets a fresh set of
    dependencies, then wires overrides for settings, embedder, vector store,
    and the full RAG chain (with a mocked LLM client).
    """
    # Clear cached singletons so overrides take effect
    deps.get_settings.cache_clear()
    deps.get_embedder.cache_clear()
    deps.get_vector_store.cache_clear()
    deps.get_reranker.cache_clear()
    deps.get_retriever.cache_clear()
    deps.get_llm_client.cache_clear()
    deps.get_rag_chain.cache_clear()
    deps.get_ingestion_processor.cache_clear()

    from backend.src.main import create_app

    app = create_app()

    # -- Shared objects used across overrides ----------------------------------
    test_settings = Settings(
        OPENROUTER_API_KEY="test-key-not-real",
        VECTOR_STORE_BACKEND="qdrant",
    )
    embedder = Embedder(model_name=test_settings.EMBEDDING_MODEL)
    store = QdrantStore(host=None, port=None, collection_name="test_integration")
    store.initialize(dimension=embedder.dimension)
    reranker = Reranker(model_name=test_settings.RERANKER_MODEL)
    retriever = HybridRetriever(
        embedder=embedder,
        vector_store=store,
        reranker=reranker,
    )
    llm_client = OpenRouterClient(
        api_key=test_settings.OPENROUTER_API_KEY,
        model=test_settings.OPENROUTER_MODEL,
    )
    rag_chain = RAGChain(retriever=retriever, llm_client=llm_client)
    processor = IngestionProcessor(
        embedder=embedder,
        vector_store=store,
        chunk_size=test_settings.CHUNK_SIZE,
        chunk_overlap=test_settings.CHUNK_OVERLAP,
    )

    # -- Override FastAPI dependencies -----------------------------------------
    app.dependency_overrides[deps.get_settings] = lambda: test_settings
    app.dependency_overrides[deps.get_embedder] = lambda: embedder
    app.dependency_overrides[deps.get_vector_store] = lambda: store
    app.dependency_overrides[deps.get_reranker] = lambda: reranker
    app.dependency_overrides[deps.get_retriever] = lambda: retriever
    app.dependency_overrides[deps.get_llm_client] = lambda: llm_client
    app.dependency_overrides[deps.get_rag_chain] = lambda: rag_chain
    app.dependency_overrides[deps.get_ingestion_processor] = lambda: processor

    yield app

    # Cleanup
    app.dependency_overrides.clear()


@pytest.fixture()
def client(test_app):
    """Return a synchronous test client wired to the overridden app."""
    return TestClient(test_app)


# ---------------------------------------------------------------------------
# Helper: upload a text document
# ---------------------------------------------------------------------------

def _upload_txt(client: TestClient, filename: str, content: bytes) -> dict:
    """Upload a plain-text file and return the parsed JSON response body."""
    resp = client.post(
        "/api/documents/upload",
        files={"file": (filename, content, "text/plain")},
    )
    assert resp.status_code == 200
    return resp.json()


# ===========================================================================
# Tests
# ===========================================================================


class TestHealthEndpoint:
    """Verify the health-check endpoint returns the expected shape."""

    def test_health_returns_200_and_expected_fields(self, client):
        resp = client.get("/api/health")
        assert resp.status_code == 200

        data = resp.json()
        assert data["status"] == "healthy"
        assert "vector_store" in data
        assert "document_count" in data["vector_store"]
        assert "embedding_model" in data
        assert "embedding_dimension" in data


class TestDocumentUpload:
    """Upload documents through the API and verify side-effects."""

    def test_upload_txt_returns_ingested(self, client):
        data = _upload_txt(
            client,
            "python_intro.txt",
            b"Python is a high-level programming language created by Guido van Rossum in 1991.",
        )
        assert data["status"] == "ingested"
        assert "document_id" in data
        assert data["filename"] == "python_intro.txt"

    def test_upload_markdown(self, client):
        resp = client.post(
            "/api/documents/upload",
            files={
                "file": (
                    "notes.md",
                    b"# Machine Learning\n\nML is a subset of artificial intelligence.",
                    "text/markdown",
                )
            },
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "ingested"

    def test_upload_csv(self, client):
        csv_content = b"name,role\nAlice,Engineer\nBob,Designer"
        resp = client.post(
            "/api/documents/upload",
            files={"file": ("team.csv", csv_content, "text/csv")},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "ingested"

    def test_upload_unsupported_format_returns_400(self, client):
        resp = client.post(
            "/api/documents/upload",
            files={"file": ("data.xyz", b"stuff", "application/octet-stream")},
        )
        assert resp.status_code == 400
        assert "Unsupported" in resp.json()["detail"]

    def test_upload_increases_vector_count(self, client):
        """Health endpoint should reflect stored chunks after ingestion."""
        count_before = client.get("/api/health").json()["vector_store"]["document_count"]

        _upload_txt(client, "facts.txt", b"Qdrant is a vector database written in Rust.")

        count_after = client.get("/api/health").json()["vector_store"]["document_count"]
        assert count_after > count_before


class TestDocumentListing:
    """Verify the document listing endpoint after ingestion."""

    def test_list_documents_empty_initially(self, client):
        resp = client.get("/api/documents")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 0
        assert data["documents"] == []

    def test_list_documents_after_upload(self, client):
        _upload_txt(
            client,
            "test_listing.txt",
            b"Some content that should appear in the document list after ingestion.",
        )

        resp = client.get("/api/documents")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] >= 1
        # Each listed document should have metadata (at least a source key)
        assert any("source" in doc for doc in data["documents"])


class TestChatEndpoint:
    """Exercise the chat (RAG) endpoint with a mocked LLM backend."""

    def test_chat_returns_sse_stream_with_sources_and_tokens(self, client):
        """Upload a document, then ask a question and verify the SSE stream."""
        # Ingest a document so the retriever has something to find
        _upload_txt(
            client,
            "python.txt",
            b"Python is a high-level programming language designed by Guido van Rossum.",
        )

        # Fake LLM streaming — yields three tokens
        async def fake_generate_stream(messages, temperature=0.7, max_tokens=2048):
            for token in ["Python", " is", " great."]:
                yield token

        with patch.object(
            OpenRouterClient,
            "generate_stream",
            side_effect=lambda *a, **kw: fake_generate_stream(*a, **kw),
        ):
            resp = client.post(
                "/api/chat",
                json={"query": "What is Python?"},
            )
            assert resp.status_code == 200

            body = resp.text

            # The SSE body should contain the sources event
            assert "event: sources" in body

            # It should also stream token events
            assert "event: token" in body

            # And end with a done event
            assert "event: done" in body

    def test_chat_sse_sources_contain_valid_json(self, client):
        """Verify the sources payload is valid JSON with the expected shape."""
        _upload_txt(
            client,
            "qdrant_info.txt",
            b"Qdrant is a vector similarity search engine written in Rust.",
        )

        async def fake_generate_stream(messages, temperature=0.7, max_tokens=2048):
            yield "Qdrant is a vector DB."

        with patch.object(
            OpenRouterClient,
            "generate_stream",
            side_effect=lambda *a, **kw: fake_generate_stream(*a, **kw),
        ):
            resp = client.post(
                "/api/chat",
                json={"query": "What is Qdrant?"},
            )
            assert resp.status_code == 200

            # Parse the SSE body line-by-line to find the sources event data
            sources_data = None
            lines = resp.text.splitlines()
            for i, line in enumerate(lines):
                if line.strip() == "event: sources":
                    # Next non-empty line starting with "data:" holds the payload
                    for subsequent in lines[i + 1 :]:
                        subsequent = subsequent.strip()
                        if subsequent.startswith("data:"):
                            sources_data = json.loads(subsequent[len("data:") :].strip())
                            break
                    break

            assert sources_data is not None, "No sources event found in SSE stream"
            assert "sources" in sources_data
            # There should be at least one source from the uploaded doc
            assert len(sources_data["sources"]) >= 1
            for src in sources_data["sources"]:
                assert "content" in src
                assert "score" in src
                assert "metadata" in src

    def test_chat_sse_tokens_form_coherent_answer(self, client):
        """Verify that all streamed token events concatenate into the expected answer."""
        _upload_txt(
            client, "ai.txt", b"Artificial intelligence is a broad field of computer science."
        )

        expected_tokens = ["AI", " is", " fascinating", "."]

        async def fake_generate_stream(messages, temperature=0.7, max_tokens=2048):
            for token in expected_tokens:
                yield token

        with patch.object(
            OpenRouterClient,
            "generate_stream",
            side_effect=lambda *a, **kw: fake_generate_stream(*a, **kw),
        ):
            resp = client.post(
                "/api/chat",
                json={"query": "What is AI?"},
            )

            collected_tokens = []
            for line in resp.text.splitlines():
                line = line.strip()
                if line.startswith("data:") and "token" in line:
                    payload = json.loads(line[len("data:"):].strip())
                    if "token" in payload:
                        collected_tokens.append(payload["token"])

            assert collected_tokens == expected_tokens

    def test_chat_with_conversation_history(self, client):
        """Verify the endpoint accepts and passes conversation history."""
        _upload_txt(client, "general.txt", b"FastAPI is a modern Python web framework.")

        async def fake_generate_stream(messages, temperature=0.7, max_tokens=2048):
            yield "It uses ASGI."

        with patch.object(
            OpenRouterClient,
            "generate_stream",
            side_effect=lambda *a, **kw: fake_generate_stream(*a, **kw),
        ):
            resp = client.post(
                "/api/chat",
                json={
                    "query": "Tell me more about it.",
                    "conversation_history": [
                        {"role": "user", "content": "What is FastAPI?"},
                        {"role": "assistant", "content": "FastAPI is a web framework."},
                    ],
                },
            )
            assert resp.status_code == 200
            assert "event: done" in resp.text

    def test_chat_with_empty_store_still_streams(self, client):
        """Even with no documents the endpoint should not crash — it streams an
        empty sources event and whatever the LLM produces."""

        async def fake_generate_stream(messages, temperature=0.7, max_tokens=2048):
            yield "I don't have information about that."

        with patch.object(
            OpenRouterClient,
            "generate_stream",
            side_effect=lambda *a, **kw: fake_generate_stream(*a, **kw),
        ):
            resp = client.post(
                "/api/chat",
                json={"query": "Random question with no context?"},
            )
            assert resp.status_code == 200
            assert "event: sources" in resp.text
            assert "event: done" in resp.text


class TestDocumentDeletion:
    """Verify the delete endpoint removes chunks from the store."""

    def test_delete_returns_success(self, client):
        data = _upload_txt(client, "to_delete.txt", b"Temporary content to be removed.")
        doc_id = data["document_id"]

        resp = client.delete(f"/api/documents/{doc_id}")
        assert resp.status_code == 200
        assert resp.json()["status"] == "deleted"
        assert resp.json()["document_id"] == doc_id
