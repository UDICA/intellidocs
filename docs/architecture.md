# Architecture Documentation

This document describes the internal architecture of IntelliDocs, a Retrieval-Augmented Generation (RAG) system for intelligent document understanding and retrieval.

---

## System Overview

IntelliDocs is structured as a three-tier application:

1. **Frontend** — A Next.js single-page application providing the chat interface, document upload, and settings panel.
2. **Backend** — A FastAPI service that orchestrates the RAG pipeline, exposes REST endpoints, and streams responses via Server-Sent Events (SSE).
3. **Vector Store** — A Qdrant instance (or ChromaDB fallback) that persists document embeddings and supports hybrid search.

```
┌──────────────┐     HTTP / SSE     ┌──────────────────┐
│   Frontend   │ ◄────────────────► │   FastAPI Backend │
│  (Next.js)   │                    │                   │
└──────────────┘                    │  ┌─────────────┐  │
                                    │  │  Ingestion   │  │
                                    │  │  Pipeline    │  │
                                    │  └──────┬───────┘  │
                                    │         │          │
                                    │  ┌──────▼───────┐  │
                                    │  │  Embedder    │  │
                                    │  │ (sentence-   │  │
                                    │  │  transformers)│  │
                                    │  └──────┬───────┘  │
                                    │         │          │
                                    │  ┌──────▼───────┐  │        ┌───────────┐
                                    │  │ Vector Store │──┼───────►│  Qdrant   │
                                    │  │   (base)     │  │        │  / Chroma │
                                    │  └──────────────┘  │        └───────────┘
                                    │                    │
                                    │  ┌─────────────┐   │        ┌───────────┐
                                    │  │  Retrieval   │   │        │ OpenRouter│
                                    │  │  + Reranker  │   │        │   LLM    │
                                    │  └──────┬───────┘  │        └─────▲─────┘
                                    │         │          │              │
                                    │  ┌──────▼───────┐  │              │
                                    │  │  RAG Chain   │──┼──────────────┘
                                    │  └──────────────┘  │
                                    └──────────────────────┘
```

---

## Data Flow

### Ingestion Pipeline

The ingestion pipeline converts raw documents into searchable vectors. It runs when a user uploads a document through the UI or invokes the CLI.

```
Document File
    │
    ▼
┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐
│  Loader  │────►│ Chunker  │────►│ Embedder │────►│  Vector  │
│          │     │          │     │          │     │  Store   │
└──────────┘     └──────────┘     └──────────┘     └──────────┘
```

**Step-by-step:**

1. **Loader** (`backend/src/ingestion/loader.py`) — Detects file format (PDF, DOCX, TXT, Markdown, CSV) and extracts raw text content along with metadata (filename, page numbers, file type).

2. **Chunker** (`backend/src/ingestion/chunker.py`) — Splits extracted text into overlapping chunks using recursive character splitting. The chunk size and overlap are configurable via `CHUNK_SIZE` and `CHUNK_OVERLAP` environment variables. The recursive strategy tries to split on paragraph boundaries first, then sentences, then words, preserving semantic coherence.

3. **Embedder** (`backend/src/embeddings/embedder.py`) — Encodes each chunk into two vector representations:
   - **Dense vector**: A fixed-length floating-point vector from a sentence-transformers model (default: `all-MiniLM-L6-v2`, 384 dimensions). Captures semantic meaning.
   - **Sparse vector**: A BM25-based term-frequency vector. Captures lexical matching for keyword-sensitive queries.

4. **Vector Store** (`backend/src/vectorstore/`) — Inserts the dense and sparse vectors along with the original text and metadata into Qdrant (or ChromaDB). Each document chunk becomes a point in the vector space.

### Retrieval Pipeline

The retrieval pipeline runs for every user query. It follows a **recall-then-rerank** pattern.

```
User Query
    │
    ▼
┌──────────┐     ┌──────────────┐     ┌──────────┐     ┌──────────┐
│  Embed   │────►│ Hybrid Search│────►│ Re-rank  │────►│ Generate │
│  Query   │     │ (Dense+BM25) │     │ (Cross-  │     │ (LLM via │
│          │     │              │     │  Encoder) │     │ OpenRouter│
└──────────┘     └──────────────┘     └──────────┘     └──────────┘
```

**Step-by-step:**

1. **Embed Query** — The user's question is encoded into dense and sparse vectors using the same embedding model used during ingestion.

2. **Hybrid Search** — Qdrant performs a combined dense + sparse search, returning `top_k * 3` candidates (the 3x over-fetch ensures the re-ranker has enough candidates to find the truly relevant results).

3. **Re-rank** (`backend/src/retrieval/reranker.py`) — A cross-encoder model (`cross-encoder/ms-marco-MiniLM-L-6-v2`) scores each (query, candidate) pair by reading both texts jointly. This produces far more accurate relevance estimates than the initial bi-encoder scores. The top-k results after re-ranking are kept.

4. **Generate** — The re-ranked document chunks are formatted into a prompt using a template that instructs the LLM to answer based on the provided context and cite sources. The prompt is sent to OpenRouter, which streams the response token-by-token via SSE back to the frontend.

---

## Component Descriptions

### Configuration (`backend/src/config.py`)

Centralized settings management using `pydantic-settings`. All configuration is loaded from environment variables or a `.env` file, with validation (e.g., chunk overlap must be less than chunk size). No hardcoded values.

### Document Loader (`backend/src/ingestion/loader.py`)

Multi-format document loader supporting PDF (pypdf), DOCX (python-docx), plain text, Markdown, and CSV. Each format has a dedicated loading path that extracts text and metadata. Returns a list of document objects with `content` and `metadata` fields.

### Text Chunker (`backend/src/ingestion/chunker.py`)

Recursive character text splitter that breaks documents into overlapping chunks. The recursive strategy attempts splits at paragraph boundaries (`\n\n`), then line breaks (`\n`), then sentences (`. `), and finally at the character level. This preserves semantic units where possible.

### Embedder (`backend/src/embeddings/embedder.py`)

Wraps a sentence-transformers model for dense embeddings and a BM25 tokenizer for sparse embeddings. Both representations are computed for each chunk during ingestion and for each query during retrieval. The dual representation enables hybrid search.

### Vector Store (`backend/src/vectorstore/`)

Abstract base class (`base.py`) defines the interface: `upsert`, `search`, and `delete`. Two implementations:

- **Qdrant** (`qdrant_store.py`) — Production backend using the Qdrant client. Supports named vectors for hybrid search (separate dense and sparse vector spaces). Data persists to disk via Docker volume.
- **ChromaDB** (`chroma_store.py`) — Lightweight fallback that runs in-memory. Useful for testing and development without Docker.

### Hybrid Retriever (`backend/src/retrieval/retriever.py`)

Orchestrates the two-stage retrieval process. Accepts an embedder, vector store, and optional reranker. Over-fetches candidates from the vector store, then re-ranks them with the cross-encoder. Falls back to returning vector-store results directly if no reranker is configured.

### Cross-Encoder Reranker (`backend/src/retrieval/reranker.py`)

Loads a cross-encoder model that scores (query, passage) pairs by processing both texts jointly through a transformer. This is more accurate than bi-encoder dot products but more expensive, which is why it only runs on the candidate set (not the full corpus).

### LLM Client (`backend/src/generation/llm_client.py`)

Async HTTP client for the OpenRouter API. Supports both streaming (SSE) and non-streaming generation. Handles API key authentication, model selection, and error handling. Any model available on OpenRouter can be used by changing the `OPENROUTER_MODEL` environment variable.

### Prompt Templates (`backend/src/generation/prompts.py`)

Constructs the messages array sent to the LLM. The system prompt defines the assistant's behavior (answer from context, cite sources, admit when information is not found). Retrieved document chunks are formatted with source metadata and injected into the user message.

### RAG Chain (`backend/src/generation/rag_chain.py`)

Top-level orchestrator that wires retrieval and generation together. Exposes:
- `retrieve_sources()` — retrieval only (used by the API to send sources before streaming starts)
- `generate_stream()` — generation only from pre-retrieved sources
- `query()` — full pipeline (retrieve + generate) for non-streaming use

### API Layer (`backend/src/api/`)

FastAPI routes organized into three routers:
- **Health** (`/api/health`) — Liveness check
- **Documents** (`/api/documents`) — Upload and list ingested documents
- **Chat** (`/api/chat`) — Query endpoint with SSE streaming

Dependency injection (`deps.py`) manages the lifecycle of shared resources (embedder, vector store, retriever, LLM client, RAG chain).

### Frontend (`frontend/`)

Next.js 16 application with:
- Chat interface with message history and streaming display
- Document upload panel with file format detection
- Source citation cards showing which chunks informed each answer
- Settings panel for model and retrieval parameter configuration
- Responsive sidebar layout with keyboard navigation

---

## Configuration Reference

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `OPENROUTER_API_KEY` | `str` | `""` | OpenRouter API key (required for generation) |
| `OPENROUTER_MODEL` | `str` | `anthropic/claude-haiku-4.5` | Model identifier on OpenRouter |
| `EMBEDDING_MODEL` | `str` | `all-MiniLM-L6-v2` | Sentence-transformers model name |
| `RERANKER_MODEL` | `str` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder model name |
| `QDRANT_HOST` | `str` | `localhost` | Qdrant server hostname |
| `QDRANT_PORT` | `int` | `6333` | Qdrant server port |
| `QDRANT_COLLECTION` | `str` | `intellidocs` | Name of the Qdrant collection |
| `VECTOR_STORE_BACKEND` | `str` | `qdrant` | `qdrant` or `chroma` |
| `CHUNK_SIZE` | `int` | `512` | Maximum chunk size in characters |
| `CHUNK_OVERLAP` | `int` | `50` | Character overlap between consecutive chunks |
| `TOP_K` | `int` | `5` | Number of documents returned to the LLM |
| `SIMILARITY_THRESHOLD` | `float` | `0.3` | Minimum score for retrieval results |

Validated at startup via Pydantic: `CHUNK_OVERLAP` must be strictly less than `CHUNK_SIZE`.

---

## Design Decisions and Rationale

### Hybrid Retrieval over Pure Dense Search

Pure dense (embedding-based) retrieval excels at semantic similarity but struggles with exact keyword matches, acronyms, and proper nouns. Adding BM25 sparse vectors ensures that lexical matches are captured alongside semantic ones. The combination improves recall on a wider range of query types.

### Two-Stage Recall-then-Rerank

Bi-encoder models are fast but produce relatively coarse relevance scores because query and document are encoded independently. Cross-encoders process query and document together through the same transformer, producing much more accurate scores — but at higher computational cost. The two-stage pattern uses the fast bi-encoder to cast a wide net, then applies the expensive cross-encoder only to the top candidates. This balances speed with accuracy.

### OpenRouter as LLM Gateway

Rather than integrating directly with multiple LLM providers (OpenAI, Anthropic, Google, etc.), IntelliDocs uses OpenRouter as a single gateway. This provides access to hundreds of models through one API key and one integration point, making model switching a configuration change rather than a code change.

### Abstract Vector Store Interface

The `VectorStoreBase` abstract class allows swapping between Qdrant and ChromaDB without changing any business logic. This is valuable for testing (ChromaDB in-memory is faster and needs no Docker) and for deployment flexibility (teams may already have a preferred vector database).

### Streaming SSE Responses

The chat endpoint uses Server-Sent Events rather than WebSockets. SSE is simpler (unidirectional, works over standard HTTP, no special proxy configuration) and sufficient for the use case where the server streams tokens and the client only sends requests. The API sends sources as a JSON event first, then streams answer tokens, giving users immediate feedback while the LLM generates.

### Recursive Character Splitting

The chunker uses a recursive strategy that tries larger semantic boundaries first (paragraphs, then sentences, then words). This preserves semantic coherence within chunks better than fixed-size splitting. The configurable overlap ensures that information at chunk boundaries is not lost.

### Separation of Retrieval and Generation

The RAG chain exposes retrieval and generation as separate methods rather than bundling them. This allows the API layer to send source documents to the frontend immediately (before the LLM starts generating), providing a more responsive user experience. It also makes each stage independently testable.
