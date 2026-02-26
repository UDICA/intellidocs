# IntelliDocs — Design Document

**Date:** 2026-02-25

---

## Overview

IntelliDocs is a production-ready RAG (Retrieval-Augmented Generation) system that transforms document collections into an intelligent, queryable knowledge base. It features a FastAPI backend with a modular pipeline architecture, a Next.js frontend with polished animations, and Qdrant as the vector store.

---

## Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Embeddings | Configurable, default `all-MiniLM-L6-v2` | Lightweight for demo, swappable to `bge-large-en-v1.5` for quality |
| Vector store | Qdrant in-memory (default), Docker Qdrant (prod), ChromaDB (alt) | Consistent primary store, no Docker needed for quick start |
| LLM backend | OpenRouter only | Single API proxies 200+ models, no need for separate clients |
| Default model | `anthropic/claude-haiku-4-5-20251001` via `OPENROUTER_MODEL` | Good quality/cost balance, easily configurable |
| Hybrid search | Qdrant native sparse vectors | Leverages Qdrant's built-in capabilities, no separate BM25 index |
| Frontend | Next.js 14 + Tailwind + Framer Motion | Full control over animations and visual polish |
| Backend | FastAPI + SSE streaming | Clean REST API, SSE fits the unidirectional streaming use case |
| Document upload | UI upload + CLI ingestion | Two interfaces for the same pipeline |
| Auth | None | Demo project, frictionless quick start |
| Containers | 3 separate (frontend, backend, qdrant) | Clean service decomposition |

---

## Project Structure

```
intellidocs/
├── CLAUDE.md
├── README.md
├── LICENSE
├── pyproject.toml
├── docker-compose.yml
├── .env.example
├── .gitignore
├── backend/
│   ├── Dockerfile
│   ├── src/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── main.py
│   │   ├── api/
│   │   │   ├── __init__.py
│   │   │   ├── routes/
│   │   │   │   ├── chat.py
│   │   │   │   ├── documents.py
│   │   │   │   └── health.py
│   │   │   └── deps.py
│   │   ├── ingestion/
│   │   │   ├── __init__.py
│   │   │   ├── loader.py
│   │   │   ├── chunker.py
│   │   │   └── processor.py
│   │   ├── embeddings/
│   │   │   ├── __init__.py
│   │   │   └── embedder.py
│   │   ├── vectorstore/
│   │   │   ├── __init__.py
│   │   │   ├── base.py
│   │   │   ├── qdrant_store.py
│   │   │   └── chroma_store.py
│   │   ├── retrieval/
│   │   │   ├── __init__.py
│   │   │   ├── retriever.py
│   │   │   └── reranker.py
│   │   ├── generation/
│   │   │   ├── __init__.py
│   │   │   ├── llm_client.py
│   │   │   ├── prompts.py
│   │   │   └── rag_chain.py
│   │   └── cli.py
│   └── tests/
│       ├── test_ingestion.py
│       ├── test_retrieval.py
│       ├── test_generation.py
│       └── test_integration.py
├── frontend/
│   ├── Dockerfile
│   ├── package.json
│   ├── next.config.js
│   ├── tailwind.config.js
│   ├── tsconfig.json
│   ├── public/
│   ├── src/
│   │   ├── app/
│   │   │   ├── layout.tsx
│   │   │   ├── page.tsx
│   │   │   └── globals.css
│   │   ├── components/
│   │   │   ├── Chat/
│   │   │   │   ├── ChatWindow.tsx
│   │   │   │   ├── MessageBubble.tsx
│   │   │   │   └── InputBar.tsx
│   │   │   ├── Documents/
│   │   │   │   ├── UploadPanel.tsx
│   │   │   │   └── DocumentList.tsx
│   │   │   ├── Sources/
│   │   │   │   └── SourceCard.tsx
│   │   │   ├── Settings/
│   │   │   │   └── SettingsPanel.tsx
│   │   │   └── ui/
│   │   ├── hooks/
│   │   │   └── useChat.ts
│   │   └── lib/
│   │       └── api.ts
├── data/
│   └── sample_docs/
├── notebooks/
│   └── exploration.ipynb
└── docs/
    ├── architecture.md
    ├── plans/
    └── images/
```

---

## Backend Architecture

### API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/chat` | Query with SSE streaming response |
| POST | `/api/documents/upload` | Multipart file upload, triggers ingestion |
| GET | `/api/documents` | List ingested documents with metadata |
| DELETE | `/api/documents/{id}` | Remove document and its vectors |
| GET | `/api/health` | Health check (vector store, embedding model) |

### Ingestion Pipeline

```
File → Loader (detect format: PDF, DOCX, TXT, MD, CSV)
     → Chunker (recursive character split, configurable size/overlap)
     → Embedder (dense via sentence-transformers + sparse for BM25)
     → Qdrant (upsert with metadata)
```

### RAG Chain

```
Query → Embedder (dense + sparse)
      → Qdrant hybrid search (dense + sparse vectors)
      → Re-ranker (cross-encoder: ms-marco-MiniLM-L-6-v2)
      → Top-k chunks
      → Prompt assembly (system prompt + context + query)
      → OpenRouter API
      → SSE stream to client
```

### Configuration

Managed via `pydantic-settings`, reading from `.env`:

- `OPENROUTER_API_KEY` — required for LLM generation
- `OPENROUTER_MODEL` — default `anthropic/claude-haiku-4-5-20251001`
- `QDRANT_HOST`, `QDRANT_PORT`, `QDRANT_COLLECTION`
- `EMBEDDING_MODEL` — default `all-MiniLM-L6-v2`
- `RERANKER_MODEL` — default `cross-encoder/ms-marco-MiniLM-L-6-v2`
- `CHUNK_SIZE`, `CHUNK_OVERLAP`
- `TOP_K`, `SIMILARITY_THRESHOLD`
- `VECTOR_STORE_BACKEND` — `qdrant` (default) or `chroma`

### Dependency Injection

Vector store, embedder, and LLM client instantiated once at startup and injected into route handlers via FastAPI's `Depends()`.

---

## Frontend Architecture

### Tech Stack

- Next.js 14 (App Router) + TypeScript
- Tailwind CSS (dark theme default)
- Framer Motion (animations)

### Layout

Single-page app: sidebar (documents + settings) + main chat area.

### Components & Animations

| Component | Animation |
|-----------|-----------|
| `ChatWindow` | Staggered fade-in + slide-up for messages |
| `MessageBubble` | Hover: subtle scale + shadow lift |
| `SourceCard` | Hover glow, click to expand with spring animation |
| `InputBar` | Fixed bottom, auto-resize, send button press effect |
| `UploadPanel` | Drag-and-drop with animated border, progress bar |
| `DocumentList` | Staggered list on load, exit animation on delete |
| `SettingsPanel` | Smooth accordion sections |

### State Management

React hooks only (`useState`, `useReducer`). Custom `useChat` hook handles SSE connection, message accumulation, and streaming state.

---

## SSE Streaming Protocol

### Request

```json
POST /api/chat
{
  "query": "What does the document say about X?",
  "conversation_history": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "top_k": 5
}
```

### Response (SSE stream)

```
event: sources
data: {"sources": [{"document": "file.pdf", "page": 3, "chunk": "...", "score": 0.87}]}

event: token
data: {"token": "Based"}

event: token
data: {"token": " on"}

...

event: done
data: {}
```

Sources are sent first (before generation) so the frontend can display them immediately while the answer streams in.

---

## Document Upload Flow

- **UI upload:** Multipart POST from the frontend. For large files, the API returns immediately with a document ID and ingestion runs as a background task. Frontend polls for status.
- **CLI ingestion:** `python -m backend.src.cli ingest ./my_docs/` for bulk ingestion.

---

## Docker & Deployment

### docker-compose.yml

Three services:

- `qdrant` — official Qdrant image, port 6333, persistent volume
- `backend` — Python 3.11 slim, built with uv, port 8000, depends on qdrant
- `frontend` — Node 20 alpine, multi-stage build, port 3000, depends on backend

### Quick Start (Docker)

```bash
git clone <repo>
cp .env.example .env  # add OPENROUTER_API_KEY
docker-compose up
# Open http://localhost:3000
```

### Local Dev (no Docker)

Qdrant runs in-memory (no external service needed). Backend via `uvicorn`, frontend via `next dev`.
