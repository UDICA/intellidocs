# IntelliDocs — Intelligent Document Understanding & Retrieval



## Project Overview

A clean, well-documented, production-ready RAG (Retrieval-Augmented Generation) system that transforms document collections into an intelligent, queryable knowledge base. Designed as a public portfolio project for a Senior Data Scientist / AI Director with 13+ years of experience, demonstrating expertise in RAG, LLMs, vector databases, and end-to-end AI pipeline design.



## Purpose

Public GitHub repository to showcase RAG expertise for job applications (Data Scientist / AI Director roles). Must be polished, well-documented, and demonstrate professional-grade engineering.



## Architecture



### Core Components

1. **Document Ingestion Pipeline**

   - Support multiple file formats: PDF, DOCX, TXT, Markdown, CSV

   - Chunking strategies: recursive character splitting with configurable chunk size and overlap

   - Metadata extraction (source, page number, section headers)



2. **Embedding & Vector Store**

   - Embedding model: Use sentence-transformers (e.g., `all-MiniLM-L6-v2` for lightweight demo, or `bge-large-en-v1.5` for quality)

   - Vector database: **Qdrant** (preferred — runs locally via Docker, or in-memory for demo mode)

   - Alternative: ChromaDB as fallback (simpler setup, no Docker needed)

   - Store embeddings with metadata for filtered retrieval



3. **Retrieval Engine**

   - Semantic search with cosine similarity

   - Hybrid search: combine dense (vector) + sparse (BM25) retrieval

   - Re-ranking step using a cross-encoder (e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2`)

   - Configurable top-k and similarity threshold



4. **Generation (LLM Integration)**

   - Support multiple LLM backends:

     - **Ollama** (local, free — primary demo mode)

     - **OpenAI API** (optional, via environment variable)

     - **Anthropic API** (optional, via environment variable)

   - Prompt engineering with:

     - System prompt defining assistant behavior

     - Context injection from retrieved documents

     - Source citation in responses

   - Streaming response support



5. **Simple Web UI**

   - Streamlit or Gradio interface (Streamlit preferred)

   - Chat interface with conversation history

   - Source document display (show which chunks were used)

   - Upload new documents through the UI

   - Settings panel (model selection, retrieval parameters)



## Tech Stack

- **Language**: Python 3.11+

- **Package Manager**: uv (preferred) or pip

- **Vector DB**: Qdrant (Docker) or ChromaDB (in-memory fallback)

- **Embeddings**: sentence-transformers

- **LLM**: Ollama (local) / OpenAI / Anthropic (configurable)

- **Document Processing**: langchain document loaders, unstructured

- **UI**: Streamlit

- **Containerization**: Docker + docker-compose

- **Testing**: pytest

- **Linting**: ruff



## Project Structure

```

intellidocs/

├── CLAUDE.md

├── README.md                  # Professional README with architecture diagram, setup, usage

├── LICENSE                    # MIT License

├── pyproject.toml             # Project config with uv/pip dependencies

├── docker-compose.yml         # Qdrant + app services

├── Dockerfile                 # App container

├── .env.example               # Environment variable template

├── .gitignore

├── src/

│   ├── __init__.py

│   ├── config.py              # Configuration management (pydantic-settings)

│   ├── ingestion/

│   │   ├── __init__.py

│   │   ├── loader.py          # Multi-format document loader

│   │   ├── chunker.py         # Text chunking strategies

│   │   └── processor.py       # Ingestion pipeline orchestrator

│   ├── embeddings/

│   │   ├── __init__.py

│   │   └── embedder.py        # Embedding model wrapper

│   ├── vectorstore/

│   │   ├── __init__.py

│   │   ├── qdrant_store.py    # Qdrant implementation

│   │   └── chroma_store.py    # ChromaDB fallback

│   ├── retrieval/

│   │   ├── __init__.py

│   │   ├── retriever.py       # Hybrid retrieval (dense + sparse)

│   │   └── reranker.py        # Cross-encoder re-ranking

│   ├── generation/

│   │   ├── __init__.py

│   │   ├── llm_client.py      # Multi-backend LLM client

│   │   ├── prompts.py         # Prompt templates

│   │   └── chain.py           # RAG chain orchestration

│   └── evaluation/

│       ├── __init__.py

│       └── metrics.py         # Retrieval quality metrics (MRR, recall@k)

├── ui/

│   └── app.py                 # Streamlit application

├── tests/

│   ├── test_ingestion.py

│   ├── test_retrieval.py

│   ├── test_generation.py

│   └── test_integration.py

├── data/

│   └── sample_docs/           # Sample documents for demo

│       ├── README.md           # Explain sample data

│       └── ...                 # 3-5 sample documents (public domain)

├── notebooks/

│   └── exploration.ipynb      # Notebook showing retrieval quality analysis

└── docs/

    ├── architecture.md         # Detailed architecture documentation

    └── images/

        └── architecture_diagram.png  # System diagram

```



## README Requirements

The README.md must be **exemplary** — this is the first thing recruiters see. Include:



1. **Project title + one-line description**

2. **Architecture diagram** (Mermaid or image)

3. **Features list** with checkmarks

4. **Quick Start** (3 commands or fewer to get running)

5. **Detailed Setup** (Docker, local, cloud options)

6. **Usage Examples** with screenshots or GIFs

7. **Configuration** section

8. **How It Works** — brief technical explanation of the RAG pipeline

9. **Evaluation Results** — show retrieval metrics

10. **Tech Stack** with version badges

11. **Contributing** section

12. **License**



## Code Quality Standards

- **Type hints** on all functions

- **Docstrings** on all public methods (Google style)

- **Pydantic models** for configuration and data validation

- **Abstract base classes** for vector store and LLM backends (easy to swap implementations)

- **Logging** with structlog or standard logging

- **Error handling** with custom exceptions

- **Environment variables** for all secrets/config (never hardcoded)

- Clean separation of concerns — each module is independently testable



## Key Design Decisions to Highlight

These demonstrate senior-level thinking:

1. **Hybrid retrieval** (not just naive similarity search) — shows understanding of retrieval quality

2. **Re-ranking step** — demonstrates awareness of the recall-precision tradeoff

3. **Multiple LLM backends** — shows production thinking (vendor flexibility)

4. **Docker-compose setup** — shows DevOps/MLOps awareness

5. **Evaluation metrics** — shows you measure what matters

6. **Configurable chunking** — shows understanding that chunk strategy impacts quality



## Sample Data

Include 3-5 public domain documents for the demo:

- A Wikipedia article (plain text)

- A public research paper (PDF)

- A CSV dataset with descriptions

- A markdown technical document



## Demo Mode

The project MUST work out-of-the-box with:

```bash

git clone <repo>

cd intellidocs

docker-compose up

# Open http://localhost:8501

```

No API keys required for basic demo (uses Ollama locally or ChromaDB in-memory with a small local model).



## What NOT to Include

- No proprietary data or company-specific code

- No API keys or secrets (even in git history)

- No overly complex abstractions — keep it readable

- No Jupyter notebooks as primary code (notebooks are supplementary only)



## Tone

Professional but approachable. Code comments should explain *why*, not *what*. The README should be written for someone evaluating your technical skills.
