# IntelliDocs Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a production-ready RAG system with a FastAPI backend, Next.js frontend, and Qdrant vector store.

**Architecture:** Modular pipeline — ingestion, embedding, vector storage, hybrid retrieval with re-ranking, generation via OpenRouter, SSE streaming to a Next.js frontend. Three Docker containers (frontend, backend, qdrant).

**Tech Stack:** Python 3.11+, FastAPI, Qdrant, sentence-transformers, cross-encoder, OpenRouter API, Next.js 14, Tailwind CSS, Framer Motion, Docker.

---

## Phase 1: Project Scaffolding

### Task 1: Initialize git repo and project root files

**Files:**
- Create: `.gitignore`
- Create: `.env.example`
- Create: `LICENSE`
- Create: `pyproject.toml`

**Step 1: Initialize git repo**

```bash
cd /mnt/d/Proyectos-IA/github_public_repos/intellidocs
git init
```

**Step 2: Create .gitignore**

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.egg-info/
dist/
build/
.eggs/
*.egg
.venv/
venv/

# Environment
.env
.env.local

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Qdrant data
qdrant_data/

# Node
node_modules/
.next/
out/

# Test / Coverage
.coverage
htmlcov/
.pytest_cache/

# Data (keep sample_docs)
data/uploads/
```

**Step 3: Create .env.example**

```env
# Required: OpenRouter API key for LLM generation
OPENROUTER_API_KEY=your_openrouter_api_key_here

# LLM model (any model available on OpenRouter)
OPENROUTER_MODEL=anthropic/claude-haiku-4-5-20251001

# Embedding model (sentence-transformers model name)
EMBEDDING_MODEL=all-MiniLM-L6-v2

# Reranker model (cross-encoder model name)
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2

# Qdrant configuration
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION=intellidocs

# Vector store backend: "qdrant" or "chroma"
VECTOR_STORE_BACKEND=qdrant

# Chunking configuration
CHUNK_SIZE=512
CHUNK_OVERLAP=50

# Retrieval configuration
TOP_K=5
SIMILARITY_THRESHOLD=0.3
```

**Step 4: Create LICENSE (MIT)**

Standard MIT license with copyright holder "IntelliDocs Contributors".

**Step 5: Create pyproject.toml**

```toml
[project]
name = "intellidocs"
version = "0.1.0"
description = "Intelligent Document Understanding & Retrieval — a production-ready RAG system"
readme = "README.md"
license = {text = "MIT"}
requires-python = ">=3.11"
dependencies = [
    "fastapi>=0.115.0",
    "uvicorn[standard]>=0.32.0",
    "pydantic>=2.10.0",
    "pydantic-settings>=2.7.0",
    "qdrant-client>=1.12.0",
    "sentence-transformers>=3.3.0",
    "chromadb>=0.5.0",
    "httpx>=0.28.0",
    "python-multipart>=0.0.18",
    "pypdf>=5.0.0",
    "python-docx>=1.1.0",
    "unstructured>=0.16.0",
    "sse-starlette>=2.2.0",
    "structlog>=24.4.0",
    "rank-bm25>=0.2.2",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.3.0",
    "pytest-asyncio>=0.24.0",
    "pytest-cov>=6.0.0",
    "httpx>=0.28.0",
    "ruff>=0.8.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.ruff]
target-version = "py311"
line-length = 100

[tool.ruff.lint]
select = ["E", "F", "I", "N", "W", "UP"]

[tool.pytest.ini_options]
testpaths = ["backend/tests"]
asyncio_mode = "auto"
```

**Step 6: Commit**

```bash
git add .gitignore .env.example LICENSE pyproject.toml
git commit -m "chore: initialize project with root config files"
```

---

### Task 2: Create backend directory structure with __init__.py files

**Files:**
- Create: `backend/src/__init__.py`
- Create: `backend/src/api/__init__.py`
- Create: `backend/src/api/routes/__init__.py`
- Create: `backend/src/ingestion/__init__.py`
- Create: `backend/src/embeddings/__init__.py`
- Create: `backend/src/vectorstore/__init__.py`
- Create: `backend/src/retrieval/__init__.py`
- Create: `backend/src/generation/__init__.py`
- Create: `backend/tests/__init__.py`

**Step 1: Create all directories and empty __init__.py files**

```bash
mkdir -p backend/src/{api/routes,ingestion,embeddings,vectorstore,retrieval,generation}
mkdir -p backend/tests
touch backend/src/__init__.py
touch backend/src/api/__init__.py
touch backend/src/api/routes/__init__.py
touch backend/src/ingestion/__init__.py
touch backend/src/embeddings/__init__.py
touch backend/src/vectorstore/__init__.py
touch backend/src/retrieval/__init__.py
touch backend/src/generation/__init__.py
touch backend/tests/__init__.py
```

**Step 2: Commit**

```bash
git add backend/
git commit -m "chore: create backend directory structure"
```

---

## Phase 2: Backend Core — Configuration & Ingestion

### Task 3: Configuration module (pydantic-settings)

**Files:**
- Create: `backend/src/config.py`
- Create: `backend/tests/test_config.py`

**Step 1: Write the failing test**

```python
# backend/tests/test_config.py
import os
import pytest


def test_config_loads_defaults():
    """Config should provide sensible defaults for all optional fields."""
    from backend.src.config import Settings

    settings = Settings(OPENROUTER_API_KEY="test-key")
    assert settings.OPENROUTER_API_KEY == "test-key"
    assert settings.OPENROUTER_MODEL == "anthropic/claude-haiku-4-5-20251001"
    assert settings.EMBEDDING_MODEL == "all-MiniLM-L6-v2"
    assert settings.RERANKER_MODEL == "cross-encoder/ms-marco-MiniLM-L-6-v2"
    assert settings.QDRANT_HOST == "localhost"
    assert settings.QDRANT_PORT == 6333
    assert settings.QDRANT_COLLECTION == "intellidocs"
    assert settings.VECTOR_STORE_BACKEND == "qdrant"
    assert settings.CHUNK_SIZE == 512
    assert settings.CHUNK_OVERLAP == 50
    assert settings.TOP_K == 5
    assert settings.SIMILARITY_THRESHOLD == 0.3


def test_config_from_env(monkeypatch):
    """Config should read values from environment variables."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "env-key")
    monkeypatch.setenv("CHUNK_SIZE", "1024")
    monkeypatch.setenv("TOP_K", "10")

    from backend.src.config import Settings

    settings = Settings()
    assert settings.OPENROUTER_API_KEY == "env-key"
    assert settings.CHUNK_SIZE == 1024
    assert settings.TOP_K == 10


def test_config_validates_chunk_overlap():
    """Chunk overlap must be less than chunk size."""
    from backend.src.config import Settings

    with pytest.raises(ValueError):
        Settings(OPENROUTER_API_KEY="test", CHUNK_SIZE=100, CHUNK_OVERLAP=150)
```

**Step 2: Run test to verify it fails**

```bash
cd /mnt/d/Proyectos-IA/github_public_repos/intellidocs
python -m pytest backend/tests/test_config.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'backend.src.config'`

**Step 3: Write the implementation**

```python
# backend/src/config.py
from typing import Literal

from pydantic import model_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    # OpenRouter
    OPENROUTER_API_KEY: str = ""
    OPENROUTER_MODEL: str = "anthropic/claude-haiku-4-5-20251001"

    # Embeddings
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Qdrant
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION: str = "intellidocs"

    # Vector store
    VECTOR_STORE_BACKEND: Literal["qdrant", "chroma"] = "qdrant"

    # Chunking
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50

    # Retrieval
    TOP_K: int = 5
    SIMILARITY_THRESHOLD: float = 0.3

    @model_validator(mode="after")
    def validate_chunk_overlap(self) -> "Settings":
        if self.CHUNK_OVERLAP >= self.CHUNK_SIZE:
            raise ValueError(
                f"CHUNK_OVERLAP ({self.CHUNK_OVERLAP}) must be less than "
                f"CHUNK_SIZE ({self.CHUNK_SIZE})"
            )
        return self
```

**Step 4: Run test to verify it passes**

```bash
python -m pytest backend/tests/test_config.py -v
```

Expected: 3 passed

**Step 5: Commit**

```bash
git add backend/src/config.py backend/tests/test_config.py
git commit -m "feat: add configuration module with pydantic-settings"
```

---

### Task 4: Document loader (multi-format)

**Files:**
- Create: `backend/src/ingestion/loader.py`
- Create: `backend/tests/test_ingestion.py`
- Create: `data/sample_docs/` (test fixtures)

**Step 1: Create test fixture files**

```bash
mkdir -p data/sample_docs
```

Create a small `data/sample_docs/test.txt` with a few paragraphs of text, and a `data/sample_docs/test.md` with markdown content. These serve as test fixtures.

**Step 2: Write the failing test**

```python
# backend/tests/test_ingestion.py
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def sample_txt(tmp_path: Path) -> Path:
    f = tmp_path / "sample.txt"
    f.write_text("This is a test document.\nIt has two lines.")
    return f


@pytest.fixture
def sample_md(tmp_path: Path) -> Path:
    f = tmp_path / "sample.md"
    f.write_text("# Title\n\nSome markdown content.\n\n## Section\n\nMore text here.")
    return f


@pytest.fixture
def sample_csv(tmp_path: Path) -> Path:
    f = tmp_path / "sample.csv"
    f.write_text("name,description\nAlice,Engineer\nBob,Designer")
    return f


class TestDocumentLoader:
    def test_load_txt(self, sample_txt: Path):
        from backend.src.ingestion.loader import DocumentLoader

        loader = DocumentLoader()
        docs = loader.load(sample_txt)
        assert len(docs) >= 1
        assert "test document" in docs[0].content
        assert docs[0].metadata["source"] == str(sample_txt)
        assert docs[0].metadata["format"] == "txt"

    def test_load_markdown(self, sample_md: Path):
        from backend.src.ingestion.loader import DocumentLoader

        loader = DocumentLoader()
        docs = loader.load(sample_md)
        assert len(docs) >= 1
        assert "markdown content" in docs[0].content

    def test_load_csv(self, sample_csv: Path):
        from backend.src.ingestion.loader import DocumentLoader

        loader = DocumentLoader()
        docs = loader.load(sample_csv)
        assert len(docs) >= 1
        assert "Alice" in docs[0].content

    def test_load_unsupported_format(self, tmp_path: Path):
        from backend.src.ingestion.loader import DocumentLoader

        f = tmp_path / "file.xyz"
        f.write_text("content")
        loader = DocumentLoader()
        with pytest.raises(ValueError, match="Unsupported"):
            loader.load(f)

    def test_load_directory(self, tmp_path: Path):
        from backend.src.ingestion.loader import DocumentLoader

        (tmp_path / "a.txt").write_text("Doc A")
        (tmp_path / "b.txt").write_text("Doc B")
        loader = DocumentLoader()
        docs = loader.load_directory(tmp_path)
        assert len(docs) == 2
```

**Step 3: Run test to verify it fails**

```bash
python -m pytest backend/tests/test_ingestion.py -v
```

Expected: FAIL — `ModuleNotFoundError`

**Step 4: Write the implementation**

```python
# backend/src/ingestion/loader.py
from __future__ import annotations

import csv
import io
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".txt", ".md", ".markdown", ".csv", ".pdf", ".docx"}


@dataclass
class Document:
    """A loaded document with content and metadata."""

    content: str
    metadata: dict[str, str | int] = field(default_factory=dict)


class DocumentLoader:
    """Load documents from various file formats."""

    def load(self, path: Path) -> list[Document]:
        """Load a single file and return a list of Documents."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        ext = path.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file format: {ext}")

        loader = {
            ".txt": self._load_text,
            ".md": self._load_text,
            ".markdown": self._load_text,
            ".csv": self._load_csv,
            ".pdf": self._load_pdf,
            ".docx": self._load_docx,
        }[ext]

        docs = loader(path)
        for doc in docs:
            doc.metadata.setdefault("source", str(path))
            doc.metadata.setdefault("format", ext.lstrip("."))
        return docs

    def load_directory(self, directory: Path) -> list[Document]:
        """Load all supported files from a directory."""
        directory = Path(directory)
        docs: list[Document] = []
        for path in sorted(directory.iterdir()):
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
                try:
                    docs.extend(self.load(path))
                except Exception:
                    logger.warning("Failed to load %s", path, exc_info=True)
        return docs

    def _load_text(self, path: Path) -> list[Document]:
        content = path.read_text(encoding="utf-8")
        return [Document(content=content)]

    def _load_csv(self, path: Path) -> list[Document]:
        content = path.read_text(encoding="utf-8")
        reader = csv.DictReader(io.StringIO(content))
        rows = [
            " | ".join(f"{k}: {v}" for k, v in row.items())
            for row in reader
        ]
        combined = "\n".join(rows)
        return [Document(content=combined)]

    def _load_pdf(self, path: Path) -> list[Document]:
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        docs: list[Document] = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                docs.append(Document(
                    content=text,
                    metadata={"page": i + 1},
                ))
        return docs

    def _load_docx(self, path: Path) -> list[Document]:
        from docx import Document as DocxDocument

        doc = DocxDocument(str(path))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        content = "\n\n".join(paragraphs)
        return [Document(content=content)]
```

**Step 5: Run test to verify it passes**

```bash
python -m pytest backend/tests/test_ingestion.py -v
```

Expected: 5 passed

**Step 6: Commit**

```bash
git add backend/src/ingestion/loader.py backend/tests/test_ingestion.py data/
git commit -m "feat: add multi-format document loader"
```

---

### Task 5: Text chunker (recursive character splitting)

**Files:**
- Create: `backend/src/ingestion/chunker.py`
- Modify: `backend/tests/test_ingestion.py` (add chunker tests)

**Step 1: Write the failing test**

Add to `backend/tests/test_ingestion.py`:

```python
class TestTextChunker:
    def test_chunk_short_text(self):
        from backend.src.ingestion.chunker import TextChunker

        chunker = TextChunker(chunk_size=100, chunk_overlap=20)
        chunks = chunker.chunk("Short text.")
        assert len(chunks) == 1
        assert chunks[0].content == "Short text."

    def test_chunk_long_text_produces_overlapping_chunks(self):
        from backend.src.ingestion.chunker import TextChunker

        text = "A" * 50 + " " + "B" * 50 + " " + "C" * 50
        chunker = TextChunker(chunk_size=60, chunk_overlap=10)
        chunks = chunker.chunk(text)
        assert len(chunks) > 1
        # Verify overlap: end of chunk N overlaps with start of chunk N+1
        for i in range(len(chunks) - 1):
            tail = chunks[i].content[-10:]
            assert tail in chunks[i + 1].content or len(chunks[i].content) <= 60

    def test_chunk_preserves_metadata(self):
        from backend.src.ingestion.chunker import TextChunker

        chunker = TextChunker(chunk_size=20, chunk_overlap=5)
        chunks = chunker.chunk(
            "Word " * 20,
            metadata={"source": "test.txt"},
        )
        assert all(c.metadata["source"] == "test.txt" for c in chunks)
        assert all("chunk_index" in c.metadata for c in chunks)

    def test_chunk_splits_on_separators(self):
        from backend.src.ingestion.chunker import TextChunker

        text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
        chunker = TextChunker(chunk_size=30, chunk_overlap=5)
        chunks = chunker.chunk(text)
        # Should prefer splitting on double newlines
        assert any("Paragraph one." in c.content for c in chunks)
```

**Step 2: Run test to verify it fails**

```bash
python -m pytest backend/tests/test_ingestion.py::TestTextChunker -v
```

Expected: FAIL

**Step 3: Write the implementation**

```python
# backend/src/ingestion/chunker.py
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Chunk:
    """A text chunk with metadata."""

    content: str
    metadata: dict[str, str | int] = field(default_factory=dict)


class TextChunker:
    """Recursive character text splitter with configurable overlap."""

    SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(
        self,
        text: str,
        metadata: dict[str, str | int] | None = None,
    ) -> list[Chunk]:
        """Split text into overlapping chunks using recursive separators."""
        base_metadata = metadata or {}
        raw_chunks = self._split_recursive(text, self.SEPARATORS)
        merged = self._merge_chunks(raw_chunks)

        return [
            Chunk(
                content=c,
                metadata={**base_metadata, "chunk_index": i},
            )
            for i, c in enumerate(merged)
        ]

    def _split_recursive(self, text: str, separators: list[str]) -> list[str]:
        """Recursively split text using the first separator that works."""
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []

        if not separators:
            # Hard split at chunk_size as last resort
            return [
                text[i : i + self.chunk_size]
                for i in range(0, len(text), self.chunk_size)
            ]

        sep = separators[0]
        remaining_seps = separators[1:]

        if sep == "":
            return self._split_recursive(text, [])

        parts = text.split(sep)
        results: list[str] = []
        current = ""

        for part in parts:
            candidate = current + sep + part if current else part
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    results.append(current)
                if len(part) > self.chunk_size:
                    results.extend(self._split_recursive(part, remaining_seps))
                    current = ""
                else:
                    current = part

        if current.strip():
            results.append(current)

        return results

    def _merge_chunks(self, chunks: list[str]) -> list[str]:
        """Add overlap between consecutive chunks."""
        if len(chunks) <= 1:
            return chunks

        result: list[str] = [chunks[0]]
        for i in range(1, len(chunks)):
            prev = chunks[i - 1]
            overlap_text = prev[-self.chunk_overlap :] if len(prev) > self.chunk_overlap else prev
            # Find a clean word boundary in the overlap
            space_idx = overlap_text.find(" ")
            if space_idx > 0:
                overlap_text = overlap_text[space_idx + 1 :]
            merged = overlap_text + " " + chunks[i] if overlap_text else chunks[i]
            result.append(merged.strip())

        return result
```

**Step 4: Run test to verify it passes**

```bash
python -m pytest backend/tests/test_ingestion.py::TestTextChunker -v
```

Expected: 4 passed

**Step 5: Commit**

```bash
git add backend/src/ingestion/chunker.py backend/tests/test_ingestion.py
git commit -m "feat: add recursive text chunker with overlap"
```

---

### Task 6: Embedder (sentence-transformers wrapper)

**Files:**
- Create: `backend/src/embeddings/embedder.py`
- Create: `backend/tests/test_embeddings.py`

**Step 1: Write the failing test**

```python
# backend/tests/test_embeddings.py
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
        # Sparse embedding should have indices and values
        assert "indices" in sparse[0]
        assert "values" in sparse[0]
        assert len(sparse[0]["indices"]) == len(sparse[0]["values"])
        assert len(sparse[0]["indices"]) > 0
```

**Step 2: Run test to verify it fails**

```bash
python -m pytest backend/tests/test_embeddings.py -v
```

Expected: FAIL

**Step 3: Write the implementation**

```python
# backend/src/embeddings/embedder.py
from __future__ import annotations

import logging
from collections import Counter

import numpy as np

logger = logging.getLogger(__name__)


class Embedder:
    """Wrapper around sentence-transformers for dense and sparse embeddings."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        logger.info("Loaded embedding model %s (dim=%d)", model_name, self.dimension)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate dense embeddings for a list of texts. Returns normalized vectors."""
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()

    def sparse_embed(self, texts: list[str]) -> list[dict]:
        """Generate sparse BM25-style embeddings using term frequencies.

        Returns list of dicts with 'indices' and 'values' keys,
        compatible with Qdrant sparse vectors.
        """
        results: list[dict] = []
        for text in texts:
            tokens = self._tokenize(text)
            tf = Counter(tokens)
            indices = sorted(tf.keys())
            values = [float(tf[i]) for i in indices]
            results.append({"indices": indices, "values": values})
        return results

    def _tokenize(self, text: str) -> list[int]:
        """Tokenize text into integer token IDs using the model's tokenizer."""
        encoded = self.model.tokenizer(
            text,
            add_special_tokens=False,
            return_attention_mask=False,
        )
        return encoded["input_ids"]
```

**Step 4: Run test to verify it passes**

```bash
python -m pytest backend/tests/test_embeddings.py -v
```

Expected: 4 passed

**Step 5: Commit**

```bash
git add backend/src/embeddings/embedder.py backend/tests/test_embeddings.py
git commit -m "feat: add embedder with dense and sparse embedding support"
```

---

### Task 7: Vector store abstract base class

**Files:**
- Create: `backend/src/vectorstore/base.py`

**Step 1: Write the abstract base class**

```python
# backend/src/vectorstore/base.py
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class SearchResult:
    """A single search result from the vector store."""

    content: str
    score: float
    metadata: dict[str, str | int | float] = field(default_factory=dict)


class VectorStoreBase(ABC):
    """Abstract interface for vector store backends."""

    @abstractmethod
    def initialize(self, dimension: int) -> None:
        """Create the collection/index if it doesn't exist."""

    @abstractmethod
    def upsert(
        self,
        ids: list[str],
        contents: list[str],
        dense_vectors: list[list[float]],
        sparse_vectors: list[dict] | None = None,
        metadatas: list[dict] | None = None,
    ) -> None:
        """Insert or update documents with their vectors."""

    @abstractmethod
    def search(
        self,
        dense_vector: list[float],
        sparse_vector: dict | None = None,
        top_k: int = 5,
        score_threshold: float = 0.0,
    ) -> list[SearchResult]:
        """Search for similar documents using dense and optionally sparse vectors."""

    @abstractmethod
    def delete(self, ids: list[str]) -> None:
        """Delete documents by their IDs."""

    @abstractmethod
    def count(self) -> int:
        """Return the number of documents in the store."""

    @abstractmethod
    def list_documents(self) -> list[dict]:
        """Return metadata for all unique source documents."""
```

**Step 2: Commit**

```bash
git add backend/src/vectorstore/base.py
git commit -m "feat: add abstract base class for vector store"
```

---

### Task 8: Qdrant vector store implementation

**Files:**
- Create: `backend/src/vectorstore/qdrant_store.py`
- Create: `backend/tests/test_vectorstore.py`

**Step 1: Write the failing test**

```python
# backend/tests/test_vectorstore.py
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
```

**Step 2: Run test to verify it fails**

```bash
python -m pytest backend/tests/test_vectorstore.py -v
```

Expected: FAIL

**Step 3: Write the implementation**

```python
# backend/src/vectorstore/qdrant_store.py
from __future__ import annotations

import logging
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    NamedSparseVector,
    NamedVector,
    PointStruct,
    SparseIndexParams,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from backend.src.vectorstore.base import SearchResult, VectorStoreBase

logger = logging.getLogger(__name__)


class QdrantStore(VectorStoreBase):
    """Qdrant vector store with dense and sparse vector support."""

    def __init__(
        self,
        host: str | None = "localhost",
        port: int | None = 6333,
        collection_name: str = "intellidocs",
    ) -> None:
        if host is None:
            self.client = QdrantClient(location=":memory:")
        else:
            self.client = QdrantClient(host=host, port=port)
        self.collection_name = collection_name

    def initialize(self, dimension: int) -> None:
        """Create collection with dense + sparse vector config."""
        collections = [c.name for c in self.client.get_collections().collections]
        if self.collection_name in collections:
            logger.info("Collection '%s' already exists", self.collection_name)
            return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config={
                "dense": VectorParams(size=dimension, distance=Distance.COSINE),
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(index=SparseIndexParams()),
            },
        )
        logger.info("Created collection '%s' (dim=%d)", self.collection_name, dimension)

    def upsert(
        self,
        ids: list[str],
        contents: list[str],
        dense_vectors: list[list[float]],
        sparse_vectors: list[dict] | None = None,
        metadatas: list[dict] | None = None,
    ) -> None:
        points: list[PointStruct] = []
        for i, doc_id in enumerate(ids):
            payload = {"content": contents[i]}
            if metadatas and i < len(metadatas):
                payload.update(metadatas[i])

            vectors: dict = {"dense": dense_vectors[i]}
            if sparse_vectors and i < len(sparse_vectors):
                sv = sparse_vectors[i]
                vectors["sparse"] = SparseVector(
                    indices=sv["indices"],
                    values=sv["values"],
                )

            # Convert string ID to a deterministic UUID
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, doc_id))
            points.append(PointStruct(id=point_id, vector=vectors, payload=payload))

        self.client.upsert(collection_name=self.collection_name, points=points)

    def search(
        self,
        dense_vector: list[float],
        sparse_vector: dict | None = None,
        top_k: int = 5,
        score_threshold: float = 0.0,
    ) -> list[SearchResult]:
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=dense_vector,
            using="dense",
            limit=top_k,
            score_threshold=score_threshold,
        )

        return [
            SearchResult(
                content=point.payload.get("content", ""),
                score=point.score,
                metadata={
                    k: v for k, v in point.payload.items() if k != "content"
                },
            )
            for point in results.points
        ]

    def delete(self, ids: list[str]) -> None:
        point_ids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, doc_id)) for doc_id in ids]
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=point_ids,
        )

    def count(self) -> int:
        info = self.client.get_collection(self.collection_name)
        return info.points_count

    def list_documents(self) -> list[dict]:
        """Scroll all points and return unique source documents."""
        records, _ = self.client.scroll(
            collection_name=self.collection_name,
            limit=10000,
            with_payload=True,
        )
        seen: dict[str, dict] = {}
        for r in records:
            source = r.payload.get("source", "unknown")
            if source not in seen:
                seen[source] = {
                    k: v for k, v in r.payload.items() if k != "content"
                }
        return list(seen.values())
```

**Step 4: Run test to verify it passes**

```bash
python -m pytest backend/tests/test_vectorstore.py -v
```

Expected: 4 passed

**Step 5: Commit**

```bash
git add backend/src/vectorstore/qdrant_store.py backend/tests/test_vectorstore.py
git commit -m "feat: add Qdrant vector store with hybrid dense+sparse support"
```

---

### Task 9: ChromaDB fallback implementation

**Files:**
- Create: `backend/src/vectorstore/chroma_store.py`
- Modify: `backend/tests/test_vectorstore.py` (add ChromaDB tests)

**Step 1: Write the failing test**

Add to `backend/tests/test_vectorstore.py`:

```python
@pytest.fixture
def chroma_store():
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
```

**Step 2: Run test to verify it fails**

```bash
python -m pytest backend/tests/test_vectorstore.py::TestChromaStore -v
```

**Step 3: Write the implementation**

```python
# backend/src/vectorstore/chroma_store.py
from __future__ import annotations

import logging

import chromadb

from backend.src.vectorstore.base import SearchResult, VectorStoreBase

logger = logging.getLogger(__name__)


class ChromaStore(VectorStoreBase):
    """ChromaDB in-memory vector store (fallback, dense-only)."""

    def __init__(self, collection_name: str = "intellidocs") -> None:
        self.client = chromadb.Client()
        self.collection_name = collection_name
        self._collection = None

    def initialize(self, dimension: int) -> None:
        self._collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("ChromaDB collection '%s' ready", self.collection_name)

    @property
    def collection(self):
        if self._collection is None:
            raise RuntimeError("Call initialize() before using the store")
        return self._collection

    def upsert(
        self,
        ids: list[str],
        contents: list[str],
        dense_vectors: list[list[float]],
        sparse_vectors: list[dict] | None = None,
        metadatas: list[dict] | None = None,
    ) -> None:
        # ChromaDB does not support sparse vectors — ignored
        self.collection.upsert(
            ids=ids,
            embeddings=dense_vectors,
            documents=contents,
            metadatas=metadatas,
        )

    def search(
        self,
        dense_vector: list[float],
        sparse_vector: dict | None = None,
        top_k: int = 5,
        score_threshold: float = 0.0,
    ) -> list[SearchResult]:
        results = self.collection.query(
            query_embeddings=[dense_vector],
            n_results=top_k,
        )
        search_results: list[SearchResult] = []
        if results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                distance = results["distances"][0][i] if results["distances"] else 0.0
                score = 1.0 - distance  # ChromaDB returns distance, convert to similarity
                if score >= score_threshold:
                    metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                    search_results.append(
                        SearchResult(content=doc, score=score, metadata=metadata)
                    )
        return search_results

    def delete(self, ids: list[str]) -> None:
        self.collection.delete(ids=ids)

    def count(self) -> int:
        return self.collection.count()

    def list_documents(self) -> list[dict]:
        all_data = self.collection.get(include=["metadatas"])
        seen: dict[str, dict] = {}
        for meta in all_data["metadatas"] or []:
            source = meta.get("source", "unknown")
            if source not in seen:
                seen[source] = meta
        return list(seen.values())
```

**Step 4: Run test to verify it passes**

```bash
python -m pytest backend/tests/test_vectorstore.py -v
```

Expected: 7 passed (4 Qdrant + 3 ChromaDB)

**Step 5: Commit**

```bash
git add backend/src/vectorstore/chroma_store.py backend/tests/test_vectorstore.py
git commit -m "feat: add ChromaDB fallback vector store"
```

---

### Task 10: Ingestion processor (pipeline orchestrator)

**Files:**
- Create: `backend/src/ingestion/processor.py`
- Modify: `backend/tests/test_ingestion.py` (add processor tests)

**Step 1: Write the failing test**

Add to `backend/tests/test_ingestion.py`:

```python
class TestIngestionProcessor:
    def test_process_single_file(self, sample_txt: Path):
        from backend.src.embeddings.embedder import Embedder
        from backend.src.ingestion.processor import IngestionProcessor
        from backend.src.vectorstore.qdrant_store import QdrantStore

        store = QdrantStore(host=None, port=None, collection_name="test_ingest")
        embedder = Embedder(model_name="all-MiniLM-L6-v2")
        store.initialize(dimension=embedder.dimension)

        processor = IngestionProcessor(
            embedder=embedder,
            vector_store=store,
            chunk_size=100,
            chunk_overlap=20,
        )
        doc_id = processor.process_file(sample_txt)

        assert doc_id is not None
        assert store.count() > 0

    def test_process_directory(self, tmp_path: Path):
        from backend.src.embeddings.embedder import Embedder
        from backend.src.ingestion.processor import IngestionProcessor
        from backend.src.vectorstore.qdrant_store import QdrantStore

        (tmp_path / "a.txt").write_text("Document A content here, some text to process.")
        (tmp_path / "b.txt").write_text("Document B content here, different text.")

        store = QdrantStore(host=None, port=None, collection_name="test_ingest_dir")
        embedder = Embedder(model_name="all-MiniLM-L6-v2")
        store.initialize(dimension=embedder.dimension)

        processor = IngestionProcessor(
            embedder=embedder,
            vector_store=store,
            chunk_size=100,
            chunk_overlap=20,
        )
        count = processor.process_directory(tmp_path)

        assert count == 2
        assert store.count() > 0
```

**Step 2: Run test to verify it fails**

```bash
python -m pytest backend/tests/test_ingestion.py::TestIngestionProcessor -v
```

**Step 3: Write the implementation**

```python
# backend/src/ingestion/processor.py
from __future__ import annotations

import hashlib
import logging
from pathlib import Path

from backend.src.embeddings.embedder import Embedder
from backend.src.ingestion.chunker import TextChunker
from backend.src.ingestion.loader import DocumentLoader
from backend.src.vectorstore.base import VectorStoreBase

logger = logging.getLogger(__name__)


class IngestionProcessor:
    """Orchestrates the document ingestion pipeline: load → chunk → embed → store."""

    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStoreBase,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
    ) -> None:
        self.loader = DocumentLoader()
        self.chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.embedder = embedder
        self.vector_store = vector_store

    def process_file(self, path: Path) -> str:
        """Ingest a single file. Returns a document ID."""
        path = Path(path)
        doc_id = hashlib.sha256(str(path).encode()).hexdigest()[:16]

        logger.info("Loading %s", path)
        documents = self.loader.load(path)

        all_chunks = []
        for doc in documents:
            chunks = self.chunker.chunk(doc.content, metadata=doc.metadata)
            all_chunks.extend(chunks)

        if not all_chunks:
            logger.warning("No chunks produced from %s", path)
            return doc_id

        logger.info("Embedding %d chunks from %s", len(all_chunks), path)
        texts = [c.content for c in all_chunks]
        dense_vectors = self.embedder.embed(texts)
        sparse_vectors = self.embedder.sparse_embed(texts)

        ids = [f"{doc_id}_{i}" for i in range(len(all_chunks))]
        metadatas = [
            {**c.metadata, "document_id": doc_id}
            for c in all_chunks
        ]

        self.vector_store.upsert(
            ids=ids,
            contents=texts,
            dense_vectors=dense_vectors,
            sparse_vectors=sparse_vectors,
            metadatas=metadatas,
        )
        logger.info("Stored %d chunks for %s", len(all_chunks), path)
        return doc_id

    def process_directory(self, directory: Path) -> int:
        """Ingest all supported files in a directory. Returns count of files processed."""
        directory = Path(directory)
        count = 0
        for path in sorted(directory.iterdir()):
            if path.is_file() and path.suffix.lower() in {".txt", ".md", ".csv", ".pdf", ".docx"}:
                try:
                    self.process_file(path)
                    count += 1
                except Exception:
                    logger.error("Failed to process %s", path, exc_info=True)
        return count
```

**Step 4: Run test to verify it passes**

```bash
python -m pytest backend/tests/test_ingestion.py::TestIngestionProcessor -v
```

Expected: 2 passed

**Step 5: Commit**

```bash
git add backend/src/ingestion/processor.py backend/tests/test_ingestion.py
git commit -m "feat: add ingestion processor pipeline orchestrator"
```

---

## Phase 3: Backend — Retrieval & Generation

### Task 11: Re-ranker (cross-encoder)

**Files:**
- Create: `backend/src/retrieval/reranker.py`
- Create: `backend/tests/test_retrieval.py`

**Step 1: Write the failing test**

```python
# backend/tests/test_retrieval.py
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
        # The Python-related results should rank higher for this query
        assert "Python" in reranked[0].content

    def test_rerank_preserves_metadata(self):
        from backend.src.retrieval.reranker import Reranker

        reranker = Reranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
        results = [
            SearchResult(content="Test content", score=0.5, metadata={"source": "test.txt"}),
        ]
        reranked = reranker.rerank(query="test", results=results, top_k=1)
        assert reranked[0].metadata["source"] == "test.txt"
```

**Step 2: Run test to verify it fails**

```bash
python -m pytest backend/tests/test_retrieval.py::TestReranker -v
```

**Step 3: Write the implementation**

```python
# backend/src/retrieval/reranker.py
from __future__ import annotations

import logging

from backend.src.vectorstore.base import SearchResult

logger = logging.getLogger(__name__)


class Reranker:
    """Cross-encoder re-ranker for improving retrieval precision."""

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
        """Re-rank search results using cross-encoder scores."""
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
```

**Step 4: Run test to verify it passes**

```bash
python -m pytest backend/tests/test_retrieval.py::TestReranker -v
```

Expected: 2 passed

**Step 5: Commit**

```bash
git add backend/src/retrieval/reranker.py backend/tests/test_retrieval.py
git commit -m "feat: add cross-encoder re-ranker"
```

---

### Task 12: Hybrid retriever

**Files:**
- Create: `backend/src/retrieval/retriever.py`
- Modify: `backend/tests/test_retrieval.py`

**Step 1: Write the failing test**

Add to `backend/tests/test_retrieval.py`:

```python
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

        # Insert test documents
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
```

**Step 2: Run test to verify it fails**

```bash
python -m pytest backend/tests/test_retrieval.py::TestHybridRetriever -v
```

**Step 3: Write the implementation**

```python
# backend/src/retrieval/retriever.py
from __future__ import annotations

import logging

from backend.src.embeddings.embedder import Embedder
from backend.src.retrieval.reranker import Reranker
from backend.src.vectorstore.base import SearchResult, VectorStoreBase

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Hybrid retrieval: dense + sparse search, then cross-encoder re-ranking."""

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
        """Retrieve relevant documents using hybrid search + re-ranking."""
        dense_vector = self.embedder.embed([query])[0]
        sparse_vector = self.embedder.sparse_embed([query])[0]

        # Fetch more candidates than needed so re-ranking has room to work
        fetch_k = top_k * self.initial_fetch_multiplier
        results = self.vector_store.search(
            dense_vector=dense_vector,
            sparse_vector=sparse_vector,
            top_k=fetch_k,
            score_threshold=score_threshold,
        )

        if not results:
            return []

        if self.reranker:
            results = self.reranker.rerank(query=query, results=results, top_k=top_k)
        else:
            results = results[:top_k]

        logger.info("Retrieved %d results for query: %s", len(results), query[:80])
        return results
```

**Step 4: Run test to verify it passes**

```bash
python -m pytest backend/tests/test_retrieval.py -v
```

Expected: 4 passed

**Step 5: Commit**

```bash
git add backend/src/retrieval/retriever.py backend/tests/test_retrieval.py
git commit -m "feat: add hybrid retriever with dense+sparse search and re-ranking"
```

---

### Task 13: OpenRouter LLM client

**Files:**
- Create: `backend/src/generation/llm_client.py`
- Create: `backend/tests/test_generation.py`

**Step 1: Write the failing test**

```python
# backend/tests/test_generation.py
from unittest.mock import AsyncMock, patch

import pytest


class TestOpenRouterClient:
    def test_client_initialization(self):
        from backend.src.generation.llm_client import OpenRouterClient

        client = OpenRouterClient(api_key="test-key", model="test/model")
        assert client.api_key == "test-key"
        assert client.model == "test/model"

    @pytest.mark.asyncio
    async def test_generate_stream(self):
        from backend.src.generation.llm_client import OpenRouterClient

        client = OpenRouterClient(api_key="test-key", model="test/model")

        # Mock the httpx async client to return a fake SSE stream
        mock_response_lines = [
            b'data: {"choices":[{"delta":{"content":"Hello"}}]}',
            b'data: {"choices":[{"delta":{"content":" world"}}]}',
            b"data: [DONE]",
        ]

        async def mock_aiter_lines():
            for line in mock_response_lines:
                yield line

        mock_response = AsyncMock()
        mock_response.aiter_lines = mock_aiter_lines
        mock_response.status_code = 200
        mock_response.raise_for_status = lambda: None

        with patch.object(client, "_stream_request", return_value=mock_response):
            tokens = []
            async for token in client.generate_stream(
                messages=[{"role": "user", "content": "Hi"}]
            ):
                tokens.append(token)

            assert "Hello" in tokens
            assert " world" in tokens
```

**Step 2: Run test to verify it fails**

```bash
python -m pytest backend/tests/test_generation.py -v
```

**Step 3: Write the implementation**

```python
# backend/src/generation/llm_client.py
from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx

logger = logging.getLogger(__name__)

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"


class OpenRouterClient:
    """Async client for OpenRouter's chat completion API with streaming."""

    def __init__(
        self,
        api_key: str,
        model: str = "anthropic/claude-haiku-4-5-20251001",
    ) -> None:
        self.api_key = api_key
        self.model = model

    async def generate_stream(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> AsyncIterator[str]:
        """Stream tokens from OpenRouter. Yields individual token strings."""
        async with self._stream_request(messages, temperature, max_tokens) as response:
            async for line in response.aiter_lines():
                line = line.strip() if isinstance(line, str) else line.decode().strip()
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    payload = json.loads(data)
                    delta = payload.get("choices", [{}])[0].get("delta", {})
                    content = delta.get("content")
                    if content:
                        yield content
                except (json.JSONDecodeError, IndexError, KeyError):
                    logger.debug("Skipping malformed SSE chunk: %s", data[:100])

    async def generate(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        """Non-streaming generation. Returns the full response text."""
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                OPENROUTER_API_URL,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "stream": False,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]

    @asynccontextmanager
    async def _stream_request(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ):
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream(
                "POST",
                OPENROUTER_API_URL,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "stream": True,
                },
            ) as response:
                response.raise_for_status()
                yield response
```

**Step 4: Run test to verify it passes**

```bash
python -m pytest backend/tests/test_generation.py -v
```

Expected: 2 passed

**Step 5: Commit**

```bash
git add backend/src/generation/llm_client.py backend/tests/test_generation.py
git commit -m "feat: add OpenRouter LLM client with streaming support"
```

---

### Task 14: Prompt templates

**Files:**
- Create: `backend/src/generation/prompts.py`

**Step 1: Write the implementation**

```python
# backend/src/generation/prompts.py
from __future__ import annotations

from backend.src.vectorstore.base import SearchResult

SYSTEM_PROMPT = """You are IntelliDocs, an intelligent document assistant. You answer questions based on the provided context documents.

Rules:
- Answer ONLY based on the provided context. If the context doesn't contain enough information, say so clearly.
- Cite your sources by referencing the document name and page/section when available.
- Be concise and direct. Avoid unnecessary preamble.
- If asked about something not in the context, say "I don't have information about that in the provided documents."
"""


def build_rag_prompt(
    query: str,
    sources: list[SearchResult],
    conversation_history: list[dict[str, str]] | None = None,
) -> list[dict[str, str]]:
    """Build the full message list for the LLM, including system prompt, context, and history."""
    context_parts: list[str] = []
    for i, source in enumerate(sources, 1):
        src_label = source.metadata.get("source", "unknown")
        page = source.metadata.get("page", "")
        page_str = f" (page {page})" if page else ""
        context_parts.append(f"[{i}] {src_label}{page_str}:\n{source.content}")

    context_block = "\n\n---\n\n".join(context_parts)

    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]

    if conversation_history:
        messages.extend(conversation_history)

    user_message = f"""Context documents:

{context_block}

---

Question: {query}"""

    messages.append({"role": "user", "content": user_message})
    return messages
```

**Step 2: Commit**

```bash
git add backend/src/generation/prompts.py
git commit -m "feat: add prompt templates for RAG chain"
```

---

### Task 15: RAG chain orchestrator

**Files:**
- Create: `backend/src/generation/rag_chain.py`

**Step 1: Write the implementation**

```python
# backend/src/generation/rag_chain.py
from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass

from backend.src.generation.llm_client import OpenRouterClient
from backend.src.generation.prompts import build_rag_prompt
from backend.src.retrieval.retriever import HybridRetriever
from backend.src.vectorstore.base import SearchResult

logger = logging.getLogger(__name__)


@dataclass
class RAGResponse:
    """Container for a complete RAG response."""

    answer: str
    sources: list[SearchResult]


class RAGChain:
    """Orchestrates retrieval + generation for RAG queries."""

    def __init__(
        self,
        retriever: HybridRetriever,
        llm_client: OpenRouterClient,
    ) -> None:
        self.retriever = retriever
        self.llm_client = llm_client

    def retrieve_sources(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.0,
    ) -> list[SearchResult]:
        """Retrieve relevant sources for a query."""
        return self.retriever.retrieve(
            query=query,
            top_k=top_k,
            score_threshold=score_threshold,
        )

    async def generate_stream(
        self,
        query: str,
        sources: list[SearchResult],
        conversation_history: list[dict[str, str]] | None = None,
    ) -> AsyncIterator[str]:
        """Stream the LLM response given a query and pre-retrieved sources."""
        messages = build_rag_prompt(
            query=query,
            sources=sources,
            conversation_history=conversation_history,
        )
        async for token in self.llm_client.generate_stream(messages):
            yield token

    async def query(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.0,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> RAGResponse:
        """Full RAG pipeline: retrieve + generate (non-streaming)."""
        sources = self.retrieve_sources(query, top_k, score_threshold)
        messages = build_rag_prompt(query, sources, conversation_history)
        answer = await self.llm_client.generate(messages)
        return RAGResponse(answer=answer, sources=sources)
```

**Step 2: Commit**

```bash
git add backend/src/generation/rag_chain.py
git commit -m "feat: add RAG chain orchestrator"
```

---

## Phase 4: API Layer

### Task 16: FastAPI dependency injection

**Files:**
- Create: `backend/src/api/deps.py`

**Step 1: Write the implementation**

```python
# backend/src/api/deps.py
from __future__ import annotations

import logging
from functools import lru_cache

from backend.src.config import Settings
from backend.src.embeddings.embedder import Embedder
from backend.src.generation.llm_client import OpenRouterClient
from backend.src.generation.rag_chain import RAGChain
from backend.src.ingestion.processor import IngestionProcessor
from backend.src.retrieval.reranker import Reranker
from backend.src.retrieval.retriever import HybridRetriever
from backend.src.vectorstore.base import VectorStoreBase

logger = logging.getLogger(__name__)


@lru_cache
def get_settings() -> Settings:
    return Settings()


@lru_cache
def get_embedder() -> Embedder:
    settings = get_settings()
    return Embedder(model_name=settings.EMBEDDING_MODEL)


@lru_cache
def get_vector_store() -> VectorStoreBase:
    settings = get_settings()
    embedder = get_embedder()

    if settings.VECTOR_STORE_BACKEND == "chroma":
        from backend.src.vectorstore.chroma_store import ChromaStore

        store = ChromaStore(collection_name=settings.QDRANT_COLLECTION)
    else:
        from backend.src.vectorstore.qdrant_store import QdrantStore

        store = QdrantStore(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT,
            collection_name=settings.QDRANT_COLLECTION,
        )

    store.initialize(dimension=embedder.dimension)
    return store


@lru_cache
def get_reranker() -> Reranker:
    settings = get_settings()
    return Reranker(model_name=settings.RERANKER_MODEL)


@lru_cache
def get_retriever() -> HybridRetriever:
    return HybridRetriever(
        embedder=get_embedder(),
        vector_store=get_vector_store(),
        reranker=get_reranker(),
    )


@lru_cache
def get_llm_client() -> OpenRouterClient:
    settings = get_settings()
    return OpenRouterClient(
        api_key=settings.OPENROUTER_API_KEY,
        model=settings.OPENROUTER_MODEL,
    )


@lru_cache
def get_rag_chain() -> RAGChain:
    return RAGChain(
        retriever=get_retriever(),
        llm_client=get_llm_client(),
    )


@lru_cache
def get_ingestion_processor() -> IngestionProcessor:
    settings = get_settings()
    return IngestionProcessor(
        embedder=get_embedder(),
        vector_store=get_vector_store(),
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
    )
```

**Step 2: Commit**

```bash
git add backend/src/api/deps.py
git commit -m "feat: add FastAPI dependency injection wiring"
```

---

### Task 17: API routes — health check

**Files:**
- Create: `backend/src/api/routes/health.py`

**Step 1: Write the implementation**

```python
# backend/src/api/routes/health.py
from __future__ import annotations

from fastapi import APIRouter, Depends

from backend.src.api.deps import get_embedder, get_vector_store
from backend.src.embeddings.embedder import Embedder
from backend.src.vectorstore.base import VectorStoreBase

router = APIRouter(tags=["health"])


@router.get("/api/health")
def health_check(
    vector_store: VectorStoreBase = Depends(get_vector_store),
    embedder: Embedder = Depends(get_embedder),
) -> dict:
    return {
        "status": "healthy",
        "vector_store": {
            "document_count": vector_store.count(),
        },
        "embedding_model": embedder.model_name,
        "embedding_dimension": embedder.dimension,
    }
```

**Step 2: Commit**

```bash
git add backend/src/api/routes/health.py
git commit -m "feat: add health check endpoint"
```

---

### Task 18: API routes — documents (upload, list, delete)

**Files:**
- Create: `backend/src/api/routes/documents.py`

**Step 1: Write the implementation**

```python
# backend/src/api/routes/documents.py
from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, UploadFile

from backend.src.api.deps import get_ingestion_processor, get_vector_store
from backend.src.ingestion.processor import IngestionProcessor
from backend.src.vectorstore.base import VectorStoreBase

router = APIRouter(prefix="/api/documents", tags=["documents"])

SUPPORTED_EXTENSIONS = {".txt", ".md", ".markdown", ".csv", ".pdf", ".docx"}


@router.post("/upload")
async def upload_document(
    file: UploadFile,
    background_tasks: BackgroundTasks,
    processor: IngestionProcessor = Depends(get_ingestion_processor),
) -> dict:
    """Upload and ingest a document."""
    filename = file.filename or "unnamed"
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format: {ext}. Supported: {SUPPORTED_EXTENSIONS}",
        )

    content = await file.read()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext, prefix="intellidocs_")
    tmp.write(content)
    tmp.flush()
    tmp_path = Path(tmp.name)
    tmp.close()

    # Run ingestion in background for large files
    doc_id = processor.process_file(tmp_path)

    # Clean up temp file after processing
    tmp_path.unlink(missing_ok=True)

    return {"document_id": doc_id, "filename": filename, "status": "ingested"}


@router.get("")
def list_documents(
    vector_store: VectorStoreBase = Depends(get_vector_store),
) -> dict:
    """List all ingested documents."""
    docs = vector_store.list_documents()
    return {"documents": docs, "count": len(docs)}


@router.delete("/{document_id}")
def delete_document(
    document_id: str,
    vector_store: VectorStoreBase = Depends(get_vector_store),
) -> dict:
    """Delete a document and all its chunks from the vector store."""
    vector_store.delete(ids=[document_id])
    return {"status": "deleted", "document_id": document_id}
```

**Step 2: Commit**

```bash
git add backend/src/api/routes/documents.py
git commit -m "feat: add document upload, list, and delete endpoints"
```

---

### Task 19: API routes — chat (SSE streaming)

**Files:**
- Create: `backend/src/api/routes/chat.py`

**Step 1: Write the implementation**

```python
# backend/src/api/routes/chat.py
from __future__ import annotations

import json
import logging

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from backend.src.api.deps import get_rag_chain, get_settings
from backend.src.config import Settings
from backend.src.generation.rag_chain import RAGChain

logger = logging.getLogger(__name__)
router = APIRouter(tags=["chat"])


class ChatRequest(BaseModel):
    query: str
    conversation_history: list[dict[str, str]] = Field(default_factory=list)
    top_k: int | None = None


@router.post("/api/chat")
async def chat(
    request: ChatRequest,
    rag_chain: RAGChain = Depends(get_rag_chain),
    settings: Settings = Depends(get_settings),
) -> EventSourceResponse:
    """Chat endpoint with SSE streaming. Sends sources first, then streams tokens."""
    top_k = request.top_k or settings.TOP_K

    async def event_generator():
        # Step 1: Retrieve sources
        sources = rag_chain.retrieve_sources(
            query=request.query,
            top_k=top_k,
            score_threshold=settings.SIMILARITY_THRESHOLD,
        )

        # Send sources event
        sources_data = [
            {
                "content": s.content,
                "score": round(s.score, 4),
                "metadata": s.metadata,
            }
            for s in sources
        ]
        yield {"event": "sources", "data": json.dumps({"sources": sources_data})}

        # Step 2: Stream generation
        async for token in rag_chain.generate_stream(
            query=request.query,
            sources=sources,
            conversation_history=request.conversation_history or None,
        ):
            yield {"event": "token", "data": json.dumps({"token": token})}

        yield {"event": "done", "data": "{}"}

    return EventSourceResponse(event_generator())
```

**Step 2: Commit**

```bash
git add backend/src/api/routes/chat.py
git commit -m "feat: add chat endpoint with SSE streaming"
```

---

### Task 20: FastAPI main app

**Files:**
- Create: `backend/src/main.py`

**Step 1: Write the implementation**

```python
# backend/src/main.py
from __future__ import annotations

import logging

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.src.api.routes import chat, documents, health

structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_log_level,
        structlog.dev.ConsoleRenderer(),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
)
logging.basicConfig(level=logging.INFO)


def create_app() -> FastAPI:
    app = FastAPI(
        title="IntelliDocs",
        description="Intelligent Document Understanding & Retrieval API",
        version="0.1.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router)
    app.include_router(documents.router)
    app.include_router(chat.router)

    return app


app = create_app()
```

**Step 2: Commit**

```bash
git add backend/src/main.py
git commit -m "feat: add FastAPI main app with CORS and route registration"
```

---

### Task 21: CLI for bulk ingestion

**Files:**
- Create: `backend/src/cli.py`

**Step 1: Write the implementation**

```python
# backend/src/cli.py
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def ingest(args: argparse.Namespace) -> None:
    """Ingest documents from a file or directory."""
    from backend.src.api.deps import get_ingestion_processor

    processor = get_ingestion_processor()
    path = Path(args.path)

    if path.is_file():
        doc_id = processor.process_file(path)
        logger.info("Ingested %s → %s", path, doc_id)
    elif path.is_dir():
        count = processor.process_directory(path)
        logger.info("Ingested %d files from %s", count, path)
    else:
        logger.error("Path not found: %s", path)
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="intellidocs",
        description="IntelliDocs CLI — document ingestion and management",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Ingest documents into the vector store")
    ingest_parser.add_argument(
        "path",
        type=str,
        help="Path to a file or directory to ingest",
    )
    ingest_parser.set_defaults(func=ingest)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
```

**Step 2: Commit**

```bash
git add backend/src/cli.py
git commit -m "feat: add CLI for bulk document ingestion"
```

---

## Phase 5: Frontend

### Task 22: Initialize Next.js project

**Step 1: Scaffold the project**

```bash
cd /mnt/d/Proyectos-IA/github_public_repos/intellidocs
npx create-next-app@latest frontend --typescript --tailwind --app --src-dir --no-eslint --use-npm
cd frontend
npm install framer-motion
```

**Step 2: Clean up generated boilerplate**

Remove default Next.js content from `src/app/page.tsx` and `src/app/globals.css`. Keep only the Tailwind imports in globals.css.

**Step 3: Commit**

```bash
cd /mnt/d/Proyectos-IA/github_public_repos/intellidocs
git add frontend/
git commit -m "chore: initialize Next.js frontend with Tailwind and Framer Motion"
```

---

### Task 23: API client and types

**Files:**
- Create: `frontend/src/lib/api.ts`
- Create: `frontend/src/lib/types.ts`

**Step 1: Write types**

```typescript
// frontend/src/lib/types.ts
export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
}

export interface Source {
  content: string;
  score: number;
  metadata: Record<string, string | number>;
}

export interface DocumentInfo {
  source: string;
  format?: string;
  document_id?: string;
}

export interface ChatRequest {
  query: string;
  conversation_history: { role: string; content: string }[];
  top_k?: number;
}
```

**Step 2: Write API client**

```typescript
// frontend/src/lib/api.ts
const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export async function* streamChat(
  query: string,
  conversationHistory: { role: string; content: string }[],
  topK?: number
): AsyncGenerator<{ event: string; data: any }> {
  const response = await fetch(`${API_BASE}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      query,
      conversation_history: conversationHistory,
      top_k: topK,
    }),
  });

  if (!response.ok) throw new Error(`Chat request failed: ${response.status}`);
  if (!response.body) throw new Error("No response body");

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";

    let currentEvent = "";
    for (const line of lines) {
      if (line.startsWith("event: ")) {
        currentEvent = line.slice(7).trim();
      } else if (line.startsWith("data: ")) {
        const data = line.slice(6).trim();
        try {
          yield { event: currentEvent, data: JSON.parse(data) };
        } catch {
          // skip malformed JSON
        }
      }
    }
  }
}

export async function uploadDocument(file: File): Promise<any> {
  const formData = new FormData();
  formData.append("file", file);
  const response = await fetch(`${API_BASE}/api/documents/upload`, {
    method: "POST",
    body: formData,
  });
  if (!response.ok) throw new Error(`Upload failed: ${response.status}`);
  return response.json();
}

export async function listDocuments(): Promise<any> {
  const response = await fetch(`${API_BASE}/api/documents`);
  if (!response.ok) throw new Error(`List failed: ${response.status}`);
  return response.json();
}

export async function deleteDocument(documentId: string): Promise<void> {
  const response = await fetch(`${API_BASE}/api/documents/${documentId}`, {
    method: "DELETE",
  });
  if (!response.ok) throw new Error(`Delete failed: ${response.status}`);
}

export async function healthCheck(): Promise<any> {
  const response = await fetch(`${API_BASE}/api/health`);
  if (!response.ok) throw new Error(`Health check failed: ${response.status}`);
  return response.json();
}
```

**Step 3: Commit**

```bash
git add frontend/src/lib/
git commit -m "feat: add API client and TypeScript types"
```

---

### Task 24: useChat hook

**Files:**
- Create: `frontend/src/hooks/useChat.ts`

**Step 1: Write the hook**

```typescript
// frontend/src/hooks/useChat.ts
"use client";

import { useCallback, useReducer } from "react";
import { streamChat } from "@/lib/api";
import { Message, Source } from "@/lib/types";

interface ChatState {
  messages: Message[];
  isStreaming: boolean;
  currentSources: Source[];
}

type ChatAction =
  | { type: "ADD_USER_MESSAGE"; content: string }
  | { type: "START_STREAMING" }
  | { type: "SET_SOURCES"; sources: Source[] }
  | { type: "APPEND_TOKEN"; token: string }
  | { type: "FINISH_STREAMING" }
  | { type: "CLEAR" };

function chatReducer(state: ChatState, action: ChatAction): ChatState {
  switch (action.type) {
    case "ADD_USER_MESSAGE":
      return {
        ...state,
        messages: [
          ...state.messages,
          {
            id: crypto.randomUUID(),
            role: "user",
            content: action.content,
          },
        ],
      };
    case "START_STREAMING":
      return {
        ...state,
        isStreaming: true,
        currentSources: [],
        messages: [
          ...state.messages,
          {
            id: crypto.randomUUID(),
            role: "assistant",
            content: "",
          },
        ],
      };
    case "SET_SOURCES":
      return {
        ...state,
        currentSources: action.sources,
        messages: state.messages.map((m, i) =>
          i === state.messages.length - 1
            ? { ...m, sources: action.sources }
            : m
        ),
      };
    case "APPEND_TOKEN": {
      const messages = [...state.messages];
      const last = messages[messages.length - 1];
      messages[messages.length - 1] = {
        ...last,
        content: last.content + action.token,
      };
      return { ...state, messages };
    }
    case "FINISH_STREAMING":
      return { ...state, isStreaming: false };
    case "CLEAR":
      return { messages: [], isStreaming: false, currentSources: [] };
    default:
      return state;
  }
}

export function useChat() {
  const [state, dispatch] = useReducer(chatReducer, {
    messages: [],
    isStreaming: false,
    currentSources: [],
  });

  const sendMessage = useCallback(
    async (content: string, topK?: number) => {
      dispatch({ type: "ADD_USER_MESSAGE", content });
      dispatch({ type: "START_STREAMING" });

      const history = state.messages.map((m) => ({
        role: m.role,
        content: m.content,
      }));

      try {
        for await (const { event, data } of streamChat(content, history, topK)) {
          if (event === "sources") {
            dispatch({ type: "SET_SOURCES", sources: data.sources });
          } else if (event === "token") {
            dispatch({ type: "APPEND_TOKEN", token: data.token });
          }
        }
      } catch (error) {
        dispatch({
          type: "APPEND_TOKEN",
          token: "\n\n*Error: Failed to get response. Please try again.*",
        });
      } finally {
        dispatch({ type: "FINISH_STREAMING" });
      }
    },
    [state.messages]
  );

  const clearChat = useCallback(() => {
    dispatch({ type: "CLEAR" });
  }, []);

  return {
    messages: state.messages,
    isStreaming: state.isStreaming,
    currentSources: state.currentSources,
    sendMessage,
    clearChat,
  };
}
```

**Step 2: Commit**

```bash
git add frontend/src/hooks/
git commit -m "feat: add useChat hook with SSE streaming support"
```

---

### Task 25: UI components — Chat

**Files:**
- Create: `frontend/src/components/Chat/ChatWindow.tsx`
- Create: `frontend/src/components/Chat/MessageBubble.tsx`
- Create: `frontend/src/components/Chat/InputBar.tsx`

Implement all three chat components with Framer Motion animations:
- `ChatWindow`: scrollable message list with staggered fade-in
- `MessageBubble`: user/assistant bubbles with hover scale effect and markdown rendering
- `InputBar`: auto-resize textarea, send button with press animation

**Note to implementer:** Use `framer-motion` `motion.div` with `initial`, `animate`, and `whileHover` props. Keep animations subtle (0.02-0.05 scale on hover, 0.3s transitions). Use Tailwind dark theme classes (`bg-gray-900`, `bg-gray-800`, etc.).

**Step 1: Implement components** (see design doc for animation specs)

**Step 2: Commit**

```bash
git add frontend/src/components/Chat/
git commit -m "feat: add chat UI components with animations"
```

---

### Task 26: UI components — Documents

**Files:**
- Create: `frontend/src/components/Documents/UploadPanel.tsx`
- Create: `frontend/src/components/Documents/DocumentList.tsx`

Implement:
- `UploadPanel`: drag-and-drop file zone with animated border on dragover, progress indicator
- `DocumentList`: staggered list animation, delete button with exit animation

**Step 1: Implement components**

**Step 2: Commit**

```bash
git add frontend/src/components/Documents/
git commit -m "feat: add document upload and list components"
```

---

### Task 27: UI components — Sources & Settings

**Files:**
- Create: `frontend/src/components/Sources/SourceCard.tsx`
- Create: `frontend/src/components/Settings/SettingsPanel.tsx`

Implement:
- `SourceCard`: expandable card showing chunk text, score, metadata. Hover glow, spring expand.
- `SettingsPanel`: accordion sections for model config and retrieval params.

**Step 1: Implement components**

**Step 2: Commit**

```bash
git add frontend/src/components/Sources/ frontend/src/components/Settings/
git commit -m "feat: add source cards and settings panel components"
```

---

### Task 28: Main page layout

**Files:**
- Modify: `frontend/src/app/layout.tsx`
- Modify: `frontend/src/app/page.tsx`
- Modify: `frontend/src/app/globals.css`

Wire everything together:
- `layout.tsx`: dark theme HTML, Inter font, metadata
- `page.tsx`: sidebar (documents + settings) + main chat area layout
- `globals.css`: Tailwind base imports + custom scrollbar styles + animation keyframes

**Step 1: Implement layout and page**

**Step 2: Commit**

```bash
git add frontend/src/app/
git commit -m "feat: wire up main page layout with sidebar and chat"
```

---

## Phase 6: Docker & Deployment

### Task 29: Backend Dockerfile

**Files:**
- Create: `backend/Dockerfile`

**Step 1: Write the Dockerfile**

```dockerfile
FROM python:3.11-slim

WORKDIR /app

RUN pip install uv

COPY pyproject.toml .
RUN uv pip install --system -e ".[dev]"

COPY backend/ backend/

EXPOSE 8000

CMD ["uvicorn", "backend.src.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Note: The `pyproject.toml` is copied from the repo root since it defines all Python dependencies.

**Step 2: Commit**

```bash
git add backend/Dockerfile
git commit -m "chore: add backend Dockerfile"
```

---

### Task 30: Frontend Dockerfile

**Files:**
- Create: `frontend/Dockerfile`

**Step 1: Write the Dockerfile**

```dockerfile
FROM node:20-alpine AS deps
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci

FROM node:20-alpine AS builder
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .
ARG NEXT_PUBLIC_API_URL=http://localhost:8000
ENV NEXT_PUBLIC_API_URL=$NEXT_PUBLIC_API_URL
RUN npm run build

FROM node:20-alpine AS runner
WORKDIR /app
ENV NODE_ENV=production
COPY --from=builder /app/next.config.js ./
COPY --from=builder /app/public ./public
COPY --from=builder /app/.next ./.next
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/package.json ./

EXPOSE 3000
CMD ["npm", "start"]
```

**Step 2: Commit**

```bash
git add frontend/Dockerfile
git commit -m "chore: add frontend multi-stage Dockerfile"
```

---

### Task 31: docker-compose.yml

**Files:**
- Create: `docker-compose.yml`

**Step 1: Write docker-compose**

```yaml
services:
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage

  backend:
    build:
      context: .
      dockerfile: backend/Dockerfile
    ports:
      - "8000:8000"
    env_file: .env
    environment:
      - QDRANT_HOST=qdrant
      - QDRANT_PORT=6333
    depends_on:
      - qdrant

  frontend:
    build:
      context: ./frontend
      args:
        NEXT_PUBLIC_API_URL: http://localhost:8000
    ports:
      - "3000:3000"
    depends_on:
      - backend

volumes:
  qdrant_data:
```

**Step 2: Commit**

```bash
git add docker-compose.yml
git commit -m "chore: add docker-compose with three services"
```

---

## Phase 7: Sample Data & Documentation

### Task 32: Sample documents

**Files:**
- Create: `data/sample_docs/README.md`
- Create: `data/sample_docs/python_overview.txt`
- Create: `data/sample_docs/machine_learning_basics.md`
- Create: `data/sample_docs/world_cities.csv`

Provide 3 small public-domain sample documents covering different formats (TXT, MD, CSV). Content should be factual, ~500 words each, covering distinct topics so retrieval quality can be demonstrated.

**Step 1: Create sample files**

**Step 2: Commit**

```bash
git add data/sample_docs/
git commit -m "feat: add sample documents for demo"
```

---

### Task 33: README.md

**Files:**
- Create: `README.md`

Write a professional README following the spec requirements:
1. Project title + description
2. Architecture diagram (Mermaid)
3. Features list
4. Quick Start (3 commands)
5. Detailed Setup
6. Usage Examples
7. Configuration
8. How It Works
9. Tech Stack with badges
10. License

**Step 1: Write the README**

**Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add professional README with architecture diagram"
```

---

### Task 34: Architecture documentation

**Files:**
- Create: `docs/architecture.md`

Detailed technical documentation of the RAG pipeline, data flow, and system design. Reference the design document decisions.

**Step 1: Write the doc**

**Step 2: Commit**

```bash
git add docs/architecture.md
git commit -m "docs: add architecture documentation"
```

---

## Phase 8: Integration Testing

### Task 35: Integration tests

**Files:**
- Create: `backend/tests/test_integration.py`

Write integration tests that exercise the full pipeline:
1. Ingest a sample document
2. Query the API
3. Verify sources are returned
4. Verify the SSE stream protocol

Use an in-memory Qdrant store and mock the OpenRouter API call.

**Step 1: Write the tests**

```python
# backend/tests/test_integration.py
import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def app():
    """Create a test app with in-memory vector store."""
    # Override settings to use in-memory Qdrant
    from backend.src.api.deps import get_settings, get_vector_store
    from backend.src.config import Settings
    from backend.src.main import create_app
    from backend.src.vectorstore.qdrant_store import QdrantStore

    test_settings = Settings(
        OPENROUTER_API_KEY="test-key",
        QDRANT_HOST="",  # triggers in-memory
        VECTOR_STORE_BACKEND="qdrant",
    )

    app = create_app()

    store = QdrantStore(host=None, port=None, collection_name="test_integration")
    from backend.src.api.deps import get_embedder
    embedder = get_embedder()
    store.initialize(dimension=embedder.dimension)

    app.dependency_overrides[get_settings] = lambda: test_settings
    app.dependency_overrides[get_vector_store] = lambda: store

    return app


@pytest.fixture
def client(app):
    return TestClient(app)


class TestIntegration:
    def test_health_check(self, client):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"

    def test_upload_and_list(self, client):
        content = b"This is a test document about artificial intelligence."
        resp = client.post(
            "/api/documents/upload",
            files={"file": ("test.txt", content, "text/plain")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ingested"

        resp = client.get("/api/documents")
        assert resp.status_code == 200

    def test_chat_returns_sse_stream(self, client):
        # First upload a document
        content = b"Python is a high-level programming language created by Guido van Rossum."
        client.post(
            "/api/documents/upload",
            files={"file": ("python.txt", content, "text/plain")},
        )

        # Then query — mock the LLM call
        with patch(
            "backend.src.generation.llm_client.OpenRouterClient.generate_stream"
        ) as mock_stream:
            async def fake_stream(*args, **kwargs):
                yield "Python"
                yield " is"
                yield " great."

            mock_stream.return_value = fake_stream()

            resp = client.post(
                "/api/chat",
                json={"query": "What is Python?"},
            )
            assert resp.status_code == 200
```

**Step 2: Run integration tests**

```bash
python -m pytest backend/tests/test_integration.py -v
```

**Step 3: Commit**

```bash
git add backend/tests/test_integration.py
git commit -m "test: add integration tests for full RAG pipeline"
```

---

## Summary

| Phase | Tasks | Description |
|-------|-------|-------------|
| 1 | 1-2 | Project scaffolding (git, config files, directory structure) |
| 2 | 3-10 | Backend core (config, loader, chunker, embedder, vector stores, processor) |
| 3 | 11-15 | Retrieval & generation (reranker, retriever, LLM client, prompts, RAG chain) |
| 4 | 16-21 | API layer (deps, routes, main app, CLI) |
| 5 | 22-28 | Frontend (Next.js setup, API client, hooks, all UI components) |
| 6 | 29-31 | Docker (backend, frontend, docker-compose) |
| 7 | 32-34 | Sample data & documentation |
| 8 | 35 | Integration testing |

**Total: 35 tasks across 8 phases.**
