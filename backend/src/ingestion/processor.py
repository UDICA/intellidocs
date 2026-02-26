"""Ingestion pipeline orchestrator: load -> chunk -> embed -> store.

Wires together the document loader, text chunker, embedding model, and vector
store into a single coherent pipeline. Supports processing individual files or
entire directories of supported documents.
"""

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
    """Orchestrates the document ingestion pipeline: load -> chunk -> embed -> store.

    Takes a file (or directory of files), loads the raw text via
    :class:`DocumentLoader`, splits it into overlapping chunks via
    :class:`TextChunker`, produces dense and sparse embeddings via
    :class:`Embedder`, and stores everything in the configured vector store.

    Args:
        embedder: Embedding model wrapper for dense and sparse vectors.
        vector_store: Vector store backend (Qdrant, ChromaDB, etc.).
        chunk_size: Maximum characters per text chunk.
        chunk_overlap: Character overlap between consecutive chunks.
    """

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
        """Ingest a single file through the full pipeline.

        Loads the file, splits its content into chunks, generates dense and
        sparse embeddings for each chunk, and upserts them into the vector
        store. The document ID is a truncated SHA-256 hash of the file path,
        providing a deterministic identifier for re-ingestion scenarios.

        Args:
            path: Path to the file to ingest.

        Returns:
            A 16-character hex document ID derived from the file path.
        """
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
        """Ingest all supported files in a directory.

        Iterates through the directory (non-recursive, sorted order) and
        processes each file with a supported extension. Failures on individual
        files are logged and skipped so that one bad file does not abort the
        entire batch.

        Args:
            directory: Path to the directory to scan.

        Returns:
            Number of files successfully processed.
        """
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
