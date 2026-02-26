# backend/src/api/routes/documents.py
from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile

from backend.src.api.deps import get_ingestion_processor, get_vector_store
from backend.src.ingestion.processor import IngestionProcessor
from backend.src.vectorstore.base import VectorStoreBase

router = APIRouter(prefix="/api/documents", tags=["documents"])

SUPPORTED_EXTENSIONS = {".txt", ".md", ".markdown", ".csv", ".pdf", ".docx"}


@router.post("/upload")
async def upload_document(
    file: UploadFile,
    processor: IngestionProcessor = Depends(get_ingestion_processor),
) -> dict:
    """Upload and ingest a document."""
    filename = file.filename or "unnamed"
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format: {ext}. Supported: {sorted(SUPPORTED_EXTENSIONS)}",
        )

    content = await file.read()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext, prefix="intellidocs_")
    tmp.write(content)
    tmp.flush()
    tmp_path = Path(tmp.name)
    tmp.close()

    try:
        doc_id = processor.process_file(tmp_path)
    finally:
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
