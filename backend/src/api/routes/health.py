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
