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
