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
    """Container for a complete RAG response.

    Attributes:
        answer: The generated text from the LLM.
        sources: The retrieved document chunks that informed the answer,
            so the caller can display citations alongside the text.
    """

    answer: str
    sources: list[SearchResult]


class RAGChain:
    """Orchestrates retrieval + generation for RAG queries.

    This is the top-level entry point for the RAG pipeline.  It wires
    together the hybrid retriever (dense + sparse search with optional
    re-ranking) and the LLM client, exposing both streaming and
    non-streaming interfaces.

    Typical usage::

        chain = RAGChain(retriever=retriever, llm_client=llm)
        response = await chain.query("What is the refund policy?")
        print(response.answer)
        for src in response.sources:
            print(src.metadata["source"])

    Args:
        retriever: A ``HybridRetriever`` configured with an embedder,
            vector store, and optional reranker.
        llm_client: An ``OpenRouterClient`` configured with an API key
            and model identifier.
    """

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
        """Retrieve relevant sources for a query.

        This is exposed separately so callers (e.g. the API layer) can
        display sources before the LLM finishes generating, which gives
        a more responsive user experience.

        Args:
            query: Natural language query string.
            top_k: Number of final results to return.
            score_threshold: Minimum similarity score for initial retrieval.

        Returns:
            Ranked list of ``SearchResult`` objects.
        """
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
        """Stream the LLM response given a query and pre-retrieved sources.

        This method does NOT perform retrieval -- the caller must pass in
        already-retrieved sources.  This design allows the API layer to
        send sources to the frontend immediately, then start streaming
        the generated answer.

        Args:
            query: The user's question.
            sources: Pre-retrieved document chunks.
            conversation_history: Optional prior messages for multi-turn
                conversations.

        Yields:
            Individual token strings as they arrive from the LLM.
        """
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
        """Full RAG pipeline: retrieve + generate (non-streaming).

        Convenience method that runs both stages and returns the
        complete answer with its sources in one call.

        Args:
            query: The user's question.
            top_k: Number of sources to retrieve.
            score_threshold: Minimum similarity score for retrieval.
            conversation_history: Optional prior messages for multi-turn
                conversations.

        Returns:
            A ``RAGResponse`` containing the answer text and source
            documents.
        """
        sources = self.retrieve_sources(query, top_k, score_threshold)
        messages = build_rag_prompt(query, sources, conversation_history)
        answer = await self.llm_client.generate(messages)
        return RAGResponse(answer=answer, sources=sources)
