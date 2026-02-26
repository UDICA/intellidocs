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
