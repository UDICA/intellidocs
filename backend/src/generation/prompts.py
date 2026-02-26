from __future__ import annotations

from backend.src.vectorstore.base import SearchResult

SYSTEM_PROMPT = """You are IntelliDocs, an intelligent document assistant. You answer questions \
based on the provided context documents.

Rules:
- Answer ONLY based on the provided context. If the context doesn't contain enough information, \
say so clearly.
- Cite your sources by referencing the document name and page/section when available.
- Be concise and direct. Avoid unnecessary preamble.
- If asked about something not in the context, say "I don't have information about that in the \
provided documents."
"""


def build_rag_prompt(
    query: str,
    sources: list[SearchResult],
    conversation_history: list[dict[str, str]] | None = None,
) -> list[dict[str, str]]:
    """Build the full message list for the LLM.

    Constructs a conversation in the standard ``messages`` format
    (system / user / assistant) that includes:

    1. A system prompt defining the assistant's behaviour and rules.
    2. Any prior conversation history (for multi-turn follow-ups).
    3. A user message containing the retrieved context chunks and the
       new question.

    Each source chunk is numbered so the LLM can cite ``[1]``, ``[2]``,
    etc. in its answer.

    Args:
        query: The user's natural-language question.
        sources: Retrieved document chunks with metadata.
        conversation_history: Optional prior messages for multi-turn
            conversations.

    Returns:
        A list of message dicts ready to send to the LLM API.
    """
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
