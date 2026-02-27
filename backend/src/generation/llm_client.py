from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx

logger = logging.getLogger(__name__)

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"


class OpenRouterClient:
    """Async client for OpenRouter's chat completion API with streaming.

    OpenRouter exposes an OpenAI-compatible REST API, which lets us swap
    between hundreds of models (Claude, GPT, Llama, Mistral, ...) by
    changing a single ``model`` string.  This client supports both
    streaming (SSE) and non-streaming generation.

    Args:
        api_key: OpenRouter API key.
        model: Model identifier, e.g. ``"anthropic/claude-haiku-4.5"``.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "anthropic/claude-haiku-4.5",
    ) -> None:
        self.api_key = api_key
        self.model = model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def generate_stream(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> AsyncIterator[str]:
        """Stream tokens from OpenRouter.

        Yields individual token strings as they arrive via Server-Sent
        Events (SSE).  The caller can iterate with ``async for`` and
        forward each chunk to the end-user for a real-time typing effect.
        """
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

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @asynccontextmanager
    async def _stream_request(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ):
        """Open an HTTP streaming connection to OpenRouter."""
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
