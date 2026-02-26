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

        # Mock the httpx async stream to return fake SSE data
        mock_lines = [
            'data: {"choices":[{"delta":{"content":"Hello"}}]}',
            'data: {"choices":[{"delta":{"content":" world"}}]}',
            "data: [DONE]",
        ]

        async def fake_aiter_lines():
            for line in mock_lines:
                yield line

        mock_response = AsyncMock()
        mock_response.aiter_lines = fake_aiter_lines
        mock_response.status_code = 200
        mock_response.raise_for_status = lambda: None

        # We need to mock _stream_request as an async context manager
        async def mock_stream_request(*args, **kwargs):
            return mock_response

        with patch.object(client, "_stream_request") as mock_sr:
            # Make _stream_request an async context manager that yields mock_response
            mock_cm = AsyncMock()
            mock_cm.__aenter__ = AsyncMock(return_value=mock_response)
            mock_cm.__aexit__ = AsyncMock(return_value=False)
            mock_sr.return_value = mock_cm

            tokens = []
            async for token in client.generate_stream(
                messages=[{"role": "user", "content": "Hi"}]
            ):
                tokens.append(token)

            assert "Hello" in tokens
            assert " world" in tokens
