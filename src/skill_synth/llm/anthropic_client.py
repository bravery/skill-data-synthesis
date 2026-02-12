"""Anthropic API client implementation."""

from __future__ import annotations

from anthropic import AsyncAnthropic

from .base import BaseLLMClient


class AnthropicClient(BaseLLMClient):
    """Anthropic Messages API wrapper."""

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-5-20250929") -> None:
        self._client = AsyncAnthropic(api_key=api_key)
        self._model = model

    async def generate(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.8,
        max_tokens: int = 2048,
    ) -> str:
        # Extract system message if present
        system = ""
        user_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system = msg["content"]
            else:
                user_messages.append(msg)

        kwargs: dict = dict(
            model=self._model,
            messages=user_messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        if system:
            kwargs["system"] = system

        resp = await self._client.messages.create(**kwargs)
        return resp.content[0].text

    async def close(self) -> None:
        await self._client.close()
