"""Abstract LLM client interface."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any


class BaseLLMClient(ABC):
    """Abstract base class for LLM API clients."""

    @abstractmethod
    async def generate(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.8,
        max_tokens: int = 2048,
    ) -> str:
        """Generate a text completion from messages."""

    async def generate_json(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.8,
        max_tokens: int = 2048,
    ) -> dict[str, Any] | list[Any]:
        """Generate a JSON response from messages.

        Appends a JSON instruction to the last user message and parses the result.
        """
        response = await self.generate(
            messages, temperature=temperature, max_tokens=max_tokens
        )
        return _extract_json(response)

    @abstractmethod
    async def close(self) -> None:
        """Close underlying HTTP connections."""


def _extract_json(text: str) -> dict[str, Any] | list[Any]:
    """Extract JSON from LLM response, handling markdown code blocks."""
    text = text.strip()
    # Strip markdown code fences
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first line (```json or ```) and last line (```)
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return json.loads(text)
