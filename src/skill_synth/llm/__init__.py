"""LLM client factory."""

from __future__ import annotations

from ..config import LLMConfig
from .base import BaseLLMClient


def create_llm_client(config: LLMConfig) -> BaseLLMClient:
    """Create an LLM client based on config provider setting."""
    if config.provider == "openai":
        from .openai_client import OpenAIClient

        return OpenAIClient(api_key=config.api_key, model=config.model)
    elif config.provider == "anthropic":
        from .anthropic_client import AnthropicClient

        return AnthropicClient(api_key=config.api_key, model=config.model)
    else:
        raise ValueError(f"Unknown LLM provider: {config.provider}")


__all__ = ["BaseLLMClient", "create_llm_client"]
