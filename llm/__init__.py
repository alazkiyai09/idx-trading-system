"""
LLM Client Package

Provides abstracted LLM client for the IDX Trading System.

Usage:
    from llm import create_client, LLMProvider

    # Create default client (Claude)
    client = create_client()

    # Create specific provider
    client = create_client(LLMProvider.GLM)

    # Generate text
    response = client.generate("Analyze this stock")

    # Generate structured JSON
    data = client.generate_json("Rate this company", schema={...})
"""

import logging
from typing import Optional

from llm.base_client import (
    BaseLLMClient,
    LLMAPIError,
    LLMAuthenticationError,
    LLMConfig,
    LLMError,
    LLMMessage,
    LLMProvider,
    LLMRateLimitError,
    LLMResponse,
    LLMValidationError,
    ModelInfo,
)
from llm.cost_tracker import CostTracker
from llm.prompt_manager import PromptManager
from llm.response_validator import ResponseValidator
from llm.retry_handler import RetryHandler, RetryConfig

logger = logging.getLogger(__name__)

__all__ = [
    # Core types
    "BaseLLMClient",
    "LLMConfig",
    "LLMMessage",
    "LLMProvider",
    "LLMResponse",
    "ModelInfo",
    # Exceptions
    "LLMError",
    "LLMAPIError",
    "LLMAuthenticationError",
    "LLMRateLimitError",
    "LLMValidationError",
    # Utilities
    "CostTracker",
    "PromptManager",
    "ResponseValidator",
    "RetryHandler",
    "RetryConfig",
    # Factory
    "create_client",
    "create_client_with_fallback",
]


def create_client(
    provider: Optional[LLMProvider] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> BaseLLMClient:
    """Factory function to create an LLM client.

    Args:
        provider: LLM provider. Defaults to Claude.
        api_key: API key. Uses settings if not provided.
        model: Model name. Uses provider default if not provided.

    Returns:
        LLM client instance.

    Raises:
        ValueError: If provider is not supported.
    """
    if provider is None:
        provider = LLMProvider.CLAUDE

    if provider == LLMProvider.CLAUDE:
        from llm.claude_client import ClaudeClient
        kwargs = {}
        if api_key:
            kwargs["api_key"] = api_key
        if model:
            kwargs["model"] = model
        return ClaudeClient(**kwargs)

    elif provider == LLMProvider.GLM:
        from llm.glm_client import GLMClient
        kwargs = {}
        if api_key:
            kwargs["api_key"] = api_key
        if model:
            kwargs["model"] = model
        return GLMClient(**kwargs)

    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")


def create_client_with_fallback(
    primary: LLMProvider = LLMProvider.CLAUDE,
    fallback: LLMProvider = LLMProvider.GLM,
) -> "FallbackClient":
    """Create an LLM client with automatic fallback.

    If the primary provider fails, automatically falls back to the
    secondary provider.

    Args:
        primary: Primary LLM provider.
        fallback: Fallback LLM provider.

    Returns:
        FallbackClient that tries primary then fallback.
    """
    return FallbackClient(
        primary=create_client(primary),
        fallback=create_client(fallback),
    )


class FallbackClient:
    """LLM client with automatic provider fallback.

    Tries the primary client first. If it fails, falls back to
    the secondary client.
    """

    def __init__(
        self,
        primary: BaseLLMClient,
        fallback: BaseLLMClient,
    ) -> None:
        self._primary = primary
        self._fallback = fallback
        self._retry_handler = RetryHandler()

    @property
    def provider(self) -> LLMProvider:
        return self._primary.provider

    @property
    def model_info(self) -> ModelInfo:
        return self._primary.model_info

    def generate(
        self,
        prompt: str,
        config: Optional[LLMConfig] = None,
        messages: Optional[list] = None,
    ) -> LLMResponse:
        """Generate text with automatic fallback."""
        try:
            return self._retry_handler.execute(
                self._primary.generate, prompt, config, messages
            )
        except (LLMAPIError, LLMRateLimitError) as e:
            logger.warning(
                f"Primary provider ({self._primary.provider.value}) failed: {e}. "
                f"Falling back to {self._fallback.provider.value}."
            )
            if self._fallback.is_available():
                return self._fallback.generate(prompt, config, messages)
            raise

    def generate_json(
        self,
        prompt: str,
        schema: Optional[dict] = None,
        config: Optional[LLMConfig] = None,
    ) -> dict:
        """Generate structured JSON with automatic fallback."""
        try:
            return self._retry_handler.execute(
                self._primary.generate_json, prompt, schema, config
            )
        except (LLMAPIError, LLMRateLimitError, LLMValidationError) as e:
            logger.warning(
                f"Primary provider ({self._primary.provider.value}) failed: {e}. "
                f"Falling back to {self._fallback.provider.value}."
            )
            if self._fallback.is_available():
                return self._fallback.generate_json(prompt, schema, config)
            raise

    def is_available(self) -> bool:
        """Check if any provider is available."""
        return self._primary.is_available() or self._fallback.is_available()
