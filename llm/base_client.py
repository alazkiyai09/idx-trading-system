"""
Base LLM Client Module

Defines the Protocol (interface) for all LLM clients.
Provides shared types used across all implementations.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Dict, List, Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


class LLMProvider(Enum):
    """Supported LLM providers."""
    CLAUDE = "claude"
    GLM = "glm"


@dataclass
class ModelInfo:
    """Information about an LLM model.

    Attributes:
        provider: LLM provider.
        model_name: Full model name (e.g., 'claude-sonnet-4-20250514').
        max_tokens: Maximum tokens the model supports.
        input_cost_per_1k: Cost per 1K input tokens (USD).
        output_cost_per_1k: Cost per 1K output tokens (USD).
    """
    provider: LLMProvider
    model_name: str
    max_tokens: int = 4096
    input_cost_per_1k: float = 0.0
    output_cost_per_1k: float = 0.0


@dataclass
class LLMMessage:
    """A single message in a conversation.

    Attributes:
        role: Message role (system, user, assistant).
        content: Message content.
    """
    role: str
    content: str


@dataclass
class LLMResponse:
    """Response from an LLM API call.

    Attributes:
        content: Generated text content.
        model: Model used for generation.
        input_tokens: Number of input tokens used.
        output_tokens: Number of output tokens generated.
        total_tokens: Total tokens used.
        cost_usd: Estimated cost in USD.
        finish_reason: Why generation stopped.
        raw_response: Raw API response for debugging.
    """
    content: str
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    finish_reason: str = ""
    raw_response: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        """Calculate total tokens if not provided."""
        if self.total_tokens == 0:
            self.total_tokens = self.input_tokens + self.output_tokens


@dataclass
class LLMConfig:
    """Configuration for an LLM API call.

    Attributes:
        model: Model name to use.
        max_tokens: Maximum tokens to generate.
        temperature: Sampling temperature (0-1).
        top_p: Top-p sampling parameter.
        stop_sequences: Sequences that stop generation.
        system_prompt: System prompt to set context.
    """
    model: Optional[str] = None
    max_tokens: int = 4096
    temperature: float = 0.7
    top_p: float = 1.0
    stop_sequences: List[str] = field(default_factory=list)
    system_prompt: Optional[str] = None


class LLMError(Exception):
    """Base exception for LLM errors."""
    pass


class LLMAuthenticationError(LLMError):
    """API key is invalid or missing."""
    pass


class LLMRateLimitError(LLMError):
    """Rate limit exceeded."""
    pass


class LLMAPIError(LLMError):
    """General API error."""

    def __init__(self, message: str, status_code: Optional[int] = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class LLMValidationError(LLMError):
    """Response validation failed."""
    pass


@runtime_checkable
class BaseLLMClient(Protocol):
    """Protocol defining the interface for all LLM clients.

    All LLM provider implementations must satisfy this protocol.
    This enables provider-agnostic code throughout the system.

    Example:
        client: BaseLLMClient = ClaudeClient(api_key="...")
        response = client.generate("Analyze this stock")
        json_data = client.generate_json("Rate this company", schema={...})
    """

    @property
    def provider(self) -> LLMProvider:
        """Get the provider for this client."""
        ...

    @property
    def model_info(self) -> ModelInfo:
        """Get model information."""
        ...

    def generate(
        self,
        prompt: str,
        config: Optional[LLMConfig] = None,
        messages: Optional[List[LLMMessage]] = None,
    ) -> LLMResponse:
        """Generate text from a prompt.

        Args:
            prompt: User prompt text.
            config: Optional configuration overrides.
            messages: Optional conversation history.

        Returns:
            LLMResponse with generated content.

        Raises:
            LLMAuthenticationError: If API key is invalid.
            LLMRateLimitError: If rate limit is exceeded.
            LLMAPIError: If API call fails.
        """
        ...

    def generate_json(
        self,
        prompt: str,
        schema: Optional[Dict[str, Any]] = None,
        config: Optional[LLMConfig] = None,
    ) -> Dict[str, Any]:
        """Generate structured JSON output.

        Args:
            prompt: User prompt requesting JSON output.
            schema: Optional JSON schema for validation.
            config: Optional configuration overrides.

        Returns:
            Parsed JSON dictionary.

        Raises:
            LLMValidationError: If response is not valid JSON.
            LLMAPIError: If API call fails.
        """
        ...

    def is_available(self) -> bool:
        """Check if the client is configured and available.

        Returns:
            True if the client can make API calls.
        """
        ...
