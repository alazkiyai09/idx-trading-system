"""
Claude (Anthropic) LLM Client

Implementation of BaseLLMClient for Anthropic's Claude models.
"""

import json
import logging
from typing import Any, Dict, List, Optional

from llm.base_client import (
    BaseLLMClient,
    LLMAPIError,
    LLMAuthenticationError,
    LLMConfig,
    LLMMessage,
    LLMProvider,
    LLMRateLimitError,
    LLMResponse,
    LLMValidationError,
    ModelInfo,
)

logger = logging.getLogger(__name__)

# Claude model pricing (USD per 1K tokens)
CLAUDE_MODELS = {
    "claude-sonnet-4-20250514": ModelInfo(
        provider=LLMProvider.CLAUDE,
        model_name="claude-sonnet-4-20250514",
        max_tokens=8192,
        input_cost_per_1k=0.003,
        output_cost_per_1k=0.015,
    ),
    "claude-3-5-haiku-20241022": ModelInfo(
        provider=LLMProvider.CLAUDE,
        model_name="claude-3-5-haiku-20241022",
        max_tokens=8192,
        input_cost_per_1k=0.001,
        output_cost_per_1k=0.005,
    ),
    "claude-3-opus-20240229": ModelInfo(
        provider=LLMProvider.CLAUDE,
        model_name="claude-3-opus-20240229",
        max_tokens=4096,
        input_cost_per_1k=0.015,
        output_cost_per_1k=0.075,
    ),
}

DEFAULT_MODEL = "claude-sonnet-4-20250514"


class ClaudeClient:
    """Anthropic Claude LLM client.

    Implements the BaseLLMClient protocol for Claude models.

    Example:
        client = ClaudeClient(api_key="sk-ant-...")
        response = client.generate("Analyze BBCA stock fundamentals")
        print(response.content)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
    ) -> None:
        """Initialize Claude client.

        Args:
            api_key: Anthropic API key. Uses settings if not provided.
            model: Model name (default: claude-sonnet-4-20250514).
        """
        if api_key is None:
            from config.settings import settings
            api_key = settings.anthropic_api_key

        self._api_key = api_key
        self._model = model
        self._client = None
        self._model_info = CLAUDE_MODELS.get(
            model,
            ModelInfo(
                provider=LLMProvider.CLAUDE,
                model_name=model,
                max_tokens=4096,
            ),
        )

    def _get_client(self) -> Any:
        """Lazily initialize the Anthropic client."""
        if self._client is None:
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=self._api_key)
            except ImportError:
                raise LLMAPIError(
                    "anthropic package not installed. Run: pip install anthropic"
                )
            except Exception as e:
                raise LLMAuthenticationError(f"Failed to initialize Anthropic client: {e}")
        return self._client

    @property
    def provider(self) -> LLMProvider:
        """Get the provider for this client."""
        return LLMProvider.CLAUDE

    @property
    def model_info(self) -> ModelInfo:
        """Get model information."""
        return self._model_info

    def generate(
        self,
        prompt: str,
        config: Optional[LLMConfig] = None,
        messages: Optional[List[LLMMessage]] = None,
    ) -> LLMResponse:
        """Generate text using Claude.

        Args:
            prompt: User prompt text.
            config: Optional configuration overrides.
            messages: Optional conversation history.

        Returns:
            LLMResponse with generated content.
        """
        config = config or LLMConfig()
        client = self._get_client()

        # Build messages list
        api_messages = []
        if messages:
            for msg in messages:
                api_messages.append({
                    "role": msg.role if msg.role != "system" else "user",
                    "content": msg.content,
                })

        # Add the current prompt
        api_messages.append({"role": "user", "content": prompt})

        # Build API call kwargs
        kwargs: Dict[str, Any] = {
            "model": config.model or self._model,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "messages": api_messages,
        }

        if config.system_prompt:
            kwargs["system"] = config.system_prompt

        if config.stop_sequences:
            kwargs["stop_sequences"] = config.stop_sequences

        try:
            response = client.messages.create(**kwargs)

            # Extract content
            content = ""
            if response.content:
                content = response.content[0].text

            # Calculate cost
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            cost = (
                (input_tokens / 1000) * self._model_info.input_cost_per_1k
                + (output_tokens / 1000) * self._model_info.output_cost_per_1k
            )

            return LLMResponse(
                content=content,
                model=response.model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost,
                finish_reason=response.stop_reason or "",
                raw_response={"id": response.id, "type": response.type},
            )

        except Exception as e:
            error_str = str(e)
            if "authentication" in error_str.lower() or "api key" in error_str.lower():
                raise LLMAuthenticationError(f"Claude authentication failed: {e}")
            elif "rate" in error_str.lower() and "limit" in error_str.lower():
                raise LLMRateLimitError(f"Claude rate limit exceeded: {e}")
            else:
                raise LLMAPIError(f"Claude API error: {e}")

    def generate_json(
        self,
        prompt: str,
        schema: Optional[Dict[str, Any]] = None,
        config: Optional[LLMConfig] = None,
    ) -> Dict[str, Any]:
        """Generate structured JSON output using Claude.

        Args:
            prompt: User prompt requesting JSON output.
            schema: Optional JSON schema for validation.
            config: Optional configuration overrides.

        Returns:
            Parsed JSON dictionary.
        """
        # Enhance prompt to request JSON
        json_prompt = prompt
        if schema:
            json_prompt += f"\n\nRespond with valid JSON matching this schema:\n{json.dumps(schema, indent=2)}"
        else:
            json_prompt += "\n\nRespond with valid JSON only. No markdown, no explanation."

        response = self.generate(json_prompt, config=config)

        # Parse JSON from response
        content = response.content.strip()

        # Strip markdown code fences if present
        if content.startswith("```json"):
            content = content[7:]
        elif content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()

        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            raise LLMValidationError(
                f"Failed to parse Claude response as JSON: {e}\nContent: {content[:500]}"
            )

    def is_available(self) -> bool:
        """Check if Claude client is configured and available."""
        return bool(self._api_key and self._api_key.strip())
