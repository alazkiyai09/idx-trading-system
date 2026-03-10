"""
GLM (Anthropic-compatible) LLM Client

Implementation of BaseLLMClient for GLM models via Anthropic-compatible API.
The GLM API at api.z.ai uses the Anthropic message format.
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

# GLM model definitions
GLM_MODELS = {
    "glm-5": ModelInfo(
        provider=LLMProvider.GLM,
        model_name="glm-5",
        max_tokens=4096,
        input_cost_per_1k=0.001,
        output_cost_per_1k=0.002,
    ),
    "glm-4-plus": ModelInfo(
        provider=LLMProvider.GLM,
        model_name="glm-4-plus",
        max_tokens=4096,
        input_cost_per_1k=0.001,
        output_cost_per_1k=0.002,
    ),
    "glm-4": ModelInfo(
        provider=LLMProvider.GLM,
        model_name="glm-4",
        max_tokens=4096,
        input_cost_per_1k=0.001,
        output_cost_per_1k=0.002,
    ),
}

DEFAULT_MODEL = "glm-5"
DEFAULT_BASE_URL = "https://api.z.ai/api/anthropic"


class GLMClient:
    """GLM LLM client using Anthropic-compatible API.

    Implements the BaseLLMClient protocol for GLM models served
    via an Anthropic-compatible endpoint (e.g., api.z.ai).

    Example:
        client = GLMClient(api_key="your-glm-key")
        response = client.generate("Analyze BBCA stock fundamentals")
        print(response.content)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
    ) -> None:
        """Initialize GLM client.

        Args:
            api_key: GLM API key. Uses settings if not provided.
            model: Model name (default: glm-5).
            base_url: API base URL (Anthropic-compatible).
        """
        if api_key is None:
            from config.settings import settings
            api_key = settings.glm_api_key

        self._api_key = api_key
        self._model = model
        self._base_url = base_url
        self._client = None
        self._model_info = GLM_MODELS.get(
            model,
            ModelInfo(
                provider=LLMProvider.GLM,
                model_name=model,
                max_tokens=4096,
            ),
        )

    def _get_client(self) -> Any:
        """Lazily initialize the Anthropic-compatible client."""
        if self._client is None:
            try:
                import anthropic
                self._client = anthropic.Anthropic(
                    api_key=self._api_key,
                    base_url=self._base_url,
                )
            except ImportError:
                raise LLMAPIError(
                    "anthropic package not installed. Run: pip install anthropic"
                )
            except Exception as e:
                raise LLMAuthenticationError(f"Failed to initialize GLM client: {e}")
        return self._client

    @property
    def provider(self) -> LLMProvider:
        """Get the provider for this client."""
        return LLMProvider.GLM

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
        """Generate text using GLM via Anthropic-compatible API.

        Args:
            prompt: User prompt text.
            config: Optional configuration overrides.
            messages: Optional conversation history.

        Returns:
            LLMResponse with generated content.
        """
        config = config or LLMConfig()
        client = self._get_client()

        # Build messages list (Anthropic format)
        api_messages: List[Dict[str, str]] = []

        if messages:
            for msg in messages:
                api_messages.append({
                    "role": msg.role,
                    "content": msg.content,
                })

        api_messages.append({"role": "user", "content": prompt})

        try:
            # Build API call kwargs
            kwargs: Dict[str, Any] = {
                "model": config.model or self._model,
                "messages": api_messages,
                "max_tokens": config.max_tokens,
            }

            # Add system prompt if provided
            if config.system_prompt:
                kwargs["system"] = config.system_prompt

            # Add optional parameters
            if config.temperature is not None:
                kwargs["temperature"] = config.temperature
            if config.top_p is not None:
                kwargs["top_p"] = config.top_p
            if config.stop_sequences:
                kwargs["stop_sequences"] = config.stop_sequences

            response = client.messages.create(**kwargs)

            # Extract content from Anthropic-format response
            content = ""
            if response.content:
                for block in response.content:
                    if hasattr(block, "text"):
                        content += block.text

            stop_reason = getattr(response, "stop_reason", "") or ""

            # Token usage
            input_tokens = 0
            output_tokens = 0
            if response.usage:
                input_tokens = getattr(response.usage, "input_tokens", 0) or 0
                output_tokens = getattr(response.usage, "output_tokens", 0) or 0

            # Calculate cost
            cost = (
                (input_tokens / 1000) * self._model_info.input_cost_per_1k
                + (output_tokens / 1000) * self._model_info.output_cost_per_1k
            )

            return LLMResponse(
                content=content,
                model=getattr(response, "model", self._model) or self._model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost,
                finish_reason=stop_reason,
                raw_response={"id": getattr(response, "id", "")},
            )

        except Exception as e:
            error_str = str(e)
            if "authentication" in error_str.lower() or "api key" in error_str.lower():
                raise LLMAuthenticationError(f"GLM authentication failed: {e}")
            elif "rate" in error_str.lower() and "limit" in error_str.lower():
                raise LLMRateLimitError(f"GLM rate limit exceeded: {e}")
            else:
                raise LLMAPIError(f"GLM API error: {e}")

    def generate_json(
        self,
        prompt: str,
        schema: Optional[Dict[str, Any]] = None,
        config: Optional[LLMConfig] = None,
    ) -> Dict[str, Any]:
        """Generate structured JSON output using GLM.

        Args:
            prompt: User prompt requesting JSON output.
            schema: Optional JSON schema for validation.
            config: Optional configuration overrides.

        Returns:
            Parsed JSON dictionary.
        """
        json_prompt = prompt
        if schema:
            json_prompt += f"\n\nRespond with valid JSON matching this schema:\n{json.dumps(schema, indent=2)}"
        else:
            json_prompt += "\n\nRespond with valid JSON only. No markdown, no explanation."

        response = self.generate(json_prompt, config=config)

        content = response.content.strip()
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
                f"Failed to parse GLM response as JSON: {e}\nContent: {content[:500]}"
            )

    def is_available(self) -> bool:
        """Check if GLM client is configured and available."""
        return bool(self._api_key and self._api_key.strip())
