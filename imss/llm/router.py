"""LLM Router — async OpenAI-compatible client for GLM-5."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field

from openai import AsyncOpenAI

from imss.config import IMSSSettings

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Parsed LLM response."""

    content: str
    parsed_json: dict | None
    input_tokens: int
    output_tokens: int
    model: str
    latency_ms: float


@dataclass
class CostTracker:
    """Tracks LLM API costs per run."""

    total_calls: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    calls_by_tier: dict[int, int] = field(default_factory=lambda: {1: 0, 2: 0})
    parse_successes: int = 0
    parse_failures: int = 0

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens

    @property
    def estimated_cost_usd(self) -> float:
        # GLM-5 approximate pricing
        return (self.total_input_tokens * 0.001 + self.total_output_tokens * 0.002) / 1000

    @property
    def json_parse_rate(self) -> float:
        total = self.parse_successes + self.parse_failures
        return self.parse_successes / total if total > 0 else 1.0


def strip_json_fences(text: str) -> str:
    """Remove markdown code fences from JSON response."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*\n?", "", text)
    text = re.sub(r"\n?```\s*$", "", text)
    return text.strip()


def parse_agent_json(text: str) -> dict | None:
    """Parse agent JSON response, handling common GLM-5 quirks."""
    cleaned = strip_json_fences(text)
    try:
        data = json.loads(cleaned)
        # Validate required keys
        required = {"action", "stock", "quantity", "confidence", "reasoning"}
        if not required.issubset(data.keys()):
            logger.warning("Missing required keys in response: %s", data.keys())
            return None
        if data["action"] not in ("BUY", "SELL", "HOLD"):
            logger.warning("Invalid action: %s", data["action"])
            return None
        return data
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        logger.warning("JSON parse failed: %s — raw: %s", e, cleaned[:200])
        return None


class LLMRouter:
    """Async LLM router for GLM-5 via OpenAI-compatible API."""

    def __init__(self, settings: IMSSSettings | None = None):
        if settings is None:
            settings = IMSSSettings()
        self._client = AsyncOpenAI(
            api_key=settings.glm_api_key,
            base_url=settings.glm_base_url,
            timeout=settings.llm_request_timeout,
        )
        self._model = settings.glm_model
        self._max_retries = 3
        self.cost_tracker = CostTracker()

    async def call(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        tier: int = 1,
    ) -> LLMResponse:
        """Make a single LLM call with retry logic.

        Returns LLMResponse with parsed_json if valid, else None.
        """
        last_error: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                start = time.monotonic()
                response = await self._client.chat.completions.create(
                    model=self._model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                latency = (time.monotonic() - start) * 1000

                content = response.choices[0].message.content or ""
                usage = response.usage
                input_tokens = usage.prompt_tokens if usage else 0
                output_tokens = usage.completion_tokens if usage else 0

                # Track costs
                self.cost_tracker.total_calls += 1
                self.cost_tracker.total_input_tokens += input_tokens
                self.cost_tracker.total_output_tokens += output_tokens
                self.cost_tracker.calls_by_tier[tier] = (
                    self.cost_tracker.calls_by_tier.get(tier, 0) + 1
                )

                # Parse JSON
                parsed = parse_agent_json(content)
                if parsed is not None:
                    self.cost_tracker.parse_successes += 1
                else:
                    self.cost_tracker.parse_failures += 1

                return LLMResponse(
                    content=content,
                    parsed_json=parsed,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    model=self._model,
                    latency_ms=latency,
                )

            except Exception as e:
                last_error = e
                wait = 2**attempt
                logger.warning(
                    "LLM call attempt %d failed: %s. Retrying in %ds...",
                    attempt + 1,
                    e,
                    wait,
                )
                await asyncio.sleep(wait)

        # All retries exhausted
        logger.error("LLM call failed after %d retries: %s", self._max_retries, last_error)
        self.cost_tracker.total_calls += 1
        self.cost_tracker.parse_failures += 1
        return LLMResponse(
            content="",
            parsed_json=None,
            input_tokens=0,
            output_tokens=0,
            model=self._model,
            latency_ms=0,
        )
