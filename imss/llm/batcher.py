"""Batched async LLM execution with concurrency control."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from imss.llm.router import LLMResponse, LLMRouter

logger = logging.getLogger(__name__)


@dataclass
class LLMRequest:
    """A pending LLM request."""

    agent_id: str
    system_prompt: str
    user_prompt: str
    temperature: float
    max_tokens: int
    tier: int


class LLMBatcher:
    """Execute batches of LLM calls with semaphore-based concurrency."""

    def __init__(self, router: LLMRouter, max_concurrent: int = 5):
        self._router = router
        self._semaphore = asyncio.Semaphore(max_concurrent)

    async def _limited_call(self, request: LLMRequest) -> LLMResponse:
        async with self._semaphore:
            return await self._router.call(
                system_prompt=request.system_prompt,
                user_prompt=request.user_prompt,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                tier=request.tier,
            )

    async def execute_batch(
        self, requests: list[LLMRequest]
    ) -> list[LLMResponse]:
        """Execute a batch of LLM requests with concurrency limit.

        Failed calls return a response with parsed_json=None.
        """
        results = await asyncio.gather(
            *[self._limited_call(req) for req in requests],
            return_exceptions=True,
        )
        processed: list[LLMResponse] = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(
                    "Batch call failed for agent %s: %s",
                    requests[i].agent_id,
                    result,
                )
                processed.append(
                    LLMResponse(
                        content="",
                        parsed_json=None,
                        input_tokens=0,
                        output_tokens=0,
                        model="error",
                        latency_ms=0,
                    )
                )
            else:
                processed.append(result)
        return processed
