"""Tests for Phase 2B Tier 2 archetypes: dividend_holder and sector_rotator."""

import pytest
from imss.agents.tier2.archetypes import ARCHETYPES, create_tier2_agents, Tier2Agent


class TestNewArchetypes:
    def test_dividend_holder_exists(self):
        assert "dividend_holder" in ARCHETYPES

    def test_dividend_holder_config(self):
        cfg = ARCHETYPES["dividend_holder"]
        assert cfg.decision_latency == 2
        assert cfg.sentiment_bias_range == (0.1, 0.3)

    def test_sector_rotator_exists(self):
        assert "sector_rotator" in ARCHETYPES

    def test_sector_rotator_config(self):
        cfg = ARCHETYPES["sector_rotator"]
        assert cfg.decision_latency == 1
        assert cfg.sentiment_bias_range == (-0.1, 0.1)

    def test_all_five_archetypes_registered(self):
        expected = {"momentum_chaser", "panic_seller", "news_reactive", "dividend_holder", "sector_rotator"}
        assert set(ARCHETYPES.keys()) == expected

    def test_create_tier2_agents_dividend_holder(self):
        agents = create_tier2_agents("dividend_holder", 4)
        assert len(agents) == 4
        for a in agents:
            assert a.archetype_key == "dividend_holder"
            assert a.decision_latency == 2
            assert 100_000_000 <= a.working_memory.cash <= 500_000_000

    def test_create_tier2_agents_sector_rotator(self):
        agents = create_tier2_agents("sector_rotator", 4)
        assert len(agents) == 4
        for a in agents:
            assert a.archetype_key == "sector_rotator"
            assert a.decision_latency == 1
            assert 150_000_000 <= a.working_memory.cash <= 600_000_000


class TestDecisionLatency:
    @pytest.mark.asyncio
    async def test_latency_filters_recent_events(self):
        """Agent with decision_latency=2 only sees events where step - _step >= 2."""
        from unittest.mock import AsyncMock
        from imss.llm.router import LLMResponse

        agents = create_tier2_agents("dividend_holder", 1, seed=99)
        agent = agents[0]
        assert agent.decision_latency == 2

        mock_response = LLMResponse(
            content='{"action": "HOLD", "stock": "BBRI", "quantity": 0, "confidence": 0.5, "reasoning": "hold", "sentiment_update": 0.0}',
            parsed_json={"action": "HOLD", "stock": "BBRI", "quantity": 0, "confidence": 0.5, "reasoning": "hold", "sentiment_update": 0.0},
            input_tokens=50, output_tokens=25, model="glm-5", latency_ms=50,
        )
        mock_router = AsyncMock()
        mock_router.call = AsyncMock(return_value=mock_response)
        agent.set_router(mock_router)

        events = [
            {"title": "Old event", "category": "EARNINGS", "_step": 0, "sentiment_score": 0.5},
            {"title": "Recent event", "category": "NEWS", "_step": 2, "sentiment_score": -0.3},
        ]
        # At step 2: old event passes (2-0=2 >= 2), recent filtered (2-2=0 < 2)
        await agent.decide(
            {"symbol": "BBRI", "ohlcv": {"close": 5000}, "pct_change_1d": 0, "pct_change_5d": 0},
            events, step=2,
        )
        # Verify the LLM prompt only got the old event
        call_args = mock_router.call.call_args
        user_prompt = call_args.kwargs.get("user_prompt", call_args[1] if len(call_args) > 1 else "")
        assert "Old event" in user_prompt
        assert "Recent event" not in user_prompt

    @pytest.mark.asyncio
    async def test_zero_latency_passes_all_events(self):
        """Agent with decision_latency=0 receives all events (nothing filtered)."""
        from unittest.mock import AsyncMock
        from imss.llm.router import LLMResponse

        agents = create_tier2_agents("momentum_chaser", 1, seed=99)
        agent = agents[0]
        assert agent.decision_latency == 0

        mock_response = LLMResponse(
            content='{"action": "HOLD", "stock": "BBRI", "quantity": 0, "confidence": 0.5, "reasoning": "hold", "sentiment_update": 0.0}',
            parsed_json={"action": "HOLD", "stock": "BBRI", "quantity": 0, "confidence": 0.5, "reasoning": "hold", "sentiment_update": 0.0},
            input_tokens=50, output_tokens=25, model="glm-5", latency_ms=50,
        )
        mock_router = AsyncMock()
        mock_router.call = AsyncMock(return_value=mock_response)
        agent.set_router(mock_router)

        events = [
            {"title": "Event A", "category": "NEWS", "_step": 2, "sentiment_score": 0.1},
            {"title": "Event B", "category": "MACRO", "_step": 2, "sentiment_score": 0.2},
        ]
        await agent.decide(
            {"symbol": "BBRI", "ohlcv": {"close": 5000}, "pct_change_1d": 0, "pct_change_5d": 0},
            events, step=2,
        )
        call_args = mock_router.call.call_args
        user_prompt = call_args.kwargs.get("user_prompt", call_args[1] if len(call_args) > 1 else "")
        # Both events should be present since latency=0
        assert "Event A" in user_prompt
