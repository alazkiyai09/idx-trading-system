"""Tests for Phase 2B Tier 1 agents: Dr. Lim and MarketBot."""

import pytest
from imss.agents.tier1.personas import PERSONAS, create_tier1_agent


class TestDrLimPersona:
    def test_dr_lim_exists_in_personas(self):
        assert "dr_lim" in PERSONAS

    def test_dr_lim_config_fields(self):
        pc = PERSONAS["dr_lim"]
        assert pc.name == "Dr. Lim"
        assert pc.risk_tolerance == 0.4
        assert pc.stop_loss_pct == 20
        assert pc.initial_cash == 5_000_000_000
        assert pc.initial_holdings == {"BBRI": 300_000}

    def test_dr_lim_agent_creation(self):
        agent = create_tier1_agent("dr_lim")
        assert agent.tier == 1
        assert agent.persona_type == "tier1_dr_lim"
        assert agent.working_memory.cash == 5_000_000_000
        assert agent.working_memory.holdings["BBRI"] == 300_000


class TestTier1PromptFundamentals:
    def test_prompt_includes_fundamentals_when_provided(self):
        from imss.llm.prompts.tier1_decision import build_tier1_user_prompt

        fundamentals = {
            "pe_ratio": 12.5,
            "pb_ratio": 2.1,
            "dividend_yield_pct": 5.2,
            "roe_pct": 18.3,
            "market_cap_trillion_idr": 620,
        }
        prompt = build_tier1_user_prompt(
            step=1, simulated_date="2024-07-01", cash=5e9,
            holdings_formatted="BBRI: 300,000", portfolio_value=6.5e9,
            unrealized_pnl=2.5, stock_symbol="BBRI",
            ohlcv={"open": 5100, "high": 5200, "low": 5050, "close": 5150, "volume": 80000000},
            pct_change_5d=1.5, pct_change_20d=3.2,
            events_formatted="No events",
            fundamentals=fundamentals,
        )
        assert "P/E Ratio: 12.5" in prompt
        assert "Dividend Yield: 5.2%" in prompt
        assert "ROE: 18.3%" in prompt

    def test_prompt_omits_fundamentals_when_none(self):
        from imss.llm.prompts.tier1_decision import build_tier1_user_prompt

        prompt = build_tier1_user_prompt(
            step=1, simulated_date="2024-07-01", cash=5e9,
            holdings_formatted="None", portfolio_value=5e9,
            unrealized_pnl=0.0, stock_symbol="BBRI",
            ohlcv={"open": 5100, "high": 5200, "low": 5050, "close": 5150, "volume": 80000000},
            pct_change_5d=0.0, pct_change_20d=0.0,
            events_formatted="",
            fundamentals=None,
        )
        assert "Fundamental Data" not in prompt


from unittest.mock import AsyncMock, patch
from imss.llm.router import LLMResponse


class TestMarketBotAgent:
    def test_marketbot_exists_in_personas(self):
        assert "marketbot" in PERSONAS

    def test_marketbot_config_fields(self):
        pc = PERSONAS["marketbot"]
        assert pc.name == "MarketBot"
        assert pc.risk_tolerance == 0.2
        assert pc.stop_loss_pct == 1
        assert pc.initial_cash == 20_000_000_000
        assert pc.initial_holdings == {"BBRI": 200_000}

    @pytest.mark.asyncio
    async def test_marketbot_sells_on_positive_imbalance(self):
        from imss.agents.tier1.marketbot import MarketBotAgent
        agent = create_tier1_agent("marketbot")
        assert isinstance(agent, MarketBotAgent)

        mock_response = LLMResponse(
            content='{"action": "SELL", "stock": "BBRI", "quantity": 1000, "confidence": 0.7, "reasoning": "providing liquidity", "sentiment_update": 0.0}',
            parsed_json={"action": "SELL", "stock": "BBRI", "quantity": 1000, "confidence": 0.7, "reasoning": "providing liquidity", "sentiment_update": 0.0},
            input_tokens=100, output_tokens=50, model="glm-5", latency_ms=100,
        )
        mock_router = AsyncMock()
        mock_router.call = AsyncMock(return_value=mock_response)
        agent.set_router(mock_router)

        market_state = {
            "symbol": "BBRI", "date": "2024-07-01",
            "ohlcv": {"open": 5100, "high": 5200, "low": 5050, "close": 5150, "volume": 80000000},
            "prices": {"BBRI": 5150}, "price_history": [5100, 5150],
            "volume_history": [80000000], "pct_change_1d": 0.5,
            "pct_change_5d": 1.0, "pct_change_20d": 2.0,
            "prev_aggregate_order_imbalance": 0.5,
        }
        action = await agent.decide(market_state, [], 1)
        assert action.action == "SELL"

    @pytest.mark.asyncio
    async def test_marketbot_buys_on_negative_imbalance(self):
        from imss.agents.tier1.marketbot import MarketBotAgent
        agent = create_tier1_agent("marketbot")

        mock_response = LLMResponse(
            content='{"action": "BUY", "stock": "BBRI", "quantity": 500, "confidence": 0.6, "reasoning": "absorbing sell pressure", "sentiment_update": 0.0}',
            parsed_json={"action": "BUY", "stock": "BBRI", "quantity": 500, "confidence": 0.6, "reasoning": "absorbing sell pressure", "sentiment_update": 0.0},
            input_tokens=100, output_tokens=50, model="glm-5", latency_ms=100,
        )
        mock_router = AsyncMock()
        mock_router.call = AsyncMock(return_value=mock_response)
        agent.set_router(mock_router)

        market_state = {
            "symbol": "BBRI", "date": "2024-07-01",
            "ohlcv": {"open": 5100, "high": 5200, "low": 5050, "close": 5150, "volume": 80000000},
            "prices": {"BBRI": 5150}, "price_history": [5100, 5150],
            "volume_history": [80000000], "pct_change_1d": -0.5,
            "pct_change_5d": -1.0, "pct_change_20d": -2.0,
            "prev_aggregate_order_imbalance": -0.5,
        }
        action = await agent.decide(market_state, [], 1)
        assert action.action == "BUY"

    @pytest.mark.asyncio
    async def test_marketbot_holds_on_near_zero_imbalance(self):
        from imss.agents.tier1.marketbot import MarketBotAgent
        agent = create_tier1_agent("marketbot")
        # No router needed — HOLD doesn't call LLM

        market_state = {
            "symbol": "BBRI", "date": "2024-07-01",
            "ohlcv": {"open": 5100, "high": 5200, "low": 5050, "close": 5150, "volume": 80000000},
            "prices": {"BBRI": 5150}, "price_history": [5100, 5150],
            "volume_history": [80000000], "pct_change_1d": 0.0,
            "pct_change_5d": 0.0, "pct_change_20d": 0.0,
            "prev_aggregate_order_imbalance": 0.05,
        }
        action = await agent.decide(market_state, [], 1)
        assert action.action == "HOLD"

    @pytest.mark.asyncio
    async def test_marketbot_reduces_sizing_in_high_volatility(self):
        """Volatility gate halves quantity when 5-day change exceeds 5%."""
        agent = create_tier1_agent("marketbot")

        mock_response = LLMResponse(
            content='{"action": "SELL", "stock": "BBRI", "quantity": 2000, "confidence": 0.7, "reasoning": "providing liquidity", "sentiment_update": 0.0}',
            parsed_json={"action": "SELL", "stock": "BBRI", "quantity": 2000, "confidence": 0.7, "reasoning": "providing liquidity", "sentiment_update": 0.0},
            input_tokens=100, output_tokens=50, model="glm-5", latency_ms=100,
        )
        mock_router = AsyncMock()
        mock_router.call = AsyncMock(return_value=mock_response)
        agent.set_router(mock_router)

        market_state = {
            "symbol": "BBRI", "date": "2024-07-01",
            "ohlcv": {"open": 5100, "high": 5200, "low": 5050, "close": 5150, "volume": 80000000},
            "prices": {"BBRI": 5150},
            "pct_change_5d": 7.0,  # > 5% triggers volatility gate
            "prev_aggregate_order_imbalance": 0.5,
        }
        action = await agent.decide(market_state, [], 1)
        assert action.action == "SELL"
        assert action.quantity == 1000  # halved from 2000, still lot-aligned
