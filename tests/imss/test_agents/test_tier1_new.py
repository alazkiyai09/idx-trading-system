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
