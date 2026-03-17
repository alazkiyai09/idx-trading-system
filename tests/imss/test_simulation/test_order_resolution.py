"""Test order resolution for backtest mode."""

import pytest
from imss.simulation.order_book import resolve_backtest_orders
from imss.agents.base import AgentAction, WorkingMemory, BaseAgent
from typing import Any


class DummyAgent(BaseAgent):
    tier: int = 3
    async def decide(self, market_state, events, step):
        return AgentAction(agent_id=self.id, step=step, action="HOLD", stock="BBRI")


def _make_agent(cash: float, holdings: dict | None = None) -> DummyAgent:
    return DummyAgent(
        id="test", name="Test", persona_type="test",
        working_memory=WorkingMemory(cash=cash, holdings=holdings or {}),
    )


def test_buy_order_fills_at_close():
    agent = _make_agent(cash=1_000_000)
    action = AgentAction(agent_id="test", step=1, action="BUY", stock="BBRI", quantity=100, confidence=0.8, reasoning="test")
    result = resolve_backtest_orders([agent], [action], close_prices={"BBRI": 5000})
    assert agent.working_memory.cash == 1_000_000 - (100 * 5000)
    assert agent.working_memory.holdings["BBRI"] == 100
    assert result[0].price == 5000


def test_buy_rejected_insufficient_cash():
    agent = _make_agent(cash=100_000)
    action = AgentAction(agent_id="test", step=1, action="BUY", stock="BBRI", quantity=100, confidence=0.8, reasoning="test")
    result = resolve_backtest_orders([agent], [action], close_prices={"BBRI": 5000})
    assert agent.working_memory.cash == 100_000  # unchanged
    assert agent.working_memory.holdings.get("BBRI", 0) == 0


def test_sell_order_fills():
    agent = _make_agent(cash=0, holdings={"BBRI": 500})
    action = AgentAction(agent_id="test", step=1, action="SELL", stock="BBRI", quantity=200, confidence=0.8, reasoning="test")
    result = resolve_backtest_orders([agent], [action], close_prices={"BBRI": 5000})
    assert agent.working_memory.cash == 200 * 5000
    assert agent.working_memory.holdings["BBRI"] == 300


def test_sell_capped_to_holdings():
    agent = _make_agent(cash=0, holdings={"BBRI": 100})
    action = AgentAction(agent_id="test", step=1, action="SELL", stock="BBRI", quantity=500, confidence=0.8, reasoning="test")
    result = resolve_backtest_orders([agent], [action], close_prices={"BBRI": 5000})
    assert agent.working_memory.holdings["BBRI"] == 0
    assert agent.working_memory.cash == 100 * 5000


def test_hold_action_no_change():
    agent = _make_agent(cash=1_000_000, holdings={"BBRI": 100})
    action = AgentAction(agent_id="test", step=1, action="HOLD", stock="BBRI", quantity=0, confidence=0.5, reasoning="wait")
    resolve_backtest_orders([agent], [action], close_prices={"BBRI": 5000})
    assert agent.working_memory.cash == 1_000_000
    assert agent.working_memory.holdings["BBRI"] == 100


def test_lot_size_enforcement():
    agent = _make_agent(cash=10_000_000)
    action = AgentAction(agent_id="test", step=1, action="BUY", stock="BBRI", quantity=150, confidence=0.8, reasoning="test")
    result = resolve_backtest_orders([agent], [action], close_prices={"BBRI": 5000})
    # Should round 150 down to 100
    assert agent.working_memory.holdings.get("BBRI", 0) == 100
