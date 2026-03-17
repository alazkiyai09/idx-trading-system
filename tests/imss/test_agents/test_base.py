"""Test agent base classes and working memory."""

import pytest
from imss.agents.base import AgentAction, WorkingMemory, round_to_lot, round_to_tick


def test_round_to_lot_rounds_down():
    assert round_to_lot(150) == 100
    assert round_to_lot(99) == 0
    assert round_to_lot(200) == 200
    assert round_to_lot(1050) == 1000


def test_round_to_tick():
    assert round_to_tick(100) == 100    # tick=1
    assert round_to_tick(303) == 302    # tick=2
    assert round_to_tick(1003) == 1000  # tick=5
    assert round_to_tick(3007) == 3000  # tick=10
    assert round_to_tick(5013) == 5000  # tick=25
    assert round_to_tick(5025) == 5025  # tick=25


def test_working_memory_portfolio_value():
    wm = WorkingMemory(cash=1_000_000)
    wm.holdings = {"BBRI": 100}
    prices = {"BBRI": 5000}
    value = wm.compute_portfolio_value(prices)
    assert value == 1_000_000 + 100 * 5000


def test_working_memory_recent_actions_capped():
    wm = WorkingMemory(cash=1_000_000)
    for i in range(15):
        wm.add_action(AgentAction(
            agent_id="test", step=i, action="HOLD", stock="BBRI",
            quantity=0, confidence=0.5, reasoning="test",
            sentiment_update=0.0,
        ))
    assert len(wm.recent_actions) == 10


def test_agent_action_defaults():
    action = AgentAction(
        agent_id="pak_budi", step=1, action="BUY", stock="BBRI",
        quantity=100, confidence=0.7, reasoning="test",
        sentiment_update=0.3,
    )
    assert action.price == 0.0
