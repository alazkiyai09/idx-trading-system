"""Test Tier 3 rule-based heuristic agents."""

import pytest
from imss.agents.tier3.heuristic import (
    MomentumFollower,
    MeanReversion,
    RandomWalkAgent,
    VolumeFollower,
    create_tier3_agents,
)
from imss.agents.base import WorkingMemory, round_to_lot


def _make_market_state(closes: list[float], volumes: list[int] | None = None) -> dict:
    """Helper: build market_state from price history."""
    if volumes is None:
        volumes = [80_000_000] * len(closes)
    return {
        "symbol": "BBRI",
        "ohlcv": {"open": closes[-1], "high": closes[-1] + 50, "low": closes[-1] - 50, "close": closes[-1], "volume": volumes[-1]},
        "price_history": closes,
        "volume_history": volumes,
        "prices": {"BBRI": closes[-1]},
    }


@pytest.mark.asyncio
async def test_momentum_follower_buys_on_uptrend():
    agent = MomentumFollower(
        id="mf_001", name="MF 1", persona_type="tier3_momentum",
        working_memory=WorkingMemory(cash=100_000_000, holdings={}),
    )
    # 5-day uptrend: +4%
    closes = [5000, 5050, 5100, 5150, 5200]
    ms = _make_market_state(closes)
    action = await agent.decide(ms, [], step=5)
    assert action.action == "BUY"
    assert action.quantity > 0
    assert action.quantity % 100 == 0


@pytest.mark.asyncio
async def test_momentum_follower_sells_on_downtrend():
    agent = MomentumFollower(
        id="mf_002", name="MF 2", persona_type="tier3_momentum",
        working_memory=WorkingMemory(cash=50_000_000, holdings={"BBRI": 1000}),
    )
    closes = [5200, 5150, 5100, 5050, 5000]  # -3.8%
    ms = _make_market_state(closes)
    action = await agent.decide(ms, [], step=5)
    assert action.action == "SELL"


@pytest.mark.asyncio
async def test_momentum_follower_holds_on_flat():
    agent = MomentumFollower(
        id="mf_003", name="MF 3", persona_type="tier3_momentum",
        working_memory=WorkingMemory(cash=100_000_000, holdings={}),
    )
    closes = [5000, 5010, 5005, 5015, 5010]  # +0.2%
    ms = _make_market_state(closes)
    action = await agent.decide(ms, [], step=5)
    assert action.action == "HOLD"


@pytest.mark.asyncio
async def test_mean_reversion_buys_on_dip():
    agent = MeanReversion(
        id="mr_001", name="MR 1", persona_type="tier3_mean_reversion",
        working_memory=WorkingMemory(cash=100_000_000, holdings={}),
    )
    # Price well below 20-day MA
    closes = [5200] * 15 + [5100, 5050, 5000, 4950, 4900]
    ms = _make_market_state(closes)
    action = await agent.decide(ms, [], step=20)
    assert action.action == "BUY"


@pytest.mark.asyncio
async def test_all_tier3_quantities_are_lot_aligned():
    agents = create_tier3_agents(total=20, seed=42)
    closes = [5000, 5050, 5100, 5150, 5200]
    ms = _make_market_state(closes)
    for agent in agents:
        action = await agent.decide(ms, [], step=5)
        assert action.quantity % 100 == 0, f"{agent.id} produced non-lot quantity {action.quantity}"


def test_create_tier3_agents_distribution():
    agents = create_tier3_agents(total=50, seed=42)
    assert len(agents) == 50
    types = [a.persona_type for a in agents]
    # round(50*0.30)=15, round(50*0.25)=12, round(50*0.30)=15, round(50*0.15)=8 = 50
    assert types.count("tier3_momentum_follower") == 15
    assert types.count("tier3_mean_reversion") == 12
    assert types.count("tier3_random_walk") == 15
    assert types.count("tier3_volume_follower") == 8
