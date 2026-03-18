"""Integration test for multi-run simulation."""

import pytest
from unittest.mock import AsyncMock, patch
from datetime import datetime

from imss.config import SimulationConfig
from imss.simulation.engine import SimulationEngine
from imss.simulation.aggregator import MultiRunResult
from imss.llm.router import LLMResponse


MOCK_LLM_RESPONSE = LLMResponse(
    content='{"action": "HOLD", "stock": "BBRI", "quantity": 0, "confidence": 0.5, "reasoning": "test hold", "sentiment_update": 0.0}',
    parsed_json={"action": "HOLD", "stock": "BBRI", "quantity": 0, "confidence": 0.5, "reasoning": "test hold", "sentiment_update": 0.0},
    input_tokens=100, output_tokens=50, model="glm-5", latency_ms=100,
)


@pytest.mark.asyncio
async def test_run_multi_two_runs(tmp_path):
    """run_multi completes 2 runs and returns aggregated result."""
    db_url = f"sqlite+aiosqlite:///{tmp_path}/test.db"

    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
    from imss.db.models import Base, StockOHLCV

    engine_db = create_async_engine(db_url)
    async with engine_db.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    sf = async_sessionmaker(engine_db, expire_on_commit=False)
    async with sf() as session:
        async with session.begin():
            for d, c in [
                ("2024-07-01", 5150), ("2024-07-02", 5175), ("2024-07-03", 5200),
                ("2024-07-04", 5225), ("2024-07-05", 5250),
            ]:
                session.add(StockOHLCV(
                    symbol="BBRI", timestamp=datetime.fromisoformat(d),
                    open=c - 25, high=c + 25, low=c - 50, close=c,
                    volume=85_000_000, adjusted_close=c,
                ))
    await engine_db.dispose()

    config = SimulationConfig(
        target_stocks=["BBRI"],
        backtest_start="2024-07-01",
        backtest_end="2024-07-05",
        tier1_personas=["pak_budi"],
        tier2_per_archetype=0,
        tier2_archetypes=[],
        tier3_total=5,
        num_parallel_runs=2,
    )

    with patch.dict("os.environ", {"GLM_API_KEY": "test", "IMSS_DATABASE_URL": db_url}):
        sim_engine = SimulationEngine()
        sim_engine._router.call = AsyncMock(return_value=MOCK_LLM_RESPONSE)

        result = await sim_engine.run_multi(config)

    assert isinstance(result, MultiRunResult)
    assert result.num_runs == 2
    assert len(result.individual_results) == 2
    assert result.individual_results[0].run_number == 0
    assert result.individual_results[1].run_number == 1
    assert result.total_batch_cost_usd >= 0
    assert "tier1_pak_budi" in result.agent_stats


@pytest.mark.asyncio
async def test_different_seeds_produce_different_tier3(tmp_path):
    """Different run numbers produce different Tier 3 random agent behavior."""
    db_url = f"sqlite+aiosqlite:///{tmp_path}/test.db"

    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
    from imss.db.models import Base, StockOHLCV

    engine_db = create_async_engine(db_url)
    async with engine_db.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    sf = async_sessionmaker(engine_db, expire_on_commit=False)
    async with sf() as session:
        async with session.begin():
            for d, c in [
                ("2024-07-01", 5150), ("2024-07-02", 5175), ("2024-07-03", 5200),
            ]:
                session.add(StockOHLCV(
                    symbol="BBRI", timestamp=datetime.fromisoformat(d),
                    open=c - 25, high=c + 25, low=c - 50, close=c,
                    volume=85_000_000, adjusted_close=c,
                ))
    await engine_db.dispose()

    config = SimulationConfig(
        target_stocks=["BBRI"],
        backtest_start="2024-07-01",
        backtest_end="2024-07-03",
        tier1_personas=[],
        tier2_per_archetype=0,
        tier2_archetypes=[],
        tier3_total=10,
        num_parallel_runs=2,
    )

    with patch.dict("os.environ", {"GLM_API_KEY": "test", "IMSS_DATABASE_URL": db_url}):
        sim_engine = SimulationEngine()

        result = await sim_engine.run_multi(config)

    # Different seeds cause different initial cash for Tier 3 agents (via create_tier3_agents seed param).
    # Note: RandomWalkAgent._get_rng() uses hash(self.id) not the run seed, so the
    # difference comes from initial cash, not from action randomness.
    r0_cash = {a.id: a.final_cash for a in result.individual_results[0].agents_final}
    r1_cash = {a.id: a.final_cash for a in result.individual_results[1].agents_final}
    differences = sum(1 for aid in r0_cash if abs(r0_cash.get(aid, 0) - r1_cash.get(aid, 0)) > 1.0)
    assert differences > 0, "Different seeds should produce different Tier 3 initial cash → different outcomes"
