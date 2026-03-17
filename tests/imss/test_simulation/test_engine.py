"""Integration test for simulation engine with mocked LLM."""

import pytest
from unittest.mock import AsyncMock, patch
from datetime import datetime

from imss.config import SimulationConfig
from imss.simulation.engine import SimulationEngine
from imss.llm.router import LLMResponse


MOCK_LLM_RESPONSE = LLMResponse(
    content='{"action": "HOLD", "stock": "BBRI", "quantity": 0, "confidence": 0.5, "reasoning": "test hold", "sentiment_update": 0.0}',
    parsed_json={"action": "HOLD", "stock": "BBRI", "quantity": 0, "confidence": 0.5, "reasoning": "test hold", "sentiment_update": 0.0},
    input_tokens=100,
    output_tokens=50,
    model="glm-5",
    latency_ms=100,
)


@pytest.mark.asyncio
async def test_engine_runs_with_mocked_llm(tmp_path):
    """Engine completes a minimal run with mocked LLM and file-based DB."""
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
    )

    with patch.dict("os.environ", {"GLM_API_KEY": "test", "IMSS_DATABASE_URL": db_url}):
        sim_engine = SimulationEngine()
        # Mock the LLM router
        sim_engine._router.call = AsyncMock(return_value=MOCK_LLM_RESPONSE)

        result = await sim_engine.run_single(config)

    assert result.status == "COMPLETED"
    assert result.step_count >= 3  # at least some trading days
    for agent in result.agents_final:
        assert agent.final_cash >= 0
