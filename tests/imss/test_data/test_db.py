"""Test IMSS database model creation."""

import pytest
from sqlalchemy import inspect
from sqlalchemy.ext.asyncio import create_async_engine

from imss.db.models import Base, create_tables


@pytest.mark.asyncio
async def test_create_tables_creates_all_expected_tables(tmp_path):
    """All 8 IMSS tables are created in SQLite."""
    db_path = tmp_path / "test.db"
    url = f"sqlite+aiosqlite:///{db_path}"
    await create_tables(url)

    engine = create_async_engine(url)
    async with engine.connect() as conn:
        table_names = await conn.run_sync(
            lambda sync_conn: inspect(sync_conn).get_table_names()
        )
    await engine.dispose()

    expected = {
        "stocks_ohlcv",
        "events",
        "event_entities",
        "causal_links",
        "simulation_runs",
        "agent_configs",
        "simulation_step_logs",
        "stock_fundamentals",
    }
    assert expected == set(table_names)


@pytest.mark.asyncio
async def test_stock_fundamentals_table_exists(tmp_path):
    """StockFundamentals table is created alongside other tables."""
    url = f"sqlite+aiosqlite:///{tmp_path}/test.db"
    from sqlalchemy.ext.asyncio import create_async_engine
    from imss.db.models import Base

    engine = create_async_engine(url)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with engine.connect() as conn:
        result = await conn.run_sync(
            lambda sync_conn: sync_conn.execute(
                __import__("sqlalchemy").text(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='stock_fundamentals'"
                )
            ).fetchone()
        )
    assert result is not None, "stock_fundamentals table should exist"
    await engine.dispose()
