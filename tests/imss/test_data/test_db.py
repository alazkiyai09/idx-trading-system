"""Test IMSS database model creation."""

import pytest
from sqlalchemy import inspect
from sqlalchemy.ext.asyncio import create_async_engine

from imss.db.models import Base, create_tables


@pytest.mark.asyncio
async def test_create_tables_creates_all_expected_tables(tmp_path):
    """All 7 IMSS tables are created in SQLite."""
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
    }
    assert expected == set(table_names)
