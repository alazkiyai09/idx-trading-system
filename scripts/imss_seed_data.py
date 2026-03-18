"""Seed IMSS database with BBRI price data and curated events.

Usage:
    python3 scripts/imss_seed_data.py
"""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.progress import Progress
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from imss.config import get_settings
from imss.data.price_feed import fetch_idx_prices, store_prices
from imss.db.models import Base, Event, EventEntity

console = Console()
SEED_EVENTS_PATH = Path("data/seed_events/bbri_events_2024.json")


async def seed_database() -> None:
    settings = get_settings()

    # 1. Create tables
    console.print("[bold blue]Creating database tables...[/]")
    engine = create_async_engine(settings.database_url, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    # 2. Download price data
    console.print("[bold blue]Downloading BBRI.JK price data (2024-01-01 to 2025-12-31)...[/]")
    price_data = fetch_idx_prices(["BBRI"], "2024-01-01", "2025-12-31")

    if "BBRI" in price_data:
        async with session_factory() as session:
            async with session.begin():
                count = await store_prices(session, "BBRI", price_data["BBRI"])
        console.print(f"[green]Stored {count} price rows for BBRI[/]")
    else:
        console.print("[red]Failed to download BBRI price data[/]")
        return

    # 3. Load and store events
    console.print("[bold blue]Loading seed events...[/]")
    events_raw = json.loads(SEED_EVENTS_PATH.read_text())

    # 4. Store events in DB (ChromaDB embeddings deferred — requires embedding API)
    with Progress() as progress:
        task = progress.add_task("Processing events...", total=len(events_raw))
        async with session_factory() as session:
            async with session.begin():
                for evt_data in events_raw:
                    event_id = str(uuid.uuid4())
                    ts = datetime.fromisoformat(evt_data["timestamp"])

                    event = Event(
                        id=event_id,
                        timestamp=ts,
                        category=evt_data["category"],
                        source=evt_data["source"],
                        title=evt_data["title"],
                        summary=evt_data["summary"],
                        sentiment_score=evt_data["sentiment_score"],
                        magnitude_score=evt_data["magnitude_score"],
                        embedding_id=event_id,
                    )
                    session.add(event)

                    for entity in evt_data.get("affected_entities", []):
                        session.add(
                            EventEntity(
                                event_id=event_id,
                                entity_type=entity["type"],
                                entity_symbol=entity["symbol"],
                            )
                        )

                    progress.advance(task)

    console.print(f"[green]Loaded {len(events_raw)} events into DB[/]")

    # 6. Load fundamentals
    fundamentals_path = Path("data/seed_events/bbri_fundamentals.json")
    if fundamentals_path.exists():
        console.print("[bold blue]Loading fundamentals data...[/]")
        fund_data = json.loads(fundamentals_path.read_text())
        from imss.db.models import StockFundamentals
        async with session_factory() as session:
            async with session.begin():
                session.add(StockFundamentals(
                    symbol=fund_data["symbol"],
                    period=fund_data["period"],
                    pe_ratio=fund_data["pe_ratio"],
                    pb_ratio=fund_data["pb_ratio"],
                    dividend_yield_pct=fund_data["dividend_yield_pct"],
                    roe_pct=fund_data["roe_pct"],
                    market_cap_trillion_idr=fund_data["market_cap_trillion_idr"],
                ))
        console.print("[green]Loaded fundamentals for BBRI[/]")

    await engine.dispose()
    console.print("[bold green]Seed complete![/]")


if __name__ == "__main__":
    asyncio.run(seed_database())
