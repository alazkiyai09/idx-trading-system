"""SQLAlchemy async models for IMSS."""

from __future__ import annotations

import uuid
from datetime import date, datetime

from sqlalchemy import (
    BigInteger,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.ext.asyncio import AsyncAttrs, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(AsyncAttrs, DeclarativeBase):
    pass


class StockOHLCV(Base):
    __tablename__ = "stocks_ohlcv"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(10), index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, index=True)
    open: Mapped[float] = mapped_column(Float)
    high: Mapped[float] = mapped_column(Float)
    low: Mapped[float] = mapped_column(Float)
    close: Mapped[float] = mapped_column(Float)
    volume: Mapped[int] = mapped_column(BigInteger)
    adjusted_close: Mapped[float] = mapped_column(Float)

    __table_args__ = (
        UniqueConstraint("symbol", "timestamp", name="uq_ohlcv_symbol_ts"),
    )


class Event(Base):
    __tablename__ = "events"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    timestamp: Mapped[datetime] = mapped_column(DateTime, index=True)
    category: Mapped[str] = mapped_column(String(20))
    source: Mapped[str] = mapped_column(String(100))
    title: Mapped[str] = mapped_column(Text)
    summary: Mapped[str] = mapped_column(Text)
    raw_content: Mapped[str | None] = mapped_column(Text, nullable=True)
    sentiment_score: Mapped[float] = mapped_column(Float)
    magnitude_score: Mapped[float] = mapped_column(Float)
    embedding_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow
    )


class EventEntity(Base):
    __tablename__ = "event_entities"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("events.id"), index=True
    )
    entity_type: Mapped[str] = mapped_column(String(10))  # STOCK, SECTOR
    entity_symbol: Mapped[str] = mapped_column(String(20))


class CausalLink(Base):
    __tablename__ = "causal_links"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    event_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("events.id"), nullable=True
    )
    event_category: Mapped[str] = mapped_column(String(20))
    stock_symbol: Mapped[str] = mapped_column(String(10))
    lag_days: Mapped[int] = mapped_column(Integer)
    direction: Mapped[str] = mapped_column(String(10))
    correlation_strength: Mapped[float] = mapped_column(Float)
    confidence: Mapped[float] = mapped_column(Float)
    sample_count: Mapped[int] = mapped_column(Integer, default=0)
    last_updated: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow
    )


class SimulationRun(Base):
    __tablename__ = "simulation_runs"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    config_json: Mapped[str] = mapped_column(Text)
    mode: Mapped[str] = mapped_column(String(10))
    status: Mapped[str] = mapped_column(String(20), default="PENDING")
    started_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    results_summary_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    total_llm_calls: Mapped[int] = mapped_column(Integer, default=0)
    total_tokens_used: Mapped[int] = mapped_column(Integer, default=0)
    estimated_cost_usd: Mapped[float] = mapped_column(Float, default=0.0)


class AgentConfig(Base):
    __tablename__ = "agent_configs"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    simulation_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("simulation_runs.id"), index=True
    )
    agent_id: Mapped[str] = mapped_column(String(50))
    tier: Mapped[int] = mapped_column(Integer)
    persona_type: Mapped[str] = mapped_column(String(50))
    parameters_json: Mapped[str] = mapped_column(Text)


class SimulationStepLog(Base):
    __tablename__ = "simulation_step_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    simulation_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("simulation_runs.id"), index=True
    )
    run_number: Mapped[int] = mapped_column(Integer)
    step_number: Mapped[int] = mapped_column(Integer)
    simulated_date: Mapped[date] = mapped_column(Date)
    market_state_json: Mapped[str] = mapped_column(Text)
    agent_actions_json: Mapped[str] = mapped_column(Text)
    events_active_json: Mapped[str] = mapped_column(Text)
    aggregate_sentiment: Mapped[float] = mapped_column(Float, default=0.0)
    aggregate_order_imbalance: Mapped[float] = mapped_column(Float, default=0.0)

    __table_args__ = (
        Index("ix_step_sim_run_step", "simulation_id", "run_number", "step_number"),
    )


class StockFundamentals(Base):
    __tablename__ = "stock_fundamentals"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(10), index=True)
    period: Mapped[str] = mapped_column(String(10))  # e.g. "2024-Q2"
    pe_ratio: Mapped[float] = mapped_column(Float)
    pb_ratio: Mapped[float] = mapped_column(Float)
    dividend_yield_pct: Mapped[float] = mapped_column(Float)
    roe_pct: Mapped[float] = mapped_column(Float)
    market_cap_trillion_idr: Mapped[float] = mapped_column(Float)


# --- Database initialization ---

async def get_engine(database_url: str):
    """Create async engine."""
    return create_async_engine(database_url, echo=False)


async def get_session_factory(engine) -> async_sessionmaker:
    """Create session factory."""
    return async_sessionmaker(engine, expire_on_commit=False)


async def create_tables(database_url: str) -> None:
    """Create all tables. Idempotent."""
    engine = create_async_engine(database_url, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    await engine.dispose()
