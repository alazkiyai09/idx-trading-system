"""Simulation engine orchestrator."""

from __future__ import annotations

import json
import logging
import uuid
from datetime import date, datetime
from typing import Any, Callable

import pandas as pd
from pydantic import BaseModel
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from imss.agents.base import BaseAgent
from imss.agents.tier1.personas import create_tier1_agent
from imss.agents.tier2.archetypes import create_tier2_agents
from imss.agents.tier3.heuristic import create_tier3_agents
from imss.config import IMSSSettings, SimulationConfig
from imss.db.models import (
    AgentConfig,
    Base,
    Event,
    SimulationRun,
    SimulationStepLog,
    StockOHLCV,
)
from imss.llm.batcher import LLMBatcher
from imss.llm.router import LLMRouter
from imss.simulation.loop import MarketData, run_simulation_loop

logger = logging.getLogger(__name__)


class AgentSummary(BaseModel):
    id: str
    tier: int
    persona_type: str
    final_cash: float
    holdings: dict[str, int]
    pnl_pct: float


class SimulationResult(BaseModel):
    simulation_id: str
    status: str
    total_steps: int
    agents_final: list[AgentSummary]
    total_llm_calls: int
    total_tokens_used: int
    estimated_cost_usd: float
    json_parse_success_rate: float
    step_count: int


class SimulationEngine:
    """Main orchestrator for running IMSS simulations."""

    def __init__(self, settings: IMSSSettings | None = None):
        self.settings = settings or IMSSSettings()
        self._router = LLMRouter(self.settings)
        self._batcher = LLMBatcher(self._router, self.settings.max_concurrent_llm_calls)

    async def run_single(
        self,
        config: SimulationConfig,
        run_number: int = 0,
        on_step: Callable | None = None,
    ) -> SimulationResult:
        """Execute a single backtest simulation run."""

        engine = create_async_engine(self.settings.database_url, echo=False)
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        session_factory = async_sessionmaker(engine, expire_on_commit=False)

        # 1. Initialize agents
        agents: list[BaseAgent] = []

        for persona_key in config.tier1_personas:
            agent = create_tier1_agent(persona_key)
            agent.set_router(self._router)
            agents.append(agent)

        for archetype in config.tier2_archetypes:
            t2_agents = create_tier2_agents(archetype, config.tier2_per_archetype)
            for a in t2_agents:
                a.set_router(self._router)
            agents.extend(t2_agents)

        t3_agents = create_tier3_agents(
            total=config.tier3_total,
            distribution=config.tier3_distribution,
        )
        agents.extend(t3_agents)

        logger.info(
            "Initialized %d agents: %d T1, %d T2, %d T3",
            len(agents),
            len([a for a in agents if a.tier == 1]),
            len([a for a in agents if a.tier == 2]),
            len([a for a in agents if a.tier == 3]),
        )

        # 2. Load market data
        async with session_factory() as session:
            start_dt = datetime.strptime(config.backtest_start, "%Y-%m-%d")
            end_dt = datetime.strptime(config.backtest_end, "%Y-%m-%d")
            stmt = (
                select(StockOHLCV)
                .where(StockOHLCV.symbol == config.target_stocks[0])
                .where(StockOHLCV.timestamp >= start_dt)
                .where(StockOHLCV.timestamp <= end_dt)
                .order_by(StockOHLCV.timestamp)
            )
            result = await session.execute(stmt)
            rows = result.scalars().all()

        if not rows:
            logger.error("No price data found for %s", config.target_stocks[0])
            await engine.dispose()
            return SimulationResult(
                simulation_id="", status="FAILED", total_steps=0,
                agents_final=[], total_llm_calls=0, total_tokens_used=0,
                estimated_cost_usd=0, json_parse_success_rate=0, step_count=0,
            )

        # Build DataFrame
        data = [{
            "Open": r.open, "High": r.high, "Low": r.low, "Close": r.close,
            "Volume": r.volume, "Adj Close": r.adjusted_close,
        } for r in rows]
        index = [r.timestamp for r in rows]
        df = pd.DataFrame(data, index=pd.DatetimeIndex(index))
        market_data = MarketData(df, config.target_stocks[0])

        # 3. Load events
        async with session_factory() as session:
            stmt = (
                select(Event)
                .where(Event.timestamp >= start_dt)
                .where(Event.timestamp <= end_dt)
                .order_by(Event.timestamp)
            )
            result = await session.execute(stmt)
            event_rows = result.scalars().all()

        events_by_date: dict[date, list[dict]] = {}
        for evt in event_rows:
            d = evt.timestamp.date()
            if d not in events_by_date:
                events_by_date[d] = []
            events_by_date[d].append({
                "id": evt.id,
                "category": evt.category,
                "title": evt.title,
                "summary": evt.summary,
                "sentiment_score": evt.sentiment_score,
                "magnitude_score": evt.magnitude_score,
            })

        trading_days = [d for d in market_data.trading_days
                        if date.fromisoformat(config.backtest_start) <= d <= date.fromisoformat(config.backtest_end)]

        # 4. Create simulation run record
        sim_id = str(uuid.uuid4())
        async with session_factory() as session:
            async with session.begin():
                session.add(SimulationRun(
                    id=sim_id,
                    config_json=config.model_dump_json(),
                    mode=config.mode,
                    status="RUNNING",
                ))
                # Store agent configs
                for agent in agents:
                    session.add(AgentConfig(
                        simulation_id=sim_id,
                        agent_id=agent.id,
                        tier=agent.tier,
                        persona_type=agent.persona_type,
                        parameters_json=json.dumps({
                            "cash": agent.working_memory.cash,
                            "holdings": agent.working_memory.holdings,
                        }),
                    ))

        # 5. Run simulation loop
        step_logs = await run_simulation_loop(
            agents=agents,
            market_data=market_data,
            events_by_date=events_by_date,
            trading_days=trading_days,
            router=self._router,
            batcher=self._batcher,
            on_step=on_step,
        )

        # 6. Save step logs (batched write)
        async with session_factory() as session:
            async with session.begin():
                for log in step_logs:
                    session.add(SimulationStepLog(
                        simulation_id=sim_id,
                        run_number=run_number,
                        step_number=log["step_number"],
                        simulated_date=date.fromisoformat(log["simulated_date"]),
                        market_state_json=json.dumps(log["market_state"]),
                        agent_actions_json=json.dumps(log["agent_actions"]),
                        events_active_json=json.dumps(log["events_active"]),
                        aggregate_sentiment=log["aggregate_sentiment"],
                        aggregate_order_imbalance=log["aggregate_order_imbalance"],
                    ))

        # 7. Finalize
        cost_tracker = self._router.cost_tracker
        final_prices = {market_data.symbol: float(df.iloc[-1]["Close"])}

        agent_summaries = []
        for agent in agents:
            pv = agent.working_memory.compute_portfolio_value(final_prices)
            agent_summaries.append(AgentSummary(
                id=agent.id, tier=agent.tier, persona_type=agent.persona_type,
                final_cash=agent.working_memory.cash,
                holdings=dict(agent.working_memory.holdings),
                pnl_pct=0.0,  # Detailed P&L in Phase 2
            ))

        async with session_factory() as session:
            async with session.begin():
                await session.execute(
                    update(SimulationRun)
                    .where(SimulationRun.id == sim_id)
                    .values(
                        status="COMPLETED",
                        completed_at=datetime.utcnow(),
                        total_llm_calls=cost_tracker.total_calls,
                        total_tokens_used=cost_tracker.total_tokens,
                        estimated_cost_usd=cost_tracker.estimated_cost_usd,
                    )
                )

        await engine.dispose()

        return SimulationResult(
            simulation_id=sim_id,
            status="COMPLETED",
            total_steps=len(trading_days),
            agents_final=agent_summaries,
            total_llm_calls=cost_tracker.total_calls,
            total_tokens_used=cost_tracker.total_tokens,
            estimated_cost_usd=cost_tracker.estimated_cost_usd,
            json_parse_success_rate=cost_tracker.json_parse_rate,
            step_count=len(step_logs),
        )
