"""Turn-based simulation loop."""

from __future__ import annotations

import logging
import random
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from imss.agents.base import AgentAction, BaseAgent
from imss.llm.batcher import LLMBatcher
from imss.llm.router import LLMRouter
from imss.simulation.order_book import resolve_backtest_orders
from imss.simulation.propagation import distribute_events

logger = logging.getLogger(__name__)


class MarketData:
    """Wrapper around price DataFrame for simulation access."""

    def __init__(self, df: pd.DataFrame, symbol: str):
        self.df = df.sort_index()
        self.symbol = symbol
        self._dates = [d.date() if hasattr(d, "date") else d for d in self.df.index]

    @property
    def trading_days(self) -> list[date]:
        return self._dates

    def get_ohlcv(self, d: date) -> dict[str, Any] | None:
        matches = self.df[self.df.index.date == d] if hasattr(self.df.index, "date") else self.df.loc[[d]]
        if matches.empty:
            return None
        row = matches.iloc[0]
        return {
            "open": float(row["Open"]),
            "high": float(row["High"]),
            "low": float(row["Low"]),
            "close": float(row["Close"]),
            "volume": int(row["Volume"]),
        }

    def get_price_history(self, d: date, lookback: int = 20) -> list[float]:
        """Get last N close prices up to and including date d."""
        idx = None
        for i, td in enumerate(self._dates):
            if td <= d:
                idx = i
        if idx is None:
            return []
        start = max(0, idx - lookback + 1)
        return [float(self.df.iloc[i]["Close"]) for i in range(start, idx + 1)]

    def get_volume_history(self, d: date, lookback: int = 20) -> list[int]:
        idx = None
        for i, td in enumerate(self._dates):
            if td <= d:
                idx = i
        if idx is None:
            return []
        start = max(0, idx - lookback + 1)
        return [int(self.df.iloc[i]["Volume"]) for i in range(start, idx + 1)]


async def run_simulation_loop(
    agents: list[BaseAgent],
    market_data: MarketData,
    events_by_date: dict[date, list[dict[str, Any]]],
    trading_days: list[date],
    router: LLMRouter,
    batcher: LLMBatcher,
    on_step: Any = None,
    seed: int = 42,
    fundamentals: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Execute the turn-based simulation loop.

    Returns list of step log dicts.
    """
    rng = random.Random(seed)
    step_logs: list[dict[str, Any]] = []
    prev_aggregate_order_imbalance = 0.0

    tier1_agents = [a for a in agents if a.tier == 1]
    tier2_agents = [a for a in agents if a.tier == 2]
    tier3_agents = [a for a in agents if a.tier == 3]

    for step, sim_date in enumerate(trading_days):
        # 1. Environment update
        ohlcv = market_data.get_ohlcv(sim_date)
        if ohlcv is None:
            logger.warning("No price data for %s, skipping", sim_date)
            continue

        price_history = market_data.get_price_history(sim_date)
        volume_history = market_data.get_volume_history(sim_date)
        close = ohlcv["close"]
        prices_map = {market_data.symbol: close}

        # Compute derived signals
        pct_change_1d = ((price_history[-1] / price_history[-2]) - 1) * 100 if len(price_history) >= 2 else 0
        pct_change_5d = ((price_history[-1] / price_history[-5]) - 1) * 100 if len(price_history) >= 5 else 0
        pct_change_20d = ((price_history[-1] / price_history[-20]) - 1) * 100 if len(price_history) >= 20 else 0

        market_state = {
            "symbol": market_data.symbol,
            "date": str(sim_date),
            "ohlcv": ohlcv,
            "prices": prices_map,
            "price_history": price_history,
            "volume_history": volume_history,
            "pct_change_1d": pct_change_1d,
            "pct_change_5d": pct_change_5d,
            "pct_change_20d": pct_change_20d,
        }
        market_state["prev_aggregate_order_imbalance"] = prev_aggregate_order_imbalance
        market_state["fundamentals"] = fundamentals

        # 2. Get events for this date
        day_events = events_by_date.get(sim_date, [])
        for evt in day_events:
            evt["_injection_step"] = step
            evt["_step"] = step

        # 3. Agent execution by tier
        all_actions: list[AgentAction] = []

        # Tier 3: synchronous, no events
        for agent in tier3_agents:
            action = await agent.decide(market_state, [], step)
            all_actions.append(action)

        # Tier 2: sequential LLM with event distribution
        if tier2_agents:
            for agent in tier2_agents:
                t2_events = distribute_events(day_events, step, tier=2, rng=rng)
                action = await agent.decide(market_state, t2_events, step)
                all_actions.append(action)

        # Tier 1: sequential LLM with full context
        for agent in tier1_agents:
            t1_events = distribute_events(day_events, step, tier=1, rng=rng)
            action = await agent.decide(market_state, t1_events, step)
            all_actions.append(action)

        # 4. Order resolution
        resolved = resolve_backtest_orders(agents, all_actions, prices_map)

        # 5. Post-step: compute aggregates
        sentiments = [a.sentiment_update for a in resolved if a.action != "HOLD"]
        agg_sentiment = float(np.mean(sentiments)) if sentiments else 0.0

        buy_vol = sum(a.quantity for a in resolved if a.action == "BUY")
        sell_vol = sum(a.quantity for a in resolved if a.action == "SELL")
        total_vol = buy_vol + sell_vol
        order_imbalance = (buy_vol - sell_vol) / total_vol if total_vol > 0 else 0.0
        prev_aggregate_order_imbalance = order_imbalance

        step_log = {
            "step_number": step,
            "simulated_date": str(sim_date),
            "market_state": {"symbol": market_data.symbol, "close": close, "volume": ohlcv["volume"]},
            "agent_actions": [a.model_dump() for a in resolved],
            "events_active": [{"title": e.get("title", ""), "category": e.get("category", "")} for e in day_events],
            "aggregate_sentiment": agg_sentiment,
            "aggregate_order_imbalance": order_imbalance,
        }
        step_logs.append(step_log)

        # Callback for CLI display
        if on_step:
            on_step(step, sim_date, ohlcv, resolved, agents)

    return step_logs
