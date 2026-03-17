"""Tier 3 rule-based heuristic agents — no LLM, pure Python."""

from __future__ import annotations

import random
from typing import Any

import numpy as np

from imss.agents.base import AgentAction, BaseAgent, WorkingMemory, round_to_lot

IDX_LOT_SIZE = 100


class MomentumFollower(BaseAgent):
    """Buy on uptrend, sell on downtrend."""

    tier: int = 3
    lookback_days: int = 5
    buy_threshold: float = 0.03
    sell_threshold: float = -0.03
    position_pct: float = 0.05

    async def decide(self, market_state: dict[str, Any], events: list, step: int) -> AgentAction:
        stock = market_state.get("symbol", "BBRI")
        history = market_state.get("price_history", [])
        close = market_state.get("ohlcv", {}).get("close", 0)

        if len(history) < self.lookback_days:
            return AgentAction(agent_id=self.id, step=step, action="HOLD", stock=stock)

        pct_change = (history[-1] - history[-self.lookback_days]) / history[-self.lookback_days]

        if pct_change > self.buy_threshold:
            qty = round_to_lot(int(self.working_memory.cash * self.position_pct / close)) if close > 0 else 0
            if qty > 0:
                return AgentAction(agent_id=self.id, step=step, action="BUY", stock=stock, quantity=qty, confidence=min(abs(pct_change) * 10, 1.0), reasoning=f"Momentum: {pct_change:+.2%} over {self.lookback_days}d")
        elif pct_change < self.sell_threshold:
            held = self.working_memory.holdings.get(stock, 0)
            qty = round_to_lot(min(held, round_to_lot(int(held * 0.5)))) if held > 0 else 0
            if qty > 0:
                return AgentAction(agent_id=self.id, step=step, action="SELL", stock=stock, quantity=qty, confidence=min(abs(pct_change) * 10, 1.0), reasoning=f"Momentum reversal: {pct_change:+.2%}")

        return AgentAction(agent_id=self.id, step=step, action="HOLD", stock=stock)


class MeanReversion(BaseAgent):
    """Trade toward the mean when price deviates."""

    tier: int = 3
    lookback_days: int = 20
    z_threshold: float = 1.5
    position_pct: float = 0.05

    async def decide(self, market_state: dict[str, Any], events: list, step: int) -> AgentAction:
        stock = market_state.get("symbol", "BBRI")
        history = market_state.get("price_history", [])
        close = market_state.get("ohlcv", {}).get("close", 0)

        if len(history) < self.lookback_days:
            return AgentAction(agent_id=self.id, step=step, action="HOLD", stock=stock)

        window = history[-self.lookback_days:]
        ma = np.mean(window)
        std = np.std(window)
        if std == 0:
            return AgentAction(agent_id=self.id, step=step, action="HOLD", stock=stock)

        z_score = (close - ma) / std

        if z_score < -self.z_threshold:
            qty = round_to_lot(int(self.working_memory.cash * self.position_pct / close)) if close > 0 else 0
            if qty > 0:
                return AgentAction(agent_id=self.id, step=step, action="BUY", stock=stock, quantity=qty, confidence=min(abs(z_score) / 3, 1.0), reasoning=f"Mean reversion: z={z_score:.2f}, price below MA")
        elif z_score > self.z_threshold:
            held = self.working_memory.holdings.get(stock, 0)
            qty = round_to_lot(int(held * 0.5)) if held > 0 else 0
            if qty > 0:
                return AgentAction(agent_id=self.id, step=step, action="SELL", stock=stock, quantity=qty, confidence=min(abs(z_score) / 3, 1.0), reasoning=f"Mean reversion: z={z_score:.2f}, price above MA")

        return AgentAction(agent_id=self.id, step=step, action="HOLD", stock=stock)


class RandomWalkAgent(BaseAgent):
    """Random trades for liquidity/noise."""

    tier: int = 3
    action_probability: float = 0.1
    buy_bias: float = 0.5
    position_pct: float = 0.03
    _rng: random.Random | None = None

    model_config = {"arbitrary_types_allowed": True}

    def _get_rng(self) -> random.Random:
        if self._rng is None:
            self._rng = random.Random(hash(self.id))
        return self._rng

    async def decide(self, market_state: dict[str, Any], events: list, step: int) -> AgentAction:
        stock = market_state.get("symbol", "BBRI")
        close = market_state.get("ohlcv", {}).get("close", 0)
        rng = self._get_rng()

        if rng.random() > self.action_probability:
            return AgentAction(agent_id=self.id, step=step, action="HOLD", stock=stock)

        if rng.random() < self.buy_bias:
            qty = round_to_lot(int(self.working_memory.cash * self.position_pct / close)) if close > 0 else 0
            if qty > 0:
                return AgentAction(agent_id=self.id, step=step, action="BUY", stock=stock, quantity=qty, confidence=0.3, reasoning="Random walk buy")
        else:
            held = self.working_memory.holdings.get(stock, 0)
            qty = round_to_lot(int(held * 0.3)) if held > 0 else 0
            if qty > 0:
                return AgentAction(agent_id=self.id, step=step, action="SELL", stock=stock, quantity=qty, confidence=0.3, reasoning="Random walk sell")

        return AgentAction(agent_id=self.id, step=step, action="HOLD", stock=stock)


class VolumeFollower(BaseAgent):
    """Buy on volume spikes."""

    tier: int = 3
    volume_lookback: int = 10
    spike_threshold: float = 2.0
    position_pct: float = 0.05

    async def decide(self, market_state: dict[str, Any], events: list, step: int) -> AgentAction:
        stock = market_state.get("symbol", "BBRI")
        volumes = market_state.get("volume_history", [])
        close = market_state.get("ohlcv", {}).get("close", 0)
        current_vol = market_state.get("ohlcv", {}).get("volume", 0)

        if len(volumes) < self.volume_lookback:
            return AgentAction(agent_id=self.id, step=step, action="HOLD", stock=stock)

        avg_vol = np.mean(volumes[-self.volume_lookback:])
        if avg_vol == 0:
            return AgentAction(agent_id=self.id, step=step, action="HOLD", stock=stock)

        vol_ratio = current_vol / avg_vol
        if vol_ratio > self.spike_threshold:
            qty = round_to_lot(int(self.working_memory.cash * self.position_pct / close)) if close > 0 else 0
            if qty > 0:
                return AgentAction(agent_id=self.id, step=step, action="BUY", stock=stock, quantity=qty, confidence=min(vol_ratio / 4, 1.0), reasoning=f"Volume spike: {vol_ratio:.1f}x average")

        return AgentAction(agent_id=self.id, step=step, action="HOLD", stock=stock)


# --- Factory ---

TIER3_CLASSES: dict[str, type[BaseAgent]] = {
    "momentum_follower": MomentumFollower,
    "mean_reversion": MeanReversion,
    "random_walk": RandomWalkAgent,
    "volume_follower": VolumeFollower,
}


def create_tier3_agents(
    total: int = 50,
    distribution: dict[str, float] | None = None,
    seed: int = 42,
) -> list[BaseAgent]:
    """Generate Tier 3 agents per distribution percentages.

    Counts rounded to nearest int, remainder goes to momentum_follower.
    """
    if distribution is None:
        distribution = {
            "momentum_follower": 0.30,
            "mean_reversion": 0.25,
            "random_walk": 0.30,
            "volume_follower": 0.15,
        }

    rng = random.Random(seed)
    agents: list[BaseAgent] = []
    counts: dict[str, int] = {}
    allocated = 0

    for heuristic, pct in distribution.items():
        count = round(total * pct)
        counts[heuristic] = count
        allocated += count

    # Assign remainder to momentum_follower
    if allocated != total:
        counts["momentum_follower"] += total - allocated

    for heuristic, count in counts.items():
        cls = TIER3_CLASSES[heuristic]
        for i in range(count):
            cash = rng.uniform(20_000_000, 80_000_000)
            agents.append(
                cls(
                    id=f"{heuristic}_{i:03d}",
                    name=f"{heuristic.replace('_', ' ').title()} #{i + 1}",
                    persona_type=f"tier3_{heuristic}",
                    working_memory=WorkingMemory(cash=cash),
                )
            )

    return agents
