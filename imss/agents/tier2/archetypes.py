"""Tier 2 typed agent archetypes — LLM-powered retail/mid-market agents."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from imss.agents.base import AgentAction, BaseAgent, WorkingMemory, default_hold_action, round_to_lot
from imss.llm.prompts.tier2_decision import (
    build_tier2_system_prompt,
    build_tier2_user_prompt,
)
from imss.llm.router import LLMRouter


@dataclass
class ArchetypeConfig:
    """Archetype template for generating Tier 2 agents."""

    key: str
    name: str
    one_liner: str
    risk_tolerance_range: tuple[float, float]
    sentiment_bias_range: tuple[float, float]
    decision_latency: int
    initial_cash_range: tuple[float, float]


ARCHETYPES: dict[str, ArchetypeConfig] = {
    "momentum_chaser": ArchetypeConfig(
        key="momentum_chaser",
        name="Momentum Chaser",
        one_liner="You follow price momentum. Rising stocks attract you, falling stocks repel you.",
        risk_tolerance_range=(0.6, 0.9),
        sentiment_bias_range=(0.0, 0.3),
        decision_latency=0,
        initial_cash_range=(50_000_000, 200_000_000),
    ),
    "panic_seller": ArchetypeConfig(
        key="panic_seller",
        name="Panic Seller",
        one_liner="You are risk-averse and react strongly to bad news. Preservation of capital is everything.",
        risk_tolerance_range=(0.1, 0.3),
        sentiment_bias_range=(-0.3, 0.0),
        decision_latency=0,
        initial_cash_range=(50_000_000, 150_000_000),
    ),
    "news_reactive": ArchetypeConfig(
        key="news_reactive",
        name="News Reactive Trader",
        one_liner="You trade the news. Headlines drive your decisions. First to react, sometimes wrong.",
        risk_tolerance_range=(0.5, 0.8),
        sentiment_bias_range=(-0.1, 0.1),
        decision_latency=0,
        initial_cash_range=(30_000_000, 100_000_000),
    ),
    "dividend_holder": ArchetypeConfig(
        key="dividend_holder",
        name="Dividend Holder",
        one_liner="You buy high-dividend stocks and hold long term. You rarely sell unless dividends are cut.",
        risk_tolerance_range=(0.3, 0.5),
        sentiment_bias_range=(0.1, 0.3),
        decision_latency=2,
        initial_cash_range=(100_000_000, 500_000_000),
    ),
    "sector_rotator": ArchetypeConfig(
        key="sector_rotator",
        name="Sector Rotator",
        one_liner="You shift allocation based on economic cycle and macro data. Rate cuts mean buy banking, inflation means sell growth.",
        risk_tolerance_range=(0.4, 0.7),
        sentiment_bias_range=(-0.1, 0.1),
        decision_latency=1,
        initial_cash_range=(150_000_000, 600_000_000),
    ),
}


class Tier2Agent(BaseAgent):
    """LLM-powered typed agent with simplified prompt."""

    tier: int = 2
    archetype_key: str = ""
    archetype_one_liner: str = ""
    archetype_name: str = ""
    risk_tolerance: float = 0.5
    decision_latency: int = 0
    _router: LLMRouter | None = None

    model_config = {"arbitrary_types_allowed": True}

    def set_router(self, router: LLMRouter) -> None:
        self._router = router

    async def decide(
        self,
        market_state: dict[str, Any],
        events: list[dict[str, Any]],
        step: int,
    ) -> AgentAction:
        # Apply decision latency BEFORE router check — filter too-recent events
        if self.decision_latency > 0:
            events = [e for e in events if step - e.get("_step", 0) >= self.decision_latency]

        if self._router is None:
            return default_hold_action(self.id, market_state.get("symbol", "BBRI"), step)

        stock = market_state.get("symbol", "BBRI")
        ohlcv = market_state.get("ohlcv", {})

        # Holdings summary
        parts = [f"{s}: {q}" for s, q in self.working_memory.holdings.items()]
        holdings_summary = ", ".join(parts) if parts else "None"

        # Events brief (max 2)
        events_brief_parts = []
        for evt in events[:2]:
            events_brief_parts.append(f"{evt.get('title', '')}")
        events_brief = "; ".join(events_brief_parts)

        # Recent decisions
        recent_parts = []
        for act in self.working_memory.recent_actions[-3:]:
            recent_parts.append(f"Step {act.step}: {act.action} {act.stock}")
        recent_brief = "; ".join(recent_parts)

        system_prompt = build_tier2_system_prompt(
            archetype_name=self.archetype_name,
            archetype_one_liner=self.archetype_one_liner,
        )
        user_prompt = build_tier2_user_prompt(
            cash=self.working_memory.cash,
            holdings_summary=holdings_summary,
            stock_symbol=stock,
            close=ohlcv.get("close", 0),
            pct_change_1d=market_state.get("pct_change_1d", 0),
            pct_change_5d=market_state.get("pct_change_5d", 0),
            events_brief=events_brief,
            recent_decisions_brief=recent_brief,
        )

        response = await self._router.call(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.5,
            max_tokens=512,
            tier=2,
        )

        if response.parsed_json is None:
            return default_hold_action(self.id, stock, step)

        data = response.parsed_json
        return AgentAction(
            agent_id=self.id,
            step=step,
            action=data["action"],
            stock=data.get("stock", stock),
            quantity=round_to_lot(int(data.get("quantity", 0))),
            confidence=float(data.get("confidence", 0.5)),
            reasoning=data.get("reasoning", ""),
            sentiment_update=float(data.get("sentiment_update", 0.0)),
        )


def create_tier2_agents(
    archetype_key: str,
    count: int,
    seed: int = 42,
) -> list[Tier2Agent]:
    """Factory: generate N agents from an archetype template."""
    config = ARCHETYPES[archetype_key]
    rng = random.Random(seed)
    agents = []
    for i in range(count):
        cash = rng.uniform(*config.initial_cash_range)
        agents.append(
            Tier2Agent(
                id=f"{archetype_key}_{i:03d}",
                name=f"{config.name} #{i + 1}",
                persona_type=f"tier2_{archetype_key}",
                archetype_key=archetype_key,
                archetype_name=config.name,
                archetype_one_liner=config.one_liner,
                risk_tolerance=rng.uniform(*config.risk_tolerance_range),
                decision_latency=config.decision_latency,
                working_memory=WorkingMemory(cash=cash),
            )
        )
    return agents
