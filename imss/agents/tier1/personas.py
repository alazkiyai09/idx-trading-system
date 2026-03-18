"""Tier 1 named agent personas — LLM-powered institutional/key players."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from imss.agents.base import AgentAction, BaseAgent, WorkingMemory, default_hold_action, round_to_lot
from imss.llm.prompts.tier1_decision import (
    build_tier1_system_prompt,
    build_tier1_user_prompt,
)
from imss.llm.router import LLMRouter


@dataclass
class PersonaConfig:
    """Static persona configuration."""

    key: str
    name: str
    description: str
    behavioral_rules: str
    risk_tolerance: float
    max_allocation: int
    stop_loss_pct: int
    holding_period: str
    initial_cash: float
    initial_holdings: dict[str, int]


PERSONAS: dict[str, PersonaConfig] = {
    "pak_budi": PersonaConfig(
        key="pak_budi",
        name="Pak Budi",
        description="Senior fund manager at a major Indonesian state-owned pension fund. 25 years of experience in IDX. Conservative, regulation-aware, long-term oriented.",
        behavioral_rules=(
            "- You prioritize capital preservation over aggressive returns\n"
            "- You pay close attention to OJK and BI regulatory signals\n"
            "- You rarely panic sell; you believe in fundamental value and Indonesian economic growth\n"
            "- You are skeptical of momentum-driven moves and prefer to buy on dips\n"
            "- You hold positions for weeks to months, not days\n"
            "- You are uncomfortable with more than 30% of portfolio in a single stock\n"
            "- When uncertain, you HOLD rather than act"
        ),
        risk_tolerance=0.3,
        max_allocation=30,
        stop_loss_pct=15,
        holding_period="weeks to months",
        initial_cash=10_000_000_000,
        initial_holdings={"BBRI": 500_000},
    ),
    "sarah": PersonaConfig(
        key="sarah",
        name="Sarah",
        description="Portfolio manager at a Singapore-based emerging markets fund. Analytical, USD-return focused, sensitive to currency risk and foreign flow data.",
        behavioral_rules=(
            "- You evaluate IDX stocks in USD terms\n"
            "- You track foreign fund flow data closely; net foreign outflows make you nervous\n"
            "- You are more aggressive than local institutions\n"
            "- You follow global macro trends and correlate IDX with regional markets\n"
            "- You are quick to cut losses if your thesis is invalidated\n"
            "- Political instability or regulatory uncertainty triggers position reduction\n"
            "- You like liquid large-caps; you won't touch low-volume stocks"
        ),
        risk_tolerance=0.6,
        max_allocation=40,
        stop_loss_pct=8,
        holding_period="days to weeks",
        initial_cash=15_000_000_000,
        initial_holdings={"BBRI": 200_000},
    ),
    "andi": PersonaConfig(
        key="andi",
        name="Andi",
        description="Full-time retail day trader. Active on Stockbit community. Momentum-driven, influenced by social media sentiment and influencer calls.",
        behavioral_rules=(
            "- You chase momentum — if a stock is running, you want in\n"
            "- You follow what popular Stockbit influencers recommend\n"
            "- You panic sell quickly when things go against you\n"
            "- You overtrade — you prefer action to sitting still\n"
            "- You have strong recency bias\n"
            "- You use simple technical signals: breaking resistance or support broken\n"
            "- You tend to buy at highs and sell at lows due to emotional decision-making\n"
            "- Small position sizes relative to institutional agents"
        ),
        risk_tolerance=0.8,
        max_allocation=50,
        stop_loss_pct=5,
        holding_period="hours to days",
        initial_cash=100_000_000,
        initial_holdings={},
    ),
    "dr_lim": PersonaConfig(
        key="dr_lim",
        name="Dr. Lim",
        description="Independent equity analyst with PhD in finance. Deep fundamental analysis. Goes against the crowd when valuations are extreme.",
        behavioral_rules=(
            "- You are a contrarian — you buy when fear is high and valuations are low\n"
            "- You evaluate stocks on P/E, P/B, dividend yield, and ROE fundamentals\n"
            "- You are patient — you hold losing positions for months if your thesis is intact\n"
            "- You ignore short-term noise and social media hype\n"
            "- You increase position sizes when fear is high and valuations are low\n"
            "- You are a statistical thinker who believes in mean reversion\n"
            "- When the market is euphoric and prices are high, you reduce positions"
        ),
        risk_tolerance=0.4,
        max_allocation=35,
        stop_loss_pct=20,
        holding_period="months",
        initial_cash=5_000_000_000,
        initial_holdings={"BBRI": 300_000},
    ),
}


class Tier1Agent(BaseAgent):
    """LLM-powered named agent with full persona prompt."""

    tier: int = 1
    persona_config: PersonaConfig
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
        if self._router is None:
            return default_hold_action(self.id, market_state.get("symbol", "BBRI"), step)

        pc = self.persona_config
        stock = market_state.get("symbol", "BBRI")

        # Format holdings
        holdings_parts = []
        for sym, qty in self.working_memory.holdings.items():
            price = market_state.get("prices", {}).get(sym, 0)
            holdings_parts.append(f"{sym}: {qty:,} shares (≈ IDR {qty * price:,.0f})")
        holdings_formatted = "; ".join(holdings_parts) if holdings_parts else "None"

        # Portfolio value
        prices_map = market_state.get("prices", {})
        portfolio_value = self.working_memory.compute_portfolio_value(prices_map)

        # Unrealized P&L
        initial_value = pc.initial_cash + sum(
            qty * prices_map.get(sym, 0) for sym, qty in pc.initial_holdings.items()
        )
        unrealized_pnl = ((portfolio_value - initial_value) / initial_value * 100) if initial_value > 0 else 0.0

        # Format events
        events_lines = []
        for evt in events:
            events_lines.append(f"- [{evt.get('category', 'NEWS')}] {evt.get('title', '')} (sentiment: {evt.get('sentiment_score', 0):.1f})")
        events_formatted = "\n".join(events_lines)

        system_prompt = build_tier1_system_prompt(
            agent_name=pc.name,
            persona_description=pc.description,
            behavioral_rules=pc.behavioral_rules,
            risk_tolerance=pc.risk_tolerance,
            max_allocation=pc.max_allocation,
            stop_loss_pct=pc.stop_loss_pct,
            holding_period=pc.holding_period,
        )

        ohlcv = market_state.get("ohlcv", {})

        # Pass fundamentals only to Dr. Lim
        agent_fundamentals = market_state.get("fundamentals") if self.persona_config.key == "dr_lim" else None

        user_prompt = build_tier1_user_prompt(
            step=step,
            simulated_date=market_state.get("date", ""),
            cash=self.working_memory.cash,
            holdings_formatted=holdings_formatted,
            portfolio_value=portfolio_value,
            unrealized_pnl=unrealized_pnl,
            stock_symbol=stock,
            ohlcv=ohlcv,
            pct_change_5d=market_state.get("pct_change_5d", 0),
            pct_change_20d=market_state.get("pct_change_20d", 0),
            events_formatted=events_formatted,
            fundamentals=agent_fundamentals,
        )

        response = await self._router.call(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.7,
            max_tokens=1024,
            tier=1,
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


def create_tier1_agent(persona_key: str) -> Tier1Agent:
    """Factory: create a Tier 1 agent from persona key."""
    config = PERSONAS[persona_key]
    return Tier1Agent(
        id=config.key,
        name=config.name,
        persona_type=f"tier1_{config.key}",
        persona_config=config,
        working_memory=WorkingMemory(
            cash=config.initial_cash,
            holdings=dict(config.initial_holdings),
        ),
    )
