"""MarketBot — hybrid market-maker agent with rule-based direction + LLM reasoning."""

from __future__ import annotations

from typing import Any

from imss.agents.base import AgentAction, default_hold_action, round_to_lot
from imss.agents.tier1.personas import Tier1Agent
from imss.llm.prompts.marketbot_decision import (
    build_marketbot_system_prompt,
    build_marketbot_user_prompt,
)


class MarketBotAgent(Tier1Agent):
    """Hybrid market maker: rule-based direction, LLM-reasoned sizing."""

    async def decide(
        self,
        market_state: dict[str, Any],
        events: list[dict[str, Any]],
        step: int,
    ) -> AgentAction:
        stock = market_state.get("symbol", "BBRI")
        imbalance = market_state.get("prev_aggregate_order_imbalance", 0.0)
        pct_change_5d = market_state.get("pct_change_5d", 0.0)

        # 1. Rule-based direction
        if imbalance > 0.1:
            direction = "SELL"
        elif imbalance < -0.1:
            direction = "BUY"
        else:
            return default_hold_action(self.id, stock, step)

        # 2. If no router, return with default quantity
        if self._router is None:
            return default_hold_action(self.id, stock, step)

        # 3. Format holdings
        parts = [f"{s}: {q:,} shares" for s, q in self.working_memory.holdings.items()]
        holdings_formatted = "; ".join(parts) if parts else "None"

        close_price = market_state.get("ohlcv", {}).get("close", 0)

        system_prompt = build_marketbot_system_prompt()
        user_prompt = build_marketbot_user_prompt(
            direction=direction,
            imbalance=imbalance,
            pct_change_5d=pct_change_5d,
            cash=self.working_memory.cash,
            holdings_formatted=holdings_formatted,
            stock_symbol=stock,
            close_price=close_price,
        )

        response = await self._router.call(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.3,
            max_tokens=512,
            tier=1,
        )

        if response.parsed_json is None:
            return default_hold_action(self.id, stock, step)

        data = response.parsed_json
        quantity = round_to_lot(int(data.get("quantity", 0)))

        # Volatility gate: reduce sizing by 50% in high volatility
        if abs(pct_change_5d) > 5:
            quantity = round_to_lot(quantity // 2)

        return AgentAction(
            agent_id=self.id,
            step=step,
            action=direction,  # Use rule-based direction, not LLM's
            stock=stock,
            quantity=quantity,
            confidence=float(data.get("confidence", 0.5)),
            reasoning=data.get("reasoning", ""),
            sentiment_update=float(data.get("sentiment_update", 0.0)),
        )
