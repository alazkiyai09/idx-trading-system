"""Order resolution for backtest mode."""

from __future__ import annotations

import logging

from imss.agents.base import AgentAction, BaseAgent, round_to_lot

logger = logging.getLogger(__name__)


def resolve_backtest_orders(
    agents: list[BaseAgent],
    actions: list[AgentAction],
    close_prices: dict[str, float],
) -> list[AgentAction]:
    """Resolve orders at historical close prices.

    Modifies agent working_memory in place. Returns actions with fill prices set.
    """
    agent_map = {a.id: a for a in agents}
    resolved: list[AgentAction] = []

    for action in actions:
        agent = agent_map.get(action.agent_id)
        if agent is None:
            logger.warning("Unknown agent_id: %s", action.agent_id)
            continue

        stock = action.stock
        price = close_prices.get(stock, 0)
        if price <= 0:
            logger.warning("No price for %s, skipping action", stock)
            resolved.append(action)
            continue

        if action.action == "BUY":
            qty = round_to_lot(action.quantity)
            cost = qty * price
            if qty > 0 and agent.working_memory.cash >= cost:
                agent.working_memory.cash -= cost
                agent.working_memory.holdings[stock] = (
                    agent.working_memory.holdings.get(stock, 0) + qty
                )
                action.price = price
                action.quantity = qty
                logger.debug("%s BUY %s x%d @ %.0f", action.agent_id, stock, qty, price)
            else:
                logger.debug(
                    "%s BUY rejected: qty=%d, cost=%.0f, cash=%.0f",
                    action.agent_id, qty, cost, agent.working_memory.cash,
                )
                action.action = "HOLD"
                action.quantity = 0
        elif action.action == "SELL":
            held = agent.working_memory.holdings.get(stock, 0)
            qty = round_to_lot(min(action.quantity, held))
            if qty > 0:
                agent.working_memory.cash += qty * price
                agent.working_memory.holdings[stock] = held - qty
                action.price = price
                action.quantity = qty
                logger.debug("%s SELL %s x%d @ %.0f", action.agent_id, stock, qty, price)
            else:
                action.action = "HOLD"
                action.quantity = 0
        else:
            # HOLD — no-op
            action.price = price

        # Update sentiment
        agent.working_memory.current_sentiment = action.sentiment_update
        agent.working_memory.add_action(action)
        resolved.append(action)

    return resolved
