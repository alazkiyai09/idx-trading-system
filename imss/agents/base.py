"""Agent base classes, actions, and IDX market utilities."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict


# --- IDX Market Utilities ---

IDX_LOT_SIZE = 100


def round_to_lot(quantity: int) -> int:
    """Round quantity down to nearest IDX lot (100 shares)."""
    return (quantity // IDX_LOT_SIZE) * IDX_LOT_SIZE


def round_to_tick(price: float) -> float:
    """Round price down to valid IDX tick size."""
    if price < 200:
        tick = 1
    elif price < 500:
        tick = 2
    elif price < 2000:
        tick = 5
    elif price < 5000:
        tick = 10
    else:
        tick = 25
    return (int(price) // tick) * tick


# --- Agent Data Models ---


class AgentAction(BaseModel):
    """Action produced by any agent."""

    agent_id: str
    step: int
    action: Literal["BUY", "SELL", "HOLD"]
    stock: str
    quantity: int = 0  # multiple of IDX_LOT_SIZE
    price: float = 0.0  # fill price, set after order resolution
    confidence: float = 0.5
    reasoning: str = ""
    sentiment_update: float = 0.0


class WorkingMemory(BaseModel):
    """In-memory state maintained during a simulation run."""

    current_step: int = 0
    cash: float = 0.0
    holdings: dict[str, int] = {}
    portfolio_value_history: list[tuple[int, float]] = []
    recent_actions: list[AgentAction] = []
    recent_observations: list[dict[str, Any]] = []
    current_sentiment: float = 0.0

    def compute_portfolio_value(self, prices: dict[str, float]) -> float:
        """Cash + sum(holdings * price)."""
        holdings_value = sum(
            qty * prices.get(sym, 0) for sym, qty in self.holdings.items()
        )
        return self.cash + holdings_value

    def add_action(self, action: AgentAction) -> None:
        """Append action, keeping last 10."""
        self.recent_actions.append(action)
        if len(self.recent_actions) > 10:
            self.recent_actions = self.recent_actions[-10:]

    def add_observation(self, obs: dict[str, Any]) -> None:
        """Append observation, keeping last 20."""
        self.recent_observations.append(obs)
        if len(self.recent_observations) > 20:
            self.recent_observations = self.recent_observations[-20:]


def default_hold_action(agent_id: str, stock: str, step: int) -> AgentAction:
    """Fallback HOLD action for failed LLM calls."""
    return AgentAction(
        agent_id=agent_id,
        step=step,
        action="HOLD",
        stock=stock,
        quantity=0,
        confidence=0.0,
        reasoning="Default HOLD — LLM call failed or response unparseable",
        sentiment_update=0.0,
    )


class BaseAgent(BaseModel, ABC):
    """Abstract base for all agent tiers."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str
    tier: int
    name: str
    persona_type: str
    working_memory: WorkingMemory

    @abstractmethod
    async def decide(
        self,
        market_state: dict[str, Any],
        events: list[dict[str, Any]],
        step: int,
    ) -> AgentAction:
        """Produce a trading decision for this step."""
        ...

    def execute(self, action: AgentAction, fill_price: float) -> None:
        """Update portfolio based on filled order."""
        stock = action.stock
        if action.action == "BUY" and action.quantity > 0:
            cost = action.quantity * fill_price
            if self.working_memory.cash >= cost:
                self.working_memory.cash -= cost
                self.working_memory.holdings[stock] = (
                    self.working_memory.holdings.get(stock, 0) + action.quantity
                )
        elif action.action == "SELL" and action.quantity > 0:
            held = self.working_memory.holdings.get(stock, 0)
            qty = min(action.quantity, held)
            if qty > 0:
                self.working_memory.cash += qty * fill_price
                self.working_memory.holdings[stock] = held - qty

        # Update sentiment
        self.working_memory.current_sentiment = action.sentiment_update
        action.price = fill_price
        self.working_memory.add_action(action)
