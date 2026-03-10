"""Portfolio-related API schemas."""

from datetime import date, datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class PositionResponse(BaseModel):
    """A single portfolio position."""
    symbol: str
    entry_price: float
    current_price: Optional[float] = None
    shares: int
    entry_date: date
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0
    stop_loss: Optional[float] = None
    target: Optional[float] = None


class PortfolioResponse(BaseModel):
    """Portfolio summary."""
    total_value: float
    cash: float
    invested: float
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    total_pnl: float = 0.0
    total_pnl_pct: float = 0.0
    num_positions: int = 0
    positions: List[PositionResponse] = Field(default_factory=list)


class TradeHistoryItem(BaseModel):
    """A single trade in the history."""
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    shares: int
    pnl: float
    pnl_pct: float
    entry_date: date
    exit_date: date
    holding_days: int = 0


class TradeHistoryResponse(BaseModel):
    """Trade history response."""
    trades: List[TradeHistoryItem]
    total: int
    total_pnl: float = 0.0
    win_rate: float = 0.0
