"""Backtest-related API schemas."""

from datetime import date
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class BacktestRequest(BaseModel):
    """Request to run a backtest."""
    symbols: List[str] = Field(description="Symbols to backtest")
    mode: str = Field(default="swing", description="Trading mode")
    start_date: date = Field(description="Backtest start date")
    end_date: date = Field(description="Backtest end date")
    initial_capital: float = Field(default=100_000_000, description="Initial capital in IDR")


class BacktestMetrics(BaseModel):
    """Backtest performance metrics."""
    total_return_pct: float = 0.0
    annualized_return_pct: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0
    avg_holding_days: float = 0.0
    calmar_ratio: float = 0.0


class BacktestResponse(BaseModel):
    """Backtest results response."""
    id: str
    status: str
    mode: str
    start_date: date
    end_date: date
    initial_capital: float
    final_capital: float = 0.0
    metrics: Optional[BacktestMetrics] = None
    equity_curve: Optional[List[Dict[str, Any]]] = None
    errors: List[str] = Field(default_factory=list)
