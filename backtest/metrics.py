"""
Backtest Metrics Module

Calculates comprehensive performance metrics for backtesting including
Sharpe, Sortino, Calmar ratios, drawdown analysis, and trade statistics.
"""

import logging
import numpy as np
from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional, Any, Tuple

from core.data.models import Trade

logger = logging.getLogger(__name__)


@dataclass
class TradeMetrics:
    """Trade-level metrics.

    Attributes:
        total_trades: Total number of trades.
        winning_trades: Number of winning trades.
        losing_trades: Number of losing trades.
        win_rate: Percentage of winning trades.
        avg_win: Average winning trade return %.
        avg_loss: Average losing trade return %.
        avg_trade: Average trade return %.
        largest_win: Largest winning trade return %.
        largest_loss: Largest losing trade return %.
        profit_factor: Gross wins / Gross losses.
        expectancy: Expected value per trade.
    """

    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    avg_trade: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0
    avg_holding_days: float = 0.0


@dataclass
class DrawdownMetrics:
    """Drawdown-related metrics.

    Attributes:
        max_drawdown: Maximum drawdown as decimal.
        max_drawdown_pct: Maximum drawdown as percentage.
        avg_drawdown: Average drawdown during drawdown periods.
        max_drawdown_duration: Maximum drawdown duration in days.
        recovery_factor: Total return / Max drawdown.
        ulcer_index: Measure of downside risk.
    """

    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    avg_drawdown: float = 0.0
    max_drawdown_duration: int = 0
    recovery_factor: float = 0.0
    ulcer_index: float = 0.0
    drawdown_periods: int = 0


@dataclass
class RiskAdjustedMetrics:
    """Risk-adjusted performance metrics.

    Attributes:
        sharpe_ratio: Risk-adjusted return (rf=0).
        sortino_ratio: Downside risk-adjusted return.
        calmar_ratio: Annual return / Max drawdown.
        sterling_ratio: Average return / Average max drawdown.
        burke_ratio: Modified risk measure.
    """

    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    sterling_ratio: float = 0.0
    burke_ratio: float = 0.0
    annual_return: float = 0.0
    annual_volatility: float = 0.0


@dataclass
class PerformanceMetrics:
    """Complete performance metrics.

    Attributes:
        initial_capital: Starting capital.
        final_capital: Ending capital.
        total_return: Total return as decimal.
        total_return_pct: Total return as percentage.
        trade: Trade-level metrics.
        drawdown: Drawdown metrics.
        risk_adjusted: Risk-adjusted metrics.
    """

    initial_capital: float = 0.0
    final_capital: float = 0.0
    total_return: float = 0.0
    total_return_pct: float = 0.0
    cagr: float = 0.0
    trade: TradeMetrics = field(default_factory=TradeMetrics)
    drawdown: DrawdownMetrics = field(default_factory=DrawdownMetrics)
    risk_adjusted: RiskAdjustedMetrics = field(default_factory=RiskAdjustedMetrics)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "initial_capital": self.initial_capital,
            "final_capital": self.final_capital,
            "total_return_pct": self.total_return_pct,
            "cagr": self.cagr,
            "trade": {
                "total_trades": self.trade.total_trades,
                "win_rate": self.trade.win_rate,
                "profit_factor": self.trade.profit_factor,
                "avg_trade": self.trade.avg_trade,
                "expectancy": self.trade.expectancy,
            },
            "drawdown": {
                "max_drawdown_pct": self.drawdown.max_drawdown_pct,
                "max_drawdown_duration": self.drawdown.max_drawdown_duration,
                "recovery_factor": self.drawdown.recovery_factor,
            },
            "risk_adjusted": {
                "sharpe_ratio": self.risk_adjusted.sharpe_ratio,
                "sortino_ratio": self.risk_adjusted.sortino_ratio,
                "calmar_ratio": self.risk_adjusted.calmar_ratio,
                "annual_return": self.risk_adjusted.annual_return,
            },
        }

    def summary(self) -> str:
        """Generate summary string."""
        lines = [
            "=" * 60,
            "PERFORMANCE METRICS",
            "=" * 60,
            "",
            "RETURNS:",
            f"  Total Return: {self.total_return_pct:+.2f}%",
            f"  CAGR: {self.cagr:+.2f}%",
            f"  Final Capital: {self.final_capital:,.0f} IDR",
            "",
            "TRADE STATISTICS:",
            f"  Total Trades: {self.trade.total_trades}",
            f"  Win Rate: {self.trade.win_rate:.1%}",
            f"  Profit Factor: {self.trade.profit_factor:.2f}",
            f"  Avg Trade: {self.trade.avg_trade:+.2f}%",
            f"  Expectancy: {self.trade.expectancy:+.2f}%",
            "",
            "DRAWDOWN:",
            f"  Max Drawdown: {self.drawdown.max_drawdown_pct:.2f}%",
            f"  Max Duration: {self.drawdown.max_drawdown_duration} days",
            f"  Recovery Factor: {self.drawdown.recovery_factor:.2f}",
            "",
            "RISK-ADJUSTED:",
            f"  Sharpe Ratio: {self.risk_adjusted.sharpe_ratio:.2f}",
            f"  Sortino Ratio: {self.risk_adjusted.sortino_ratio:.2f}",
            f"  Calmar Ratio: {self.risk_adjusted.calmar_ratio:.2f}",
            "",
            "=" * 60,
        ]
        return "\n".join(lines)


def calculate_metrics(
    trades: List[Trade],
    equity_curve: List[Dict[str, Any]],
    initial_capital: float,
    trading_days_per_year: int = 252,
) -> Dict[str, Any]:
    """Calculate all performance metrics.

    Args:
        trades: List of completed trades.
        equity_curve: Daily equity values.
        initial_capital: Starting capital.
        trading_days_per_year: Trading days per year.

    Returns:
        Dictionary of metrics.
    """
    metrics = PerformanceMetrics(initial_capital=initial_capital)

    if not equity_curve:
        return metrics.to_dict()

    # Basic returns
    final_capital = equity_curve[-1]["equity"]
    metrics.final_capital = final_capital
    metrics.total_return = (final_capital - initial_capital) / initial_capital
    metrics.total_return_pct = metrics.total_return * 100

    # CAGR
    if len(equity_curve) > 1:
        first_date = equity_curve[0]["date"]
        last_date = equity_curve[-1]["date"]

        # Parse dates
        if isinstance(first_date, str):
            first_date = date.fromisoformat(first_date)
        if isinstance(last_date, str):
            last_date = date.fromisoformat(last_date)

        days = (last_date - first_date).days
        if days > 0:
            years = days / 365.0
            if years > 0 and final_capital > 0:
                metrics.cagr = ((final_capital / initial_capital) ** (1 / years) - 1) * 100

    # Trade metrics
    metrics.trade = _calculate_trade_metrics(trades)

    # Drawdown metrics
    metrics.drawdown = _calculate_drawdown_metrics(
        equity_curve,
        metrics.total_return,
    )

    # Risk-adjusted metrics
    metrics.risk_adjusted = _calculate_risk_adjusted_metrics(
        equity_curve,
        metrics.drawdown.max_drawdown,
        trading_days_per_year,
    )

    return metrics.to_dict()


def _calculate_trade_metrics(trades: List[Trade]) -> TradeMetrics:
    """Calculate trade-level metrics.

    Args:
        trades: List of trades.

    Returns:
        TradeMetrics instance.
    """
    metrics = TradeMetrics()

    if not trades:
        return metrics

    metrics.total_trades = len(trades)

    # Separate wins and losses
    wins = [t for t in trades if t.net_pnl > 0]
    losses = [t for t in trades if t.net_pnl <= 0]

    metrics.winning_trades = len(wins)
    metrics.losing_trades = len(losses)
    metrics.win_rate = len(wins) / len(trades) if trades else 0

    # Average returns
    if wins:
        metrics.avg_win = np.mean([t.return_pct for t in wins])
        metrics.largest_win = max(t.return_pct for t in wins)

    if losses:
        metrics.avg_loss = np.mean([t.return_pct for t in losses])
        metrics.largest_loss = min(t.return_pct for t in losses)

    # Overall average
    metrics.avg_trade = np.mean([t.return_pct for t in trades])

    # Profit factor
    gross_wins = sum(t.net_pnl for t in wins)
    gross_losses = abs(sum(t.net_pnl for t in losses))

    if gross_losses > 0:
        metrics.profit_factor = gross_wins / gross_losses
    elif gross_wins > 0:
        metrics.profit_factor = float('inf')
    else:
        metrics.profit_factor = 0

    # Expectancy
    if trades:
        metrics.expectancy = np.mean([t.return_pct for t in trades])

    # Average holding days
    if trades:
        metrics.avg_holding_days = np.mean([t.holding_days for t in trades])

    return metrics


def _calculate_drawdown_metrics(
    equity_curve: List[Dict[str, Any]],
    total_return: float,
) -> DrawdownMetrics:
    """Calculate drawdown metrics.

    Args:
        equity_curve: Daily equity values.
        total_return: Total return as decimal.

    Returns:
        DrawdownMetrics instance.
    """
    metrics = DrawdownMetrics()

    if len(equity_curve) < 2:
        return metrics

    # Extract equity values
    equities = [e["equity"] for e in equity_curve]
    equity_array = np.array(equities)

    # Calculate running maximum
    running_max = np.maximum.accumulate(equity_array)

    # Calculate drawdowns
    drawdowns = (running_max - equity_array) / running_max
    drawdowns = np.nan_to_num(drawdowns, nan=0.0)

    metrics.max_drawdown = np.max(drawdowns)
    metrics.max_drawdown_pct = metrics.max_drawdown * 100

    # Average drawdown (when in drawdown)
    in_drawdown = drawdowns > 0
    if np.any(in_drawdown):
        metrics.avg_drawdown = np.mean(drawdowns[in_drawdown])

    # Count drawdown periods
    metrics.drawdown_periods = _count_drawdown_periods(drawdowns)

    # Max drawdown duration
    metrics.max_drawdown_duration = _calculate_max_drawdown_duration(drawdowns)

    # Recovery factor
    if metrics.max_drawdown > 0:
        metrics.recovery_factor = total_return / metrics.max_drawdown
    else:
        metrics.recovery_factor = float('inf') if total_return > 0 else 0

    # Ulcer index
    metrics.ulcer_index = np.sqrt(np.mean(drawdowns ** 2)) * 100

    return metrics


def _calculate_risk_adjusted_metrics(
    equity_curve: List[Dict[str, Any]],
    max_drawdown: float,
    trading_days_per_year: int,
) -> RiskAdjustedMetrics:
    """Calculate risk-adjusted metrics.

    Args:
        equity_curve: Daily equity values.
        max_drawdown: Maximum drawdown as decimal.
        trading_days_per_year: Trading days per year.

    Returns:
        RiskAdjustedMetrics instance.
    """
    metrics = RiskAdjustedMetrics()

    if len(equity_curve) < 2:
        return metrics

    # Extract equity values
    equities = np.array([e["equity"] for e in equity_curve])

    # Calculate daily returns
    daily_returns = np.diff(equities) / equities[:-1]
    daily_returns = np.nan_to_num(daily_returns, nan=0.0)

    if len(daily_returns) == 0:
        return metrics

    # Annual return
    total_return = (equities[-1] / equities[0]) - 1

    # Days in backtest
    first_date = equity_curve[0]["date"]
    last_date = equity_curve[-1]["date"]

    if isinstance(first_date, str):
        first_date = date.fromisoformat(first_date)
    if isinstance(last_date, str):
        last_date = date.fromisoformat(last_date)

    days = (last_date - first_date).days

    if days > 0:
        years = days / 365.0
        if years > 0:
            metrics.annual_return = ((equities[-1] / equities[0]) ** (1 / years) - 1) * 100

    # Annual volatility
    if len(daily_returns) > 1:
        daily_vol = np.std(daily_returns, ddof=1)
        metrics.annual_volatility = daily_vol * np.sqrt(trading_days_per_year) * 100

    # Sharpe Ratio (assuming risk-free = 0)
    if metrics.annual_volatility > 0:
        metrics.sharpe_ratio = metrics.annual_return / metrics.annual_volatility

    # Sortino Ratio (downside deviation only)
    negative_returns = daily_returns[daily_returns < 0]
    if len(negative_returns) > 0:
        downside_dev = np.std(negative_returns, ddof=1) * np.sqrt(trading_days_per_year) * 100
        if downside_dev > 0:
            metrics.sortino_ratio = metrics.annual_return / downside_dev

    # Calmar Ratio
    if max_drawdown > 0:
        metrics.calmar_ratio = metrics.annual_return / (max_drawdown * 100)
    elif metrics.annual_return > 0:
        metrics.calmar_ratio = float('inf')

    # Sterling Ratio (using average of top 5 drawdowns)
    # Simplified version using max drawdown
    if max_drawdown > 0:
        metrics.sterling_ratio = metrics.annual_return / (max_drawdown * 100)

    # Burke Ratio
    if len(negative_returns) > 0:
        burke_denom = np.sqrt(np.sum(negative_returns ** 2)) * 100
        if burke_denom > 0:
            metrics.burke_ratio = metrics.annual_return / burke_denom

    return metrics


def _count_drawdown_periods(drawdowns: np.ndarray) -> int:
    """Count number of drawdown periods.

    Args:
        drawdowns: Array of drawdown values.

    Returns:
        Number of drawdown periods.
    """
    if len(drawdowns) == 0:
        return 0

    # A new period starts when drawdown goes from 0 to > 0
    periods = 0
    in_drawdown = False

    for dd in drawdowns:
        if dd > 0 and not in_drawdown:
            periods += 1
            in_drawdown = True
        elif dd == 0:
            in_drawdown = False

    return periods


def _calculate_max_drawdown_duration(drawdowns: np.ndarray) -> int:
    """Calculate maximum drawdown duration in days.

    Args:
        drawdowns: Array of drawdown values.

    Returns:
        Maximum duration in days.
    """
    if len(drawdowns) == 0:
        return 0

    max_duration = 0
    current_duration = 0

    for dd in drawdowns:
        if dd > 0:
            current_duration += 1
            max_duration = max(max_duration, current_duration)
        else:
            current_duration = 0

    return max_duration


def calculate_returns_by_period(
    equity_curve: List[Dict[str, Any]],
    period: str = "monthly",
) -> Dict[str, float]:
    """Calculate returns by period.

    Args:
        equity_curve: Daily equity values.
        period: Period type ('daily', 'weekly', 'monthly', 'yearly').

    Returns:
        Dictionary of period to return.
    """
    if len(equity_curve) < 2:
        return {}

    returns = {}
    period_values: Dict[str, Tuple[float, float]] = {}

    for entry in equity_curve:
        entry_date = entry["date"]
        if isinstance(entry_date, str):
            entry_date = date.fromisoformat(entry_date)

        if period == "daily":
            key = entry_date.isoformat()
        elif period == "weekly":
            key = f"{entry_date.year}-W{entry_date.isocalendar()[1]}"
        elif period == "monthly":
            key = f"{entry_date.year}-{entry_date.month:02d}"
        elif period == "yearly":
            key = str(entry_date.year)
        else:
            key = entry_date.isoformat()

        if key not in period_values:
            period_values[key] = (entry["equity"], entry["equity"])
        else:
            # Update end value
            period_values[key] = (period_values[key][0], entry["equity"])

    for key, (start, end) in period_values.items():
        if start > 0:
            returns[key] = (end - start) / start * 100

    return returns


def calculate_trade_statistics_by_setup(
    trades: List[Trade],
) -> Dict[str, Dict[str, Any]]:
    """Calculate statistics grouped by setup type.

    Args:
        trades: List of trades.

    Returns:
        Dictionary of setup type to statistics.
    """
    stats = {}

    # Group trades by setup type
    by_setup: Dict[str, List[Trade]] = {}
    for trade in trades:
        setup = trade.setup_type.value if trade.setup_type else "UNKNOWN"
        if setup not in by_setup:
            by_setup[setup] = []
        by_setup[setup].append(trade)

    # Calculate stats for each setup
    for setup, setup_trades in by_setup.items():
        wins = [t for t in setup_trades if t.net_pnl > 0]

        stats[setup] = {
            "count": len(setup_trades),
            "wins": len(wins),
            "win_rate": len(wins) / len(setup_trades) if setup_trades else 0,
            "avg_return": np.mean([t.return_pct for t in setup_trades]) if setup_trades else 0,
            "total_pnl": sum(t.net_pnl for t in setup_trades),
        }

    return stats
