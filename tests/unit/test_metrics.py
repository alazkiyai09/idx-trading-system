"""
Tests for Backtest Metrics Module
"""

import pytest
import numpy as np
from datetime import date, timedelta

from backtest.metrics import (
    TradeMetrics,
    DrawdownMetrics,
    RiskAdjustedMetrics,
    PerformanceMetrics,
    calculate_metrics,
    _calculate_trade_metrics,
    _calculate_drawdown_metrics,
    _calculate_risk_adjusted_metrics,
    calculate_returns_by_period,
    calculate_trade_statistics_by_setup,
)
from core.data.models import Trade, SetupType, FlowSignal, OrderSide


def create_test_trade(
    return_pct: float = 5.0,
    net_pnl: float = 500_000,
    holding_days: int = 3,
    setup_type: SetupType = SetupType.MOMENTUM,
) -> Trade:
    """Create a test trade."""
    return Trade(
        trade_id="TEST-001",
        symbol="TEST",
        entry_date=date.today() - timedelta(days=holding_days),
        entry_price=9000.0,
        exit_date=date.today(),
        exit_price=9000.0 * (1 + return_pct / 100),
        exit_reason="test",
        quantity=100,
        side=OrderSide.BUY,
        gross_pnl=net_pnl + 100,  # Add fees
        fees=100,
        net_pnl=net_pnl,
        return_pct=return_pct,
        holding_days=holding_days,
        max_favorable=0,
        max_adverse=0,
        signal_score=75.0,
        setup_type=setup_type,
        rsi_at_entry=50.0,
        flow_signal=FlowSignal.NEUTRAL,
        flow_consecutive_days=0,
    )


class TestTradeMetrics:
    """Tests for TradeMetrics dataclass."""

    def test_empty_metrics(self):
        """Test empty metrics."""
        metrics = TradeMetrics()
        assert metrics.total_trades == 0
        assert metrics.win_rate == 0.0

    def test_with_trades(self):
        """Test metrics calculation from trades."""
        trades = [
            create_test_trade(return_pct=5.0, net_pnl=500_000),
            create_test_trade(return_pct=-2.0, net_pnl=-200_000),
            create_test_trade(return_pct=8.0, net_pnl=800_000),
        ]

        metrics = _calculate_trade_metrics(trades)

        assert metrics.total_trades == 3
        assert metrics.winning_trades == 2
        assert metrics.losing_trades == 1
        assert abs(metrics.win_rate - 2/3) < 0.01


class TestDrawdownMetrics:
    """Tests for DrawdownMetrics dataclass."""

    def test_empty_metrics(self):
        """Test empty metrics."""
        metrics = DrawdownMetrics()
        assert metrics.max_drawdown == 0.0
        assert metrics.max_drawdown_pct == 0.0

    def test_with_equity_curve(self):
        """Test metrics from equity curve."""
        equity_curve = [
            {"date": "2023-01-01", "equity": 100_000_000},
            {"date": "2023-01-02", "equity": 105_000_000},
            {"date": "2023-01-03", "equity": 95_000_000},  # Drawdown
            {"date": "2023-01-04", "equity": 98_000_000},
            {"date": "2023-01-05", "equity": 102_000_000},
        ]

        metrics = _calculate_drawdown_metrics(equity_curve, 0.02)

        assert metrics.max_drawdown > 0
        assert metrics.max_drawdown_pct > 0


class TestRiskAdjustedMetrics:
    """Tests for RiskAdjustedMetrics dataclass."""

    def test_empty_metrics(self):
        """Test empty metrics."""
        metrics = RiskAdjustedMetrics()
        assert metrics.sharpe_ratio == 0.0
        assert metrics.sortino_ratio == 0.0

    def test_with_equity_curve(self):
        """Test metrics from equity curve."""
        # Create upward trending equity curve
        equity_curve = []
        base = 100_000_000
        for i in range(100):
            equity_curve.append({
                "date": (date(2023, 1, 1) + timedelta(days=i)).isoformat(),
                "equity": base * (1 + i * 0.001),
            })

        metrics = _calculate_risk_adjusted_metrics(equity_curve, 0.05, 252)

        assert metrics.annual_return > 0


class TestCalculateMetrics:
    """Tests for calculate_metrics function."""

    def test_empty_data(self):
        """Test with empty data."""
        metrics = calculate_metrics([], [], 100_000_000)

        assert metrics["initial_capital"] == 100_000_000
        assert metrics["total_return_pct"] == 0.0

    def test_with_trades_and_equity(self):
        """Test with trades and equity curve."""
        trades = [
            create_test_trade(return_pct=5.0, net_pnl=500_000),
            create_test_trade(return_pct=-2.0, net_pnl=-200_000),
        ]

        equity_curve = [
            {"date": "2023-01-01", "equity": 100_000_000},
            {"date": "2023-01-02", "equity": 100_500_000},
            {"date": "2023-01-03", "equity": 100_300_000},
        ]

        metrics = calculate_metrics(trades, equity_curve, 100_000_000)

        assert metrics["initial_capital"] == 100_000_000
        assert "trade" in metrics
        assert "drawdown" in metrics
        assert "risk_adjusted" in metrics

    def test_profit_factor(self):
        """Test profit factor calculation."""
        # 2 wins, 1 loss
        trades = [
            create_test_trade(return_pct=10.0, net_pnl=1_000_000),
            create_test_trade(return_pct=5.0, net_pnl=500_000),
            create_test_trade(return_pct=-10.0, net_pnl=-1_000_000),
        ]

        metrics = _calculate_trade_metrics(trades)

        # Profit factor = (1000K + 500K) / 1000K = 1.5
        assert metrics.profit_factor == pytest.approx(1.5, rel=0.1)

    def test_all_winning_trades(self):
        """Test with all winning trades."""
        trades = [
            create_test_trade(return_pct=5.0, net_pnl=500_000),
            create_test_trade(return_pct=3.0, net_pnl=300_000),
        ]

        metrics = _calculate_trade_metrics(trades)

        assert metrics.win_rate == 1.0
        assert metrics.profit_factor == float('inf')

    def test_all_losing_trades(self):
        """Test with all losing trades."""
        trades = [
            create_test_trade(return_pct=-5.0, net_pnl=-500_000),
            create_test_trade(return_pct=-3.0, net_pnl=-300_000),
        ]

        metrics = _calculate_trade_metrics(trades)

        assert metrics.win_rate == 0.0
        assert metrics.profit_factor == 0.0


class TestCalculateReturnsByPeriod:
    """Tests for calculate_returns_by_period function."""

    def test_empty_curve(self):
        """Test with empty curve."""
        returns = calculate_returns_by_period([])
        assert returns == {}

    def test_monthly_returns(self):
        """Test monthly return calculation."""
        equity_curve = [
            {"date": "2023-01-01", "equity": 100_000_000},
            {"date": "2023-01-15", "equity": 105_000_000},
            {"date": "2023-01-31", "equity": 103_000_000},
            {"date": "2023-02-15", "equity": 108_000_000},
            {"date": "2023-02-28", "equity": 110_000_000},
        ]

        returns = calculate_returns_by_period(equity_curve, "monthly")

        # Format is "2023-01" for January
        assert "2023-01" in returns
        assert "2023-02" in returns


class TestCalculateTradeStatisticsBySetup:
    """Tests for calculate_trade_statistics_by_setup function."""

    def test_empty_trades(self):
        """Test with no trades."""
        stats = calculate_trade_statistics_by_setup([])
        assert stats == {}

    def test_by_setup(self):
        """Test grouping by setup type."""
        trades = [
            create_test_trade(setup_type=SetupType.MOMENTUM, return_pct=5.0),
            create_test_trade(setup_type=SetupType.MOMENTUM, return_pct=-2.0),
            create_test_trade(setup_type=SetupType.BREAKOUT, return_pct=8.0),
        ]

        stats = calculate_trade_statistics_by_setup(trades)

        assert "MOMENTUM" in stats
        assert "BREAKOUT" in stats
        assert stats["MOMENTUM"]["count"] == 2
        assert stats["BREAKOUT"]["count"] == 1


class TestPerformanceMetrics:
    """Tests for PerformanceMetrics dataclass."""

    def test_to_dict(self):
        """Test dictionary conversion."""
        metrics = PerformanceMetrics(
            initial_capital=100_000_000,
            final_capital=110_000_000,
            total_return=0.10,
            total_return_pct=10.0,
        )

        d = metrics.to_dict()

        assert d["initial_capital"] == 100_000_000
        assert d["total_return_pct"] == 10.0

    def test_summary(self):
        """Test summary generation."""
        metrics = PerformanceMetrics(
            initial_capital=100_000_000,
            final_capital=110_000_000,
            total_return_pct=10.0,
        )

        summary = metrics.summary()

        assert "PERFORMANCE" in summary
        assert "10.00" in summary


class TestEdgeCases:
    """Test edge cases."""

    def test_single_trade(self):
        """Test with single trade."""
        trades = [create_test_trade(return_pct=5.0)]

        metrics = _calculate_trade_metrics(trades)

        assert metrics.total_trades == 1
        assert metrics.win_rate == 1.0

    def test_single_equity_point(self):
        """Test with single equity point."""
        equity_curve = [{"date": "2023-01-01", "equity": 100_000_000}]

        metrics = _calculate_drawdown_metrics(equity_curve, 0)

        assert metrics.max_drawdown == 0.0

    def test_zero_volatility(self):
        """Test with zero volatility."""
        # Flat equity curve
        equity_curve = [
            {"date": f"2023-01-{i:02d}", "equity": 100_000_000}
            for i in range(1, 11)
        ]

        metrics = _calculate_risk_adjusted_metrics(equity_curve, 0, 252)

        # No change = no return
        assert metrics.annual_return == 0.0
