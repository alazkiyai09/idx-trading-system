"""Tests for Virtual Trading page helper functions."""
import pytest
import pandas as pd
from datetime import datetime
from unittest.mock import patch, MagicMock

from dashboard.pages.trading_helpers import (
    calculate_order_value,
    validate_order,
    calculate_portfolio_value,
    calculate_performance_metrics,
    process_trade_history,
    generate_equity_curve,
)


class TestCalculateOrderValue:
    """Tests for calculate_order_value function."""

    def test_buy_order_value(self):
        """Test order value calculation for BUY order."""
        result = calculate_order_value(1000, 9000, "BUY")

        assert result["gross_value"] == 9_000_000
        assert result["fee_rate"] == 0.0015  # 0.15% for buy
        assert result["fees"] == 13_500  # 0.15% of 9M
        assert result["net_value"] == 9_013_500

    def test_sell_order_value(self):
        """Test order value calculation for SELL order."""
        result = calculate_order_value(1000, 9000, "SELL")

        assert result["gross_value"] == 9_000_000
        assert result["fee_rate"] == 0.0025  # 0.25% for sell
        assert result["fees"] == 22_500  # 0.25% of 9M
        assert result["net_value"] == 8_977_500

    def test_lot_size_order(self):
        """Test order with standard lot size (100 shares)."""
        result = calculate_order_value(100, 10000, "BUY")

        assert result["gross_value"] == 1_000_000

    def test_large_order(self):
        """Test large order value calculation."""
        result = calculate_order_value(10000, 5000, "BUY")

        assert result["gross_value"] == 50_000_000
        assert result["fees"] == 75_000


class TestValidateOrder:
    """Tests for validate_order function."""

    def test_valid_buy_order(self):
        """Test validation of valid BUY order."""
        result = validate_order(
            symbol="BBCA",
            side="BUY",
            quantity=1000,
            price=9000,
            capital=10_000_000
        )

        assert result["valid"] is True
        assert len(result["errors"]) == 0

    def test_missing_symbol(self):
        """Test validation with missing symbol."""
        result = validate_order(
            symbol="",
            side="BUY",
            quantity=1000,
            price=9000,
            capital=10_000_000
        )

        assert result["valid"] is False
        assert "Symbol is required" in result["errors"]

    def test_invalid_side(self):
        """Test validation with invalid side."""
        result = validate_order(
            symbol="BBCA",
            side="HOLD",
            quantity=1000,
            price=9000,
            capital=10_000_000
        )

        assert result["valid"] is False
        assert any("BUY or SELL" in e for e in result["errors"])

    def test_non_lot_quantity(self):
        """Test validation with quantity not multiple of lot size."""
        result = validate_order(
            symbol="BBCA",
            side="BUY",
            quantity=150,  # Not multiple of 100
            price=9000,
            capital=10_000_000
        )

        assert result["valid"] is False
        assert any("lot size" in e.lower() for e in result["errors"])

    def test_zero_quantity(self):
        """Test validation with zero quantity."""
        result = validate_order(
            symbol="BBCA",
            side="BUY",
            quantity=0,
            price=9000,
            capital=10_000_000
        )

        assert result["valid"] is False
        assert any("positive" in e.lower() for e in result["errors"])

    def test_zero_price(self):
        """Test validation with zero price."""
        result = validate_order(
            symbol="BBCA",
            side="BUY",
            quantity=1000,
            price=0,
            capital=10_000_000
        )

        assert result["valid"] is False

    def test_insufficient_capital(self):
        """Test validation with insufficient capital."""
        result = validate_order(
            symbol="BBCA",
            side="BUY",
            quantity=1000,
            price=9000,
            capital=5_000_000  # Not enough
        )

        assert result["valid"] is False
        assert any("capital" in e.lower() for e in result["errors"])

    def test_sell_without_position(self):
        """Test SELL validation without existing position."""
        # When there are no existing positions, SELL should be invalid
        result = validate_order(
            symbol="BBCA",
            side="SELL",
            quantity=1000,
            price=9000,
            capital=10_000_000,
            existing_positions=[]  # No positions
        )

        # SELL without position should be invalid
        assert result["valid"] is False
        assert any("position" in e.lower() for e in result["errors"])

    def test_sell_exceeding_position(self):
        """Test SELL validation when quantity exceeds position."""
        positions = [{"symbol": "BBCA", "quantity": 500}]

        result = validate_order(
            symbol="BBCA",
            side="SELL",
            quantity=1000,
            price=9000,
            capital=10_000_000,
            existing_positions=positions
        )

        assert result["valid"] is False
        assert any("Insufficient" in e for e in result["errors"])

    def test_valid_sell_order(self):
        """Test validation of valid SELL order."""
        positions = [{"symbol": "BBCA", "quantity": 2000}]

        result = validate_order(
            symbol="BBCA",
            side="SELL",
            quantity=1000,
            price=9000,
            capital=10_000_000,
            existing_positions=positions
        )

        assert result["valid"] is True


class TestCalculatePortfolioValue:
    """Tests for calculate_portfolio_value function."""

    def test_cash_only(self):
        """Test portfolio with cash only."""
        result = calculate_portfolio_value(
            capital=10_000_000,
            positions=[],
            current_prices={}
        )

        assert result["cash"] == 10_000_000
        assert result["positions_value"] == 0
        assert result["total_value"] == 10_000_000
        assert result["unrealized_pnl"] == 0

    def test_with_positions(self):
        """Test portfolio with positions."""
        positions = [
            {"symbol": "BBCA", "quantity": 1000, "avg_price": 9000},
        ]
        current_prices = {"BBCA": 9500}

        result = calculate_portfolio_value(
            capital=1_000_000,
            positions=positions,
            current_prices=current_prices
        )

        assert result["positions_value"] == 9_500_000
        assert result["total_value"] == 10_500_000
        assert result["unrealized_pnl"] == 500_000

    def test_with_multiple_positions(self):
        """Test portfolio with multiple positions."""
        positions = [
            {"symbol": "BBCA", "quantity": 1000, "avg_price": 9000},
            {"symbol": "TLKM", "quantity": 500, "avg_price": 4000},
        ]
        current_prices = {"BBCA": 9500, "TLKM": 4500}

        result = calculate_portfolio_value(
            capital=5_000_000,
            positions=positions,
            current_prices=current_prices
        )

        # BBCA: 1000 * 9500 = 9,500,000, PnL: 500,000
        # TLKM: 500 * 4500 = 2,250,000, PnL: 250,000
        assert result["positions_value"] == 11_750_000
        assert result["unrealized_pnl"] == 750_000

    def test_with_loss(self):
        """Test portfolio with losing position."""
        positions = [
            {"symbol": "BBCA", "quantity": 1000, "avg_price": 9000},
        ]
        current_prices = {"BBCA": 8500}

        result = calculate_portfolio_value(
            capital=1_000_000,
            positions=positions,
            current_prices=current_prices
        )

        assert result["unrealized_pnl"] == -500_000

    def test_missing_current_price(self):
        """Test portfolio when current price is missing."""
        positions = [
            {"symbol": "BBCA", "quantity": 1000, "avg_price": 9000},
        ]
        current_prices = {}  # Missing BBCA price

        result = calculate_portfolio_value(
            capital=10_000_000,
            positions=positions,
            current_prices=current_prices
        )

        # Position with missing price should be valued at 0
        assert result["positions_value"] == 0


class TestCalculatePerformanceMetrics:
    """Tests for calculate_performance_metrics function."""

    def test_empty_trades(self):
        """Test metrics with no trades."""
        result = calculate_performance_metrics([])

        assert result["total_trades"] == 0
        assert result["win_rate"] == 0
        assert result["profit_factor"] == 0

    def test_all_winning_trades(self):
        """Test metrics with all winning trades."""
        trades = [
            {"pnl": 100000},
            {"pnl": 150000},
            {"pnl": 200000},
        ]

        result = calculate_performance_metrics(trades)

        assert result["total_trades"] == 3
        assert result["winning_trades"] == 3
        assert result["losing_trades"] == 0
        assert result["win_rate"] == 1.0
        assert result["total_pnl"] == 450000
        assert result["profit_factor"] == float('inf')

    def test_all_losing_trades(self):
        """Test metrics with all losing trades."""
        trades = [
            {"pnl": -100000},
            {"pnl": -150000},
            {"pnl": -200000},
        ]

        result = calculate_performance_metrics(trades)

        assert result["winning_trades"] == 0
        assert result["losing_trades"] == 3
        assert result["win_rate"] == 0.0
        assert result["total_pnl"] == -450000
        assert result["profit_factor"] == 0

    def test_mixed_trades(self):
        """Test metrics with mixed winning and losing trades."""
        trades = [
            {"pnl": 200000},   # Win
            {"pnl": -100000},  # Loss
            {"pnl": 150000},   # Win
            {"pnl": -50000},   # Loss
        ]

        result = calculate_performance_metrics(trades)

        assert result["total_trades"] == 4
        assert result["winning_trades"] == 2
        assert result["losing_trades"] == 2
        assert result["win_rate"] == 0.5
        assert result["total_pnl"] == 200000
        # Profit factor = gross_profit / gross_loss = 350000 / 150000 = 2.33
        assert result["profit_factor"] == pytest.approx(2.33, rel=0.1)

    def test_calculates_avg_win_loss(self):
        """Test that average win/loss are calculated correctly."""
        trades = [
            {"pnl": 100000},
            {"pnl": 200000},
            {"pnl": -50000},
            {"pnl": -100000},
        ]

        result = calculate_performance_metrics(trades)

        assert result["avg_win"] == 150000  # (100000 + 200000) / 2
        assert result["avg_loss"] == -75000  # (-50000 + -100000) / 2


class TestProcessTradeHistory:
    """Tests for process_trade_history function."""

    def test_empty_history(self):
        """Test processing empty trade history."""
        result = process_trade_history([])

        assert result.empty

    def test_with_trades(self):
        """Test processing trade history with trades."""
        trades = [
            {"symbol": "BBCA", "side": "BUY", "quantity": 1000, "price": 9000, "pnl": 0, "timestamp": "2024-01-01"},
            {"symbol": "BBCA", "side": "SELL", "quantity": 1000, "price": 9500, "pnl": 500000, "timestamp": "2024-01-05"},
        ]

        result = process_trade_history(trades)

        assert len(result) == 2
        assert "symbol" in result.columns
        assert "pnl" in result.columns

    def test_sorts_by_timestamp(self):
        """Test that history is sorted by timestamp."""
        trades = [
            {"symbol": "TLKM", "side": "BUY", "quantity": 500, "price": 4000, "pnl": 0, "timestamp": "2024-01-10"},
            {"symbol": "BBCA", "side": "BUY", "quantity": 1000, "price": 9000, "pnl": 0, "timestamp": "2024-01-01"},
        ]

        result = process_trade_history(trades)

        # Most recent should be first
        assert result.iloc[0]["symbol"] == "TLKM"


class TestGenerateEquityCurve:
    """Tests for generate_equity_curve function."""

    def test_empty_trades(self):
        """Test equity curve with no trades."""
        result = generate_equity_curve(10_000_000, [])

        assert len(result) == 1
        assert result[0]["value"] == 10_000_000

    def test_with_trades(self):
        """Test equity curve with trades."""
        initial_capital = 10_000_000
        trades = [
            {"timestamp": "2024-01-01", "pnl": 100000},
            {"timestamp": "2024-01-05", "pnl": 200000},
            {"timestamp": "2024-01-10", "pnl": -50000},
        ]

        result = generate_equity_curve(initial_capital, trades)

        assert len(result) == 3
        assert result[0]["value"] == 10_100_000  # 10M + 100k
        assert result[1]["value"] == 10_300_000  # 10.1M + 200k
        assert result[2]["value"] == 10_250_000  # 10.3M - 50k

    def test_cumulative_calculation(self):
        """Test that equity curve is cumulative."""
        initial_capital = 1_000_000
        trades = [
            {"timestamp": "2024-01-01", "pnl": 100000},
            {"timestamp": "2024-01-02", "pnl": 100000},
            {"timestamp": "2024-01-03", "pnl": 100000},
        ]

        result = generate_equity_curve(initial_capital, trades)

        # Should show 1.1M, 1.2M, 1.3M
        assert result[0]["value"] == 1_100_000
        assert result[1]["value"] == 1_200_000
        assert result[2]["value"] == 1_300_000
