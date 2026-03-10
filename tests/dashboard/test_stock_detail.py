"""Tests for Stock Detail page helper functions."""
import pytest
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from dashboard.pages.stock_detail_helpers import (
    process_price_history,
    calculate_price_change,
    calculate_position_metrics,
    validate_trade_params,
    process_foreign_flow_data,
    calculate_flow_summary,
    format_sentiment_score,
)


class TestProcessPriceHistory:
    """Tests for process_price_history function."""

    def test_with_valid_data(self):
        """Test processing valid price data."""
        price_data = [
            {"date": "2024-01-01", "open": 9000, "high": 9100, "low": 8950, "close": 9050, "volume": 1000000},
            {"date": "2024-01-02", "open": 9050, "high": 9200, "low": 9000, "close": 9150, "volume": 1200000},
            {"date": "2024-01-03", "open": 9150, "high": 9300, "low": 9100, "close": 9250, "volume": 1100000},
        ]

        result = process_price_history(price_data, days=30)

        assert len(result) == 3
        assert "date" in result.columns
        assert "close" in result.columns
        assert result["close"].iloc[-1] == 9250

    def test_with_empty_data(self):
        """Test processing empty price data."""
        result = process_price_history([], days=30)

        assert result.empty

    def test_with_days_limit(self):
        """Test that days parameter limits data."""
        # Create 100 days of data with proper date format
        base_date = datetime(2024, 1, 1)
        price_data = []
        for i in range(100):
            date = base_date + timedelta(days=i)
            price_data.append({
                "date": date.strftime("%Y-%m-%d"),
                "open": 9000 + i,
                "high": 9100 + i,
                "low": 8950 + i,
                "close": 9050 + i,
                "volume": 1000000
            })

        result = process_price_history(price_data, days=50)

        assert len(result) == 50

    def test_date_sorting(self):
        """Test that data is sorted by date."""
        price_data = [
            {"date": "2024-01-15", "open": 9000, "high": 9100, "low": 8950, "close": 9050, "volume": 1000000},
            {"date": "2024-01-01", "open": 8800, "high": 8900, "low": 8750, "close": 8850, "volume": 800000},
            {"date": "2024-01-10", "open": 8900, "high": 9000, "low": 8850, "close": 8950, "volume": 900000},
        ]

        result = process_price_history(price_data, days=10)

        # Should be sorted with oldest first
        assert result["date"].iloc[0] < result["date"].iloc[-1]


class TestCalculatePriceChange:
    """Tests for calculate_price_change function."""

    def test_positive_change(self):
        """Test positive price change calculation."""
        result = calculate_price_change(9500, 9000)

        assert result["change"] == 500
        assert result["change_pct"] == pytest.approx(5.56, rel=0.01)

    def test_negative_change(self):
        """Test negative price change calculation."""
        result = calculate_price_change(8500, 9000)

        assert result["change"] == -500
        assert result["change_pct"] == pytest.approx(-5.56, rel=0.01)

    def test_zero_change(self):
        """Test zero price change."""
        result = calculate_price_change(9000, 9000)

        assert result["change"] == 0
        assert result["change_pct"] == 0

    def test_zero_previous_price(self):
        """Test with zero previous price (edge case)."""
        result = calculate_price_change(9000, 0)

        assert result["change"] == 0
        assert result["change_pct"] == 0


class TestCalculatePositionMetrics:
    """Tests for calculate_position_metrics function."""

    def test_profitable_position(self):
        """Test profitable position metrics."""
        result = calculate_position_metrics(9000, 9500, 1000)

        assert result["position_value"] == 9_500_000
        assert result["cost_basis"] == 9_000_000
        assert result["pnl"] == 500_000
        assert result["pnl_pct"] == pytest.approx(5.56, rel=0.01)

    def test_losing_position(self):
        """Test losing position metrics."""
        result = calculate_position_metrics(9000, 8500, 1000)

        assert result["pnl"] == -500_000
        assert result["pnl_pct"] == pytest.approx(-5.56, rel=0.01)

    def test_zero_entry_price(self):
        """Test with zero entry price (edge case)."""
        result = calculate_position_metrics(0, 9000, 1000)

        assert result["pnl_pct"] == 0


class TestValidateTradeParams:
    """Tests for validate_trade_params function."""

    def test_valid_buy_order(self):
        """Test valid buy order parameters."""
        result = validate_trade_params("BBCA", "BUY", 1000, 9000, 10_000_000)

        assert result["valid"] is True
        assert len(result["errors"]) == 0
        assert result["order_value"] == 9_000_000

    def test_invalid_symbol(self):
        """Test with invalid symbol."""
        result = validate_trade_params("", "BUY", 1000, 9000, 10_000_000)

        assert result["valid"] is False
        assert "Invalid symbol" in result["errors"]

    def test_invalid_side(self):
        """Test with invalid side."""
        result = validate_trade_params("BBCA", "HOLD", 1000, 9000, 10_000_000)

        assert result["valid"] is False
        assert "Side must be BUY or SELL" in result["errors"]

    def test_invalid_quantity_not_lot(self):
        """Test with quantity not multiple of lot size."""
        result = validate_trade_params("BBCA", "BUY", 150, 9000, 10_000_000)

        assert result["valid"] is False
        assert any("lot size" in e for e in result["errors"])

    def test_insufficient_capital(self):
        """Test with insufficient capital."""
        result = validate_trade_params("BBCA", "BUY", 1000, 9000, 5_000_000)

        assert result["valid"] is False
        assert any("capital" in e.lower() for e in result["errors"])

    def test_zero_price(self):
        """Test with zero price."""
        result = validate_trade_params("BBCA", "BUY", 1000, 0, 10_000_000)

        assert result["valid"] is False
        assert any("positive" in e.lower() for e in result["errors"])


class TestProcessForeignFlowData:
    """Tests for process_foreign_flow_data function."""

    def test_with_valid_data(self):
        """Test processing valid foreign flow data."""
        flow_data = [
            {"date": "2024-01-01", "foreign_buy": 10_000_000_000, "foreign_sell": 5_000_000_000},
            {"date": "2024-01-02", "foreign_buy": 8_000_000_000, "foreign_sell": 6_000_000_000},
        ]

        result = process_foreign_flow_data(flow_data)

        assert len(result) == 2
        assert "foreign_net" in result.columns
        assert result["foreign_net"].iloc[0] == 5_000_000_000

    def test_with_empty_data(self):
        """Test processing empty flow data."""
        result = process_foreign_flow_data([])

        assert result.empty

    def test_net_flow_calculation(self):
        """Test that net flow is calculated correctly."""
        flow_data = [
            {"date": "2024-01-01", "foreign_buy": 100, "foreign_sell": 50},
        ]

        result = process_foreign_flow_data(flow_data)

        assert result["foreign_net"].iloc[0] == 50


class TestCalculateFlowSummary:
    """Tests for calculate_flow_summary function."""

    def test_inflow_summary(self):
        """Test summary for net inflow."""
        flow_df = pd.DataFrame({
            "date": pd.date_range(start="2024-01-01", periods=5, freq="D"),
            "foreign_net": [1e9, 1.5e9, 2e9, 1.8e9, 2.2e9],
        })

        result = calculate_flow_summary(flow_df)

        assert result["direction"] == "inflow"
        assert result["net_flow"] > 0
        assert result["streak"] == 5

    def test_outflow_summary(self):
        """Test summary for net outflow."""
        flow_df = pd.DataFrame({
            "date": pd.date_range(start="2024-01-01", periods=5, freq="D"),
            "foreign_net": [-1e9, -1.5e9, -2e9, -1.8e9, -2.2e9],
        })

        result = calculate_flow_summary(flow_df)

        assert result["direction"] == "outflow"
        assert result["net_flow"] < 0
        assert result["streak"] == 5

    def test_mixed_flow_streak(self):
        """Test streak calculation with mixed flows."""
        flow_df = pd.DataFrame({
            "date": pd.date_range(start="2024-01-01", periods=5, freq="D"),
            "foreign_net": [1e9, 1.5e9, -1e9, 1e9, 2e9],  # Last 2 are positive
        })

        result = calculate_flow_summary(flow_df)

        assert result["streak"] == 2

    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        result = calculate_flow_summary(pd.DataFrame())

        assert result["net_flow"] == 0
        assert result["direction"] == "neutral"


class TestFormatSentimentScore:
    """Tests for format_sentiment_score function."""

    def test_bullish_score(self):
        """Test bullish sentiment (score >= 70)."""
        result = format_sentiment_score(75)

        assert result["label"] == "Bullish"
        assert result["emoji"] == "🟢"

    def test_slightly_bullish_score(self):
        """Test slightly bullish sentiment (55-69)."""
        result = format_sentiment_score(60)

        assert result["label"] == "Slightly Bullish"

    def test_neutral_score(self):
        """Test neutral sentiment (45-54)."""
        result = format_sentiment_score(50)

        assert result["label"] == "Neutral"
        assert result["emoji"] == "🟡"

    def test_slightly_bearish_score(self):
        """Test slightly bearish sentiment (30-44)."""
        result = format_sentiment_score(35)

        assert result["label"] == "Slightly Bearish"

    def test_bearish_score(self):
        """Test bearish sentiment (< 30)."""
        result = format_sentiment_score(20)

        assert result["label"] == "Bearish"
        assert result["emoji"] == "🔴"

    def test_boundary_values(self):
        """Test boundary values."""
        assert format_sentiment_score(70)["label"] == "Bullish"
        assert format_sentiment_score(69)["label"] == "Slightly Bullish"
        assert format_sentiment_score(55)["label"] == "Slightly Bullish"
        assert format_sentiment_score(54)["label"] == "Neutral"
        assert format_sentiment_score(30)["label"] == "Slightly Bearish"
        assert format_sentiment_score(29)["label"] == "Bearish"
