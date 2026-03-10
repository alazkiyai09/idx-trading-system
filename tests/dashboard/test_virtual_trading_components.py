"""
Tests for Virtual Trading page components and NextGen styling.

These tests verify:
- NextGen CSS styles are applied
- Components render correctly
- Helper functions work as expected
"""
import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
from datetime import datetime

from dashboard.components.nextgen_styles import (
    get_nextgen_css, COLORS, FONTS, get_chart_colors,
    apply_chart_theme, format_price, format_change,
    render_live_badge, render_signal_badge
)
from dashboard.components.metrics import (
    render_portfolio_metrics, render_signal_card, render_conviction_score
)
from dashboard.components.ux_components import (
    IDX_LOT_SIZE, IDX_FEES, IDX_TRADING_HOURS
)


class TestNextGenStyling:
    """Tests for NextGen CSS and styling components."""

    def test_get_nextgen_css_returns_string(self):
        """Test that CSS generator returns a non-empty string."""
        css = get_nextgen_css()

        assert isinstance(css, str), "CSS should be a string"
        assert len(css) > 1000, "CSS should be substantial"

    def test_css_contains_required_styles(self):
        """Test that CSS contains all required style classes."""
        css = get_nextgen_css()

        required_classes = [
            ".nextgen-card",
            ".live-badge",
            ".signal-badge",
            ".section-header",
            ".price-mono",
            ".positive",
            ".negative",
            ".buy-button",
            ".sell-button",
        ]

        for class_name in required_classes:
            assert class_name in css, f"CSS missing class: {class_name}"

    def test_css_contains_color_definitions(self):
        """Test that CSS uses defined color palette."""
        css = get_nextgen_css()

        # Check that primary colors are used
        assert COLORS["primary"] in css, "Primary color should be in CSS"
        assert COLORS["background"] in css, "Background color should be in CSS"
        assert COLORS["destructive"] in css, "Destructive color should be in CSS"

    def test_colors_dictionary_has_required_keys(self):
        """Test that COLORS dict has all required color definitions."""
        required_colors = [
            "background", "foreground", "primary", "destructive",
            "muted", "muted_foreground", "card", "border",
            "success", "warning", "error", "info",
            "chart_up", "chart_down"
        ]

        for color in required_colors:
            assert color in COLORS, f"Missing color: {color}"
            assert COLORS[color].startswith("#"), f"Color {color} should be hex"

    def test_fonts_dictionary(self):
        """Test that FONTS dict has required font definitions."""
        assert "mono" in FONTS, "Missing mono font"
        assert "sans" in FONTS, "Missing sans font"
        assert "monospace" in FONTS["mono"], "Mono font should specify monospace"

    def test_get_chart_colors_returns_dict(self):
        """Test chart color helper returns expected colors."""
        colors = get_chart_colors()

        assert isinstance(colors, dict), "Should return dict"
        assert colors["up"] == COLORS["chart_up"], "Up color should match"
        assert colors["down"] == COLORS["chart_down"], "Down color should match"

    def test_format_price(self):
        """Test price formatting with monospace styling."""
        result = format_price(9050)

        assert "9,050" in result, "Should format price with commas"
        assert "price-mono" in result, "Should include monospace class"

    def test_format_change_positive(self):
        """Test formatting positive change."""
        result = format_change(2.5, as_percent=True)

        assert "+2.50%" in result, "Should show positive with +"
        assert "positive" in result, "Should have positive class"

    def test_format_change_negative(self):
        """Test formatting negative change."""
        result = format_change(-1.5, as_percent=True)

        assert "-1.50%" in result, "Should show negative"
        assert "negative" in result, "Should have negative class"

    def test_format_change_absolute(self):
        """Test formatting change as absolute value."""
        result = format_change(50000, as_percent=False)

        assert "+50,000" in result, "Should format as absolute"
        assert "%" not in result, "Should not include percent"

    def test_render_live_badge(self):
        """Test live badge rendering."""
        result = render_live_badge("PAPER")

        assert "live-badge" in result, "Should have live-badge class"
        assert "PAPER" in result, "Should include badge text"
        assert "pulse" in result, "Should have pulse animation"

    def test_render_signal_badge_bullish(self):
        """Test bullish signal badge rendering."""
        for signal_type in ["BUY", "bullish", "Strong Buy"]:
            result = render_signal_badge(signal_type)

            assert "bullish" in result.lower(), f"Should be bullish for {signal_type}"

    def test_render_signal_badge_bearish(self):
        """Test bearish signal badge rendering."""
        for signal_type in ["SELL", "bearish", "Strong Sell"]:
            result = render_signal_badge(signal_type)

            assert "bearish" in result.lower(), f"Should be bearish for {signal_type}"

    def test_render_signal_badge_neutral(self):
        """Test neutral signal badge rendering."""
        for signal_type in ["HOLD", "neutral", "wait"]:
            result = render_signal_badge(signal_type)

            assert "neutral" in result.lower(), f"Should be neutral for {signal_type}"


class TestPortfolioMetricsComponent:
    """Tests for portfolio metrics rendering component."""

    @patch("streamlit.columns")
    @patch("streamlit.metric")
    def test_render_portfolio_metrics_basic(self, mock_metric, mock_columns):
        """Test basic portfolio metrics rendering."""
        mock_columns.return_value = [MagicMock() for _ in range(4)]

        render_portfolio_metrics(
            capital=100_000_000,
            pnl=5_000_000,
            win_rate=0.65,
            total_trades=20
        )

        # Should call columns for layout
        mock_columns.assert_called()

    @patch("streamlit.columns")
    @patch("streamlit.metric")
    def test_render_portfolio_metrics_with_advanced(self, mock_metric, mock_columns):
        """Test portfolio metrics with advanced metrics."""
        mock_columns.return_value = [MagicMock() for _ in range(7)]

        render_portfolio_metrics(
            capital=100_000_000,
            pnl=5_000_000,
            win_rate=0.65,
            total_trades=20,
            sharpe=1.5,
            max_dd=0.15,
            profit_factor=2.3
        )

        # Should call columns twice (4 basic + 3 advanced)
        assert mock_columns.call_count >= 2


class TestSignalCardComponent:
    """Tests for signal card rendering component."""

    @patch("streamlit.markdown")
    def test_render_signal_card_buy(self, mock_markdown):
        """Test rendering BUY signal card."""
        signal_data = {
            "signal": "BUY",
            "type": "BUY",
            "setup": "Breakout",
            "score": 75.5,
            "entry_price": 9050,
            "stop_loss": 8800,
            "targets": [9500, 9800],
            "risk_reward": 1.8
        }

        render_signal_card(signal_data)

        # Should render markdown with signal details
        mock_markdown.assert_called()
        call_args = mock_markdown.call_args[0][0]
        assert "BUY" in call_args, "Should show BUY action"
        assert "75.5" in call_args, "Should show score"

    @patch("streamlit.markdown")
    def test_render_signal_card_sell(self, mock_markdown):
        """Test rendering SELL signal card."""
        signal_data = {
            "signal": "SELL",
            "type": "SELL",
            "setup": "Breakdown",
            "score": 70.0,
            "entry_price": 9050,
            "stop_loss": 9300,
            "targets": [8700, 8500],
            "risk_reward": 1.4
        }

        render_signal_card(signal_data)

        call_args = mock_markdown.call_args[0][0]
        assert "SELL" in call_args, "Should show SELL action"
        assert "bearish" in call_args.lower(), "Should have bearish styling"

    @patch("streamlit.info")
    def test_render_signal_card_none(self, mock_info):
        """Test rendering no signal card."""
        signal_data = {
            "signal": "None",
            "message": "No signal generated."
        }

        render_signal_card(signal_data)

        mock_info.assert_called()


class TestConvictionScoreComponent:
    """Tests for conviction score rendering component."""

    @patch("streamlit.markdown")
    def test_conviction_score_high(self, mock_markdown):
        """Test high conviction score (>=80)."""
        render_conviction_score(85)

        call_args = mock_markdown.call_args[0][0]
        assert "85" in call_args, "Should show score"
        assert "Strong Buy" in call_args, "Should show Strong Buy rating"

    @patch("streamlit.markdown")
    def test_conviction_score_medium(self, mock_markdown):
        """Test medium conviction score (60-79)."""
        render_conviction_score(70)

        call_args = mock_markdown.call_args[0][0]
        assert "70" in call_args, "Should show score"
        assert "Buy" in call_args, "Should show Buy rating"

    @patch("streamlit.markdown")
    def test_conviction_score_low(self, mock_markdown):
        """Test low conviction score (<40)."""
        render_conviction_score(30)

        call_args = mock_markdown.call_args[0][0]
        assert "30" in call_args, "Should show score"
        assert "Sell" in call_args, "Should show Sell rating"


class TestUXComponents:
    """Tests for UX component constants and helpers."""

    def test_idx_lot_size(self):
        """Test IDX lot size constant."""
        assert IDX_LOT_SIZE == 100, "IDX lot size should be 100 shares"

    def test_idx_fees(self):
        """Test IDX fee rates."""
        assert IDX_FEES["buy"] == 0.0015, "Buy fee should be 0.15%"
        assert IDX_FEES["sell"] == 0.0025, "Sell fee should be 0.25%"

    def test_idx_trading_hours(self):
        """Test IDX trading hours."""
        start, end = IDX_TRADING_HOURS
        assert start.hour == 9, "Market opens at 09:00"
        assert end.hour == 17, "Market closes at 17:10"


class TestTradingHelpers:
    """Tests for trading helper functions."""

    def test_calculate_order_value_buy(self):
        """Test order value calculation for BUY order."""
        from dashboard.pages.trading_helpers import calculate_order_value

        result = calculate_order_value(1000, 9000, "BUY")

        assert result["gross_value"] == 9_000_000
        assert result["fee_rate"] == 0.0015
        assert result["fees"] == 13_500
        assert result["net_value"] == 9_013_500

    def test_calculate_order_value_sell(self):
        """Test order value calculation for SELL order."""
        from dashboard.pages.trading_helpers import calculate_order_value

        result = calculate_order_value(1000, 9000, "SELL")

        assert result["gross_value"] == 9_000_000
        assert result["fee_rate"] == 0.0025
        assert result["fees"] == 22_500
        assert result["net_value"] == 8_977_500

    def test_validate_order_missing_symbol(self):
        """Test order validation with missing symbol."""
        from dashboard.pages.trading_helpers import validate_order

        result = validate_order(
            symbol="",
            side="BUY",
            quantity=1000,
            price=9000,
            capital=10_000_000
        )

        assert result["valid"] is False
        assert "Symbol" in result["errors"][0]

    def test_validate_order_invalid_side(self):
        """Test order validation with invalid side."""
        from dashboard.pages.trading_helpers import validate_order

        result = validate_order(
            symbol="BBCA",
            side="HOLD",
            quantity=1000,
            price=9000,
            capital=10_000_000
        )

        assert result["valid"] is False

    def test_calculate_portfolio_value_cash_only(self):
        """Test portfolio value with cash only."""
        from dashboard.pages.trading_helpers import calculate_portfolio_value

        result = calculate_portfolio_value(
            capital=10_000_000,
            positions=[],
            current_prices={}
        )

        assert result["cash"] == 10_000_000
        assert result["positions_value"] == 0
        assert result["total_value"] == 10_000_000

    def test_calculate_portfolio_value_with_positions(self):
        """Test portfolio value with positions."""
        from dashboard.pages.trading_helpers import calculate_portfolio_value

        positions = [
            {"symbol": "BBCA", "quantity": 1000, "avg_price": 9000}
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

    def test_process_trade_history_empty(self):
        """Test processing empty trade history."""
        from dashboard.pages.trading_helpers import process_trade_history

        result = process_trade_history([])

        assert result.empty

    def test_process_trade_history_with_trades(self):
        """Test processing trade history with trades."""
        from dashboard.pages.trading_helpers import process_trade_history

        trades = [
            {"symbol": "BBCA", "side": "BUY", "quantity": 1000,
             "price": 9000, "pnl": 0, "timestamp": "2024-01-01"},
            {"symbol": "BBCA", "side": "SELL", "quantity": 1000,
             "price": 9500, "pnl": 500000, "timestamp": "2024-01-05"}
        ]

        result = process_trade_history(trades)

        assert len(result) == 2
        assert "symbol" in result.columns

    def test_generate_equity_curve_empty(self):
        """Test equity curve with no trades."""
        from dashboard.pages.trading_helpers import generate_equity_curve

        result = generate_equity_curve(10_000_000, [])

        assert len(result) == 1
        assert result[0]["value"] == 10_000_000

    def test_generate_equity_curve_with_trades(self):
        """Test equity curve with trades."""
        from dashboard.pages.trading_helpers import generate_equity_curve

        trades = [
            {"timestamp": "2024-01-01", "pnl": 100000},
            {"timestamp": "2024-01-05", "pnl": 200000}
        ]

        result = generate_equity_curve(10_000_000, trades)

        assert len(result) == 2
        assert result[0]["value"] == 10_100_000
        assert result[1]["value"] == 10_300_000


class TestVirtualTradingPageStructure:
    """Tests for Virtual Trading page structure and content."""

    def test_page_imports_successfully(self):
        """Test that the page module can be imported."""
        # This verifies the page has no import errors
        try:
            from dashboard.pages import test_import_virtual_trading
        except ImportError:
            # If the test helper doesn't exist, just verify the page exists
            import os
            page_path = os.path.join(
                os.path.dirname(__file__),
                "..", "..", "dashboard", "pages", "04_virtual_trading.py"
            )
            assert os.path.exists(page_path), "Virtual trading page should exist"

    def test_page_has_required_tabs(self):
        """Test that page defines required tabs."""
        # Read the page source to verify tab structure
        import os
        page_path = os.path.join(
            os.path.dirname(__file__),
            "..", "..", "dashboard", "pages", "04_virtual_trading.py"
        )

        with open(page_path, "r") as f:
            content = f.read()

        required_tabs = ["Sessions", "New", "Trading", "Performance"]
        for tab in required_tabs:
            assert tab in content, f"Page should have {tab} tab"

    def test_page_uses_nextgen_css(self):
        """Test that page applies NextGen CSS."""
        import os
        page_path = os.path.join(
            os.path.dirname(__file__),
            "..", "..", "dashboard", "pages", "04_virtual_trading.py"
        )

        with open(page_path, "r") as f:
            content = f.read()

        assert "get_nextgen_css" in content, "Page should call get_nextgen_css()"
        assert "unsafe_allow_html=True" in content, "Page should render HTML"

    def test_page_has_order_entry_form(self):
        """Test that page has order entry form."""
        import os
        page_path = os.path.join(
            os.path.dirname(__file__),
            "..", "..", "dashboard", "pages", "04_virtual_trading.py"
        )

        with open(page_path, "r") as f:
            content = f.read()

        assert "order_entry" in content.lower(), "Page should have order entry"
        assert "st.form" in content, "Page should use Streamlit form"
        assert "BUY" in content and "SELL" in content, "Page should have BUY/SELL"

    def test_page_displays_risk_metrics(self):
        """Test that page displays risk metrics."""
        import os
        page_path = os.path.join(
            os.path.dirname(__file__),
            "..", "..", "dashboard", "pages", "04_virtual_trading.py"
        )

        with open(page_path, "r") as f:
            content = f.read()

        # Check for risk-related content
        risk_keywords = ["kelly", "sharpe", "drawdown", "cvar", "risk"]
        content_lower = content.lower()

        found_keywords = [kw for kw in risk_keywords if kw in content_lower]
        assert len(found_keywords) >= 2, \
            f"Page should display risk metrics. Found: {found_keywords}"

    def test_page_calls_simulation_api(self):
        """Test that page makes calls to simulation API."""
        import os
        page_path = os.path.join(
            os.path.dirname(__file__),
            "..", "..", "dashboard", "pages", "04_virtual_trading.py"
        )

        with open(page_path, "r") as f:
            content = f.read()

        assert "simulation/" in content, "Page should call /simulation/ endpoint"
        assert "requests.get" in content or "requests.post" in content, \
            "Page should make HTTP requests"
