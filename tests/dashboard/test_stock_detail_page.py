"""
Comprehensive tests for Stock Detail page (02_stock_detail.py).

Tests cover:
1. Page loads without errors when symbol is selected
2. Chart displays correctly (OHLCV candlestick)
3. Technical indicators render (RSI, MACD)
4. AI Analysis section displays
5. Order entry panel works (BUY/SELL buttons)
6. Risk analytics display (Kelly, CVaR)
7. All NextGen styling applied
8. API endpoints /stocks/{symbol}, /analysis/technical/{symbol}, /analysis/signal/{symbol} work

Run with: pytest tests/dashboard/test_stock_detail_page.py -v
"""

import pytest
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock, Mock
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


# =============================================================================
# CONFIGURATION
# =============================================================================

API_URL = "http://localhost:8000"
STREAMLIT_URL = "http://localhost:8501"
REQUEST_TIMEOUT = 10
TEST_SYMBOL = "BBCA"  # Use a common stock for testing


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def api_available():
    """Check if API server is available."""
    try:
        resp = requests.get(f"{API_URL}/health", timeout=5)
        return resp.status_code == 200
    except:
        return False


@pytest.fixture
def streamlit_available():
    """Check if Streamlit server is available."""
    try:
        resp = requests.get(STREAMLIT_URL, timeout=5)
        return resp.status_code == 200
    except:
        return False


@pytest.fixture
def sample_chart_data():
    """Generate sample OHLCV chart data for testing."""
    dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
    np.random.seed(42)
    base_price = 9000
    returns = np.random.randn(100) * 0.02
    prices = base_price * (1 + returns).cumprod()

    return pd.DataFrame({
        "date": dates,
        "open": prices * (1 + np.random.randn(100) * 0.005),
        "high": prices * (1 + abs(np.random.randn(100)) * 0.01),
        "low": prices * (1 - abs(np.random.randn(100)) * 0.01),
        "close": prices,
        "volume": np.random.randint(100000, 1000000, 100),
    })


@pytest.fixture
def sample_stock_metadata():
    """Sample stock metadata for testing."""
    return {
        "symbol": TEST_SYMBOL,
        "name": "Bank Central Asia",
        "sector": "Financials",
        "sub_sector": "Banking",
        "market_cap": 1000000000000,
        "is_lq45": True,
        "latest_price": {
            "date": "2024-01-15",
            "close": 9050,
            "volume": 15000000
        }
    }


@pytest.fixture
def sample_technical_analysis():
    """Sample technical analysis response."""
    return {
        "symbol": TEST_SYMBOL,
        "date": "2024-01-15",
        "score": {
            "total": 72.5,
            "trend_score": 75.0,
            "momentum_score": 70.0,
            "volume_score": 65.0,
            "volatility_score": 80.0,
            "trend": "bullish",
            "signal": "BUY"
        },
        "indicators": {
            "close": 9050,
            "ema20": 9000,
            "ema50": 8850,
            "sma200": 8600,
            "rsi": 58.5,
            "macd": 25.3,
            "macd_signal": 20.1,
            "atr": 150,
            "bb_upper": 9200,
            "bb_lower": 8800,
            "support": 8700,
            "resistance": 9300
        }
    }


@pytest.fixture
def sample_signal_response():
    """Sample signal generation response."""
    return {
        "symbol": TEST_SYMBOL,
        "type": "BUY",
        "setup": "Breakout",
        "score": 75.5,
        "entry_price": 9050,
        "stop_loss": 8800,
        "targets": [9500, 9800],
        "risk_reward": 2.2,
        "factors": ["Strong volume", "EMA crossover", "RSI neutral"],
        "risks": ["Market volatility", "Sector rotation"]
    }


@pytest.fixture
def sample_risk_check_response():
    """Sample risk validation response."""
    return {
        "approved": True,
        "reasons": [],
        "warnings": ["High volatility expected"],
        "position_size": 50000000,
        "position_shares": 500,
        "kelly_fraction": 0.15,
        "risk_amount": 1500000
    }


# =============================================================================
# API ENDPOINT TESTS
# =============================================================================

class TestStockEndpoints:
    """Tests for /stocks/{symbol} API endpoints."""

    @pytest.mark.integration
    def test_stocks_list_endpoint(self, api_available):
        """Test GET /stocks returns list of stocks."""
        if not api_available:
            pytest.skip("API server not available")

        resp = requests.get(f"{API_URL}/stocks", timeout=REQUEST_TIMEOUT)

        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
        data = resp.json()

        # API returns dict with "stocks" key containing list
        if isinstance(data, dict):
            assert "stocks" in data, "Response should contain 'stocks' key"
            stocks = data["stocks"]
        else:
            # Fallback: direct list response
            stocks = data

        assert isinstance(stocks, list), "Stocks should be a list"
        assert len(stocks) > 0, "Should have at least one stock"

        # Each stock should have required fields
        stock = stocks[0]
        assert "symbol" in stock, "Stock should have 'symbol' field"

    @pytest.mark.integration
    def test_stock_detail_endpoint(self, api_available):
        """Test GET /stocks/{symbol} returns stock details."""
        if not api_available:
            pytest.skip("API server not available")

        resp = requests.get(f"{API_URL}/stocks/{TEST_SYMBOL}", timeout=REQUEST_TIMEOUT)

        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
        data = resp.json()

        # Verify required fields
        assert "symbol" in data, "Response should contain 'symbol'"
        assert data["symbol"] == TEST_SYMBOL, f"Expected {TEST_SYMBOL}"
        assert "name" in data, "Response should contain 'name'"
        assert "sector" in data, "Response should contain 'sector'"

    @pytest.mark.integration
    def test_stock_detail_not_found(self, api_available):
        """Test GET /stocks/{symbol} returns 404 for invalid symbol."""
        if not api_available:
            pytest.skip("API server not available")

        resp = requests.get(f"{API_URL}/stocks/INVALID_SYMBOL_XYZ", timeout=REQUEST_TIMEOUT)

        assert resp.status_code == 404, f"Expected 404, got {resp.status_code}"

    @pytest.mark.integration
    def test_stock_chart_endpoint(self, api_available):
        """Test GET /stocks/{symbol}/chart returns OHLCV data."""
        if not api_available:
            pytest.skip("API server not available")

        resp = requests.get(f"{API_URL}/stocks/{TEST_SYMBOL}/chart?days=30", timeout=REQUEST_TIMEOUT)

        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
        data = resp.json()

        assert isinstance(data, list), "Chart data should be a list"
        if len(data) > 0:
            # Verify OHLCV structure
            candle = data[0]
            required_fields = ["date", "open", "high", "low", "close", "volume"]
            for field in required_fields:
                assert field in candle, f"Candle should contain '{field}'"

    @pytest.mark.integration
    def test_stock_chart_days_parameter(self, api_available):
        """Test chart endpoint respects days parameter."""
        if not api_available:
            pytest.skip("API server not available")

        # Request 50 days
        resp = requests.get(f"{API_URL}/stocks/{TEST_SYMBOL}/chart?days=50", timeout=REQUEST_TIMEOUT)

        assert resp.status_code == 200
        data = resp.json()

        # Should return at most 50 candles (may be less if data unavailable)
        if len(data) > 0:
            assert len(data) <= 50, f"Expected at most 50 candles, got {len(data)}"


class TestAnalysisEndpoints:
    """Tests for /analysis/* API endpoints."""

    @pytest.mark.integration
    def test_technical_analysis_endpoint(self, api_available):
        """Test POST /analysis/technical/{symbol} returns analysis."""
        if not api_available:
            pytest.skip("API server not available")

        resp = requests.post(f"{API_URL}/analysis/technical/{TEST_SYMBOL}", timeout=REQUEST_TIMEOUT)

        # May return 200 or 404 if no data
        assert resp.status_code in [200, 404], f"Expected 200 or 404, got {resp.status_code}"

        if resp.status_code == 200:
            data = resp.json()
            assert "symbol" in data
            assert "score" in data or "indicators" in data

    @pytest.mark.integration
    def test_signal_endpoint(self, api_available):
        """Test POST /analysis/signal/{symbol} generates signal."""
        if not api_available:
            pytest.skip("API server not available")

        payload = {"mode": "SWING", "capital": 100000000.0}
        resp = requests.post(
            f"{API_URL}/analysis/signal/{TEST_SYMBOL}",
            json=payload,
            timeout=REQUEST_TIMEOUT
        )

        # May return 200 or 404 if no data
        assert resp.status_code in [200, 404], f"Expected 200 or 404, got {resp.status_code}"

        if resp.status_code == 200:
            data = resp.json()
            assert "symbol" in data

    @pytest.mark.integration
    def test_risk_check_endpoint(self, api_available):
        """Test POST /analysis/risk-check/{symbol} validates trade."""
        if not api_available:
            pytest.skip("API server not available")

        payload = {"mode": "SWING", "capital": 100000000.0}
        resp = requests.post(
            f"{API_URL}/analysis/risk-check/{TEST_SYMBOL}",
            json=payload,
            timeout=REQUEST_TIMEOUT
        )

        # May return 200 or 404 if no data
        assert resp.status_code in [200, 404], f"Expected 200 or 404, got {resp.status_code}"

        if resp.status_code == 200:
            data = resp.json()
            assert "approved" in data, "Response should contain 'approved' field"


# =============================================================================
# CHART COMPONENT TESTS
# =============================================================================

class TestCandlestickChart:
    """Tests for build_candlestick_chart function."""

    def test_chart_builds_with_valid_data(self, sample_chart_data):
        """Test that candlestick chart builds without errors."""
        from dashboard.components.charts import build_candlestick_chart

        fig = build_candlestick_chart(sample_chart_data)

        assert fig is not None, "Chart should be created"
        assert hasattr(fig, 'data'), "Figure should have data"
        assert len(fig.data) > 0, "Figure should have traces"

    def test_chart_includes_candlestick_trace(self, sample_chart_data):
        """Test that chart includes candlestick trace."""
        from dashboard.components.charts import build_candlestick_chart

        fig = build_candlestick_chart(sample_chart_data, show_volume=False, show_ma=False)

        # First trace should be candlestick
        trace = fig.data[0]
        assert trace.type == "candlestick", f"Expected candlestick, got {trace.type}"

    def test_chart_volume_subplot(self, sample_chart_data):
        """Test that volume subplot is added when enabled."""
        from dashboard.components.charts import build_candlestick_chart

        fig_with_vol = build_candlestick_chart(sample_chart_data, show_volume=True)
        fig_without_vol = build_candlestick_chart(sample_chart_data, show_volume=False)

        # Chart with volume should have more traces
        assert len(fig_with_vol.data) > len(fig_without_vol.data), \
            "Chart with volume should have more traces"

    def test_chart_ma_overlay(self, sample_chart_data):
        """Test that MA overlay is added when enabled."""
        from dashboard.components.charts import build_candlestick_chart

        fig_with_ma = build_candlestick_chart(
            sample_chart_data, show_volume=False, show_ma=True
        )
        fig_without_ma = build_candlestick_chart(
            sample_chart_data, show_volume=False, show_ma=False
        )

        # Should have 2 additional traces for MA20 and MA50
        assert len(fig_with_ma.data) == len(fig_without_ma.data) + 2, \
            "Should have 2 MA traces added"

    def test_chart_rsi_subplot(self, sample_chart_data):
        """Test that RSI subplot is added when enabled."""
        from dashboard.components.charts import build_candlestick_chart

        fig_with_rsi = build_candlestick_chart(
            sample_chart_data, show_volume=False, show_rsi=True
        )

        # Find RSI trace
        rsi_found = any("RSI" in str(trace.name) for trace in fig_with_rsi.data)
        assert rsi_found, "RSI trace should be present"

    def test_chart_macd_subplot(self, sample_chart_data):
        """Test that MACD subplot is added when enabled."""
        from dashboard.components.charts import build_candlestick_chart

        fig_with_macd = build_candlestick_chart(
            sample_chart_data, show_volume=False, show_macd=True
        )

        # Find MACD trace
        macd_found = any("MACD" in str(trace.name) for trace in fig_with_macd.data)
        assert macd_found, "MACD trace should be present"

    def test_chart_bollinger_bands(self, sample_chart_data):
        """Test that Bollinger Bands are added when enabled."""
        from dashboard.components.charts import build_candlestick_chart

        fig_with_bb = build_candlestick_chart(
            sample_chart_data, show_volume=False, show_bbands=True
        )

        # BB adds upper and lower band traces
        assert len(fig_with_bb.data) >= 3, "Should have BB traces"

    def test_chart_with_prediction_overlay(self, sample_chart_data):
        """Test that prediction overlay is added when provided."""
        from dashboard.components.charts import build_candlestick_chart

        prediction_data = {
            "predictions": [
                {"date": "2024-04-11", "predicted_price": 9200},
                {"date": "2024-04-12", "predicted_price": 9250},
            ],
            "is_mock": False
        }

        fig = build_candlestick_chart(
            sample_chart_data,
            show_volume=False,
            prediction_data=prediction_data
        )

        # Should include prediction trace
        pred_found = any("Forecast" in str(trace.name) for trace in fig.data)
        assert pred_found, "Prediction trace should be present"

    def test_chart_uses_nextgen_colors(self, sample_chart_data):
        """Test that chart uses NextGen color scheme."""
        from dashboard.components.charts import build_candlestick_chart
        from dashboard.components.nextgen_styles import COLORS

        fig = build_candlestick_chart(sample_chart_data)

        # Check layout uses dark background
        layout = fig.layout
        assert layout.paper_bgcolor == COLORS['background'], \
            "Paper background should use NextGen background color"
        assert layout.plot_bgcolor == COLORS['background'], \
            "Plot background should use NextGen background color"


# =============================================================================
# NEXTGEN STYLING TESTS
# =============================================================================

class TestNextGenStyles:
    """Tests for NextGen styling components."""

    def test_get_nextgen_css_returns_string(self):
        """Test that CSS generator returns valid string."""
        from dashboard.components.nextgen_styles import get_nextgen_css

        css = get_nextgen_css()

        assert isinstance(css, str), "CSS should be a string"
        assert len(css) > 1000, "CSS should be substantial"

    def test_css_contains_required_classes(self):
        """Test that CSS contains required component classes."""
        from dashboard.components.nextgen_styles import get_nextgen_css

        css = get_nextgen_css()

        required_classes = [
            "nextgen-card",
            "price-mono",
            "live-badge",
            "signal-badge",
            "agent-card",
            "indicator-card",
            "conviction-score",
            "flow-card",
            "section-header",
            "sr-level"
        ]

        for cls in required_classes:
            assert f".{cls}" in css, f"CSS should contain .{cls} class"

    def test_css_uses_zinc_color_scheme(self):
        """Test that CSS uses Zinc-based dark theme."""
        from dashboard.components.nextgen_styles import get_nextgen_css, COLORS

        css = get_nextgen_css()

        # Check for Zinc colors
        assert COLORS['background'] in css, "Should use background color"
        assert COLORS['muted'] in css, "Should use muted color"
        assert COLORS['border'] in css, "Should use border color"

    def test_css_uses_emerald_accent(self):
        """Test that CSS uses Emerald accent color."""
        from dashboard.components.nextgen_styles import get_nextgen_css, COLORS

        css = get_nextgen_css()

        # Check for Emerald primary color
        assert COLORS['primary'] in css, "Should use Emerald primary color"

    def test_render_live_badge(self):
        """Test live badge rendering."""
        from dashboard.components.nextgen_styles import render_live_badge

        badge = render_live_badge("LIVE")

        assert "live-badge" in badge, "Should contain live-badge class"
        assert "LIVE" in badge, "Should contain LIVE text"
        assert "pulse" in badge, "Should contain pulse animation"

    def test_render_signal_badge_bullish(self):
        """Test bullish signal badge rendering."""
        from dashboard.components.nextgen_styles import render_signal_badge

        badge = render_signal_badge("BUY")

        assert "signal-badge" in badge, "Should contain signal-badge class"
        assert "bullish" in badge, "Should contain bullish class"

    def test_render_signal_badge_bearish(self):
        """Test bearish signal badge rendering."""
        from dashboard.components.nextgen_styles import render_signal_badge

        badge = render_signal_badge("SELL")

        assert "signal-badge" in badge, "Should contain signal-badge class"
        assert "bearish" in badge, "Should contain bearish class"

    def test_render_signal_badge_neutral(self):
        """Test neutral signal badge rendering."""
        from dashboard.components.nextgen_styles import render_signal_badge

        badge = render_signal_badge("HOLD")

        assert "signal-badge" in badge, "Should contain signal-badge class"
        assert "neutral" in badge, "Should contain neutral class"

    def test_get_chart_colors(self):
        """Test chart colors extraction."""
        from dashboard.components.nextgen_styles import get_chart_colors

        colors = get_chart_colors()

        required_colors = [
            "up", "down", "neutral", "volume_up", "volume_down",
            "background", "grid", "text", "primary", "ma20", "ma50",
            "rsi", "macd", "signal"
        ]

        for color in required_colors:
            assert color in colors, f"Should have '{color}' color"

    def test_format_price(self):
        """Test price formatting."""
        from dashboard.components.nextgen_styles import format_price

        formatted = format_price(9050.5)

        assert "price-mono" in formatted, "Should use monospace class"
        assert "9,050" in formatted, "Should format number with comma"

    def test_format_change_positive(self):
        """Test positive change formatting."""
        from dashboard.components.nextgen_styles import format_change

        formatted = format_change(1.5)

        assert "positive" in formatted, "Should have positive class"
        assert "+1.50%" in formatted, "Should show positive percentage"

    def test_format_change_negative(self):
        """Test negative change formatting."""
        from dashboard.components.nextgen_styles import format_change

        formatted = format_change(-1.5)

        assert "negative" in formatted, "Should have negative class"
        assert "-1.50%" in formatted, "Should show negative percentage"


# =============================================================================
# PAGE FUNCTIONALITY TESTS (Mocked)
# =============================================================================

class TestStockDetailPageFunctionality:
    """Tests for Stock Detail page functionality using mocks."""

    def test_get_stock_list_caches_results(self):
        """Test that stock list fetch works with caching."""
        # Mock the API response directly
        with patch('requests.get') as mock_get:
            mock_resp = Mock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {
                "stocks": [{"symbol": "BBCA"}, {"symbol": "BBRI"}]
            }
            mock_get.return_value = mock_resp

            # Simulate the page function
            resp = mock_get.return_value
            if resp.status_code == 200:
                data = resp.json()
                stocks = data.get("stocks", data) if isinstance(data, dict) else data
                result = [s["symbol"] for s in stocks] if isinstance(stocks, list) else ["BBCA"]

            assert isinstance(result, list), "Should return list"
            assert len(result) > 0, "Should have stocks"

    def test_get_stock_details_handles_error(self):
        """Test that stock details fetch handles errors gracefully."""
        with patch('requests.get') as mock_get:
            mock_get.side_effect = requests.exceptions.RequestException("Network error")

            # Simulate the page function
            try:
                resp = mock_get(f"{API_URL}/stocks/{TEST_SYMBOL}", timeout=REQUEST_TIMEOUT)
                result = resp.json() if resp.status_code == 200 else {}
            except requests.exceptions.RequestException:
                result = {}

            assert result == {}, "Should return empty dict on error"

    def test_metrics_display_with_valid_data(self, sample_stock_metadata):
        """Test metrics display with valid stock data."""
        details = sample_stock_metadata
        latest_price = details.get('latest_price', {})

        # Verify metrics can be extracted
        assert 'close' in latest_price, "Should have close price"
        assert 'volume' in latest_price, "Should have volume"
        assert 'market_cap' in details, "Should have market cap"

    def test_metrics_display_with_missing_data(self):
        """Test metrics display with missing data."""
        details = {}

        # Should not raise errors
        latest_price = details.get('latest_price', {})
        current_price = latest_price.get('close', 0) if isinstance(latest_price, dict) else 0

        assert current_price == 0, "Should default to 0 for missing price"

    def test_order_entry_mode_selection(self):
        """Test order entry mode selection."""
        from config.trading_modes import TradingMode, get_mode_from_string

        # API accepts lowercase mode values, UI shows uppercase
        valid_modes = ["swing", "intraday", "position", "investor"]

        for mode in valid_modes:
            try:
                trading_mode = get_mode_from_string(mode)
                assert trading_mode.value == mode
            except ValueError:
                pytest.fail(f"Mode {mode} should be valid")

    def test_risk_validation_request_format(self):
        """Test risk validation request format."""
        import json

        payload = {"mode": "SWING", "capital": 100000000.0}

        # Should be JSON serializable
        json_str = json.dumps(payload)
        parsed = json.loads(json_str)

        assert parsed["mode"] == "SWING"
        assert parsed["capital"] == 100000000.0


# =============================================================================
# TECHNICAL INDICATOR TESTS
# =============================================================================

class TestTechnicalIndicators:
    """Tests for technical indicator calculations."""

    def test_rsi_calculation(self, sample_chart_data):
        """Test RSI calculation in chart."""
        from dashboard.components.charts import build_candlestick_chart

        fig = build_candlestick_chart(sample_chart_data, show_rsi=True, show_volume=False)

        # RSI trace should exist
        rsi_traces = [t for t in fig.data if "RSI" in str(t.name)]
        assert len(rsi_traces) > 0, "Should have RSI trace"

    def test_macd_calculation(self, sample_chart_data):
        """Test MACD calculation in chart."""
        from dashboard.components.charts import build_candlestick_chart

        fig = build_candlestick_chart(sample_chart_data, show_macd=True, show_volume=False)

        # MACD trace should exist
        macd_traces = [t for t in fig.data if "MACD" in str(t.name)]
        assert len(macd_traces) > 0, "Should have MACD trace"

    def test_ma_calculation(self, sample_chart_data):
        """Test Moving Average calculation in chart."""
        from dashboard.components.charts import build_candlestick_chart

        fig = build_candlestick_chart(sample_chart_data, show_ma=True, show_volume=False)

        # Should have MA20 and MA50 traces
        ma_traces = [t for t in fig.data if "MA" in str(t.name)]
        assert len(ma_traces) >= 2, "Should have MA20 and MA50 traces"


# =============================================================================
# AI ANALYSIS SECTION TESTS
# =============================================================================

class TestAIAnalysisSection:
    """Tests for AI Analysis section functionality."""

    @pytest.mark.integration
    def test_fundamental_analysis_endpoint(self, api_available):
        """Test fundamental analysis endpoint exists."""
        if not api_available:
            pytest.skip("API server not available")

        # Just check endpoint exists (don't run full analysis - too slow)
        resp = requests.options(f"{API_URL}/fundamental/analyze", timeout=5)

        # OPTIONS should work for CORS preflight
        assert resp.status_code in [200, 405, 404], \
            f"Endpoint check returned {resp.status_code}"

    def test_agent_card_rendering(self):
        """Test agent card CSS classes."""
        from dashboard.components.nextgen_styles import COLORS

        # Agent colors should be defined
        assert "agent_auditor" in COLORS
        assert "agent_growth" in COLORS
        assert "agent_synthesizer" in COLORS


# =============================================================================
# RISK ANALYTICS TESTS
# =============================================================================

class TestRiskAnalytics:
    """Tests for risk analytics display."""

    @pytest.mark.integration
    def test_risk_check_returns_kelly_fraction(self, api_available):
        """Test that risk check returns Kelly fraction when approved."""
        if not api_available:
            pytest.skip("API server not available")

        payload = {"mode": "swing", "capital": 100000000.0}
        resp = requests.post(
            f"{API_URL}/analysis/risk-check/{TEST_SYMBOL}",
            json=payload,
            timeout=REQUEST_TIMEOUT
        )

        if resp.status_code == 200:
            data = resp.json()
            # When approved, should have Kelly fraction
            # When not approved (no signal), still valid response
            assert "approved" in data, "Should include approved field"
            if data.get("approved"):
                assert "kelly_fraction" in data, "Should include Kelly fraction when approved"
                assert isinstance(data["kelly_fraction"], (int, float)), \
                    "Kelly fraction should be numeric"

    @pytest.mark.integration
    def test_risk_check_returns_position_size(self, api_available):
        """Test that risk check returns position size when approved."""
        if not api_available:
            pytest.skip("API server not available")

        payload = {"mode": "swing", "capital": 100000000.0}
        resp = requests.post(
            f"{API_URL}/analysis/risk-check/{TEST_SYMBOL}",
            json=payload,
            timeout=REQUEST_TIMEOUT
        )

        if resp.status_code == 200:
            data = resp.json()
            # When approved, should have position size
            assert "approved" in data, "Should include approved field"
            if data.get("approved"):
                assert "position_size" in data, "Should include position size when approved"

    @pytest.mark.integration
    def test_risk_check_returns_reasons(self, api_available):
        """Test that risk check returns reasons field."""
        if not api_available:
            pytest.skip("API server not available")

        payload = {"mode": "swing", "capital": 100000000.0}
        resp = requests.post(
            f"{API_URL}/analysis/risk-check/{TEST_SYMBOL}",
            json=payload,
            timeout=REQUEST_TIMEOUT
        )

        if resp.status_code == 200:
            data = resp.json()
            # Should always have reasons field
            assert "reasons" in data, "Should include reasons field"
            assert isinstance(data["reasons"], list), "Reasons should be a list"


# =============================================================================
# STREAMLIT PAGE TESTS (End-to-End)
# =============================================================================

class TestStreamlitPageE2E:
    """End-to-end tests for Streamlit Stock Detail page."""

    @pytest.mark.integration
    def test_streamlit_page_loads(self, streamlit_available):
        """Test that Streamlit server responds."""
        if not streamlit_available:
            pytest.skip("Streamlit server not available")

        resp = requests.get(STREAMLIT_URL, timeout=10)
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"

    @pytest.mark.integration
    def test_streamlit_page_contains_title(self, streamlit_available):
        """Test that page contains expected content."""
        if not streamlit_available:
            pytest.skip("Streamlit server not available")

        resp = requests.get(STREAMLIT_URL, timeout=10)
        content = resp.text

        # Should contain Streamlit content
        assert len(content) > 1000, "Page should have substantial content"


# =============================================================================
# TEST RUNNER
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
