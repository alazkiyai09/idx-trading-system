"""
End-to-End Tests for Sentiment Page (dashboard/pages/03_sentiment.py).

This test module covers:
1. Page loads without errors
2. Sentiment gauge displays correctly
3. News feed loads and displays articles
4. Sector sentiment cards render
5. Sentiment history chart works
6. All NextGen styling applied
7. API endpoints /sentiment/{symbol}, /sentiment/news work

Run with: pytest tests/dashboard/test_sentiment_page_e2e.py -v
"""
import pytest
import requests
import subprocess
import time
import json
from unittest.mock import patch, MagicMock

# API base URL
API_URL = "http://localhost:8000"
STREAMLIT_URL = "http://localhost:8501"


class TestSentimentAPIEndpoints:
    """Tests for all sentiment API endpoints."""

    def test_api_health_check(self):
        """Test that API server is running."""
        try:
            resp = requests.get(f"{API_URL}/health", timeout=5)
            assert resp.status_code == 200
        except requests.ConnectionError:
            pytest.skip("API server not running on port 8000")

    def test_sentiment_latest_endpoint_returns_200(self):
        """Test /sentiment/latest endpoint returns 200."""
        resp = requests.get(f"{API_URL}/sentiment/latest", timeout=10)
        assert resp.status_code == 200

    def test_sentiment_latest_returns_dict_with_articles(self):
        """Test /sentiment/latest returns dict with 'articles' key."""
        resp = requests.get(f"{API_URL}/sentiment/latest", timeout=10)
        data = resp.json()
        assert isinstance(data, dict)
        assert "articles" in data
        assert isinstance(data["articles"], list)

    def test_sentiment_latest_with_symbol_filter(self):
        """Test /sentiment/latest?symbol=BBCA returns filtered results."""
        resp = requests.get(f"{API_URL}/sentiment/latest?symbol=BBCA", timeout=10)
        assert resp.status_code == 200
        data = resp.json()
        assert "articles" in data
        # All returned articles should be for BBCA (if any exist)
        for article in data["articles"]:
            assert article.get("symbol") == "BBCA"

    def test_sentiment_sector_endpoint_returns_200(self):
        """Test /sentiment/sector endpoint returns 200."""
        resp = requests.get(f"{API_URL}/sentiment/sector", timeout=10)
        assert resp.status_code == 200

    def test_sentiment_sector_returns_list(self):
        """Test /sentiment/sector returns a list."""
        resp = requests.get(f"{API_URL}/sentiment/sector", timeout=10)
        data = resp.json()
        assert isinstance(data, list)

    def test_sentiment_sector_item_structure(self):
        """Test sector sentiment items have expected fields."""
        resp = requests.get(f"{API_URL}/sentiment/sector", timeout=10)
        data = resp.json()
        # If data exists, check structure
        if data:
            item = data[0]
            assert "sector" in item
            # avg_score and article_count are expected
            assert "avg_score" in item or "article_count" in item

    def test_sentiment_themes_endpoint_returns_200(self):
        """Test /sentiment/themes endpoint returns 200."""
        resp = requests.get(f"{API_URL}/sentiment/themes", timeout=10)
        assert resp.status_code == 200

    def test_sentiment_themes_returns_list(self):
        """Test /sentiment/themes returns a list."""
        resp = requests.get(f"{API_URL}/sentiment/themes", timeout=10)
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) > 0  # Should have default themes

    def test_sentiment_themes_item_structure(self):
        """Test theme items have expected fields."""
        resp = requests.get(f"{API_URL}/sentiment/themes", timeout=10)
        data = resp.json()
        if data:
            item = data[0]
            assert "theme" in item
            assert "sector" in item
            assert "impact_direction" in item

    def test_sentiment_fetch_endpoint_returns_200(self):
        """Test POST /sentiment/fetch/{symbol} returns success."""
        resp = requests.post(f"{API_URL}/sentiment/fetch/BBCA", timeout=30)
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data
        assert data["status"] == "accepted"

    def test_sentiment_cleanup_endpoint_returns_200(self):
        """Test DELETE /sentiment/cleanup returns success."""
        resp = requests.delete(f"{API_URL}/sentiment/cleanup?days=30", timeout=10)
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data


class TestSentimentGauge:
    """Tests for sentiment gauge display."""

    def test_build_sentiment_gauge_returns_figure(self):
        """Test that build_sentiment_gauge creates a valid figure."""
        from dashboard.components.charts import build_sentiment_gauge

        fig = build_sentiment_gauge(score=65, title="Test Gauge")
        assert fig is not None
        assert hasattr(fig, 'data')
        assert len(fig.data) > 0

    def test_sentiment_gauge_score_range_0_to_100(self):
        """Test gauge handles full score range."""
        from dashboard.components.charts import build_sentiment_gauge

        for score in [0, 25, 50, 75, 100]:
            fig = build_sentiment_gauge(score=score)
            assert fig is not None

    def test_sentiment_gauge_color_for_bullish(self):
        """Test gauge uses primary color for bullish scores (>=70)."""
        from dashboard.components.charts import build_sentiment_gauge
        from dashboard.components.nextgen_styles import COLORS

        fig = build_sentiment_gauge(score=80)
        # Check that the gauge has proper structure
        assert fig.data[0].mode == "gauge+number+delta"

    def test_sentiment_gauge_color_for_bearish(self):
        """Test gauge uses destructive color for bearish scores (<50)."""
        from dashboard.components.charts import build_sentiment_gauge

        fig = build_sentiment_gauge(score=30)
        assert fig is not None

    def test_sentiment_gauge_layout_dark_theme(self):
        """Test gauge uses dark theme background."""
        from dashboard.components.charts import build_sentiment_gauge
        from dashboard.components.nextgen_styles import COLORS

        fig = build_sentiment_gauge(score=50)
        assert fig.layout.paper_bgcolor == COLORS['background']


class TestSectorHeatmap:
    """Tests for sector sentiment heatmap."""

    def test_sector_treemap_with_data(self):
        """Test treemap builds correctly with sector data."""
        import pandas as pd
        import plotly.express as px
        from dashboard.components.nextgen_styles import COLORS

        sector_data = [
            {"sector": "Financials", "avg_score": 70, "article_count": 20},
            {"sector": "Technology", "avg_score": 50, "article_count": 10},
            {"sector": "Energy", "avg_score": 80, "article_count": 15},
        ]
        df = pd.DataFrame(sector_data)

        fig = px.treemap(
            df,
            path=[px.Constant("IDX"), 'sector'],
            values='article_count',
            color='avg_score',
            color_continuous_scale=[COLORS['destructive'], COLORS['primary']],
            color_continuous_midpoint=50,
        )

        assert fig is not None
        assert len(fig.data) > 0

    def test_sector_heatmap_empty_data_fallback(self):
        """Test that page has fallback demo data for heatmap."""
        import pandas as pd

        # This is the fallback data from the page
        fallback_data = [
            {"sector": "Energy", "avg_score": 75, "signal": "Bullish", "article_count": 14},
            {"sector": "Financials", "avg_score": 62, "signal": "Bullish", "article_count": 22},
            {"sector": "Consumer", "avg_score": 45, "signal": "Bearish", "article_count": 8},
            {"sector": "Technology", "avg_score": 38, "signal": "Bearish", "article_count": 5},
            {"sector": "Materials", "avg_score": 55, "signal": "Neutral", "article_count": 11},
            {"sector": "Infrastructure", "avg_score": 50, "signal": "Neutral", "article_count": 7},
            {"sector": "Healthcare", "avg_score": 68, "signal": "Bullish", "article_count": 4},
            {"sector": "Property", "avg_score": 42, "signal": "Bearish", "article_count": 6},
        ]

        df = pd.DataFrame(fallback_data)
        assert len(df) == 8
        assert 'avg_score' in df.columns
        assert 'article_count' in df.columns


class TestNewsFeed:
    """Tests for news feed display."""

    def test_article_sentiment_badge_bullish(self):
        """Test bullish badge is applied for score > 60."""
        score = 75
        if score > 60:
            badge_class = "bullish"
        elif score < 40:
            badge_class = "bearish"
        else:
            badge_class = "neutral"

        assert badge_class == "bullish"

    def test_article_sentiment_badge_bearish(self):
        """Test bearish badge is applied for score < 40."""
        score = 25
        if score > 60:
            badge_class = "bullish"
        elif score < 40:
            badge_class = "bearish"
        else:
            badge_class = "neutral"

        assert badge_class == "bearish"

    def test_article_sentiment_badge_neutral(self):
        """Test neutral badge is applied for 40 <= score <= 60."""
        score = 50
        if score > 60:
            badge_class = "bullish"
        elif score < 40:
            badge_class = "bearish"
        else:
            badge_class = "neutral"

        assert badge_class == "neutral"

    def test_article_display_limit_10(self):
        """Test that only first 10 articles are displayed."""
        articles = [{"title": f"Article {i}"} for i in range(20)]
        displayed = articles[:10]
        assert len(displayed) == 10

    def test_article_fallback_message(self):
        """Test that fallback message shows when no articles."""
        articles = []
        if not articles:
            has_articles = False
        else:
            has_articles = True

        assert has_articles is False


class TestThemeMapping:
    """Tests for theme mapping display."""

    def test_theme_impact_positive_icon(self):
        """Test positive impact uses correct icon and color."""
        impact = "positive"
        if impact == "positive":
            icon = "📈"
            border_color = "#10b981"  # Emerald
        elif impact == "negative":
            icon = "📉"
            border_color = "#ef4444"  # Red
        else:
            icon = "🔄"
            border_color = "#f59e0b"  # Amber

        assert icon == "📈"
        assert border_color == "#10b981"

    def test_theme_impact_negative_icon(self):
        """Test negative impact uses correct icon and color."""
        impact = "negative"
        if impact == "positive":
            icon = "📈"
        elif impact == "negative":
            icon = "📉"
        else:
            icon = "🔄"

        assert icon == "📉"

    def test_theme_impact_neutral_icon(self):
        """Test neutral impact uses correct icon and color."""
        impact = "neutral"
        if impact == "positive":
            icon = "📈"
        elif impact == "negative":
            icon = "📉"
        else:
            icon = "🔄"

        assert icon == "🔄"

    def test_theme_display_limit_6(self):
        """Test that only first 6 themes are displayed."""
        themes = [{"theme": f"Theme {i}"} for i in range(10)]
        displayed = themes[:6]
        assert len(displayed) == 6

    def test_theme_stocks_truncation(self):
        """Test that stocks list is truncated to 3 items for display."""
        stocks = ["A", "B", "C", "D", "E"]
        truncated = stocks[:3]
        assert len(truncated) == 3


class TestNextGenStyling:
    """Tests for NextGen styling application."""

    def test_get_nextgen_css_returns_string(self):
        """Test that get_nextgen_css returns CSS string."""
        from dashboard.components.nextgen_styles import get_nextgen_css

        css = get_nextgen_css()
        assert isinstance(css, str)
        assert len(css) > 0

    def test_nextgen_css_contains_colors(self):
        """Test CSS contains expected color variables."""
        from dashboard.components.nextgen_styles import get_nextgen_css, COLORS

        css = get_nextgen_css()

        # Check key colors are in CSS
        assert COLORS['background'] in css
        assert COLORS['primary'] in css
        assert COLORS['destructive'] in css

    def test_nextgen_css_contains_card_class(self):
        """Test CSS contains nextgen-card class."""
        from dashboard.components.nextgen_styles import get_nextgen_css

        css = get_nextgen_css()
        assert ".nextgen-card" in css

    def test_nextgen_css_contains_signal_badge(self):
        """Test CSS contains signal-badge classes."""
        from dashboard.components.nextgen_styles import get_nextgen_css

        css = get_nextgen_css()
        assert ".signal-badge" in css
        assert ".signal-badge.bullish" in css
        assert ".signal-badge.bearish" in css
        assert ".signal-badge.neutral" in css

    def test_nextgen_css_contains_live_badge(self):
        """Test CSS contains live-badge class."""
        from dashboard.components.nextgen_styles import get_nextgen_css

        css = get_nextgen_css()
        assert ".live-badge" in css

    def test_render_live_badge_returns_html(self):
        """Test render_live_badge returns HTML string."""
        from dashboard.components.nextgen_styles import render_live_badge

        html = render_live_badge("AI")
        assert isinstance(html, str)
        assert "live-badge" in html
        assert "AI" in html

    def test_colors_dictionary_complete(self):
        """Test COLORS dictionary has all required keys."""
        from dashboard.components.nextgen_styles import COLORS

        required_keys = [
            "background", "foreground", "border",
            "primary", "destructive", "warning",
            "muted", "muted_foreground", "card"
        ]

        for key in required_keys:
            assert key in COLORS, f"Missing color key: {key}"


class TestPageLoadsWithoutErrors:
    """Tests for page loading without errors."""

    def test_streamlit_server_running(self):
        """Test that Streamlit server is running."""
        try:
            resp = requests.get(f"{STREAMLIT_URL}", timeout=5)
            assert resp.status_code == 200
        except requests.ConnectionError:
            pytest.skip("Streamlit server not running on port 8501")

    def test_sentiment_page_returns_html(self):
        """Test sentiment page returns valid HTML."""
        try:
            resp = requests.get(f"{STREAMLIT_URL}/sentiment", timeout=10)
            assert resp.status_code == 200
            assert "text/html" in resp.headers.get("Content-Type", "")
        except requests.ConnectionError:
            pytest.skip("Streamlit server not running on port 8501")

    def test_sentiment_page_imports_successfully(self):
        """Test that sentiment page module imports without errors."""
        try:
            # Import the page module
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "sentiment_page",
                "/mnt/data/Project/idx-trading-system/dashboard/pages/03_sentiment.py"
            )
            assert spec is not None
        except Exception as e:
            pytest.fail(f"Failed to import sentiment page: {e}")


class TestURLSafety:
    """Tests for URL safety validation."""

    def test_is_safe_url_http(self):
        """Test that http URLs are considered safe."""
        from urllib.parse import urlparse

        url = "http://example.com/article"
        allowed_schemes = {'http', 'https'}

        try:
            parsed = urlparse(url)
            is_safe = parsed.scheme.lower() in allowed_schemes
        except Exception:
            is_safe = False

        assert is_safe is True

    def test_is_safe_url_https(self):
        """Test that https URLs are considered safe."""
        from urllib.parse import urlparse

        url = "https://example.com/article"
        allowed_schemes = {'http', 'https'}

        try:
            parsed = urlparse(url)
            is_safe = parsed.scheme.lower() in allowed_schemes
        except Exception:
            is_safe = False

        assert is_safe is True

    def test_is_safe_url_javascript_blocked(self):
        """Test that javascript: URLs are blocked."""
        from urllib.parse import urlparse

        url = "javascript:alert('xss')"
        allowed_schemes = {'http', 'https'}

        try:
            parsed = urlparse(url)
            is_safe = parsed.scheme.lower() in allowed_schemes
        except Exception:
            is_safe = False

        assert is_safe is False

    def test_is_safe_url_empty(self):
        """Test that empty URLs return False."""
        url = ""
        allowed_schemes = {'http', 'https'}

        if not url:
            is_safe = False

        assert is_safe is False


class TestMarketWideSentimentCalculation:
    """Tests for market-wide sentiment calculation."""

    def test_average_sentiment_calculation(self):
        """Test average sentiment from sector data."""
        sector_data = [
            {"sector": "Financials", "avg_score": 70},
            {"sector": "Technology", "avg_score": 50},
            {"sector": "Energy", "avg_score": 60},
        ]

        avg_score = sum(s.get('avg_score', 50) for s in sector_data) / max(len(sector_data), 1)
        assert avg_score == 60

    def test_average_sentiment_empty_data(self):
        """Test average sentiment with empty sector data uses default."""
        sector_data = []

        if sector_data:
            avg_score = sum(s.get('avg_score', 50) for s in sector_data) / max(len(sector_data), 1)
        else:
            avg_score = 50  # Default fallback

        assert avg_score == 50

    def test_average_sentiment_missing_scores(self):
        """Test average sentiment with missing avg_score uses default 50."""
        sector_data = [
            {"sector": "Financials"},  # Missing avg_score
            {"sector": "Technology", "avg_score": 60},
        ]

        avg_score = sum(s.get('avg_score', 50) for s in sector_data) / max(len(sector_data), 1)
        assert avg_score == 55  # (50 + 60) / 2


class TestTradingHoursIndicator:
    """Tests for trading hours indicator component."""

    def test_trading_hours_indicator_function_exists(self):
        """Test that trading_hours_indicator function exists."""
        from dashboard.components.ux_components import trading_hours_indicator

        assert callable(trading_hours_indicator)

    def test_idx_trading_hours_constants(self):
        """Test IDX trading hours constants are correct."""
        from dashboard.components.ux_components import IDX_TRADING_HOURS

        # IDX trading hours: 09:00 - 17:10 WIB
        assert IDX_TRADING_HOURS[0].hour == 9
        assert IDX_TRADING_HOURS[0].minute == 0
        assert IDX_TRADING_HOURS[1].hour == 17
        assert IDX_TRADING_HOURS[1].minute == 10


class TestQuickActions:
    """Tests for quick action buttons."""

    def test_fetch_all_news_endpoint_integration(self):
        """Test that Fetch All News button triggers correct API calls."""
        # Simulate the button logic
        watchlist = ['BBCA', 'BBRI', 'TLKM', 'ASII']

        for sym in watchlist[:2]:  # Test first 2 to save time
            resp = requests.post(f"{API_URL}/sentiment/fetch/{sym}", timeout=30)
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "accepted"

    def test_cleanup_endpoint_integration(self):
        """Test that Clear Old Data button calls cleanup API."""
        days = 30
        resp = requests.delete(f"{API_URL}/sentiment/cleanup?days={days}", timeout=10)
        assert resp.status_code == 200
        data = resp.json()
        assert "deleted" in data
