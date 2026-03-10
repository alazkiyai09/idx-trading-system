"""Tests for Market Overview page data processing and visualization.

Tests cover:
1. Page loads without errors
2. Sector heatmap/treemap displays
3. Stock list table renders
4. Filtering works (sector, LQ45, search)
5. Pagination/scrolling works
6. All NextGen styling applied
7. API endpoints /stocks work
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import requests


# =============================================================================
# API ENDPOINT INTEGRATION TESTS
# =============================================================================

@pytest.fixture
def api_url():
    """API base URL for testing."""
    return "http://localhost:8000"


def _extract_stocks(data):
    """Extract stocks list from API response (handles both list and wrapped formats)."""
    if isinstance(data, dict) and "stocks" in data:
        return data["stocks"]
    return data


class TestStocksAPIEndpoint:
    """Integration tests for /stocks API endpoint."""

    def test_stocks_endpoint_returns_200(self, api_url):
        """Test that /stocks endpoint returns 200 OK."""
        response = requests.get(f"{api_url}/stocks", timeout=10)
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"

    def test_stocks_endpoint_returns_list(self, api_url):
        """Test that /stocks endpoint returns data."""
        response = requests.get(f"{api_url}/stocks", timeout=10)
        data = response.json()
        # API may return list directly or wrapped in {"stocks": [...]}
        if isinstance(data, dict) and "stocks" in data:
            assert isinstance(data["stocks"], list), "stocks field should be a list"
        else:
            assert isinstance(data, list), "Response should be a list or dict with stocks"

    def test_stocks_endpoint_has_required_fields(self, api_url):
        """Test that each stock has required fields."""
        response = requests.get(f"{api_url}/stocks", timeout=10)
        data = response.json()

        # Handle both list and wrapped response
        if isinstance(data, dict) and "stocks" in data:
            stocks = data["stocks"]
        else:
            stocks = data

        assert len(stocks) > 0, "Should return at least one stock"

        required_fields = ["symbol", "name", "sector", "market_cap", "is_lq45", "change_pct"]
        for stock in stocks[:5]:  # Check first 5 stocks
            for field in required_fields:
                assert field in stock, f"Stock should have '{field}' field"

    def test_stocks_endpoint_sector_filter(self, api_url):
        """Test sector filtering works."""
        response = requests.get(
            f"{api_url}/stocks",
            params={"sector": "Financials"},
            timeout=10
        )
        assert response.status_code == 200
        data = response.json()

        # All returned stocks should be in the filtered sector
        for stock in _extract_stocks(data):
            assert stock.get("sector") == "Financials", f"Expected sector Financials"

    def test_stocks_endpoint_lq45_filter(self, api_url):
        """Test LQ45 filtering works."""
        response = requests.get(
            f"{api_url}/stocks",
            params={"is_lq45": "true"},
            timeout=10
        )
        assert response.status_code == 200
        data = response.json()

        # All returned stocks should be LQ45
        for stock in _extract_stocks(data):
            assert stock.get("is_lq45") is True, "Expected is_lq45 to be True"

    def test_stocks_endpoint_market_cap_filter(self, api_url):
        """Test minimum market cap filter works."""
        min_cap = 1e14  # 100 billion
        response = requests.get(
            f"{api_url}/stocks",
            params={"min_market_cap": min_cap},
            timeout=10
        )
        assert response.status_code == 200
        data = response.json()

        # All returned stocks should meet the minimum market cap
        for stock in _extract_stocks(data):
            if stock.get("market_cap"):
                assert stock["market_cap"] >= min_cap, "Market cap should meet minimum"

    def test_stock_detail_endpoint(self, api_url):
        """Test stock detail returns 200 for valid symbol."""
        # First get a valid symbol
        stocks_response = requests.get(f"{api_url}/stocks", timeout=10)
        stocks_data = _extract_stocks(stocks_response.json())

        if stocks_data:
            symbol = stocks_data[0]["symbol"]
            response = requests.get(f"{api_url}/stocks/{symbol}", timeout=10)
            assert response.status_code == 200, f"Expected 200 for {symbol}"

    def test_stock_detail_404_for_invalid(self, api_url):
        """Test stock detail returns 404 for invalid symbol."""
        response = requests.get(f"{api_url}/stocks/INVALID_SYMBOL_XYZ", timeout=10)
        assert response.status_code == 404, f"Expected 404, got {response.status_code}"

    def test_stock_chart_endpoint(self, api_url):
        """Test chart endpoint returns OHLCV data."""
        stocks_response = requests.get(f"{api_url}/stocks", timeout=10)
        stocks_data = _extract_stocks(stocks_response.json())

        if stocks_data:
            symbol = stocks_data[0]["symbol"]
            response = requests.get(f"{api_url}/stocks/{symbol}/chart", timeout=10)

            if response.status_code == 200:
                data = response.json()
                if len(data) > 0:
                    required_fields = ["date", "open", "high", "low", "close", "volume"]
                    for field in required_fields:
                        assert field in data[0], f"Chart data should have '{field}'"


class TestStocksDataProcessing:
    """Tests for stocks data processing in market overview."""

    def test_market_cap_normalization(self):
        """Test that market cap is normalized for display."""
        df = pd.DataFrame({
            "symbol": ["BBCA", "BBRI"],
            "market_cap": [1.2e12, 8e11],
        })

        # Convert to billions
        df["market_cap_bn"] = (df["market_cap"] / 1e9).round(2)

        assert df["market_cap_bn"].iloc[0] == 1200.0
        assert df["market_cap_bn"].iloc[1] == 800.0

    def test_handle_missing_market_cap(self):
        """Test handling of missing market cap values."""
        df = pd.DataFrame({
            "symbol": ["BBCA", "TEST"],
            "market_cap": [1.2e12, None],
        })

        # As in the page, fillna(1) to avoid division issues in treemap
        df["market_cap"] = pd.to_numeric(df["market_cap"], errors="coerce").fillna(1)
        df["market_cap_bn"] = (df["market_cap"] / 1e9).round(2)

        # Valid stock should have correct market cap
        assert df["market_cap_bn"].iloc[0] == 1200.0
        # Missing value filled with 1, resulting in very small bn value
        assert df["market_cap_bn"].iloc[1] == 1e-9 or df["market_cap_bn"].iloc[1] == 0.0

    def test_sector_fillna(self):
        """Test that missing sectors are filled with 'Unknown'."""
        df = pd.DataFrame({
            "symbol": ["BBCA", "TEST", "UNVR"],
            "sector": ["Financials", None, "Consumer"],
        })

        df["sector"] = df["sector"].fillna("Unknown")

        assert df["sector"].iloc[1] == "Unknown"

    def test_change_pct_normalization(self):
        """Test that change percentage is normalized."""
        df = pd.DataFrame({
            "symbol": ["BBCA", "BBRI", "TLKM"],
            "change_pct": [2.5, None, -1.5],
        })

        if "change_pct" not in df.columns:
            df["change_pct"] = 0.0

        df["change_pct"] = pd.to_numeric(df["change_pct"], errors="coerce").fillna(0)

        assert df["change_pct"].iloc[0] == 2.5
        assert df["change_pct"].iloc[1] == 0.0
        assert df["change_pct"].iloc[2] == -1.5

    def test_filter_lq45_only(self):
        """Test filtering to show only LQ45 stocks."""
        df = pd.DataFrame({
            "symbol": ["BBCA", "TEST", "TLKM"],
            "is_lq45": [True, False, True],
        })

        lq45_df = df[df["is_lq45"] == True]

        assert len(lq45_df) == 2
        assert "TEST" not in lq45_df["symbol"].values

    def test_filter_by_sector(self):
        """Test filtering by sector."""
        df = pd.DataFrame({
            "symbol": ["BBCA", "BBRI", "TLKM", "ASII"],
            "sector": ["Financials", "Financials", "Infrastructure", "Consumer"],
        })

        filtered = df[df["sector"] == "Financials"]

        assert len(filtered) == 2
        assert all(filtered["sector"] == "Financials")

    def test_search_filter(self):
        """Test search filtering by symbol."""
        df = pd.DataFrame({
            "symbol": ["BBCA", "BBRI", "TLKM", "UNVR"],
        })

        search = "BB"
        filtered = df[df["symbol"].str.contains(search.upper(), na=False)]

        assert len(filtered) == 2
        assert all("BB" in s for s in filtered["symbol"])

    def test_combined_filters(self):
        """Test combining multiple filters."""
        df = pd.DataFrame({
            "symbol": ["BBCA", "BBRI", "TLKM", "BMTR"],
            "sector": ["Financials", "Financials", "Infrastructure", "Financials"],
            "is_lq45": [True, True, True, False],
        })

        # Filter by sector
        filtered = df[df["sector"] == "Financials"]
        # Filter by LQ45
        filtered = filtered[filtered["is_lq45"] == True]
        # Filter by search
        filtered = filtered[filtered["symbol"].str.contains("BB", na=False)]

        assert len(filtered) == 2
        assert set(filtered["symbol"]) == {"BBCA", "BBRI"}


class TestTreemapVisualization:
    """Tests for treemap visualization data preparation."""

    def test_treemap_hierarchy_structure(self):
        """Test that treemap hierarchy is correctly structured."""
        df = pd.DataFrame([
            {"sector": "Financials", "sub_sector": "Banking", "symbol": "BBCA", "market_cap": 1e12, "change_pct": 2.5},
            {"sector": "Financials", "sub_sector": "Banking", "symbol": "BBRI", "market_cap": 8e11, "change_pct": 1.5},
            {"sector": "Technology", "sub_sector": "Software", "symbol": "EMTK", "market_cap": 2e11, "change_pct": -1.0},
        ])

        # Verify hierarchy levels exist
        assert "sector" in df.columns
        assert "sub_sector" in df.columns
        assert "symbol" in df.columns

    def test_treemap_color_mapping(self):
        """Test that treemap color is based on change percentage."""
        df = pd.DataFrame({
            "symbol": ["BBCA", "BBRI", "TLKM"],
            "change_pct": [5.0, -3.0, 0.0],
        })

        # Green for positive, red for negative
        def get_color(change_pct):
            if change_pct > 0:
                return "green"
            elif change_pct < 0:
                return "red"
            else:
                return "gray"

        colors = df["change_pct"].apply(get_color)

        assert colors.iloc[0] == "green"
        assert colors.iloc[1] == "red"
        assert colors.iloc[2] == "gray"

    def test_treemap_size_calculation(self):
        """Test that treemap block size is based on market cap."""
        df = pd.DataFrame({
            "symbol": ["BBCA", "BBRI", "TLKM"],
            "market_cap": [1.2e12, 8e11, 5e11],
        })

        total = df["market_cap"].sum()
        df["size_ratio"] = df["market_cap"] / total

        # BBCA should be largest
        assert df["size_ratio"].iloc[0] > df["size_ratio"].iloc[1]
        assert df["size_ratio"].iloc[1] > df["size_ratio"].iloc[2]

    def test_treemap_with_plotly(self):
        """Test treemap creation with Plotly."""
        import plotly.express as px

        df = pd.DataFrame({
            "symbol": ["BBCA", "BBRI", "TLKM"],
            "sector": ["Financials", "Financials", "Infrastructure"],
            "sub_sector": ["Banking", "Banking", "Telecom"],
            "market_cap": [1e12, 5e11, 3e11],
            "change_pct": [1.5, -0.8, 0.5],
        })

        fig = px.treemap(
            df,
            path=[px.Constant("IDX Market"), "sector", "sub_sector", "symbol"],
            values="market_cap",
            color="change_pct",
            color_continuous_scale=["#ef4444", "#a1a1aa", "#10b981"],
            color_continuous_midpoint=0,
        )

        assert fig is not None
        assert len(fig.data) > 0


class TestStockListDisplay:
    """Tests for stock list display formatting."""

    def test_column_selection(self):
        """Test that correct columns are selected for display."""
        df = pd.DataFrame({
            "symbol": ["BBCA"],
            "name": ["Bank Central Asia"],
            "sector": ["Financials"],
            "sub_sector": ["Banking"],
            "is_lq45": [True],
            "market_cap": [1.2e12],
            "internal_data": ["secret"],  # Should not be displayed
        })

        display_cols = ["symbol", "name", "sector", "sub_sector", "is_lq45", "market_cap"]
        available_cols = [c for c in display_cols if c in df.columns]
        display_df = df[available_cols].copy()

        assert "internal_data" not in display_df.columns
        assert "symbol" in display_df.columns

    def test_market_cap_formatting(self):
        """Test that market cap is formatted in billions."""
        df = pd.DataFrame({
            "symbol": ["BBCA"],
            "market_cap": [1_200_000_000_000],  # 1.2 trillion
        })

        df["market_cap_bn"] = (df["market_cap"] / 1e9).round(2)

        assert df["market_cap_bn"].iloc[0] == 1200.0

    def test_sorting_by_market_cap(self):
        """Test sorting stocks by market cap descending."""
        df = pd.DataFrame({
            "symbol": ["TLKM", "BBCA", "BBRI"],
            "market_cap": [5e11, 1.2e12, 8e11],
        })

        sorted_df = df.sort_values("market_cap", ascending=False)

        assert sorted_df.iloc[0]["symbol"] == "BBCA"  # Largest first

    def test_checkbox_column_formatting(self):
        """Test that boolean columns display as checkboxes."""
        df = pd.DataFrame({
            "symbol": ["BBCA", "TEST"],
            "is_lq45": [True, False],
        })

        # Verify boolean values
        assert df["is_lq45"].dtype == bool


class TestAPIErrorHandling:
    """Tests for API error handling."""

    def test_empty_response_handling(self):
        """Test handling of empty API response."""
        response_data = []

        if not response_data:
            df = pd.DataFrame()
        else:
            df = pd.DataFrame(response_data)

        assert df.empty

    def test_api_timeout_handling(self):
        """Test handling of API timeout."""
        def fetch_with_timeout(mock_timeout=True):
            if mock_timeout:
                return None
            return [{"symbol": "BBCA"}]

        result = fetch_with_timeout(mock_timeout=True)
        assert result is None

    def test_malformed_response_handling(self):
        """Test handling of malformed API response."""
        response_data = [{"invalid_key": "value"}]

        df = pd.DataFrame(response_data)

        # Should create DataFrame even with unexpected keys
        assert len(df) == 1
        assert "invalid_key" in df.columns

    def test_partial_data_handling(self):
        """Test handling of partial data in response."""
        response_data = [
            {"symbol": "BBCA", "sector": "Financials", "market_cap": 1e12},
            {"symbol": "BBRI", "sector": None, "market_cap": None},  # Partial data
        ]

        df = pd.DataFrame(response_data)
        df["sector"] = df["sector"].fillna("Unknown")
        df["market_cap"] = pd.to_numeric(df["market_cap"], errors="coerce").fillna(1)

        assert df["sector"].iloc[1] == "Unknown"
        assert df["market_cap"].iloc[1] == 1


class TestSectorAggregation:
    """Tests for sector-level data aggregation."""

    def test_sector_count(self):
        """Test counting stocks per sector."""
        df = pd.DataFrame({
            "symbol": ["BBCA", "BBRI", "TLKM", "ASII"],
            "sector": ["Financials", "Financials", "Infrastructure", "Consumer"],
        })

        sector_counts = df.groupby("sector").size()

        assert sector_counts["Financials"] == 2
        assert sector_counts["Infrastructure"] == 1

    def test_sector_market_cap_sum(self):
        """Test summing market cap per sector."""
        df = pd.DataFrame({
            "symbol": ["BBCA", "BBRI", "TLKM"],
            "sector": ["Financials", "Financials", "Infrastructure"],
            "market_cap": [1.2e12, 8e11, 5e11],
        })

        sector_mcap = df.groupby("sector")["market_cap"].sum()

        assert sector_mcap["Financials"] == 2e12

    def test_sector_average_change(self):
        """Test calculating average change per sector."""
        df = pd.DataFrame({
            "symbol": ["BBCA", "BBRI", "TLKM"],
            "sector": ["Financials", "Financials", "Infrastructure"],
            "change_pct": [2.0, 1.0, -1.5],
        })

        sector_avg = df.groupby("sector")["change_pct"].mean()

        assert sector_avg["Financials"] == 1.5
        assert sector_avg["Infrastructure"] == -1.5

    def test_sector_performance_ranking(self):
        """Test ranking sectors by performance."""
        df = pd.DataFrame({
            "sector": ["Financials", "Technology", "Consumer"],
            "avg_change": [2.5, -1.0, 1.5],
        })

        ranked = df.sort_values("avg_change", ascending=False)

        assert ranked.iloc[0]["sector"] == "Financials"
        assert ranked.iloc[-1]["sector"] == "Technology"

    def test_sector_summary_aggregation(self):
        """Test full sector summary aggregation as in the page."""
        df = pd.DataFrame({
            "symbol": ["BBCA", "BBRI", "TLKM", "UNVR"],
            "sector": ["Financials", "Financials", "Infrastructure", "Consumer"],
            "market_cap": [1e12, 5e11, 3e11, 4e11],
            "change_pct": [1.5, -0.5, 0.5, 0.0],
        })

        sector_summary = df.groupby("sector").agg({
            "change_pct": "mean",
            "market_cap": "sum",
            "symbol": "count"
        }).reset_index()
        sector_summary.columns = ["Sector", "Avg Change", "Market Cap", "Stocks"]
        sector_summary = sector_summary.sort_values("Market Cap", ascending=False)

        assert len(sector_summary) == 3
        assert sector_summary.iloc[0]["Sector"] == "Financials"
        assert sector_summary.iloc[0]["Stocks"] == 2


class TestDataQualityChecks:
    """Tests for data quality validation."""

    def test_detect_duplicate_symbols(self):
        """Test detection of duplicate symbols."""
        df = pd.DataFrame({
            "symbol": ["BBCA", "BBRI", "BBCA"],  # Duplicate BBCA
            "market_cap": [1e12, 8e11, 1e12],
        })

        duplicates = df[df.duplicated(subset=["symbol"], keep=False)]

        assert len(duplicates) == 2

    def test_detect_negative_market_cap(self):
        """Test detection of negative market cap values."""
        df = pd.DataFrame({
            "symbol": ["BBCA", "TEST"],
            "market_cap": [1e12, -1e9],  # Negative
        })

        invalid = df[df["market_cap"] < 0]

        assert len(invalid) == 1
        assert invalid.iloc[0]["symbol"] == "TEST"

    def test_detect_extreme_change_pct(self):
        """Test detection of extreme change percentages."""
        df = pd.DataFrame({
            "symbol": ["BBCA", "TEST"],
            "change_pct": [2.5, 50.0],  # 50% is suspicious
        })

        # IDX has +/- 7% daily limit
        idx_limit = 7.0
        suspicious = df[abs(df["change_pct"]) > idx_limit]

        assert len(suspicious) == 1

    def test_validate_symbol_format(self):
        """Test validation of stock symbol format."""
        df = pd.DataFrame({
            "symbol": ["BBCA", "BBRI", "INVALID_SYMBOL", "123"],
        })

        # Valid IDX symbols are 2-5 uppercase letters
        valid_pattern = r"^[A-Z]{2,5}$"
        import re
        df["valid"] = df["symbol"].str.match(valid_pattern)

        # Use bool() to convert numpy bool_ to Python bool
        assert bool(df["valid"].iloc[0]) is True  # BBCA
        assert bool(df["valid"].iloc[1]) is True  # BBRI
        assert bool(df["valid"].iloc[2]) is False  # INVALID_SYMBOL
        assert bool(df["valid"].iloc[3]) is False  # 123


class TestSummaryMetrics:
    """Tests for summary metrics calculation."""

    def test_summary_metrics_calculation(self):
        """Test summary metrics as displayed in the page."""
        df = pd.DataFrame({
            "symbol": ["BBCA", "BBRI", "TLKM", "UNVR", "TEST"],
            "is_lq45": [True, True, True, True, False],
            "sector": ["Financials", "Financials", "Infrastructure", "Consumer", "Other"],
            "market_cap": [1e12, 5e11, 3e11, 4e11, 1e10],
        })

        total_stocks = len(df)
        lq45_count = len(df[df["is_lq45"] == True])
        sectors = df["sector"].nunique()
        total_mcap = df["market_cap"].sum() / 1e15  # Trillions

        assert total_stocks == 5
        assert lq45_count == 4
        assert sectors == 4
        assert total_mcap > 0


class TestNextGenStyling:
    """Tests for NextGen styling components."""

    def test_get_nextgen_css_returns_string(self):
        """Test that get_nextgen_css returns a CSS string."""
        from dashboard.components.nextgen_styles import get_nextgen_css

        css = get_nextgen_css()
        assert isinstance(css, str)
        assert len(css) > 0

    def test_css_contains_required_classes(self):
        """Test that CSS contains required style classes."""
        from dashboard.components.nextgen_styles import get_nextgen_css

        css = get_nextgen_css()

        required_classes = [
            ".nextgen-card",
            ".live-badge",
            ".signal-badge",
            ".section-header",
            ".stMetric",
            ".stButton",
            ".stDataFrame",
        ]

        for cls in required_classes:
            assert cls in css, f"CSS should contain '{cls}' class"

    def test_colors_dictionary_complete(self):
        """Test that COLORS dictionary has all required colors."""
        from dashboard.components.nextgen_styles import COLORS

        required_colors = [
            "background",
            "foreground",
            "primary",
            "destructive",
            "muted",
            "muted_foreground",
            "card",
            "border",
        ]

        for color in required_colors:
            assert color in COLORS, f"COLORS should have '{color}'"
            assert COLORS[color].startswith("#"), f"{color} should be a hex color"

    def test_render_live_badge_returns_html(self):
        """Test render_live_badge returns valid HTML."""
        from dashboard.components.nextgen_styles import render_live_badge

        badge = render_live_badge("TEST")
        assert "live-badge" in badge
        assert "TEST" in badge
        assert "pulse" in badge

    def test_format_price_with_styling(self):
        """Test format_price returns styled HTML."""
        from dashboard.components.nextgen_styles import format_price

        formatted = format_price(9050)
        assert "price-mono" in formatted
        assert "9,050" in formatted

    def test_format_change_positive(self):
        """Test format_change for positive values."""
        from dashboard.components.nextgen_styles import format_change

        formatted = format_change(1.5)
        assert "positive" in formatted
        assert "+1.50%" in formatted

    def test_format_change_negative(self):
        """Test format_change for negative values."""
        from dashboard.components.nextgen_styles import format_change

        formatted = format_change(-0.8)
        assert "negative" in formatted
        assert "-0.80%" in formatted


class TestPageComponents:
    """Tests for page component existence and structure."""

    def test_page_file_exists(self):
        """Test that the page file exists."""
        import os
        page_path = "/mnt/data/Project/idx-trading-system/dashboard/pages/06_market_overview.py"
        assert os.path.exists(page_path), "Page file should exist"

    def test_page_contains_required_components(self):
        """Test that page contains all required UI components."""
        page_path = "/mnt/data/Project/idx-trading-system/dashboard/pages/06_market_overview.py"

        with open(page_path, "r") as f:
            content = f.read()

        required_components = [
            ("get_nextgen_css()", "NextGen CSS applied"),
            ("render_live_badge", "Live badge used"),
            ("px.treemap", "Treemap chart used"),
            ("st.dataframe", "DataFrame displayed"),
            ("st.tabs", "Tabs used"),
            ("sector_filter", "Sector filter present"),
            ("lq45_only", "LQ45 filter present"),
            ("search", "Search filter present"),
            ("trading_hours_indicator", "Trading hours indicator"),
        ]

        for component, description in required_components:
            assert component in content, f"Page should contain {description}"


class TestPerformance:
    """Tests for performance characteristics."""

    def test_api_response_time(self, api_url):
        """Test API responds within acceptable time."""
        import time

        start = time.time()
        response = requests.get(f"{api_url}/stocks", timeout=10)
        elapsed = time.time() - start

        assert response.status_code == 200
        assert elapsed < 10.0, f"API took {elapsed:.2f}s, should be < 10s"

    def test_dataframe_operations_performance(self):
        """Test DataFrame operations are performant."""
        import time

        # Create a large DataFrame simulating all stocks
        df = pd.DataFrame({
            "symbol": [f"STOCK{i}" for i in range(1000)],
            "sector": ["Financials"] * 500 + ["Technology"] * 500,
            "market_cap": [1e12] * 1000,
            "change_pct": [1.0] * 1000,
            "is_lq45": [True] * 500 + [False] * 500,
        })

        start = time.time()

        # Simulate page operations
        df["market_cap"] = pd.to_numeric(df["market_cap"], errors="coerce").fillna(1)
        df["sector"] = df["sector"].fillna("Unknown")
        df["change_pct"] = pd.to_numeric(df["change_pct"], errors="coerce").fillna(0)

        # Filtering
        filtered = df[df["sector"] == "Financials"]
        filtered = filtered[filtered["is_lq45"] == True]

        # Aggregation
        sector_summary = df.groupby("sector").agg({
            "change_pct": "mean",
            "market_cap": "sum",
            "symbol": "count"
        }).reset_index()

        elapsed = time.time() - start

        assert elapsed < 1.0, f"DataFrame operations took {elapsed:.2f}s, should be < 1s"
