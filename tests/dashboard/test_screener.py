"""Tests for Stock Screener page helper functions."""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from dashboard.pages.screener_helpers import (
    apply_classification_filters,
    apply_price_filters,
    apply_technical_filters,
    calculate_filter_stats,
    get_unique_sectors,
    get_unique_sub_sectors,
    sort_analysis_results,
    format_stock_display,
)


@pytest.fixture
def sample_stocks_df():
    """Create sample stocks DataFrame for testing."""
    return pd.DataFrame([
        {"symbol": "BBCA", "name": "Bank Central Asia", "sector": "Financials", "sub_sector": "Banking", "is_lq45": True, "is_idx30": True, "market_cap": 1.2e12},
        {"symbol": "BBRI", "name": "Bank Rakyat Indonesia", "sector": "Financials", "sub_sector": "Banking", "is_lq45": True, "is_idx30": False, "market_cap": 8e11},
        {"symbol": "TLKM", "name": "Telkom Indonesia", "sector": "Infrastructure", "sub_sector": "Telecom", "is_lq45": True, "is_idx30": True, "market_cap": 5e11},
        {"symbol": "ASII", "name": "Astra International", "sector": "Consumer", "sub_sector": "Auto", "is_lq45": True, "is_idx30": True, "market_cap": 3e11},
        {"symbol": "UNVR", "name": "Unilever Indonesia", "sector": "Consumer", "sub_sector": "FMCG", "is_lq45": True, "is_idx30": True, "market_cap": 4e11},
        {"symbol": "TEST", "name": "Test Company", "sector": "Other", "sub_sector": "Other", "is_lq45": False, "is_idx30": False, "market_cap": 1e10},
    ])


class TestApplyClassificationFilters:
    """Tests for apply_classification_filters function."""

    def test_no_filters(self, sample_stocks_df):
        """Test with no filters applied."""
        filters = {"lq45": False, "idx30": False, "sector": "All", "sub_sector": "All", "board": "All"}
        result = apply_classification_filters(sample_stocks_df, filters)

        assert len(result) == len(sample_stocks_df)

    def test_lq45_filter(self, sample_stocks_df):
        """Test LQ45 filter."""
        filters = {"lq45": True, "idx30": False, "sector": "All", "sub_sector": "All", "board": "All"}
        result = apply_classification_filters(sample_stocks_df, filters)

        assert len(result) == 5  # 5 LQ45 stocks
        assert all(result['is_lq45'] == True)

    def test_idx30_filter(self, sample_stocks_df):
        """Test IDX30 filter."""
        filters = {"lq45": False, "idx30": True, "sector": "All", "sub_sector": "All", "board": "All"}
        result = apply_classification_filters(sample_stocks_df, filters)

        assert len(result) == 4  # 4 IDX30 stocks
        assert all(result['is_idx30'] == True)

    def test_sector_filter(self, sample_stocks_df):
        """Test sector filter."""
        filters = {"lq45": False, "idx30": False, "sector": "Financials", "sub_sector": "All", "board": "All"}
        result = apply_classification_filters(sample_stocks_df, filters)

        assert len(result) == 2  # BBCA and BBRI
        assert all(result['sector'] == "Financials")

    def test_combined_filters(self, sample_stocks_df):
        """Test multiple filters combined."""
        filters = {"lq45": True, "idx30": False, "sector": "Financials", "sub_sector": "All", "board": "All"}
        result = apply_classification_filters(sample_stocks_df, filters)

        assert len(result) == 2  # BBCA and BBRI (both LQ45 and Financials)

    def test_empty_result(self, sample_stocks_df):
        """Test filter that returns no results."""
        filters = {"lq45": False, "idx30": False, "sector": "Nonexistent", "sub_sector": "All", "board": "All"}
        result = apply_classification_filters(sample_stocks_df, filters)

        assert len(result) == 0


class TestApplyPriceFilters:
    """Tests for apply_price_filters function."""

    def test_min_market_cap_filter(self, sample_stocks_df):
        """Test minimum market cap filter."""
        filters = {"min_market_cap": 500}  # 500 billion
        result = apply_price_filters(sample_stocks_df, filters)

        # BBCA (1200B), BBRI (800B), TLKM (500B) >= 500B
        # ASII (300B), UNVR (400B), TEST (10B) < 500B
        assert len(result) == 3
        assert all(result['market_cap'] >= 500e9)

    def test_no_price_filters(self, sample_stocks_df):
        """Test with no price filters."""
        filters = {"min_market_cap": 0, "min_price": 0, "max_price": 0}
        result = apply_price_filters(sample_stocks_df, filters)

        assert len(result) == len(sample_stocks_df)


class TestApplyTechnicalFilters:
    """Tests for apply_technical_filters function."""

    def test_rsi_filter(self, sample_stocks_df):
        """Test RSI range filter."""
        technical_data = {
            "BBCA": {"rsi": 65},
            "BBRI": {"rsi": 45},
            "TLKM": {"rsi": 75},
            "ASII": {"rsi": 30},
            "UNVR": {"rsi": 55},
            "TEST": {"rsi": 80},
        }
        filters = {"rsi_range": (40, 70)}

        result = apply_technical_filters(sample_stocks_df, filters, technical_data)

        # Should include BBCA (65), BBRI (45), UNVR (55)
        assert len(result) == 3
        assert "BBCA" in result['symbol'].values
        assert "BBRI" in result['symbol'].values
        assert "UNVR" in result['symbol'].values

    def test_no_technical_data(self, sample_stocks_df):
        """Test with no technical data provided."""
        filters = {"rsi_range": (0, 100)}
        result = apply_technical_filters(sample_stocks_df, filters, None)

        assert len(result) == len(sample_stocks_df)


class TestCalculateFilterStats:
    """Tests for calculate_filter_stats function."""

    def test_basic_stats(self):
        """Test basic filter statistics."""
        stats = calculate_filter_stats(100, 25)

        assert stats["original_count"] == 100
        assert stats["filtered_count"] == 25
        assert stats["removed_count"] == 75
        assert stats["reduction_pct"] == 75.0

    def test_no_reduction(self):
        """Test with no filtering."""
        stats = calculate_filter_stats(50, 50)

        assert stats["reduction_pct"] == 0.0

    def test_full_reduction(self):
        """Test with all stocks filtered out."""
        stats = calculate_filter_stats(100, 0)

        assert stats["reduction_pct"] == 100.0


class TestGetUniqueSectors:
    """Tests for get_unique_sectors function."""

    def test_get_sectors(self, sample_stocks_df):
        """Test getting unique sectors."""
        sectors = get_unique_sectors(sample_stocks_df)

        # 4 unique sectors: Consumer, Financials, Infrastructure, Other
        assert len(sectors) == 4
        assert "Financials" in sectors
        assert "Infrastructure" in sectors
        assert "Consumer" in sectors
        assert "Other" in sectors

    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        df = pd.DataFrame()
        sectors = get_unique_sectors(df)

        assert sectors == []

    def test_no_sector_column(self):
        """Test with DataFrame without sector column."""
        df = pd.DataFrame({"symbol": ["BBCA"], "name": ["BCA"]})
        sectors = get_unique_sectors(df)

        assert sectors == []


class TestGetUniqueSubSectors:
    """Tests for get_unique_sub_sectors function."""

    def test_get_all_sub_sectors(self, sample_stocks_df):
        """Test getting all unique sub-sectors."""
        sub_sectors = get_unique_sub_sectors(sample_stocks_df)

        assert len(sub_sectors) >= 4
        assert "Banking" in sub_sectors
        assert "Telecom" in sub_sectors

    def test_filter_by_sector(self, sample_stocks_df):
        """Test getting sub-sectors filtered by sector."""
        sub_sectors = get_unique_sub_sectors(sample_stocks_df, "Financials")

        assert sub_sectors == ["Banking"]

    def test_filter_by_nonexistent_sector(self, sample_stocks_df):
        """Test filtering by sector that doesn't exist."""
        sub_sectors = get_unique_sub_sectors(sample_stocks_df, "Nonexistent")

        assert sub_sectors == []


class TestSortAnalysisResults:
    """Tests for sort_analysis_results function."""

    def test_sort_by_symbol(self):
        """Test sorting by symbol."""
        df = pd.DataFrame({
            "Symbol": ["TLKM", "BBCA", "BBRI"],
            "Tech Score": [85, 75, 60],
        })
        result = sort_analysis_results(df, "Symbol", ascending=True)

        assert result.iloc[0]["Symbol"] == "BBCA"

    def test_sort_by_score(self):
        """Test sorting by score descending."""
        df = pd.DataFrame({
            "Symbol": ["BBCA", "BBRI", "TLKM"],
            "Tech Score": [75, 60, 85],
        })
        result = sort_analysis_results(df, "Tech Score", ascending=False)

        assert result.iloc[0]["Symbol"] == "TLKM"
        assert result.iloc[0]["Tech Score"] == 85

    def test_sort_invalid_column(self):
        """Test sorting by non-existent column."""
        df = pd.DataFrame({"Symbol": ["BBCA"]})
        result = sort_analysis_results(df, "NonExistent")

        assert len(result) == 1

    def test_sort_with_nan(self):
        """Test sorting with NaN values."""
        df = pd.DataFrame({
            "Symbol": ["BBCA", "BBRI", "TLKM"],
            "Tech Score": [75, None, 85],
        })
        result = sort_analysis_results(df, "Tech Score", ascending=False)

        # NaN should be at the end
        assert pd.isna(result.iloc[-1]["Tech Score"])


class TestFormatStockDisplay:
    """Tests for format_stock_display function."""

    def test_default_columns(self, sample_stocks_df):
        """Test with default columns."""
        result = format_stock_display(sample_stocks_df)

        assert "symbol" in result.columns
        assert "name" in result.columns
        assert "sector" in result.columns

    def test_custom_columns(self, sample_stocks_df):
        """Test with custom columns."""
        result = format_stock_display(sample_stocks_df, ["symbol", "market_cap"])

        assert list(result.columns) == ["symbol", "market_cap"]
        assert len(result) == len(sample_stocks_df)

    def test_missing_columns(self, sample_stocks_df):
        """Test with columns that don't exist."""
        result = format_stock_display(sample_stocks_df, ["symbol", "nonexistent"])

        assert "symbol" in result.columns
        assert "nonexistent" not in result.columns


class TestScreenerIntegration:
    """Integration tests for screener workflow."""

    def test_full_filter_workflow(self, sample_stocks_df):
        """Test complete filtering workflow."""
        # Step 1: Apply classification filters
        cls_filters = {"lq45": True, "idx30": False, "sector": "All", "sub_sector": "All", "board": "All"}
        filtered = apply_classification_filters(sample_stocks_df, cls_filters)
        # 5 LQ45 stocks (excludes TEST)
        assert len(filtered) == 5

        # Step 2: Apply price filters (min market cap 300B)
        price_filters = {"min_market_cap": 300}
        filtered = apply_price_filters(filtered, price_filters)
        # 5 stocks with market cap >= 300B: BBCA (1200B), BBRI (800B), TLKM (500B), ASII (300B), UNVR (400B)
        assert len(filtered) == 5

        # Step 3: Calculate stats
        stats = calculate_filter_stats(len(sample_stocks_df), len(filtered))
        # 6 original - 5 filtered = 1 removed (16.7%)
        assert stats["reduction_pct"] == pytest.approx(16.7, rel=0.1)

    def test_sector_subsector_workflow(self, sample_stocks_df):
        """Test sector and sub-sector selection workflow."""
        # Get available sectors
        sectors = get_unique_sectors(sample_stocks_df)
        assert "Financials" in sectors

        # Get sub-sectors for Financials
        sub_sectors = get_unique_sub_sectors(sample_stocks_df, "Financials")
        assert sub_sectors == ["Banking"]

        # Filter by both
        filters = {"lq45": False, "idx30": False, "sector": "Financials", "sub_sector": "Banking", "board": "All"}
        result = apply_classification_filters(sample_stocks_df, filters)
        assert len(result) == 2
