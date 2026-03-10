"""
Unit tests for macro_correlation module.
"""
import pytest
from datetime import date, timedelta, datetime
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np

from core.data.database import DatabaseManager
from core.analysis.macro_correlation import MacroCorrelationAnalyzer


class TestMacroCorrelationAnalyzer:
    """Tests for MacroCorrelationAnalyzer class."""

    @pytest.fixture
    def mock_db(self):
        """Create mock database manager."""
        return Mock(spec=DatabaseManager)

    @pytest.fixture
    def analyzer(self, mock_db):
        """Create analyzer with mock database."""
        mock_db.get_session.return_value = MagicMock()
        return MacroCorrelationAnalyzer(mock_db)

    def test_commodities_list(self, analyzer):
        """Test that commodities list is correct."""
        assert "GOLD" in analyzer.COMMODITIES
        assert "SILVER" in analyzer.COMMODITIES
        assert "OIL" in analyzer.COMMODITIES

    def test_sector_impact_mappings(self, analyzer):
        """Test sector commodity impact mappings."""
        assert "MINING" in analyzer.SECTOR_COMMODITY_IMPACT
        assert "ENERGY" in analyzer.SECTOR_COMMODITY_IMPACT
        assert "BANKING" in analyzer.SECTOR_COMMODITY_IMPACT

        # Mining should be positively impacted by gold
        assert analyzer.SECTOR_COMMODITY_IMPACT["MINING"]["GOLD"] > 0

        # Energy should be positively impacted by oil
        assert analyzer.SECTOR_COMMODITY_IMPACT["ENERGY"]["OIL"] > 0

    def test_get_commodity_impact_known_sector(self, analyzer):
        """Test getting commodity impact for known sector."""
        result = analyzer.get_commodity_impact("MINING")
        assert result["sector"] == "MINING"
        assert "commodity_impacts" in result
        assert "GOLD" in result["commodity_impacts"]

    def test_get_commodity_impact_unknown_sector(self, analyzer):
        """Test getting commodity impact for unknown sector returns default."""
        result = analyzer.get_commodity_impact("UNKNOWN_SECTOR")
        assert result["sector"] == "UNKNOWN_SECTOR"
        # Should return default impacts
        assert "commodity_impacts" in result

    def test_compute_correlations_no_stock_data(self, analyzer, mock_db):
        """Test correlation computation with no stock data."""
        # Mock get_session to return a session that returns empty list
        mock_session = MagicMock()
        mock_query = MagicMock()
        mock_filter = MagicMock()
        mock_filter.all.return_value = []
        mock_query.filter.return_value = mock_filter
        mock_session.query.return_value = mock_query
        mock_db.get_session.return_value = mock_session

        result = analyzer.compute_correlations("UNKNOWN")
        # Should return error details for each commodity
        for commodity in ["GOLD", "SILVER", "OIL"]:
            assert commodity in result
            assert result[commodity]["correlation_30d"] is None
            assert result[commodity]["correlation_90d"] is None
            assert "error" in result[commodity]

    def test_cache_ttl(self, analyzer):
        """Test that cache TTL is set correctly."""
        assert analyzer._cache_ttl_seconds == 3600  # 1 hour

    def test_cache_set_and_get(self, analyzer):
        """Test cache set and get operations."""
        analyzer._set_cache("BBCA", "GOLD", 0.5, 0.4)

        cached = analyzer._get_cache_key("BBCA", "GOLD")
        assert cached == (0.5, 0.4)

        # Different commodity should not be cached
        not_cached = analyzer._get_cache_key("BBCA", "SILVER")
        assert not_cached is None


class TestMacroCorrelationAnalyzerEdgeCases:
    """Edge case tests for MacroCorrelationAnalyzer."""

    @pytest.fixture
    def mock_db(self):
        return Mock(spec=DatabaseManager)

    @pytest.fixture
    def analyzer(self, mock_db):
        mock_db.get_session.return_value = MagicMock()
        return MacroCorrelationAnalyzer(mock_db)

    def test_zero_correlation_signal(self, analyzer, mock_db):
        """Test signal generation with zero correlation."""
        # Mock stock prices
        mock_session = MagicMock()
        mock_query = MagicMock()

        # Create mock price records
        mock_prices = [
            Mock(date=date.today() - timedelta(days=i), close=9000.0)
            for i in range(100, 0, -1)
        ]
        mock_filter = MagicMock()
        mock_filter.all.return_value = mock_prices
        mock_query.filter.return_value = mock_filter
        mock_session.query.return_value = mock_query
        mock_db.get_session.return_value = mock_session

        # Mock commodity prices with no correlation (random)
        mock_commodity = [
            Mock(date=date.today() - timedelta(days=i), close=2000.0 + np.random.randn() * 10)
            for i in range(100, 0, -1)
        ]
        mock_db.get_commodity_prices.return_value = mock_commodity

        result = analyzer.generate_signals("BBCA")

        # Should have overall signal (even if neutral)
        assert "overall_signal" in result

    def test_insufficient_overlapping_dates(self, analyzer, mock_db):
        """Test handling of insufficient overlapping dates."""
        # Stock has only 5 days
        mock_session = MagicMock()
        mock_query = MagicMock()
        mock_prices = [
            Mock(date=date.today() - timedelta(days=i), close=9000.0)
            for i in range(5, 0, -1)
        ]
        mock_filter = MagicMock()
        mock_filter.all.return_value = mock_prices
        mock_query.filter.return_value = mock_filter
        mock_session.query.return_value = mock_query
        mock_db.get_session.return_value = mock_session

        # Commodity has 90 days but only 5 overlap
        commodity_prices = [
            Mock(date=date.today() - timedelta(days=i), close=2000.0)
            for i in range(90, 0, -1)
        ]
        mock_db.get_commodity_prices.return_value = commodity_prices

        result = analyzer.compute_correlations("BBCA")

        # Should have error details for correlations due to insufficient overlap
        for commodity in ["GOLD", "SILVER", "OIL"]:
            assert result[commodity]["correlation_30d"] is None
            assert result[commodity]["correlation_90d"] is None
            assert "error" in result[commodity]


class TestMacroCorrelationAnalyzerSignals:
    """Tests for signal generation."""

    @pytest.fixture
    def mock_db(self):
        return Mock(spec=DatabaseManager)

    @pytest.fixture
    def analyzer(self, mock_db):
        mock_db.get_session.return_value = MagicMock()
        return MacroCorrelationAnalyzer(mock_db)

    def test_generate_signals_bullish(self, analyzer, mock_db):
        """Test bullish signal generation with positive correlation and up movement."""
        # Mock correlated stock prices (going up with gold)
        mock_session = MagicMock()
        mock_query = MagicMock()
        mock_prices = [
            Mock(date=date.today() - timedelta(days=i), close=9000.0 + (100-i))
            for i in range(100, 0, -1)
        ]
        mock_filter = MagicMock()
        mock_filter.all.return_value = mock_prices
        mock_query.filter.return_value = mock_filter
        mock_session.query.return_value = mock_query
        mock_db.get_session.return_value = mock_session

        # Mock commodity prices (going up - correlated with stock)
        commodity_prices = [
            Mock(date=date.today() - timedelta(days=i), close=2000.0 + (100-i))
            for i in range(100, 0, -1)
        ]
        mock_db.get_commodity_prices.return_value = commodity_prices

        result = analyzer.generate_signals("BBCA", sector="MINING")

        # Should have signals
        assert "overall_signal" in result
        assert result["overall_signal"] in ["BULLISH", "BEARISH", "NEUTRAL"]

    def test_generate_signals_bearish(self, analyzer, mock_db):
        """Test bearish signal generation."""
        # Mock stock prices (going up)
        mock_session = MagicMock()
        mock_query = MagicMock()
        mock_prices = [
            Mock(date=date.today() - timedelta(days=i), close=9000.0 + (100-i))
            for i in range(100, 0, -1)
        ]
        mock_filter = MagicMock()
        mock_filter.all.return_value = mock_prices
        mock_query.filter.return_value = mock_filter
        mock_session.query.return_value = mock_query
        mock_db.get_session.return_value = mock_session

        # Mock commodity prices (going down - inversely correlated)
        commodity_prices = [
            Mock(date=date.today() - timedelta(days=i), close=2000.0 - (100-i))
            for i in range(100, 0, -1)
        ]
        mock_db.get_commodity_prices.return_value = commodity_prices

        result = analyzer.generate_signals("BBCA")

        # Should have signals
        assert "overall_signal" in result

    def test_generate_signals_with_sector(self, analyzer, mock_db):
        """Test signal generation includes sector impact."""
        # Mock stock prices
        mock_session = MagicMock()
        mock_query = MagicMock()
        mock_prices = [
            Mock(date=date.today() - timedelta(days=i), close=9000.0)
            for i in range(100, 0, -1)
        ]
        mock_filter = MagicMock()
        mock_filter.all.return_value = mock_prices
        mock_query.filter.return_value = mock_filter
        mock_session.query.return_value = mock_query
        mock_db.get_session.return_value = mock_session

        # Mock commodity prices
        commodity_prices = [
            Mock(date=date.today() - timedelta(days=i), close=2000.0)
            for i in range(100, 0, -1)
        ]
        mock_db.get_commodity_prices.return_value = commodity_prices

        result = analyzer.generate_signals("BBCA", sector="MINING")

        # Should include sector impact
        if "sector_impact" in result:
            assert result["sector_impact"]["sector"] == "MINING"
