"""
Unit tests for commodity_scraper module.
"""
import pytest
from datetime import date, timedelta
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np

from core.data.commodity_scraper import CommodityScraper


class TestCommodityScraper:
    """Tests for CommodityScraper class."""

    def test_commodities_constants(self):
        """Test that commodity tickers are correctly defined."""
        assert "GOLD" in CommodityScraper.COMMODITIES
        assert "SILVER" in CommodityScraper.COMMODITIES
        assert "OIL" in CommodityScraper.COMMODITIES
        assert CommodityScraper.COMMODITIES["GOLD"] == "GC=F"
        assert CommodityScraper.COMMODITIES["SILVER"] == "SI=F"
        assert CommodityScraper.COMMODITIES["OIL"] == "CL=F"

    def test_init_without_db(self):
        """Test initialization without database manager."""
        scraper = CommodityScraper()
        assert scraper.db is None

    def test_init_with_db(self):
        """Test initialization with database manager."""
        mock_db = Mock()
        scraper = CommodityScraper(db_manager=mock_db)
        assert scraper.db is mock_db

    def test_fetch_unknown_commodity_raises_error(self):
        """Test that unknown commodity raises ValueError."""
        scraper = CommodityScraper()
        with pytest.raises(ValueError, match="Unknown commodity"):
            scraper.fetch_commodity_data("UNKNOWN")

    def test_save_to_database_without_db(self):
        """Test that save without database returns 0."""
        scraper = CommodityScraper()
        result = scraper.save_to_database([{'test': 'data'}])
        assert result == 0

    def test_save_to_database_with_db(self):
        """Test save to database with mock db manager."""
        mock_db = Mock()
        mock_db.save_commodity_prices.return_value = 5

        scraper = CommodityScraper(db_manager=mock_db)
        data = [
            {'commodity': 'GOLD', 'date': date.today(), 'open': 2000, 'high': 2100, 'low': 1950, 'close': 2050, 'volume': 100000}
        ]
        result = scraper.save_to_database(data)

        assert result == 5
        mock_db.save_commodity_prices.assert_called_once_with(data)

    def test_max_retries_constant(self):
        """Test that MAX_RETRIES is set."""
        assert hasattr(CommodityScraper, 'MAX_RETRIES')
        assert CommodityScraper.MAX_RETRIES == 3

    def test_retry_delay_constant(self):
        """Test that RETRY_DELAY is set."""
        assert hasattr(CommodityScraper, 'RETRY_DELAY')
        assert CommodityScraper.RETRY_DELAY == 2


class TestCommodityScraperWithMockedYfinance:
    """Tests with mocked yfinance - using proper import mocking."""

    def test_fetch_commodity_data_success(self):
        """Test successful commodity data fetch."""
        # Create mock DataFrame
        dates = pd.date_range(end=pd.Timestamp.now(), periods=10, freq='D')
        mock_df = pd.DataFrame({
            'Date': dates,
            'Open': np.random.uniform(1900, 2100, 10),
            'High': np.random.uniform(2000, 2200, 10),
            'Low': np.random.uniform(1800, 2000, 10),
            'Close': np.random.uniform(1900, 2100, 10),
            'Volume': np.random.uniform(100000, 500000, 10),
        })

        with patch.dict('sys.modules', {'yfinance': MagicMock()}):
            import sys
            mock_yf = sys.modules['yfinance']
            mock_yf.download.return_value = mock_df

            # Re-import to get the mocked version
            import importlib
            import core.data.commodity_scraper
            importlib.reload(core.data.commodity_scraper)

            scraper = core.data.commodity_scraper.CommodityScraper()
            result = scraper.fetch_commodity_data("GOLD", days=10)

            assert len(result) == 10
            assert all('commodity' in r for r in result)
            assert all(r['commodity'] == 'GOLD' for r in result)

    def test_fetch_all_commodities(self):
        """Test fetching all commodities at once."""
        mock_df = pd.DataFrame({
            'Date': pd.date_range(end=pd.Timestamp.now(), periods=1, freq='D'),
            'Open': [2000], 'High': [2100], 'Low': [1950],
            'Close': [2050], 'Volume': [100000]
        })

        with patch.dict('sys.modules', {'yfinance': MagicMock()}):
            import sys
            mock_yf = sys.modules['yfinance']
            mock_yf.download.return_value = mock_df

            import importlib
            import core.data.commodity_scraper
            importlib.reload(core.data.commodity_scraper)

            scraper = core.data.commodity_scraper.CommodityScraper()
            result = scraper.fetch_all_commodities(days=10)

            assert "GOLD" in result
            assert "SILVER" in result
            assert "OIL" in result


class TestCommodityScraperIntegration:
    """Integration tests for commodity scraper."""

    @pytest.fixture
    def temp_db(self, tmp_path):
        """Create temporary database for testing."""
        from core.data.database import DatabaseManager
        db_path = tmp_path / "test_commodity.db"
        manager = DatabaseManager(f"sqlite:///{db_path}")
        manager.create_tables()
        return manager

    def test_full_workflow(self, temp_db):
        """Test full fetch and save workflow."""
        scraper = CommodityScraper(db_manager=temp_db)

        # Create mock data
        mock_data = [
            {
                'commodity': 'GOLD',
                'date': date.today() - timedelta(days=i),
                'open': 2000.0 + i,
                'high': 2100.0 + i,
                'low': 1950.0 + i,
                'close': 2050.0 + i,
                'volume': 100000.0
            }
            for i in range(5)
        ]

        # Save data
        inserted = scraper.save_to_database(mock_data)
        assert inserted == 5

        # Verify data was saved
        prices = temp_db.get_commodity_prices('GOLD', start_date=date.today() - timedelta(days=10))
        assert len(prices) == 5

    def test_database_round_trip(self, temp_db):
        """Test data can be saved and retrieved."""
        scraper = CommodityScraper(db_manager=temp_db)

        test_data = [
            {
                'commodity': 'SILVER',
                'date': date(2024, 1, 15),
                'open': 23.5,
                'high': 24.0,
                'low': 23.0,
                'close': 23.8,
                'volume': 500000.0
            }
        ]

        inserted = scraper.save_to_database(test_data)
        assert inserted == 1

        prices = temp_db.get_commodity_prices('SILVER', start_date=date(2024, 1, 1))
        assert len(prices) == 1
        assert prices[0].close == 23.8
