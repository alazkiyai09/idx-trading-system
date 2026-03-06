"""Integration tests for data fetching."""

import pytest
from datetime import date, timedelta
from unittest.mock import patch, MagicMock

from core.data.scraper import IDXScraper
from core.data.foreign_flow import ForeignFlowFetcher
from core.data.cache import DataCacheManager
from core.data.database import DatabaseManager
from core.data.models import OHLCV, ForeignFlow


class TestScraperIntegration:
    """Integration tests for scraper with cache."""

    @pytest.fixture
    def scraper(self):
        """Create IDXScraper instance."""
        return IDXScraper()

    @pytest.fixture
    def cache_manager(self):
        """Create DataCacheManager instance."""
        return DataCacheManager()

    @pytest.fixture
    def db_manager(self, tmp_path):
        """Create DatabaseManager instance with temp database."""
        db_path = tmp_path / "test.db"
        manager = DatabaseManager(f"sqlite:///{db_path}")
        manager.create_tables()
        return manager

    @patch("core.data.scraper.YahooFinanceSource.fetch_daily")
    def test_fetch_and_cache_flow(
        self, mock_fetch, scraper, cache_manager, db_manager
    ):
        """Test fetching data, caching it, and storing in database."""
        # Setup mock
        mock_ohlcv = [
            OHLCV(
                symbol="BBCA",
                date=date(2024, 1, 15),
                open=9000.0,
                high=9200.0,
                low=8900.0,
                close=9100.0,
                volume=10000000,
                value=91000000000.0,
            ),
            OHLCV(
                symbol="BBCA",
                date=date(2024, 1, 16),
                open=9100.0,
                high=9300.0,
                low=9000.0,
                close=9200.0,
                volume=12000000,
                value=110400000000.0,
            ),
        ]
        mock_fetch.return_value = mock_ohlcv

        # Fetch data
        data = scraper.fetch_historical("BBCA", days=30)

        # Cache data
        cache_manager.set_price_data("BBCA", "historical", data)

        # Verify cache
        cached = cache_manager.get_price_data("BBCA", "historical")
        assert cached is not None
        assert len(cached) == 2

        # Store in database
        price_records = [
            {
                "symbol": ohlcv.symbol,
                "date": ohlcv.date,
                "open": ohlcv.open,
                "high": ohlcv.high,
                "low": ohlcv.low,
                "close": ohlcv.close,
                "volume": ohlcv.volume,
                "value": ohlcv.value,
            }
            for ohlcv in data
        ]
        db_manager.save_prices(price_records)

        # Verify database
        retrieved = db_manager.get_prices("BBCA", date(2024, 1, 1))
        assert len(retrieved) == 2


class TestForeignFlowIntegration:
    """Integration tests for foreign flow with analysis."""

    @pytest.fixture
    def fetcher(self):
        """Create ForeignFlowFetcher instance."""
        return ForeignFlowFetcher()

    def test_fetch_analyze_and_cache(self, fetcher):
        """Test fetching, analyzing, and caching foreign flow."""
        # Fetch data (uses simulated data)
        end_date = date.today()
        start_date = end_date - timedelta(days=30)

        flow_data = fetcher.fetch_flow("BBCA", start_date, end_date)

        # Analyze
        analysis = fetcher.analyze_flow(flow_data)

        assert analysis is not None
        assert analysis.symbol == "BBCA"
        assert analysis.signal is not None

        # Cache
        fetcher.cache_flow_data("BBCA", flow_data)

        # Verify cache
        cached = fetcher.get_cached_flow_data("BBCA")
        assert cached is not None
        assert len(cached) > 0


class TestDatabaseIntegration:
    """Integration tests for database operations."""

    @pytest.fixture
    def db_manager(self, tmp_path):
        """Create DatabaseManager with temp database."""
        db_path = tmp_path / "test.db"
        manager = DatabaseManager(f"sqlite:///{db_path}")
        manager.create_tables()
        return manager

    def test_full_price_workflow(self, db_manager):
        """Test full workflow for price data."""
        # Create test data
        prices = [
            {
                "symbol": "TEST",
                "date": date(2024, 1, i),
                "open": 100.0 + i,
                "high": 110.0 + i,
                "low": 90.0 + i,
                "close": 105.0 + i,
                "volume": 10000 * i,
                "value": 1050000.0 * i,
            }
            for i in range(1, 6)
        ]

        # Save
        db_manager.save_prices(prices)

        # Retrieve
        retrieved = db_manager.get_prices("TEST", date(2024, 1, 1), date(2024, 1, 5))

        assert len(retrieved) == 5
        assert retrieved[0].symbol == "TEST"
        assert retrieved[-1].close == 110.0

        # Get latest
        latest = db_manager.get_latest_price("TEST")
        assert latest is not None
        assert latest.date == date(2024, 1, 5)

    def test_flow_and_trade_storage(self, db_manager):
        """Test storing flow data and trades."""
        # Store foreign flow
        flows = [
            {
                "symbol": "TEST",
                "date": date(2024, 1, 15),
                "foreign_buy": 100_000_000_000.0,
                "foreign_sell": 80_000_000_000.0,
                "foreign_net": 20_000_000_000.0,
                "total_value": 200_000_000_000.0,
                "foreign_pct": 40.0,
            }
        ]
        db_manager.save_foreign_flows(flows)

        # Verify
        retrieved = db_manager.get_foreign_flows("TEST", date(2024, 1, 1))
        assert len(retrieved) == 1
        assert retrieved[0].foreign_net == 20_000_000_000.0


class TestCacheIntegration:
    """Integration tests for caching with data operations."""

    @pytest.fixture
    def cache_manager(self):
        """Create DataCacheManager instance."""
        return DataCacheManager()

    def test_cache_invalidation(self, cache_manager):
        """Test cache invalidation workflow."""
        # Set multiple cached values
        cache_manager.set_price_data("BBCA", "current", {"price": 9100})
        cache_manager.set_price_data("BBCA", "historical", [{"date": "2024-01-15"}])
        cache_manager.set_foreign_flow("BBCA", {"net": 1000000})

        cache_manager.set_price_data("TLKM", "current", {"price": 3500})

        # Invalidate BBCA
        cache_manager.invalidate_symbol("BBCA")

        # Verify
        assert cache_manager.get_price_data("BBCA", "current") is None
        assert cache_manager.get_price_data("BBCA", "historical") is None
        assert cache_manager.get_foreign_flow("BBCA") is None
        assert cache_manager.get_price_data("TLKM", "current") is not None

    def test_analysis_caching_workflow(self, cache_manager):
        """Test caching analysis results."""
        # Perform analysis (simulated)
        analysis_result = {
            "symbol": "BBCA",
            "date": "2024-01-15",
            "technical_score": 75,
            "flow_score": 80,
            "recommendation": "BUY",
        }

        # Cache result
        cache_manager.set_analysis("signal", "BBCA_20240115", analysis_result)

        # Retrieve
        cached = cache_manager.get_analysis("signal", "BBCA_20240115")

        assert cached is not None
        assert cached["recommendation"] == "BUY"
