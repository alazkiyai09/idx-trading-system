"""Tests for IDX scraper module."""

import pytest
from datetime import date, timedelta
from unittest.mock import MagicMock, patch

from core.data.scraper import (
    DataSource,
    YahooFinanceSource,
    IDXScraper,
)
from core.data.models import OHLCV


class TestYahooFinanceSource:
    """Tests for YahooFinanceSource class."""

    @pytest.fixture
    def source(self):
        """Create YahooFinanceSource instance."""
        return YahooFinanceSource()

    def test_format_symbol(self, source):
        """Test symbol formatting with .JK suffix."""
        assert source._format_symbol("BBCA") == "BBCA.JK"
        assert source._format_symbol("baca") == "BACA.JK"
        assert source._format_symbol("TLKM.JK") == "TLKM.JK"

    def test_strip_suffix(self, source):
        """Test removing .JK suffix."""
        assert source._strip_suffix("BBCA.JK") == "BBCA"
        assert source._strip_suffix("BBCA") == "BBCA"


class TestIDXScraper:
    """Tests for IDXScraper class."""

    @pytest.fixture
    def scraper(self):
        """Create IDXScraper instance."""
        return IDXScraper()

    def test_lq45_symbols_exist(self, scraper):
        """Test LQ45 symbols list is populated."""
        assert len(scraper.LQ45_SYMBOLS) > 0
        assert "BBCA" in scraper.LQ45_SYMBOLS
        assert "TLKM" in scraper.LQ45_SYMBOLS
        assert "BBRI" in scraper.LQ45_SYMBOLS

    def test_get_universe_lq45_only(self, scraper):
        """Test getting LQ45 universe."""
        universe = scraper.get_universe(include_kompas100=False)
        assert len(universe) == len(scraper.LQ45_SYMBOLS)
        assert "BBCA" in universe

    def test_get_universe_with_kompas100(self, scraper):
        """Test getting full universe including Kompas100."""
        universe = scraper.get_universe(include_kompas100=True)
        assert len(universe) >= len(scraper.LQ45_SYMBOLS)

    @patch("core.data.scraper.YahooFinanceSource.fetch_daily")
    def test_fetch_historical(self, mock_fetch, scraper):
        """Test fetching historical data."""
        # Setup mock
        mock_data = [
            OHLCV(
                symbol="BBCA",
                date=date(2024, 1, 15),
                open=9000.0,
                high=9200.0,
                low=8900.0,
                close=9100.0,
                volume=10000000,
            ),
        ]
        mock_fetch.return_value = mock_data

        # Execute
        result = scraper.fetch_historical("BBCA", days=30)

        # Verify
        assert len(result) == 1
        assert result[0].symbol == "BBCA"
        assert result[0].close == 9100.0

    @patch("core.data.scraper.YahooFinanceSource.fetch_intraday")
    def test_fetch_current(self, mock_fetch, scraper):
        """Test fetching current data."""
        # Setup mock
        mock_data = OHLCV(
            symbol="BBCA",
            date=date.today(),
            open=9000.0,
            high=9200.0,
            low=8900.0,
            close=9100.0,
            volume=10000000,
        )
        mock_fetch.return_value = mock_data

        # Execute
        result = scraper.fetch_current("BBCA")

        # Verify
        assert result is not None
        assert result.symbol == "BBCA"
        assert result.close == 9100.0

    @patch("core.data.scraper.YahooFinanceSource.fetch_daily")
    def test_fetch_multiple(self, mock_fetch, scraper):
        """Test fetching multiple symbols."""
        # Setup mock
        mock_fetch.return_value = [
            OHLCV(
                symbol="TEST",
                date=date(2024, 1, 15),
                open=100.0,
                high=110.0,
                low=90.0,
                close=105.0,
                volume=1000,
            )
        ]

        # Execute
        symbols = ["BBCA", "TLKM"]
        result = scraper.fetch_multiple(symbols, days=30)

        # Verify
        assert len(result) == 2
        assert "BBCA" in result
        assert "TLKM" in result

    @patch("core.data.scraper.YahooFinanceSource.fetch_daily")
    def test_fetch_lq45(self, mock_fetch, scraper):
        """Test fetching LQ45 stocks."""
        # Setup mock
        mock_fetch.return_value = []

        # Execute
        result = scraper.fetch_lq45(days=30)

        # Verify
        assert len(result) == len(scraper.LQ45_SYMBOLS)
        mock_fetch.assert_called()


class TestOHLCVModel:
    """Tests for OHLCV data model used by scraper."""

    def test_ohlcv_creation(self):
        """Test creating OHLCV instance."""
        ohlcv = OHLCV(
            symbol="BBCA",
            date=date(2024, 1, 15),
            open=9000.0,
            high=9200.0,
            low=8900.0,
            close=9100.0,
            volume=10000000,
            value=91000000000.0,
        )

        assert ohlcv.symbol == "BBCA"
        assert ohlcv.date == date(2024, 1, 15)
        assert ohlcv.open == 9000.0
        assert ohlcv.high == 9200.0
        assert ohlcv.low == 8900.0
        assert ohlcv.close == 9100.0
        assert ohlcv.volume == 10000000
        assert ohlcv.value == 91000000000.0

    def test_ohlcv_optional_value(self):
        """Test OHLCV with optional value field."""
        ohlcv = OHLCV(
            symbol="TLKM",
            date=date(2024, 1, 15),
            open=3500.0,
            high=3550.0,
            low=3450.0,
            close=3500.0,
            volume=50000000,
        )

        assert ohlcv.value is None
