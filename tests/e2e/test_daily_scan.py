"""
End-to-End Tests for Daily Scan Workflow

Tests the complete daily scan flow:
- Data fetching
- Technical analysis
- Signal generation
- Risk validation
- Report generation
"""

import pytest
from datetime import date, datetime
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List

from config.trading_modes import TradingMode
from agents.coordinator import (
    Coordinator,
    CoordinatorConfig,
    ScanReport,
    ScanResult,
)


class TestDailyScanE2E:
    """End-to-end tests for daily scan."""

    @pytest.fixture
    def mock_data_fetcher(self):
        """Mock data fetcher."""
        from datetime import timedelta

        def fetch_historical(symbol, period="3mo"):
            # Generate mock OHLCV data
            import random
            base_price = random.uniform(1000, 10000)
            data = []
            end_date = date.today()

            for i in range(100):
                change = random.uniform(-0.03, 0.03)
                base_price *= (1 + change)
                # Go back from today safely
                data_date = end_date - timedelta(days=100 - i)
                data.append({
                    "date": data_date,
                    "open": base_price * 0.99,
                    "high": base_price * 1.02,
                    "low": base_price * 0.98,
                    "close": base_price,
                    "volume": random.randint(100000, 10000000),
                })
            return data

        mock = Mock()
        mock.fetch_historical = fetch_historical
        mock.fetch_current = Mock(return_value=True)
        return mock

    @pytest.fixture
    def mock_flow_analyzer(self):
        """Mock foreign flow analyzer."""
        mock = Mock()
        mock.analyze = Mock(return_value={
            "net_flow": 5000000000,
            "flow_trend": "accumulation",
            "flow_score": 75,
        })
        return mock

    @pytest.fixture
    def coordinator_config(self):
        """Create test configuration."""
        return CoordinatorConfig(
            universe=["BBCA", "BBRI", "TLKM"],
            mode=TradingMode.SWING,
            dry_run=True,
            max_signals_per_day=5,
            min_signal_score=60.0,
            enable_notifications=False,
        )

    def test_coordinator_initialization(self, coordinator_config):
        """Test coordinator can be initialized."""
        coordinator = Coordinator(config=coordinator_config)

        assert coordinator.config == coordinator_config
        assert coordinator.dry_run is True
        assert coordinator.mode == TradingMode.SWING

    def test_daily_scan_basic_flow(self, coordinator_config, mock_data_fetcher):
        """Test basic daily scan flow."""
        coordinator = Coordinator(config=coordinator_config)
        coordinator._initialized = True
        coordinator._data_manager = mock_data_fetcher

        # Mock other components
        coordinator._technical_analyzer = Mock()
        coordinator._technical_analyzer.calculate = Mock(return_value=Mock(
            rsi=50,
            macd=Mock(macd=0, signal=0, histogram=0),
            ema_20=8000,
            ema_50=7800,
            trend="neutral",
            score=70,
        ))

        coordinator._signal_generator = Mock()
        coordinator._signal_generator.generate = Mock(return_value={
            "symbol": "BBCA",
            "signal_type": "BUY",
            "entry_price": 8500,
            "stop_loss": 8200,
            "targets": [{"price": 9000, "pct": 5.9}],
            "score": 75,
            "composite_score": 75,
        })

        coordinator._risk_manager = Mock()
        coordinator._risk_manager.validate_entry = Mock(return_value=Mock(
            approved=True,
            position_size=100,
            veto_reason=None,
        ))

        # Run scan
        report = coordinator.run_daily_scan()

        assert report is not None
        assert isinstance(report, ScanReport)
        assert report.mode == TradingMode.SWING
        assert report.symbols_scanned == 3

    def test_daily_scan_no_signals(self, coordinator_config, mock_data_fetcher):
        """Test daily scan when no signals are generated."""
        coordinator = Coordinator(config=coordinator_config)
        coordinator._initialized = True
        coordinator._data_manager = mock_data_fetcher

        coordinator._technical_analyzer = Mock()
        coordinator._technical_analyzer.calculate = Mock(return_value=Mock(
            rsi=50,
            score=40,  # Low score
        ))

        coordinator._signal_generator = Mock()
        coordinator._signal_generator.generate = Mock(return_value=None)  # No signals

        report = coordinator.run_daily_scan()

        assert report.result == ScanResult.NO_SIGNALS
        assert report.signals_generated == 0
        assert report.signals_approved == 0

    def test_daily_scan_with_rejections(self, coordinator_config, mock_data_fetcher):
        """Test daily scan with risk manager rejections."""
        coordinator = Coordinator(config=coordinator_config)
        coordinator._initialized = True
        coordinator._data_manager = mock_data_fetcher

        coordinator._technical_analyzer = Mock()
        coordinator._technical_analyzer.calculate = Mock(return_value=Mock(score=70))

        coordinator._signal_generator = Mock()
        coordinator._signal_generator.generate = Mock(return_value={
            "symbol": "BBCA",
            "signal_type": "BUY",
            "entry_price": 8500,
            "score": 75,
        })

        coordinator._risk_manager = Mock()
        coordinator._risk_manager.validate_entry = Mock(return_value=Mock(
            approved=False,
            veto_reason="Position limit reached",
        ))

        report = coordinator.run_daily_scan()

        assert report.signals_generated == 3  # 3 symbols
        assert report.signals_approved == 0  # All rejected

    def test_daily_scan_with_errors(self, coordinator_config):
        """Test daily scan handles errors gracefully."""
        coordinator = Coordinator(config=coordinator_config)

        # Mock that throws error
        coordinator._initialize_components = Mock(side_effect=Exception("Init failed"))

        report = coordinator.run_daily_scan()

        assert report.result == ScanResult.FAILED
        assert len(report.errors) > 0

    def test_daily_scan_respects_max_signals(self, coordinator_config, mock_data_fetcher):
        """Test that scan respects max signals limit."""
        coordinator = Coordinator(config=coordinator_config)
        coordinator._initialized = True
        coordinator._data_manager = mock_data_fetcher

        coordinator._technical_analyzer = Mock()
        coordinator._technical_analyzer.calculate = Mock(return_value=Mock(score=80))

        coordinator._signal_generator = Mock()
        coordinator._signal_generator.generate = Mock(return_value={
            "symbol": "TEST",
            "signal_type": "BUY",
            "entry_price": 8500,
            "score": 80,
        })

        coordinator._risk_manager = Mock()
        coordinator._risk_manager.validate_entry = Mock(return_value=Mock(approved=True))

        report = coordinator.run_daily_scan()

        # Should be limited to max_signals_per_day (5)
        assert report.signals_approved <= coordinator_config.max_signals_per_day

    def test_daily_scan_report_serialization(self, coordinator_config, mock_data_fetcher):
        """Test scan report can be serialized to dict."""
        coordinator = Coordinator(config=coordinator_config)
        coordinator._initialized = True
        coordinator._data_manager = mock_data_fetcher

        coordinator._technical_analyzer = Mock()
        coordinator._technical_analyzer.calculate = Mock(return_value=Mock(score=70))

        coordinator._signal_generator = Mock()
        coordinator._signal_generator.generate = Mock(return_value=None)

        report = coordinator.run_daily_scan()
        report_dict = report.to_dict()

        assert "scan_date" in report_dict
        assert "mode" in report_dict
        assert "result" in report_dict
        assert "signals_generated" in report_dict

    def test_different_trading_modes(self, coordinator_config, mock_data_fetcher):
        """Test daily scan works with different trading modes."""
        modes = [TradingMode.SWING, TradingMode.POSITION, TradingMode.INTRADAY]

        for mode in modes:
            config = CoordinatorConfig(
                universe=["BBCA"],
                mode=mode,
                dry_run=True,
            )
            coordinator = Coordinator(config=config)
            coordinator._initialized = True
            coordinator._data_manager = mock_data_fetcher

            coordinator._technical_analyzer = Mock()
            coordinator._technical_analyzer.calculate = Mock(return_value=Mock(score=60))

            coordinator._signal_generator = Mock()
            coordinator._signal_generator.generate = Mock(return_value=None)

            report = coordinator.run_daily_scan()

            assert report.mode == mode


class TestCoordinatorUtilities:
    """Test coordinator utility methods."""

    def test_get_lq45_symbols(self):
        """Test LQ45 symbol list retrieval."""
        coordinator = Coordinator(dry_run=True)
        symbols = coordinator._get_lq45_symbols()

        assert len(symbols) > 0
        assert "BBCA" in symbols
        assert "TLKM" in symbols

    def test_get_mock_data(self):
        """Test mock data generation."""
        coordinator = Coordinator(dry_run=True)
        data = coordinator._get_mock_data("BBCA")

        assert len(data) == 100
        assert all("close" in d for d in data)
        assert all("volume" in d for d in data)

    def test_universe_default(self):
        """Test default universe is LQ45."""
        coordinator = Coordinator(dry_run=True)
        universe = coordinator.universe

        assert len(universe) > 0
        assert "BBCA" in universe

    def test_universe_custom(self):
        """Test custom universe."""
        coordinator = Coordinator(
            universe=["BBCA", "BBRI"],
            dry_run=True,
        )
        universe = coordinator.universe

        assert universe == ["BBCA", "BBRI"]


class TestScanReport:
    """Tests for ScanReport dataclass."""

    def test_scan_report_creation(self):
        """Test creating a scan report."""
        report = ScanReport(
            scan_date=date.today(),
            mode=TradingMode.SWING,
            result=ScanResult.SUCCESS,
            signals_generated=5,
            signals_approved=3,
        )

        assert report.signals_generated == 5
        assert report.signals_approved == 3
        assert report.errors == []
        assert report.signals == []

    def test_scan_report_to_dict(self):
        """Test scan report serialization."""
        report = ScanReport(
            scan_date=date(2024, 1, 15),
            mode=TradingMode.SWING,
            result=ScanResult.SUCCESS,
            signals_generated=5,
            signals_approved=3,
            execution_time_seconds=12.5,
        )

        data = report.to_dict()

        assert data["scan_date"] == "2024-01-15"
        assert data["mode"] == "swing"
        assert data["result"] == "success"
        assert data["signals_generated"] == 5
