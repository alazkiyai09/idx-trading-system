"""
Tests for Phase 7 - Integration & Testing

Tests for:
- Coordinator module
- Telegram bot
- Daily scan script
- End-to-end workflows
"""

import pytest
from datetime import date, datetime
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List
import os


# Coordinator Tests
class TestCoordinator:
    """Tests for the main coordinator."""

    @pytest.fixture
    def coordinator(self):
        from agents.coordinator import Coordinator, CoordinatorConfig
        from config.trading_modes import TradingMode

        config = CoordinatorConfig(
            universe=["BBCA", "BBRI"],
            mode=TradingMode.SWING,
            dry_run=True,
        )
        return Coordinator(config=config)

    def test_coordinator_creation(self, coordinator):
        """Test coordinator can be created."""
        assert coordinator is not None
        assert coordinator.dry_run is True

    def test_coordinator_config(self, coordinator):
        """Test coordinator has correct config."""
        assert coordinator.config.universe == ["BBCA", "BBRI"]

    def test_get_universe(self, coordinator):
        """Test universe retrieval."""
        universe = coordinator.universe
        assert "BBCA" in universe
        assert "BBRI" in universe

    def test_get_lq45_symbols(self, coordinator):
        """Test LQ45 symbols retrieval."""
        symbols = coordinator._get_lq45_symbols()
        assert len(symbols) > 0
        assert "BBCA" in symbols
        assert "TLKM" in symbols

    def test_get_mock_data(self, coordinator):
        """Test mock data generation."""
        data = coordinator._get_mock_data("BBCA")
        assert len(data) == 100
        assert all("close" in d for d in data)
        assert all("volume" in d for d in data)

    def test_scan_report_creation(self):
        """Test scan report dataclass."""
        from agents.coordinator import ScanReport, ScanResult
        from config.trading_modes import TradingMode

        report = ScanReport(
            scan_date=date.today(),
            mode=TradingMode.SWING,
            result=ScanResult.SUCCESS,
            signals_generated=5,
        )

        assert report.signals_generated == 5
        assert report.result == ScanResult.SUCCESS

    def test_scan_report_to_dict(self):
        """Test scan report serialization."""
        from agents.coordinator import ScanReport, ScanResult
        from config.trading_modes import TradingMode

        report = ScanReport(
            scan_date=date(2024, 1, 15),
            mode=TradingMode.SWING,
            result=ScanResult.SUCCESS,
            signals_generated=5,
            signals_approved=3,
        )

        data = report.to_dict()
        assert data["scan_date"] == "2024-01-15"
        assert data["mode"] == "swing"
        assert data["result"] == "success"

    def test_coordinator_config_defaults(self):
        """Test coordinator config default values."""
        from agents.coordinator import CoordinatorConfig

        config = CoordinatorConfig()
        assert config.universe == []
        assert config.dry_run is True
        assert config.max_signals_per_day == 10

    def test_scan_result_enum(self):
        """Test scan result enum values."""
        from agents.coordinator import ScanResult

        assert ScanResult.SUCCESS.value == "success"
        assert ScanResult.PARTIAL.value == "partial"
        assert ScanResult.FAILED.value == "failed"
        assert ScanResult.NO_SIGNALS.value == "no_signals"

    def test_get_portfolio_summary(self, coordinator):
        """Test portfolio summary retrieval."""
        summary = coordinator.get_portfolio_summary()

        assert isinstance(summary, dict)
        assert "total_value" in summary

    def test_execute_signals_dry_run(self, coordinator):
        """Test signal execution in dry run mode."""
        signals = [{"symbol": "BBCA", "signal_type": "BUY"}]
        results = coordinator.execute_signals(signals)

        assert len(results) == 1
        assert results[0]["status"] == "dry_run"

    def test_run_end_of_day(self, coordinator):
        """Test end of day workflow."""
        report = coordinator.run_end_of_day()

        assert isinstance(report, dict)
        assert "date" in report


class TestTelegramBot:
    """Tests for Telegram notification bot."""

    @pytest.fixture
    def telegram_config(self):
        from notifications.telegram_bot import TelegramConfig
        return TelegramConfig(
            bot_token="test_token",
            chat_id="test_chat_id",
            enabled=True,
        )

    @pytest.fixture
    def console_notifier(self):
        from notifications.telegram_bot import ConsoleNotifier
        return ConsoleNotifier(verbose=True)

    def test_telegram_config_creation(self, telegram_config):
        """Test Telegram config creation."""
        assert telegram_config.bot_token == "test_token"
        assert telegram_config.chat_id == "test_chat_id"
        assert telegram_config.enabled is True

    def test_telegram_config_from_env(self):
        """Test Telegram config from environment."""
        from notifications.telegram_bot import TelegramConfig

        with patch.dict(os.environ, {
            "TELEGRAM_BOT_TOKEN": "env_token",
            "TELEGRAM_CHAT_ID": "env_chat"
        }):
            config = TelegramConfig.from_env()
            assert config.bot_token == "env_token"
            assert config.chat_id == "env_chat"

    def test_console_notifier_send_message(self, console_notifier):
        """Test console notifier sends message."""
        from notifications.telegram_bot import MessageType

        result = console_notifier.send_message(
            "Test message",
            MessageType.INFO
        )
        assert result is True

    def test_console_notifier_send_signals(self, console_notifier):
        """Test console notifier sends signals."""
        signals = [
            {"symbol": "BBCA", "signal_type": "BUY", "price": 8500}
        ]
        result = console_notifier.send_signals(signals)
        assert result is True

    def test_console_notifier_send_risk_alert(self, console_notifier):
        """Test console notifier sends risk alert."""
        result = console_notifier.send_risk_alert(
            "POSITION_LIMIT",
            "Position limit reached",
            {"symbol": "BBCA"}
        )
        assert result is True

    def test_message_type_enum(self):
        """Test message type enum values."""
        from notifications.telegram_bot import MessageType

        assert MessageType.SIGNAL.value == "signal"
        assert MessageType.DAILY_SUMMARY.value == "daily_summary"
        assert MessageType.RISK_ALERT.value == "risk_alert"
        assert MessageType.ERROR.value == "error"

    def test_telegram_notifier_disabled(self):
        """Test Telegram notifier when disabled."""
        from notifications.telegram_bot import TelegramNotifier

        notifier = TelegramNotifier(
            bot_token=None,
            chat_id=None,
            enabled=False,
        )

        result = notifier.send_message("Test")
        assert result is False  # Should return False when disabled


class TestCoordinatorIntegration:
    """Integration tests for coordinator with other components."""

    @pytest.fixture
    def mock_components(self):
        """Create mock components for testing."""
        return {
            "data_manager": Mock(),
            "technical_analyzer": Mock(),
            "signal_generator": Mock(),
            "risk_manager": Mock(),
        }

    def test_coordinator_with_mock_data(self, mock_components):
        """Test coordinator with mocked data manager."""
        from agents.coordinator import Coordinator, CoordinatorConfig
        from config.trading_modes import TradingMode

        config = CoordinatorConfig(
            universe=["BBCA"],
            mode=TradingMode.SWING,
            dry_run=True,
        )
        coordinator = Coordinator(config=config)

        # Set mock data manager
        mock_components["data_manager"].fetch_historical = Mock(return_value=[
            {"date": date.today(), "open": 8500, "high": 8600,
             "low": 8400, "close": 8550, "volume": 1000000}
        ] * 100)

        coordinator._data_manager = mock_components["data_manager"]
        coordinator._initialized = True

        assert coordinator._data_manager is not None

    def test_coordinator_scan_symbol(self):
        """Test scanning a single symbol."""
        from agents.coordinator import Coordinator
        from config.trading_modes import TradingMode

        coordinator = Coordinator(mode=TradingMode.SWING, dry_run=True)
        coordinator._initialized = True

        # Mock components
        coordinator._data_manager = Mock()
        coordinator._data_manager.fetch_historical = Mock(return_value=[
            {"date": date.today(), "close": 8500, "volume": 1000000}
        ] * 100)

        coordinator._technical_analyzer = Mock()
        coordinator._technical_analyzer.calculate = Mock(return_value=Mock(
            rsi=50,
            score=70,
        ))

        coordinator._signal_generator = Mock()
        coordinator._signal_generator.generate = Mock(return_value={
            "symbol": "BBCA",
            "signal_type": "BUY",
            "score": 75,
        })

        signal = coordinator._scan_symbol("BBCA", TradingMode.SWING)

        assert signal is not None
        assert signal["symbol"] == "BBCA"


class TestCoordinatorConfig:
    """Tests for CoordinatorConfig."""

    def test_config_creation(self):
        """Test config can be created."""
        from agents.coordinator import CoordinatorConfig
        from config.trading_modes import TradingMode

        config = CoordinatorConfig(
            universe=["BBCA"],
            mode=TradingMode.SWING,
            dry_run=True,
            max_signals_per_day=5,
            min_signal_score=65.0,
        )

        assert config.universe == ["BBCA"]
        assert config.mode == TradingMode.SWING
        assert config.max_signals_per_day == 5
        assert config.min_signal_score == 65.0

    def test_config_defaults(self):
        """Test config default values."""
        from agents.coordinator import CoordinatorConfig

        config = CoordinatorConfig()

        assert config.universe == []
        assert config.dry_run is True
        assert config.max_signals_per_day == 10
        assert config.min_signal_score == 60.0
        assert config.enable_notifications is False


class TestConsoleNotifier:
    """Tests for ConsoleNotifier."""

    @pytest.fixture
    def notifier(self):
        from notifications.telegram_bot import ConsoleNotifier
        return ConsoleNotifier(verbose=True)

    def test_format_signals_empty(self, notifier):
        """Test formatting empty signals."""
        result = notifier._format_signals([])
        assert "No signals" in result

    def test_format_signals_with_data(self, notifier):
        """Test formatting signals with data."""
        signals = [
            {
                "symbol": "BBCA",
                "signal_type": "BUY",
                "price": 8500,
                "score": 75,
            }
        ]
        result = notifier._format_signals(signals)

        assert "BBCA" in result
        assert "BUY" in result
        assert "8,500" in result  # Formatted with comma

    def test_send_daily_summary(self, notifier):
        """Test sending daily summary."""
        result = notifier.send_daily_summary({
            "total_value": 1_000_000_000,
            "daily_pnl": 50_000_000,
        })
        assert result is True

    def test_send_error(self, notifier):
        """Test sending error notification."""
        result = notifier.send_error(
            "TEST_ERROR",
            "Test error message",
            traceback="Line 1\nLine 2"
        )
        assert result is True
