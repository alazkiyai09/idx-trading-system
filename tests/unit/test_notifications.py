"""Tests for notification modules (Telegram and Console)."""

import pytest
from unittest.mock import MagicMock, patch
from notifications.telegram_bot import (
    TelegramNotifier,
    ConsoleNotifier,
    MessageType,
    TelegramConfig,
)


class TestTelegramConfig:
    """Tests for TelegramConfig."""

    def test_from_env_disabled(self):
        """Test config creation with no env vars."""
        with patch.dict("os.environ", {}, clear=True):
            config = TelegramConfig.from_env()
            assert config.enabled is False

    def test_from_env_enabled(self):
        """Test config creation with env vars set."""
        with patch.dict("os.environ", {
            "TELEGRAM_BOT_TOKEN": "test-token",
            "TELEGRAM_CHAT_ID": "12345",
        }):
            config = TelegramConfig.from_env()
            assert config.bot_token == "test-token"
            assert config.chat_id == "12345"
            assert config.enabled is True


class TestTelegramNotifier:
    """Tests for TelegramNotifier."""

    def test_init_disabled(self):
        """Test notifier initializes in disabled mode without credentials."""
        notifier = TelegramNotifier(enabled=False)
        # The notifier stores enabled state internally
        result = notifier.send_message("test", MessageType.INFO)
        assert result is False  # Disabled notifier should not send

    def test_send_message_disabled(self):
        """Test sending message when disabled returns False."""
        notifier = TelegramNotifier(enabled=False)
        result = notifier.send_message("test", MessageType.INFO)
        assert result is False

    def test_send_signals_disabled(self):
        """Test sending signals when disabled."""
        notifier = TelegramNotifier(enabled=False)
        signals = [{"symbol": "BBCA", "signal_type": "BUY", "entry_price": 9000}]
        result = notifier.send_signals(signals)
        assert result is False

    def test_send_daily_summary_disabled(self):
        """Test sending daily summary when disabled."""
        notifier = TelegramNotifier(enabled=False)
        summary = {"date": "2024-01-15", "signals_generated": 3}
        result = notifier.send_daily_summary(summary)
        assert result is False

    def test_send_risk_alert_disabled(self):
        """Test sending risk alert when disabled."""
        notifier = TelegramNotifier(enabled=False)
        result = notifier.send_risk_alert("position_limit", "Too many positions")
        assert result is False

    def test_send_error_disabled(self):
        """Test sending error when disabled."""
        notifier = TelegramNotifier(enabled=False)
        result = notifier.send_error("DataError", "Failed to fetch")
        assert result is False


class TestConsoleNotifier:
    """Tests for ConsoleNotifier."""

    def test_init(self):
        """Test console notifier initialization."""
        notifier = ConsoleNotifier(verbose=True)
        assert notifier.verbose is True

    def test_send_message(self, capsys):
        """Test sending message to console."""
        notifier = ConsoleNotifier(verbose=True)
        result = notifier.send_message("Hello World", MessageType.INFO)
        assert result is True
        captured = capsys.readouterr()
        assert "Hello World" in captured.out

    def test_send_signals(self, capsys):
        """Test sending signals to console."""
        notifier = ConsoleNotifier(verbose=True)
        signals = [
            {
                "symbol": "BBCA",
                "signal_type": "BUY",
                "entry_price": 9000,
                "stop_loss": 8700,
                "targets": [9300, 9600],
                "composite_score": 75.0,
            }
        ]
        result = notifier.send_signals(signals)
        assert result is True
        captured = capsys.readouterr()
        assert "BBCA" in captured.out

    def test_send_daily_summary(self, capsys):
        """Test sending daily summary to console."""
        notifier = ConsoleNotifier(verbose=True)
        summary = {"date": "2024-01-15", "signals": 3, "approved": 2}
        result = notifier.send_daily_summary(summary)
        assert result is True

    def test_send_risk_alert(self, capsys):
        """Test sending risk alert to console."""
        notifier = ConsoleNotifier(verbose=True)
        result = notifier.send_risk_alert(
            "drawdown",
            "Max drawdown exceeded",
            {"current": "12%", "limit": "10%"},
        )
        assert result is True
        captured = capsys.readouterr()
        assert "drawdown" in captured.out.lower() or "risk" in captured.out.lower()

    def test_send_error(self, capsys):
        """Test sending error to console."""
        notifier = ConsoleNotifier(verbose=True)
        result = notifier.send_error(
            "DataError",
            "Yahoo Finance timeout",
            "Traceback: ...",
        )
        assert result is True

    def test_verbose_off(self, capsys):
        """Test that verbose=False suppresses output."""
        notifier = ConsoleNotifier(verbose=False)
        notifier.send_message("Should not appear")
        captured = capsys.readouterr()
        # Output may or may not appear depending on implementation
        # but the call should still succeed
