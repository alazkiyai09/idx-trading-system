"""
Telegram Bot Module

Sends trading notifications via Telegram:
- Trading signals
- Daily summaries
- Risk alerts
- Error notifications
"""

import logging
import os
from dataclasses import dataclass
from datetime import datetime, date
from typing import Dict, List, Optional, Any
from enum import Enum

logger = logging.getLogger(__name__)


class MessageType(Enum):
    """Type of notification message."""
    SIGNAL = "signal"
    DAILY_SUMMARY = "daily_summary"
    RISK_ALERT = "risk_alert"
    ERROR = "error"
    INFO = "info"


@dataclass
class TelegramConfig:
    """Configuration for Telegram bot."""
    bot_token: str = ""
    chat_id: str = ""
    enabled: bool = False
    parse_mode: str = "Markdown"

    @classmethod
    def from_env(cls) -> "TelegramConfig":
        """Create config from environment variables."""
        return cls(
            bot_token=os.getenv("TELEGRAM_BOT_TOKEN", ""),
            chat_id=os.getenv("TELEGRAM_CHAT_ID", ""),
            enabled=bool(os.getenv("TELEGRAM_BOT_TOKEN")),
        )


class TelegramNotifier:
    """Sends trading notifications via Telegram.

    Example:
        notifier = TelegramNotifier()
        notifier.send_signal({
            "symbol": "BBCA",
            "signal_type": "BUY",
            "price": 8500,
            "score": 75,
        })
    """

    def __init__(
        self,
        bot_token: Optional[str] = None,
        chat_id: Optional[str] = None,
        enabled: Optional[bool] = None,
    ) -> None:
        """Initialize Telegram notifier.

        Args:
            bot_token: Telegram bot token. Uses env var if not provided.
            chat_id: Telegram chat ID. Uses env var if not provided.
            enabled: Whether notifications are enabled. Auto-detects if not provided.
        """
        self.config = TelegramConfig(
            bot_token=bot_token or os.getenv("TELEGRAM_BOT_TOKEN", ""),
            chat_id=chat_id or os.getenv("TELEGRAM_CHAT_ID", ""),
            enabled=enabled if enabled is not None else bool(bot_token or os.getenv("TELEGRAM_BOT_TOKEN")),
        )

        self._client = None
        self._initialize_client()

    def _initialize_client(self) -> None:
        """Initialize the Telegram client."""
        if not self.config.enabled:
            logger.info("Telegram notifications disabled")
            return

        try:
            import requests
            self._client = requests
            logger.info("Telegram client initialized")
        except ImportError:
            logger.warning("requests library not available - Telegram disabled")
            self.config.enabled = False

    def send_message(
        self,
        text: str,
        message_type: MessageType = MessageType.INFO,
        parse_mode: str = "Markdown"
    ) -> bool:
        """Send a message via Telegram.

        Args:
            text: Message text to send.
            message_type: Type of message for logging.
            parse_mode: Parse mode (Markdown or HTML).

        Returns:
            True if message was sent successfully.
        """
        if not self.config.enabled or not self._client:
            logger.debug(f"Telegram disabled - would send: {text[:100]}...")
            return False

        if not self.config.bot_token or not self.config.chat_id:
            logger.warning("Telegram credentials not configured")
            return False

        url = f"https://api.telegram.org/bot{self.config.bot_token}/sendMessage"

        try:
            response = self._client.post(
                url,
                json={
                    "chat_id": self.config.chat_id,
                    "text": text,
                    "parse_mode": parse_mode,
                },
                timeout=10,
            )
            response.raise_for_status()
            logger.info(f"Telegram message sent: {message_type.value}")
            return True

        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}")
            return False

    def send_signals(self, signals: List[Dict[str, Any]]) -> bool:
        """Send trading signals notification.

        Args:
            signals: List of signal dictionaries.

        Returns:
            True if message was sent successfully.
        """
        if not signals:
            return self.send_message(
                "📊 *Daily Scan Complete*\n\nNo signals generated today.",
                MessageType.DAILY_SUMMARY
            )

        # Build message
        lines = [
            "📊 *TRADING SIGNALS*",
            f"📅 {date.today().strftime('%Y-%m-%d')}",
            f"🎯 {len(signals)} signal(s) generated\n",
        ]

        for i, signal in enumerate(signals[:10], 1):  # Limit to 10 signals
            symbol = signal.get("symbol", "UNKNOWN")
            signal_type = signal.get("signal_type", "UNKNOWN")
            price = signal.get("entry_price", signal.get("price", 0))
            score = signal.get("score", signal.get("composite_score", 0))
            stop_loss = signal.get("stop_loss", 0)
            target = signal.get("target_price", signal.get("targets", [{}])[0].get("price", 0) if signal.get("targets") else 0)

            # Emoji based on signal type
            emoji = "" if "BUY" in str(signal_type).upper() else ""

            lines.append(
                f"{i}. {emoji} *{symbol}* - {signal_type}\n"
                f"   💰 Rp {price:,.0f}\n"
                f"   🎯 Score: {score:.0f}\n"
                f"   🛑 Stop: Rp {stop_loss:,.0f}\n"
                f"   🎯 Target: Rp {target:,.0f}"
            )

        # Add disclaimer
        lines.append("\n_⚠️ Not financial advice. Do your own research._")

        return self.send_message(
            "\n".join(lines),
            MessageType.SIGNAL
        )

    def send_daily_summary(
        self,
        summary: Dict[str, Any]
    ) -> bool:
        """Send daily summary notification.

        Args:
            summary: Daily summary dictionary.

        Returns:
            True if message was sent successfully.
        """
        lines = [
            "📈 *DAILY SUMMARY*",
            f"📅 {date.today().strftime('%Y-%m-%d')}\n",
        ]

        # Portfolio value
        if "total_value" in summary:
            lines.append(f"💼 Portfolio: Rp {summary['total_value']:,.0f}")

        # Daily P&L
        if "daily_pnl" in summary:
            pnl = summary["daily_pnl"]
            emoji = "" if pnl >= 0 else ""
            lines.append(f"{emoji} Daily P&L: Rp {pnl:,.0f}")

        # Positions
        positions = summary.get("positions", [])
        if positions:
            lines.append(f"\n📊 Open Positions: {len(positions)}")
            for pos in positions[:5]:
                symbol = pos.get("symbol", "UNKNOWN")
                pnl = pos.get("unrealized_pnl", 0)
                pnl_pct = pos.get("pnl_pct", 0)
                emoji = "" if pnl >= 0 else ""
                lines.append(f"  {symbol}: {emoji} {pnl_pct:.1f}%")

        # Risk metrics
        if "risk_metrics" in summary:
            lines.append(f"\n⚡ Risk Level: {summary['risk_metrics'].get('risk_level', 'N/A')}")

        return self.send_message(
            "\n".join(lines),
            MessageType.DAILY_SUMMARY
        )

    def send_risk_alert(
        self,
        alert_type: str,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Send risk alert notification.

        Args:
            alert_type: Type of risk alert.
            message: Alert message.
            details: Additional details.

        Returns:
            True if message was sent successfully.
        """
        lines = [
            "🚨 *RISK ALERT*",
            f"⚠️ {alert_type}\n",
            message,
        ]

        if details:
            lines.append("\n*Details:*")
            for key, value in details.items():
                lines.append(f"  • {key}: {value}")

        return self.send_message(
            "\n".join(lines),
            MessageType.RISK_ALERT
        )

    def send_error(
        self,
        error_type: str,
        message: str,
        traceback: Optional[str] = None
    ) -> bool:
        """Send error notification.

        Args:
            error_type: Type of error.
            message: Error message.
            traceback: Optional traceback string.

        Returns:
            True if message was sent successfully.
        """
        lines = [
            "❌ *ERROR*",
            f"🔴 {error_type}\n",
            message,
        ]

        if traceback:
            # Truncate long tracebacks
            if len(traceback) > 500:
                traceback = traceback[:500] + "..."
            lines.append(f"\n```\n{traceback}\n```")

        return self.send_message(
            "\n".join(lines),
            MessageType.ERROR
        )


class ConsoleNotifier:
    """Console-based notifier for testing/development.

    Prints notifications to console instead of sending to Telegram.
    """

    def __init__(self, verbose: bool = True) -> None:
        """Initialize console notifier.

        Args:
            verbose: If True, print all messages.
        """
        self.verbose = verbose

    def send_message(
        self,
        text: str,
        message_type: MessageType = MessageType.INFO
    ) -> bool:
        """Print message to console.

        Args:
            text: Message text.
            message_type: Type of message.

        Returns:
            Always True.
        """
        if self.verbose:
            print(f"\n[{message_type.value.upper()}] {datetime.now().strftime('%H:%M:%S')}")
            print(text)
            print()
        return True

    def send_signals(self, signals: List[Dict[str, Any]]) -> bool:
        """Print signals to console."""
        return self.send_message(
            self._format_signals(signals),
            MessageType.SIGNAL
        )

    def _format_signals(self, signals: List[Dict[str, Any]]) -> str:
        """Format signals for console output."""
        if not signals:
            return "No signals generated"

        lines = [
            "=" * 60,
            f"TRADING SIGNALS - {date.today()}",
            f"{len(signals)} signal(s) generated",
            "=" * 60,
        ]

        for i, signal in enumerate(signals, 1):
            symbol = signal.get("symbol", "UNKNOWN")
            signal_type = signal.get("signal_type", "UNKNOWN")
            price = signal.get("entry_price", signal.get("price", 0))
            score = signal.get("score", signal.get("composite_score", 0))

            lines.append(
                f"\n{i}. {symbol} - {signal_type}\n"
                f"   Entry: Rp {price:,.0f}\n"
                f"   Score: {score:.0f}"
            )

        lines.append("\n" + "=" * 60)
        return "\n".join(lines)

    def send_daily_summary(self, summary: Dict[str, Any]) -> bool:
        """Print daily summary to console."""
        return self.send_message(
            f"Daily Summary: {summary}",
            MessageType.DAILY_SUMMARY
        )

    def send_risk_alert(
        self,
        alert_type: str,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Print risk alert to console."""
        return self.send_message(
            f"RISK ALERT [{alert_type}]: {message} | {details}",
            MessageType.RISK_ALERT
        )

    def send_error(
        self,
        error_type: str,
        message: str,
        traceback: Optional[str] = None
    ) -> bool:
        """Print error to console."""
        return self.send_message(
            f"ERROR [{error_type}]: {message}\n{traceback}",
            MessageType.ERROR
        )
