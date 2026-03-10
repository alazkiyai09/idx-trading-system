"""
Core Interfaces Module

Defines Protocol classes (interfaces) for key components.
Enables dependency injection, easy mocking, and clean architecture.
"""

from typing import Dict, List, Optional, Protocol, runtime_checkable
from datetime import date


@runtime_checkable
class DataProvider(Protocol):
    """Interface for market data providers.

    Any class that provides market data must implement these methods.
    This enables swapping between live data, cached data, and mock data.
    """

    def fetch_prices(self, symbol: str, days: int = 100) -> list:
        """Fetch historical price data.

        Args:
            symbol: Stock symbol (e.g., "BBCA").
            days: Number of days of history.

        Returns:
            List of OHLCV data points.
        """
        ...

    def fetch_foreign_flow(self, symbol: str, days: int = 30) -> list:
        """Fetch foreign flow data.

        Args:
            symbol: Stock symbol.
            days: Number of days of history.

        Returns:
            List of ForeignFlow data points.
        """
        ...


@runtime_checkable
class SignalGeneratorInterface(Protocol):
    """Interface for signal generators.

    Enables swapping between standard and forecast-enhanced generators.
    """

    def generate(
        self,
        symbol: str,
        ohlcv_data: list,
        flow_analysis: Optional[object] = None,
        fundamental_score: Optional[float] = None,
    ) -> Optional[object]:
        """Generate a trading signal.

        Args:
            symbol: Stock symbol.
            ohlcv_data: Historical OHLCV data.
            flow_analysis: Optional foreign flow analysis.
            fundamental_score: Optional fundamental score (0-100).

        Returns:
            Signal object if signal generated, None otherwise.
        """
        ...


@runtime_checkable
class RiskValidatorInterface(Protocol):
    """Interface for risk validation.

    The risk validator has veto power over all trades.
    """

    def validate_entry(
        self,
        signal: object,
        portfolio: object,
    ) -> object:
        """Validate a potential trade entry.

        Args:
            signal: Trading signal to validate.
            portfolio: Current portfolio state.

        Returns:
            ValidationResult with approved/rejected status.
        """
        ...

    def validate_exit(
        self,
        position: object,
        proposed_exit_price: float,
        exit_reason: str,
    ) -> object:
        """Validate a proposed exit.

        Args:
            position: Position to exit.
            proposed_exit_price: Proposed exit price.
            exit_reason: Reason for exit.

        Returns:
            ValidationResult with approval status.
        """
        ...

    def should_halt_trading(self, portfolio: object) -> tuple:
        """Check if trading should be halted.

        Args:
            portfolio: Current portfolio state.

        Returns:
            Tuple of (should_halt, reason).
        """
        ...


@runtime_checkable
class ExecutionEngine(Protocol):
    """Interface for trade execution.

    Enables switching between paper trader and live broker.
    """

    def execute_buy(
        self,
        symbol: str,
        shares: int,
        price: float,
    ) -> object:
        """Execute a buy order.

        Args:
            symbol: Stock to buy.
            shares: Number of shares.
            price: Limit price.

        Returns:
            Execution result.
        """
        ...

    def execute_sell(
        self,
        symbol: str,
        shares: int,
        price: float,
    ) -> object:
        """Execute a sell order.

        Args:
            symbol: Stock to sell.
            shares: Number of shares.
            price: Limit price.

        Returns:
            Execution result.
        """
        ...


@runtime_checkable
class Notifier(Protocol):
    """Interface for notifications.

    Enables switching between Telegram, email, and console notifiers.
    """

    def send_signals(self, signals: list) -> bool:
        """Send trading signals notification.

        Args:
            signals: List of signal dictionaries.

        Returns:
            True if sent successfully.
        """
        ...

    def send_daily_summary(self, summary: dict) -> bool:
        """Send daily summary.

        Args:
            summary: Summary dictionary.

        Returns:
            True if sent successfully.
        """
        ...

    def send_risk_alert(
        self,
        alert_type: str,
        message: str,
        details: Optional[dict] = None,
    ) -> bool:
        """Send risk alert.

        Args:
            alert_type: Type of alert.
            message: Alert message.
            details: Additional details.

        Returns:
            True if sent successfully.
        """
        ...

    def send_error(
        self,
        error_type: str,
        message: str,
        traceback: Optional[str] = None,
    ) -> bool:
        """Send error notification.

        Args:
            error_type: Type of error.
            message: Error message.
            traceback: Optional traceback.

        Returns:
            True if sent successfully.
        """
        ...


@runtime_checkable
class PortfolioManagerInterface(Protocol):
    """Interface for portfolio management."""

    def get_portfolio_state(self) -> object:
        """Get current portfolio state."""
        ...

    def get_open_positions(self) -> list:
        """Get all open positions."""
        ...

    def get_position(self, symbol: str) -> Optional[object]:
        """Get position for a specific symbol."""
        ...
