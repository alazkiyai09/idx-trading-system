"""
Portfolio Manager Module

Tracks portfolio state, positions, and P&L.
Provides real-time portfolio snapshots and risk metrics.
"""

import logging
from datetime import datetime, date
from typing import Dict, List, Optional

from config.settings import settings
from core.data.models import Position, PortfolioState, Trade

logger = logging.getLogger(__name__)


class PortfolioManager:
    """Manages portfolio state and tracking.

    Tracks all positions, calculates P&L, monitors drawdown,
    and generates portfolio snapshots.

    Example:
        manager = PortfolioManager(initial_capital=100_000_000)
        manager.open_position(position)
        state = manager.get_state()
        print(f"Total value: {state.total_value}, P&L: {state.total_pnl}")
    """

    def __init__(
        self,
        initial_capital: Optional[float] = None,
    ) -> None:
        """Initialize portfolio manager.

        Args:
            initial_capital: Starting capital in IDR.
        """
        self.initial_capital = initial_capital or settings.initial_capital
        self.cash = self.initial_capital
        self.positions: Dict[str, Position] = {}

        # Track peak value for drawdown
        self.peak_value = self.initial_capital

        # Daily tracking
        self._daily_start_value = self.initial_capital
        self._last_date = date.today()

        # Trade history
        self.closed_trades: List[Trade] = []

    def open_position(self, position: Position) -> None:
        """Open a new position.

        Args:
            position: Position to open.
        """
        position_value = position.entry_price * position.quantity

        if position_value > self.cash:
            raise ValueError(
                f"Insufficient cash: need {position_value:,.0f}, have {self.cash:,.0f}"
            )

        self.cash -= position_value
        self.positions[position.symbol] = position

        logger.info(
            f"Opened position: {position.symbol} - "
            f"{position.quantity} shares @ {position.entry_price:,.0f}"
        )

    def close_position(
        self,
        symbol: str,
        exit_price: float,
        exit_date: date,
        exit_reason: str,
    ) -> Optional[Trade]:
        """Close a position.

        Args:
            symbol: Symbol of position to close.
            exit_price: Exit price per share.
            exit_date: Date of exit.
            exit_reason: Reason for exit.

        Returns:
            Trade record if position existed, None otherwise.
        """
        if symbol not in self.positions:
            logger.warning(f"No position in {symbol} to close")
            return None

        position = self.positions[symbol]

        # Calculate P&L
        gross_pnl = (exit_price - position.entry_price) * position.quantity
        buy_fees = position.entry_price * position.quantity * settings.buy_fee_pct
        sell_fees = exit_price * position.quantity * settings.sell_fee_pct
        total_fees = buy_fees + sell_fees
        net_pnl = gross_pnl - total_fees
        return_pct = (net_pnl / (position.entry_price * position.quantity)) * 100

        # Return cash
        exit_value = exit_price * position.quantity - sell_fees
        self.cash += exit_value

        # Create trade record
        trade = Trade(
            trade_id=f"TRD-{symbol}-{exit_date}",
            symbol=symbol,
            entry_date=position.entry_date,
            entry_price=position.entry_price,
            exit_date=exit_date,
            exit_price=exit_price,
            exit_reason=exit_reason,
            quantity=position.quantity,
            side=position.quantity > 0 and "BUY" or "SELL",
            gross_pnl=gross_pnl,
            fees=total_fees,
            net_pnl=net_pnl,
            return_pct=return_pct,
            holding_days=(exit_date - position.entry_date).days,
            max_favorable=0.0,  # Would need to track
            max_adverse=0.0,
            signal_score=position.signal_score,
            setup_type=position.setup_type,
            rsi_at_entry=50.0,
            flow_signal=getattr(position, "flow_signal", None),
            flow_consecutive_days=0,
        )

        # Remove position
        del self.positions[symbol]
        self.closed_trades.append(trade)

        logger.info(
            f"Closed position: {symbol} @ {exit_price:,.0f} - "
            f"P&L: {net_pnl:,.0f} ({return_pct:.2f}%)"
        )

        return trade

    def update_prices(self, prices: Dict[str, float]) -> None:
        """Update current prices for all positions.

        Args:
            prices: Dictionary of symbol to current price.
        """
        for symbol, position in self.positions.items():
            if symbol in prices:
                new_price = prices[symbol]

                # Update highest price for trailing stop
                if new_price > position.highest_price:
                    position.highest_price = new_price

                position.current_price = new_price

                # Calculate unrealized P&L
                position.unrealized_pnl = (
                    new_price - position.entry_price
                ) * position.quantity
                position.unrealized_pnl_pct = (
                    (new_price - position.entry_price) / position.entry_price * 100
                )

        # Update peak value
        current_value = self.get_total_value()
        if current_value > self.peak_value:
            self.peak_value = current_value

    def get_positions_value(self) -> float:
        """Calculate total value of all positions.

        Returns:
            Total position value in IDR.
        """
        return sum(
            position.current_price * position.quantity
            for position in self.positions.values()
        )

    def get_total_value(self) -> float:
        """Calculate total portfolio value.

        Returns:
            Cash + positions value in IDR.
        """
        return self.cash + self.get_positions_value()

    def get_total_pnl(self) -> float:
        """Calculate total profit/loss.

        Returns:
            Total P&L in IDR.
        """
        return self.get_total_value() - self.initial_capital

    def get_total_pnl_pct(self) -> float:
        """Calculate total return percentage.

        Returns:
            Total return as percentage.
        """
        return (self.get_total_value() - self.initial_capital) / self.initial_capital * 100

    def get_daily_pnl(self) -> float:
        """Calculate today's P&L.

        Returns:
            Daily P&L in IDR.
        """
        return self.get_total_value() - self._daily_start_value

    def get_daily_pnl_pct(self) -> float:
        """Calculate today's return percentage.

        Returns:
            Daily return as percentage.
        """
        if self._daily_start_value == 0:
            return 0.0
        return (self.get_total_value() - self._daily_start_value) / self._daily_start_value * 100

    def get_drawdown(self) -> float:
        """Calculate current drawdown.

        Returns:
            Drawdown amount in IDR.
        """
        return self.peak_value - self.get_total_value()

    def get_drawdown_pct(self) -> float:
        """Calculate current drawdown percentage.

        Returns:
            Drawdown as percentage of peak.
        """
        if self.peak_value == 0:
            return 0.0
        return (self.peak_value - self.get_total_value()) / self.peak_value * 100

    def get_state(self) -> PortfolioState:
        """Get current portfolio state snapshot.

        Returns:
            PortfolioState with all current metrics.
        """
        positions_value = self.get_positions_value()
        total_value = self.get_total_value()

        return PortfolioState(
            timestamp=datetime.now(),
            cash=self.cash,
            total_value=total_value,
            positions_value=positions_value,
            total_pnl=self.get_total_pnl(),
            total_pnl_pct=self.get_total_pnl_pct(),
            daily_pnl=self.get_daily_pnl(),
            daily_pnl_pct=self.get_daily_pnl_pct(),
            peak_value=self.peak_value,
            drawdown=self.get_drawdown(),
            drawdown_pct=self.get_drawdown_pct(),
            open_positions=len(self.positions),
            positions=list(self.positions.values()),
        )

    def reset_daily(self) -> None:
        """Reset daily tracking at market open."""
        today = date.today()

        # Only reset if it's a new day
        if today != self._last_date:
            self._daily_start_value = self.get_total_value()
            self._last_date = today

            # Update days held for positions
            for position in self.positions.values():
                position.days_held += 1

            logger.debug(f"Daily reset: start value = {self._daily_start_value:,.0f}")

    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for a symbol.

        Args:
            symbol: Stock symbol.

        Returns:
            Position if exists, None otherwise.
        """
        return self.positions.get(symbol)

    def has_position(self, symbol: str) -> bool:
        """Check if we have a position in a symbol.

        Args:
            symbol: Stock symbol.

        Returns:
            True if position exists.
        """
        return symbol in self.positions

    def get_position_count(self) -> int:
        """Get number of open positions.

        Returns:
            Number of open positions.
        """
        return len(self.positions)

    def get_cash_available(self) -> float:
        """Get available cash.

        Returns:
            Available cash in IDR.
        """
        return self.cash

    def get_buying_power(self) -> float:
        """Get total buying power.

        Returns:
            Cash available for new positions.
        """
        # Consider max position limit
        max_new_positions = settings.max_positions - len(self.positions)
        if max_new_positions <= 0:
            return 0.0

        # Available cash limited by max position percentage
        max_single_position = self.get_total_value() * settings.max_position_pct
        return min(self.cash, max_single_position * max_new_positions)

    def deposit(self, amount: float) -> None:
        """Add cash to portfolio.

        Args:
            amount: Amount to deposit in IDR.
        """
        self.cash += amount
        self.initial_capital += amount
        logger.info(f"Deposited {amount:,.0f} IDR")

    def withdraw(self, amount: float) -> bool:
        """Withdraw cash from portfolio.

        Args:
            amount: Amount to withdraw in IDR.

        Returns:
            True if successful, False if insufficient cash.
        """
        if amount > self.cash:
            logger.warning(f"Insufficient cash for withdrawal: {amount:,.0f} > {self.cash:,.0f}")
            return False

        self.cash -= amount
        self.initial_capital -= amount
        logger.info(f"Withdrew {amount:,.0f} IDR")
        return True

    def get_summary(self) -> str:
        """Get portfolio summary string.

        Returns:
            Formatted summary string.
        """
        state = self.get_state()

        lines = [
            "=" * 50,
            "PORTFOLIO SUMMARY",
            "=" * 50,
            f"Total Value: {state.total_value:,.0f} IDR",
            f"Cash: {state.cash:,.0f} IDR ({state.cash/state.total_value*100:.1f}%)",
            f"Positions: {state.open_positions} ({state.positions_value:,.0f} IDR)",
            "",
            f"Total P&L: {state.total_pnl:+,.0f} IDR ({state.total_pnl_pct:+.2f}%)",
            f"Daily P&L: {state.daily_pnl:+,.0f} IDR ({state.daily_pnl_pct:+.2f}%)",
            "",
            f"Peak Value: {state.peak_value:,.0f} IDR",
            f"Drawdown: {state.drawdown:,.0f} IDR ({state.drawdown_pct:.2f}%)",
            "",
        ]

        if self.positions:
            lines.append("OPEN POSITIONS:")
            for pos in self.positions.values():
                pnl_pct = pos.unrealized_pnl_pct
                lines.append(
                    f"  {pos.symbol}: {pos.quantity} shares @ {pos.current_price:,.0f} "
                    f"({pnl_pct:+.2f}%)"
                )

        lines.append("=" * 50)

        return "\n".join(lines)
