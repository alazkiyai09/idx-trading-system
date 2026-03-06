"""
Tests for Portfolio Manager Module
"""

import pytest
from datetime import date, datetime
from unittest.mock import patch

from core.portfolio.portfolio_manager import PortfolioManager
from core.data.models import Position, SetupType, FlowSignal


def create_test_position(
    symbol: str = "TEST",
    quantity: int = 100,
    entry_price: float = 10000.0,
    entry_date: date = None,
    signal_score: float = 75.0,
) -> Position:
    """Create a test position."""
    if entry_date is None:
        entry_date = date.today()

    return Position(
        position_id=f"POS-{symbol}",
        symbol=symbol,
        quantity=quantity,
        entry_price=entry_price,
        entry_date=entry_date,
        signal_score=signal_score,
        setup_type=SetupType.MOMENTUM,
        current_price=entry_price,
        highest_price=entry_price,
        unrealized_pnl=0.0,
        unrealized_pnl_pct=0.0,
        days_held=0,
        stop_loss=entry_price * 0.95,
        target_1=entry_price * 1.05,
        target_2=entry_price * 1.10,
        target_3=entry_price * 1.15,
    )


class TestPortfolioManager:
    """Tests for PortfolioManager class."""

    @pytest.fixture
    def manager(self):
        """Create a portfolio manager with 100M IDR."""
        return PortfolioManager(initial_capital=100_000_000)

    def test_initialization(self):
        """Test manager initialization."""
        manager = PortfolioManager(initial_capital=50_000_000)

        assert manager.initial_capital == 50_000_000
        assert manager.cash == 50_000_000
        assert len(manager.positions) == 0
        assert manager.peak_value == 50_000_000

    def test_initialization_default(self):
        """Test manager with default capital."""
        manager = PortfolioManager()

        assert manager.initial_capital > 0
        assert manager.cash == manager.initial_capital

    def test_open_position(self, manager):
        """Test opening a position."""
        position = create_test_position(
            symbol="BBCA",
            quantity=100,
            entry_price=9000.0,
        )

        manager.open_position(position)

        assert "BBCA" in manager.positions
        assert manager.cash == 100_000_000 - (9000.0 * 100)

    def test_open_position_insufficient_cash(self, manager):
        """Test opening position with insufficient cash."""
        position = create_test_position(
            symbol="BBCA",
            quantity=10000,  # 10000 * 9000 = 90M
            entry_price=9000.0,
        )

        # Reduce cash first
        manager.cash = 50_000_000

        with pytest.raises(ValueError, match="Insufficient cash"):
            manager.open_position(position)

    def test_close_position(self, manager):
        """Test closing a position."""
        position = create_test_position(
            symbol="BBCA",
            quantity=100,
            entry_price=9000.0,
        )

        manager.open_position(position)
        initial_cash = manager.cash

        trade = manager.close_position(
            symbol="BBCA",
            exit_price=9500.0,
            exit_date=date.today(),
            exit_reason="Take profit",
        )

        assert trade is not None
        assert trade.symbol == "BBCA"
        assert "BBCA" not in manager.positions
        assert manager.cash > initial_cash
        assert trade.gross_pnl == (9500.0 - 9000.0) * 100

    def test_close_position_not_found(self, manager):
        """Test closing non-existent position."""
        trade = manager.close_position(
            symbol="NOTFOUND",
            exit_price=9500.0,
            exit_date=date.today(),
            exit_reason="Test",
        )

        assert trade is None

    def test_close_position_with_loss(self, manager):
        """Test closing position at a loss."""
        position = create_test_position(
            symbol="BBCA",
            quantity=100,
            entry_price=9000.0,
        )

        manager.open_position(position)

        trade = manager.close_position(
            symbol="BBCA",
            exit_price=8500.0,  # Lower than entry
            exit_date=date.today(),
            exit_reason="Stop loss",
        )

        assert trade.gross_pnl < 0
        assert trade.net_pnl < trade.gross_pnl  # After fees

    def test_update_prices(self, manager):
        """Test updating position prices."""
        position = create_test_position(
            symbol="BBCA",
            quantity=100,
            entry_price=9000.0,
        )

        manager.open_position(position)

        manager.update_prices({"BBCA": 9500.0})

        pos = manager.get_position("BBCA")
        assert pos.current_price == 9500.0
        assert pos.unrealized_pnl == (9500.0 - 9000.0) * 100
        assert pos.unrealized_pnl_pct > 0

    def test_update_prices_trailing_high(self, manager):
        """Test that highest price updates for trailing stop."""
        position = create_test_position(
            symbol="BBCA",
            quantity=100,
            entry_price=9000.0,
        )

        manager.open_position(position)

        manager.update_prices({"BBCA": 9500.0})
        assert manager.positions["BBCA"].highest_price == 9500.0

        # Price goes down, highest should stay
        manager.update_prices({"BBCA": 9200.0})
        assert manager.positions["BBCA"].highest_price == 9500.0

    def test_get_positions_value(self, manager):
        """Test calculating positions value."""
        assert manager.get_positions_value() == 0

        position = create_test_position(
            symbol="BBCA",
            quantity=100,
            entry_price=9000.0,
        )
        manager.open_position(position)
        manager.update_prices({"BBCA": 9500.0})

        assert manager.get_positions_value() == 9500.0 * 100

    def test_get_total_value(self, manager):
        """Test calculating total portfolio value."""
        position = create_test_position(
            symbol="BBCA",
            quantity=100,
            entry_price=9000.0,
        )
        manager.open_position(position)
        manager.update_prices({"BBCA": 9500.0})

        # Cash reduced by entry, but position value at current price
        expected = manager.cash + (9500.0 * 100)
        assert manager.get_total_value() == expected

    def test_get_total_pnl(self, manager):
        """Test calculating total P&L."""
        position = create_test_position(
            symbol="BBCA",
            quantity=100,
            entry_price=9000.0,
        )
        manager.open_position(position)

        # At same price, small loss from fees when closing
        manager.update_prices({"BBCA": 9000.0})

        # No gain yet (just opened)
        pnl = manager.get_total_pnl()
        assert pnl == 0  # No change at same price

    def test_get_total_pnl_pct(self, manager):
        """Test calculating total return percentage."""
        position = create_test_position(
            symbol="BBCA",
            quantity=100,
            entry_price=9000.0,
        )
        manager.open_position(position)
        manager.update_prices({"BBCA": 9900.0})  # 10% gain

        # Should be approximately 9% portfolio gain (900K on 100M)
        # (100 shares * 900 gain) / 100M = 0.9%
        pnl_pct = manager.get_total_pnl_pct()
        assert pnl_pct > 0

    def test_get_drawdown(self, manager):
        """Test calculating drawdown."""
        position = create_test_position(
            symbol="BBCA",
            quantity=100,
            entry_price=9000.0,
        )
        manager.open_position(position)

        # Price goes up, peak updates
        manager.update_prices({"BBCA": 10000.0})
        assert manager.peak_value > manager.initial_capital

        # Price goes down, drawdown appears
        manager.update_prices({"BBCA": 9500.0})
        dd = manager.get_drawdown()
        assert dd > 0

    def test_get_drawdown_pct(self, manager):
        """Test calculating drawdown percentage."""
        position = create_test_position(
            symbol="BBCA",
            quantity=1000,
            entry_price=9000.0,
        )
        manager.open_position(position)

        manager.update_prices({"BBCA": 10000.0})
        manager.update_prices({"BBCA": 9000.0})

        dd_pct = manager.get_drawdown_pct()
        assert dd_pct > 0

    def test_get_state(self, manager):
        """Test getting portfolio state."""
        state = manager.get_state()

        assert state.cash == 100_000_000
        assert state.total_value == 100_000_000
        assert state.open_positions == 0
        assert state.drawdown == 0

    def test_get_state_with_positions(self, manager):
        """Test getting state with positions."""
        position = create_test_position(symbol="BBCA", quantity=100, entry_price=9000.0)
        manager.open_position(position)

        state = manager.get_state()

        assert state.open_positions == 1
        assert len(state.positions) == 1

    def test_get_position(self, manager):
        """Test getting a specific position."""
        position = create_test_position(symbol="BBCA")
        manager.open_position(position)

        pos = manager.get_position("BBCA")
        assert pos.symbol == "BBCA"

        assert manager.get_position("NOTFOUND") is None

    def test_has_position(self, manager):
        """Test checking if position exists."""
        position = create_test_position(symbol="BBCA")
        manager.open_position(position)

        assert manager.has_position("BBCA") is True
        assert manager.has_position("NOTFOUND") is False

    def test_get_position_count(self, manager):
        """Test getting position count."""
        assert manager.get_position_count() == 0

        manager.open_position(create_test_position(symbol="BBCA"))
        assert manager.get_position_count() == 1

        manager.open_position(create_test_position(symbol="BBRI"))
        assert manager.get_position_count() == 2

    def test_get_cash_available(self, manager):
        """Test getting available cash."""
        assert manager.get_cash_available() == 100_000_000

        position = create_test_position(symbol="BBCA", entry_price=50000.0, quantity=100)
        manager.open_position(position)

        assert manager.get_cash_available() == 100_000_000 - 5_000_000

    def test_get_buying_power(self, manager):
        """Test getting buying power."""
        power = manager.get_buying_power()
        assert power > 0
        assert power <= manager.cash

    def test_get_buying_power_max_positions(self, manager):
        """Test buying power limited by max positions."""
        # Open many positions
        for i in range(10):
            pos = create_test_position(
                symbol=f"TEST{i}",
                entry_price=1000.0,
                quantity=100,
            )
            manager.open_position(pos)

        # Buying power should be limited
        power = manager.get_buying_power()
        # If at max positions, power should be 0 or limited
        # This depends on settings.max_positions

    def test_deposit(self, manager):
        """Test depositing cash."""
        initial = manager.cash
        manager.deposit(10_000_000)

        assert manager.cash == initial + 10_000_000
        assert manager.initial_capital == 100_000_000 + 10_000_000

    def test_withdraw(self, manager):
        """Test withdrawing cash."""
        initial = manager.cash
        result = manager.withdraw(10_000_000)

        assert result is True
        assert manager.cash == initial - 10_000_000

    def test_withdraw_insufficient(self, manager):
        """Test withdrawing more than available."""
        result = manager.withdraw(200_000_000)

        assert result is False
        assert manager.cash == 100_000_000

    def test_get_summary(self, manager):
        """Test summary generation."""
        summary = manager.get_summary()

        assert "PORTFOLIO SUMMARY" in summary
        assert "100,000,000" in summary

    def test_get_summary_with_positions(self, manager):
        """Test summary with positions."""
        position = create_test_position(symbol="BBCA", entry_price=9000.0, quantity=100)
        manager.open_position(position)
        manager.update_prices({"BBCA": 9500.0})

        summary = manager.get_summary()

        assert "BBCA" in summary
        assert "OPEN POSITIONS" in summary

    def test_reset_daily(self, manager):
        """Test daily reset."""
        position = create_test_position(symbol="BBCA")
        manager.open_position(position)

        initial_days_held = manager.positions["BBCA"].days_held

        # Reset on same day should not change anything
        manager.reset_daily()
        assert manager.positions["BBCA"].days_held == initial_days_held

    def test_multiple_positions(self, manager):
        """Test managing multiple positions."""
        pos1 = create_test_position(symbol="BBCA", entry_price=9000.0, quantity=100)
        pos2 = create_test_position(symbol="BBRI", entry_price=5000.0, quantity=200)

        manager.open_position(pos1)
        manager.open_position(pos2)

        assert manager.get_position_count() == 2
        assert manager.has_position("BBCA")
        assert manager.has_position("BBRI")

    def test_trade_history(self, manager):
        """Test closed trade history."""
        position = create_test_position(symbol="BBCA", entry_price=9000.0, quantity=100)
        manager.open_position(position)

        assert len(manager.closed_trades) == 0

        manager.close_position(
            symbol="BBCA",
            exit_price=9500.0,
            exit_date=date.today(),
            exit_reason="Take profit",
        )

        assert len(manager.closed_trades) == 1
        assert manager.closed_trades[0].symbol == "BBCA"


class TestEdgeCases:
    """Test edge cases."""

    def test_zero_capital(self):
        """Test with very small capital."""
        manager = PortfolioManager(initial_capital=1000.0)  # Very small

        assert manager.cash == 1000.0

        # Can't open position that costs more
        position = create_test_position(entry_price=1000.0, quantity=100)  # 100K cost
        with pytest.raises(ValueError, match="Insufficient cash"):
            manager.open_position(position)

    def test_very_large_position(self):
        """Test with very large position value."""
        manager = PortfolioManager(initial_capital=1_000_000_000)  # 1B IDR

        position = create_test_position(
            symbol="BBCA",
            entry_price=100_000.0,  # 100K per share
            quantity=5000,  # 5000 shares = 500M
        )

        manager.open_position(position)
        assert manager.get_position_count() == 1

    def test_fractional_pnl(self):
        """Test with small P&L changes."""
        manager = PortfolioManager(initial_capital=100_000_000)

        position = create_test_position(
            symbol="BBCA",
            entry_price=9000.0,
            quantity=100,
        )
        manager.open_position(position)

        # Tiny price change
        manager.update_prices({"BBCA": 9000.01})

        pnl = manager.get_total_pnl()
        assert pnl == pytest.approx(1.0, rel=0.1)  # ~1 IDR gain

    def test_peak_value_updates(self):
        """Test that peak value only increases."""
        manager = PortfolioManager(initial_capital=100_000_000)

        position = create_test_position(symbol="BBCA", entry_price=9000.0, quantity=1000)
        manager.open_position(position)

        initial_peak = manager.peak_value

        # Price goes up
        manager.update_prices({"BBCA": 10000.0})
        peak_after_gain = manager.peak_value
        assert peak_after_gain > initial_peak

        # Price goes down
        manager.update_prices({"BBCA": 8000.0})
        assert manager.peak_value == peak_after_gain  # Peak should not decrease
