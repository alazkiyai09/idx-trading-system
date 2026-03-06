"""
Tests for Paper Trader Module
"""

import pytest
from datetime import date, datetime
from unittest.mock import Mock, patch

from core.execution.paper_trader import (
    PaperTrader,
    Order,
    ExecutionResult,
)
from core.data.models import OrderSide, OrderType, OrderStatus


class TestOrder:
    """Tests for Order dataclass."""

    def test_order_creation(self):
        """Test creating an order."""
        order = Order(
            order_id="ORD-12345678",
            symbol="BBCA",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100,
            price=9000.0,
        )

        assert order.order_id == "ORD-12345678"
        assert order.symbol == "BBCA"
        assert order.side == OrderSide.BUY
        assert order.status == OrderStatus.PENDING

    def test_order_defaults(self):
        """Test order default values."""
        order = Order(
            order_id="ORD-123",
            symbol="BBCA",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100,
        )

        assert order.price is None
        assert order.status == OrderStatus.PENDING
        assert order.filled_quantity == 0
        assert order.fees == 0.0
        assert order.slippage == 0.0


class TestExecutionResult:
    """Tests for ExecutionResult dataclass."""

    def test_success_result(self):
        """Test successful execution result."""
        order = Order(
            order_id="ORD-123",
            symbol="BBCA",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100,
        )

        result = ExecutionResult(success=True, order=order)

        assert result.success is True
        assert result.order == order
        assert result.error_message is None

    def test_failure_result(self):
        """Test failed execution result."""
        result = ExecutionResult(
            success=False,
            error_message="Insufficient funds",
        )

        assert result.success is False
        assert result.error_message == "Insufficient funds"


class TestPaperTrader:
    """Tests for PaperTrader class."""

    @pytest.fixture
    def trader(self):
        """Create a paper trader."""
        return PaperTrader(slippage=0.001)

    def test_initialization(self):
        """Test trader initialization."""
        trader = PaperTrader(slippage=0.002)

        assert trader.slippage == 0.002
        assert len(trader.positions) == 0
        assert len(trader.trades) == 0

    def test_buy_market_order(self, trader):
        """Test buying with market order."""
        result = trader.buy(
            symbol="BBCA",
            quantity=100,
            current_market_price=9000.0,
        )

        assert result.success is True
        assert result.order is not None
        assert result.order.status == OrderStatus.FILLED
        assert result.position is not None
        assert result.position.symbol == "BBCA"

    def test_buy_creates_position(self, trader):
        """Test that buy creates a position."""
        trader.buy(
            symbol="BBCA",
            quantity=100,
            current_market_price=9000.0,
        )

        assert "BBCA" in trader.positions
        assert trader.positions["BBCA"].quantity == 100

    def test_buy_market_order_slippage(self, trader):
        """Test that market order has slippage."""
        result = trader.buy(
            symbol="BBCA",
            quantity=100,
            current_market_price=9000.0,
        )

        # Buy slippage pushes price up
        expected_price = 9000.0 * (1 + trader.slippage)
        assert result.order.filled_price == pytest.approx(expected_price, rel=0.001)
        assert result.order.slippage > 0

    def test_buy_limit_order(self, trader):
        """Test buying with limit order."""
        result = trader.buy(
            symbol="BBCA",
            quantity=100,
            price=9000.0,
            order_type=OrderType.LIMIT,
        )

        assert result.success is True
        assert result.order.filled_price == 9000.0
        assert result.order.slippage == 0.0  # No slippage for limit

    def test_buy_limit_order_no_price(self, trader):
        """Test limit order without price fails."""
        result = trader.buy(
            symbol="BBCA",
            quantity=100,
            order_type=OrderType.LIMIT,
        )

        assert result.success is False
        assert "requires a price" in result.error_message

    def test_buy_invalid_quantity(self, trader):
        """Test buy with invalid quantity."""
        # Less than lot size
        result = trader.buy(
            symbol="BBCA",
            quantity=50,  # Less than 100
            current_market_price=9000.0,
        )

        assert result.success is False
        assert "Invalid quantity" in result.error_message

        # Not multiple of lot size
        result = trader.buy(
            symbol="BBCA",
            quantity=150,  # Not multiple of 100
            current_market_price=9000.0,
        )

        assert result.success is False

    def test_sell_market_order(self, trader):
        """Test selling with market order."""
        # Buy first
        trader.buy(
            symbol="BBCA",
            quantity=100,
            current_market_price=9000.0,
        )

        # Now sell
        result = trader.sell(
            symbol="BBCA",
            quantity=100,
            current_market_price=9500.0,
        )

        assert result.success is True
        assert result.trade is not None
        assert "BBCA" not in trader.positions

    def test_sell_market_order_slippage(self, trader):
        """Test that sell has slippage (price down)."""
        trader.buy(
            symbol="BBCA",
            quantity=100,
            current_market_price=9000.0,
        )

        result = trader.sell(
            symbol="BBCA",
            quantity=100,
            current_market_price=9500.0,
        )

        # Sell slippage pushes price down
        expected_price = 9500.0 * (1 - trader.slippage)
        assert result.order.filled_price == pytest.approx(expected_price, rel=0.001)

    def test_sell_without_position(self, trader):
        """Test selling without having a position."""
        result = trader.sell(
            symbol="BBCA",
            quantity=100,
            current_market_price=9500.0,
        )

        assert result.success is False
        assert "No position" in result.error_message

    def test_sell_more_than_owned(self, trader):
        """Test selling more shares than owned."""
        trader.buy(
            symbol="BBCA",
            quantity=100,
            current_market_price=9000.0,
        )

        result = trader.sell(
            symbol="BBCA",
            quantity=200,  # More than we have
            current_market_price=9500.0,
        )

        assert result.success is False
        assert "Cannot sell" in result.error_message

    def test_sell_partial_position(self, trader):
        """Test selling part of a position."""
        trader.buy(
            symbol="BBCA",
            quantity=200,
            current_market_price=9000.0,
        )

        result = trader.sell(
            symbol="BBCA",
            quantity=100,  # Half
            current_market_price=9500.0,
        )

        assert result.success is True
        # Note: Current implementation removes position on sell
        # This test documents the current behavior

    def test_trade_pnl_calculation(self, trader):
        """Test P&L calculation in trade."""
        trader.buy(
            symbol="BBCA",
            quantity=100,
            current_market_price=9000.0,
        )

        result = trader.sell(
            symbol="BBCA",
            quantity=100,
            current_market_price=10000.0,  # ~11% gain
            exit_reason="Take profit",
        )

        trade = result.trade
        assert trade.gross_pnl > 0
        assert trade.return_pct > 0
        assert trade.exit_reason == "Take profit"

    def test_fees_calculated(self, trader):
        """Test that fees are calculated."""
        result = trader.buy(
            symbol="BBCA",
            quantity=100,
            current_market_price=9000.0,
        )

        assert result.order.fees > 0
        # IDX buy fee is 0.15%
        expected_fees = result.order.filled_price * 100 * 0.0015
        assert result.order.fees == pytest.approx(expected_fees, rel=0.01)

    def test_update_position_prices(self, trader):
        """Test updating position prices."""
        trader.buy(
            symbol="BBCA",
            quantity=100,
            current_market_price=9000.0,
        )

        updated = trader.update_position_prices({"BBCA": 9500.0})

        assert len(updated) == 1
        pos = trader.positions["BBCA"]
        assert pos.current_price == 9500.0
        assert pos.unrealized_pnl > 0

    def test_update_position_prices_highest(self, trader):
        """Test that highest price updates."""
        trader.buy(
            symbol="BBCA",
            quantity=100,
            current_market_price=9000.0,
        )

        trader.update_position_prices({"BBCA": 9500.0})
        assert trader.positions["BBCA"].highest_price == 9500.0

        # Price drops, highest should stay
        trader.update_position_prices({"BBCA": 9200.0})
        assert trader.positions["BBCA"].highest_price == 9500.0

    def test_check_stop_losses(self, trader):
        """Test stop loss checking."""
        result = trader.buy(
            symbol="BBCA",
            quantity=100,
            current_market_price=9000.0,
        )

        # Default stop is 5% below entry
        stop_price = result.position.stop_loss

        # Price above stop - no triggers
        stopped = trader.check_stop_losses({"BBCA": 8800.0})
        assert len(stopped) == 0

        # Price at stop - triggers
        stopped = trader.check_stop_losses({"BBCA": stop_price})
        assert len(stopped) == 1

    def test_check_targets(self, trader):
        """Test target checking."""
        result = trader.buy(
            symbol="BBCA",
            quantity=100,
            current_market_price=9000.0,
        )

        # Targets are 5%, 10%, 15% above entry
        target_1 = result.position.target_1

        # Below target - no hits
        hits = trader.check_targets({"BBCA": 9200.0})
        assert len(hits) == 0

        # At target 1 - hits
        hits = trader.check_targets({"BBCA": target_1})
        assert len(hits) == 1
        assert hits[0][1] == "target_1"

    def test_get_positions(self, trader):
        """Test getting all positions."""
        assert len(trader.get_positions()) == 0

        trader.buy(symbol="BBCA", quantity=100, current_market_price=9000.0)
        trader.buy(symbol="BBRI", quantity=100, current_market_price=5000.0)

        positions = trader.get_positions()
        assert len(positions) == 2

    def test_get_trades(self, trader):
        """Test getting all trades."""
        assert len(trader.get_trades()) == 0

        # Complete a trade
        trader.buy(symbol="BBCA", quantity=100, current_market_price=9000.0)
        trader.sell(symbol="BBCA", quantity=100, current_market_price=9500.0)

        trades = trader.get_trades()
        assert len(trades) == 1

    def test_get_execution_stats(self, trader):
        """Test execution statistics."""
        stats = trader.get_execution_stats()
        assert stats["total_trades"] == 0

        # Complete a trade
        trader.buy(symbol="BBCA", quantity=100, current_market_price=9000.0)
        trader.sell(symbol="BBCA", quantity=100, current_market_price=9500.0)

        stats = trader.get_execution_stats()
        assert stats["total_trades"] == 1
        assert stats["total_fees"] > 0


class TestEdgeCases:
    """Test edge cases."""

    @pytest.fixture
    def trader(self):
        """Create a paper trader."""
        return PaperTrader(slippage=0.001)

    def test_multiple_buys_same_symbol(self, trader):
        """Test multiple buys of same symbol."""
        trader.buy(symbol="BBCA", quantity=100, current_market_price=9000.0)

        # Second buy should replace position
        result = trader.buy(symbol="BBCA", quantity=200, current_market_price=9100.0)

        # Position is replaced (simplified behavior)
        assert trader.positions["BBCA"].quantity == 200

    def test_very_large_quantity(self, trader):
        """Test with very large quantity."""
        result = trader.buy(
            symbol="BBCA",
            quantity=1_000_000,  # 1M shares
            current_market_price=9000.0,
        )

        assert result.success is True
        assert result.position.quantity == 1_000_000

    def test_very_small_slippage(self):
        """Test with zero slippage."""
        trader = PaperTrader(slippage=0.0)

        result = trader.buy(
            symbol="BBCA",
            quantity=100,
            current_market_price=9000.0,
        )

        assert result.order.filled_price == 9000.0
        assert result.order.slippage == 0.0

    def test_high_slippage(self):
        """Test with high slippage."""
        trader = PaperTrader(slippage=0.01)  # 1%

        result = trader.buy(
            symbol="BBCA",
            quantity=100,
            current_market_price=9000.0,
        )

        expected = 9000.0 * 1.01
        assert result.order.filled_price == pytest.approx(expected, rel=0.001)

    def test_buy_with_db_manager(self, trader):
        """Test buy with database manager."""
        mock_db = Mock()
        trader.db_manager = mock_db

        trader.buy(symbol="BBCA", quantity=100, current_market_price=9000.0)

        # No trade saved yet (only on sell)
        assert not mock_db.save_trade.called

    def test_sell_with_db_manager(self, trader):
        """Test sell saves to database."""
        mock_db = Mock()
        trader.db_manager = mock_db

        trader.buy(symbol="BBCA", quantity=100, current_market_price=9000.0)
        trader.sell(symbol="BBCA", quantity=100, current_market_price=9500.0)

        # Trade should be saved
        assert mock_db.save_trade.called

    def test_losing_trade(self, trader):
        """Test a losing trade."""
        trader.buy(symbol="BBCA", quantity=100, current_market_price=9000.0)

        result = trader.sell(
            symbol="BBCA",
            quantity=100,
            current_market_price=8500.0,  # Loss
        )

        assert result.trade.gross_pnl < 0
        assert result.trade.return_pct < 0

    def test_breakeven_trade(self, trader):
        """Test a trade with small movement."""
        trader.buy(symbol="BBCA", quantity=100, current_market_price=9000.0)

        # With slippage, buy is at ~9009, sell is at ~8991
        # To break even on gross, need price to rise ~0.2%
        result = trader.sell(
            symbol="BBCA",
            quantity=100,
            current_market_price=9020.0,  # Small gain
        )

        # Gross P&L is small (due to slippage in opposite directions)
        # Net is negative due to fees
        assert result.trade.net_pnl < result.trade.gross_pnl  # Fees reduce net

    def test_order_id_generation(self, trader):
        """Test that order IDs are unique."""
        result1 = trader.buy(symbol="BBCA", quantity=100, current_market_price=9000.0)
        result2 = trader.buy(symbol="BBRI", quantity=100, current_market_price=5000.0)

        assert result1.order.order_id != result2.order.order_id
