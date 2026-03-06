"""
Paper Trader Module

Simulates trading execution for backtesting and paper trading.
Handles order execution with realistic slippage and fees.
"""

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, date
from typing import Dict, List, Optional

from config.settings import settings
from config.constants import IDX_BUY_FEE, IDX_SELL_FEE, IDX_LOT_SIZE
from core.data.models import (
    OrderSide,
    OrderType,
    OrderStatus,
    Signal,
    Position,
    Trade,
    SetupType,
    FlowSignal,
)
from core.data.database import DatabaseManager

logger = logging.getLogger(__name__)


@dataclass
class Order:
    """Represents a trading order.

    Attributes:
        order_id: Unique order identifier.
        symbol: Stock symbol.
        side: Buy or sell.
        order_type: Market or limit.
        quantity: Number of shares.
        price: Limit price (for limit orders).
        status: Order status.
        created_at: Order creation time.
        filled_at: Fill time (if filled).
        filled_price: Actual fill price.
        filled_quantity: Quantity filled.
        fees: Trading fees.
        slippage: Execution slippage.
    """

    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: int
    price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    created_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None
    filled_price: Optional[float] = None
    filled_quantity: int = 0
    fees: float = 0.0
    slippage: float = 0.0


@dataclass
class ExecutionResult:
    """Result of order execution.

    Attributes:
        success: Whether execution was successful.
        order: The executed order.
        position: Created/updated position (if any).
        trade: Completed trade (for sells).
        error_message: Error message if failed.
    """

    success: bool
    order: Optional[Order] = None
    position: Optional[Position] = None
    trade: Optional[Trade] = None
    error_message: Optional[str] = None


class PaperTrader:
    """Simulates trading execution.

    Handles order submission, execution simulation, and
    trade tracking for paper trading and backtesting.

    Features:
    - Simulates market and limit orders
    - Applies realistic slippage (0.1% for market orders)
    - Calculates IDX fees (0.15% buy, 0.25% sell)
    - Tracks execution quality
    - Records all trades to database

    Example:
        trader = PaperTrader(db_manager)
        result = trader.buy(symbol="BBCA", quantity=100, price=9000)
        if result.success:
            print(f"Bought at {result.order.filled_price}")
    """

    # Default slippage for market orders
    DEFAULT_SLIPPAGE = 0.001  # 0.1%

    def __init__(
        self,
        db_manager: Optional[DatabaseManager] = None,
        slippage: float = DEFAULT_SLIPPAGE,
    ) -> None:
        """Initialize paper trader.

        Args:
            db_manager: Database manager for trade storage.
            slippage: Default slippage rate for market orders.
        """
        self.db_manager = db_manager
        self.slippage = slippage

        # Track open orders and positions
        self.open_orders: Dict[str, Order] = {}
        self.positions: Dict[str, Position] = {}

        # Trade history
        self.trades: List[Trade] = []

    def buy(
        self,
        symbol: str,
        quantity: int,
        price: Optional[float] = None,
        order_type: OrderType = OrderType.MARKET,
        current_market_price: Optional[float] = None,
    ) -> ExecutionResult:
        """Execute a buy order.

        Args:
            symbol: Stock symbol.
            quantity: Number of shares to buy.
            price: Limit price (required for limit orders).
            order_type: Market or limit order.
            current_market_price: Current market price (for simulation).

        Returns:
            ExecutionResult with order and position details.
        """
        # Validate quantity
        if quantity < IDX_LOT_SIZE or quantity % IDX_LOT_SIZE != 0:
            return ExecutionResult(
                success=False,
                error_message=f"Invalid quantity {quantity}. Must be multiple of {IDX_LOT_SIZE}",
            )

        # Create order
        order = Order(
            order_id=self._generate_order_id(),
            symbol=symbol,
            side=OrderSide.BUY,
            order_type=order_type,
            quantity=quantity,
            price=price,
            created_at=datetime.now(),
        )

        # Execute based on order type
        if order_type == OrderType.MARKET:
            if current_market_price is None:
                current_market_price = price
            result = self._execute_market_order(order, current_market_price)
        else:
            result = self._handle_limit_order(order)

        return result

    def sell(
        self,
        symbol: str,
        quantity: int,
        price: Optional[float] = None,
        order_type: OrderType = OrderType.MARKET,
        current_market_price: Optional[float] = None,
        exit_reason: str = "manual",
    ) -> ExecutionResult:
        """Execute a sell order.

        Args:
            symbol: Stock symbol.
            quantity: Number of shares to sell.
            price: Limit price (required for limit orders).
            order_type: Market or limit order.
            current_market_price: Current market price.
            exit_reason: Reason for exit.

        Returns:
            ExecutionResult with order and trade details.
        """
        # Check if we have the position
        if symbol not in self.positions:
            return ExecutionResult(
                success=False,
                error_message=f"No position in {symbol} to sell",
            )

        position = self.positions[symbol]
        if quantity > position.quantity:
            return ExecutionResult(
                success=False,
                error_message=f"Cannot sell {quantity} shares, only have {position.quantity}",
            )

        # Create order
        order = Order(
            order_id=self._generate_order_id(),
            symbol=symbol,
            side=OrderSide.SELL,
            order_type=order_type,
            quantity=quantity,
            price=price,
            created_at=datetime.now(),
        )

        # Execute based on order type
        if order_type == OrderType.MARKET:
            if current_market_price is None:
                current_market_price = price
            result = self._execute_market_order(order, current_market_price)
        else:
            result = self._handle_limit_order(order)

        # If successful, close position and create trade
        if result.success and result.order:
            result = self._close_position(
                position=position,
                order=result.order,
                exit_reason=exit_reason,
            )

        return result

    def _execute_market_order(
        self,
        order: Order,
        market_price: float,
    ) -> ExecutionResult:
        """Execute a market order with slippage.

        Args:
            order: Order to execute.
            market_price: Current market price.

        Returns:
            ExecutionResult with execution details.
        """
        # Calculate slippage
        if order.side == OrderSide.BUY:
            # Buy slippage pushes price up
            fill_price = market_price * (1 + self.slippage)
        else:
            # Sell slippage pushes price down
            fill_price = market_price * (1 - self.slippage)

        # Calculate fees
        fee_rate = IDX_BUY_FEE if order.side == OrderSide.BUY else IDX_SELL_FEE
        fees = fill_price * order.quantity * fee_rate

        # Update order
        order.status = OrderStatus.FILLED
        order.filled_at = datetime.now()
        order.filled_price = fill_price
        order.filled_quantity = order.quantity
        order.fees = fees
        order.slippage = abs(fill_price - market_price) / market_price

        logger.info(
            f"Executed {order.side.value} {order.quantity} {order.symbol} @ {fill_price:.0f} "
            f"(slippage: {order.slippage:.3%}, fees: {fees:,.0f})"
        )

        # Create position if buy
        if order.side == OrderSide.BUY:
            position = self._create_position(order)
            return ExecutionResult(
                success=True,
                order=order,
                position=position,
            )

        return ExecutionResult(success=True, order=order)

    def _handle_limit_order(self, order: Order) -> ExecutionResult:
        """Handle a limit order (store for later execution).

        For paper trading, we simulate immediate fill at limit price.

        Args:
            order: Limit order to handle.

        Returns:
            ExecutionResult with order details.
        """
        if order.price is None:
            return ExecutionResult(
                success=False,
                error_message="Limit order requires a price",
            )

        # For simplicity in paper trading, fill at limit price
        fee_rate = IDX_BUY_FEE if order.side == OrderSide.BUY else IDX_SELL_FEE
        fees = order.price * order.quantity * fee_rate

        order.status = OrderStatus.FILLED
        order.filled_at = datetime.now()
        order.filled_price = order.price
        order.filled_quantity = order.quantity
        order.fees = fees
        order.slippage = 0.0  # No slippage for limit orders

        logger.info(
            f"Executed LIMIT {order.side.value} {order.quantity} {order.symbol} @ {order.price:.0f}"
        )

        # Create position if buy
        if order.side == OrderSide.BUY:
            position = self._create_position(order)
            return ExecutionResult(
                success=True,
                order=order,
                position=position,
            )

        return ExecutionResult(success=True, order=order)

    def _create_position(self, order: Order) -> Position:
        """Create a new position from a filled buy order.

        Args:
            order: Filled buy order.

        Returns:
            New Position instance.
        """
        position = Position(
            position_id=f"POS-{order.order_id}",
            symbol=order.symbol,
            entry_date=order.filled_at.date() if order.filled_at else date.today(),
            entry_price=order.filled_price,
            quantity=order.filled_quantity,
            current_price=order.filled_price,
            unrealized_pnl=0.0,
            unrealized_pnl_pct=0.0,
            stop_loss=order.filled_price * 0.95,  # Default 5% stop
            target_1=order.filled_price * 1.05,
            target_2=order.filled_price * 1.10,
            target_3=order.filled_price * 1.15,
            highest_price=order.filled_price,
            days_held=0,
            setup_type=SetupType.MOMENTUM,  # Default
            signal_score=50.0,  # Default
        )

        self.positions[order.symbol] = position
        logger.info(f"Created position: {position.position_id}")

        return position

    def _close_position(
        self,
        position: Position,
        order: Order,
        exit_reason: str,
    ) -> ExecutionResult:
        """Close a position and create a trade record.

        Args:
            position: Position to close.
            order: Sell order.
            exit_reason: Reason for exit.

        Returns:
            ExecutionResult with trade details.
        """
        # Calculate P&L
        gross_pnl = (order.filled_price - position.entry_price) * order.filled_quantity
        total_fees = order.fees + (position.entry_price * position.quantity * IDX_BUY_FEE)
        net_pnl = gross_pnl - total_fees
        return_pct = net_pnl / (position.entry_price * position.quantity) * 100

        # Create trade record
        trade = Trade(
            trade_id=f"TRD-{order.order_id}",
            symbol=position.symbol,
            entry_date=position.entry_date,
            entry_price=position.entry_price,
            exit_date=order.filled_at.date() if order.filled_at else date.today(),
            exit_price=order.filled_price,
            exit_reason=exit_reason,
            quantity=order.filled_quantity,
            side=OrderSide.BUY,  # Original side
            gross_pnl=gross_pnl,
            fees=total_fees,
            net_pnl=net_pnl,
            return_pct=return_pct,
            holding_days=(date.today() - position.entry_date).days,
            max_favorable=0.0,  # Would need to track
            max_adverse=0.0,
            signal_score=position.signal_score,
            setup_type=position.setup_type,
            rsi_at_entry=50.0,  # Would need actual
            flow_signal=FlowSignal.NEUTRAL,
            flow_consecutive_days=0,
        )

        # Remove position
        del self.positions[position.symbol]

        # Store trade
        self.trades.append(trade)
        if self.db_manager:
            self._save_trade_to_db(trade)

        logger.info(
            f"Closed position: {position.symbol} @ {order.filled_price:.0f}, "
            f"P&L: {net_pnl:,.0f} ({return_pct:.2f}%)"
        )

        return ExecutionResult(
            success=True,
            order=order,
            trade=trade,
        )

    def update_position_prices(
        self,
        prices: Dict[str, float],
    ) -> List[Position]:
        """Update current prices for all positions.

        Args:
            prices: Dictionary of symbol to current price.

        Returns:
            List of updated positions.
        """
        updated = []

        for symbol, position in self.positions.items():
            if symbol in prices:
                current_price = prices[symbol]

                # Update highest price
                if current_price > position.highest_price:
                    position.highest_price = current_price

                position.current_price = current_price

                # Calculate unrealized P&L
                position.unrealized_pnl = (
                    current_price - position.entry_price
                ) * position.quantity
                position.unrealized_pnl_pct = (
                    current_price - position.entry_price
                ) / position.entry_price * 100

                updated.append(position)

        return updated

    def check_stop_losses(
        self,
        prices: Dict[str, float],
    ) -> List[tuple]:
        """Check if any positions hit their stop loss.

        Args:
            prices: Dictionary of symbol to current price.

        Returns:
            List of (position, current_price) tuples for stopped positions.
        """
        stopped = []

        for symbol, position in self.positions.items():
            current_price = prices.get(symbol, position.current_price)

            if current_price <= position.stop_loss:
                logger.warning(
                    f"Stop loss triggered for {symbol}: "
                    f"{current_price:.0f} <= {position.stop_loss:.0f}"
                )
                stopped.append((position, current_price))

        return stopped

    def check_targets(
        self,
        prices: Dict[str, float],
    ) -> List[tuple]:
        """Check if any positions hit their targets.

        Args:
            prices: Dictionary of symbol to current price.

        Returns:
            List of (position, target_level, target_price) tuples.
        """
        targets_hit = []

        for symbol, position in self.positions.items():
            current_price = prices.get(symbol, position.current_price)

            # Check targets in order
            targets = [
                ("target_3", position.target_3),
                ("target_2", position.target_2),
                ("target_1", position.target_1),
            ]

            for target_name, target_price in targets:
                if target_price and current_price >= target_price:
                    targets_hit.append((position, target_name, target_price))
                    break  # Only hit highest target

        return targets_hit

    def _generate_order_id(self) -> str:
        """Generate unique order ID.

        Returns:
            Unique order ID string.
        """
        return f"ORD-{uuid.uuid4().hex[:8].upper()}"

    def _save_trade_to_db(self, trade: Trade) -> None:
        """Save trade to database.

        Args:
            trade: Trade to save.
        """
        if not self.db_manager:
            return

        trade_dict = {
            "trade_id": trade.trade_id,
            "symbol": trade.symbol,
            "entry_date": trade.entry_date,
            "entry_price": trade.entry_price,
            "exit_date": trade.exit_date,
            "exit_price": trade.exit_price,
            "exit_reason": trade.exit_reason,
            "quantity": trade.quantity,
            "side": trade.side.value,
            "gross_pnl": trade.gross_pnl,
            "fees": trade.fees,
            "net_pnl": trade.net_pnl,
            "return_pct": trade.return_pct,
            "holding_days": trade.holding_days,
            "max_favorable": trade.max_favorable,
            "max_adverse": trade.max_adverse,
            "signal_score": trade.signal_score,
            "setup_type": trade.setup_type.value,
            "rsi_at_entry": trade.rsi_at_entry,
            "flow_signal": trade.flow_signal.value,
            "flow_consecutive_days": trade.flow_consecutive_days,
        }

        try:
            self.db_manager.save_trade(trade_dict)
        except Exception as e:
            logger.error(f"Failed to save trade to database: {e}")

    def get_positions(self) -> List[Position]:
        """Get all open positions.

        Returns:
            List of open positions.
        """
        return list(self.positions.values())

    def get_trades(self) -> List[Trade]:
        """Get all completed trades.

        Returns:
            List of completed trades.
        """
        return self.trades.copy()

    def get_execution_stats(self) -> dict:
        """Get execution statistics.

        Returns:
            Dictionary with execution statistics.
        """
        if not self.trades:
            return {
                "total_trades": 0,
                "avg_slippage": 0.0,
                "total_fees": 0.0,
            }

        return {
            "total_trades": len(self.trades),
            "avg_slippage": sum(t.slippage or 0 for t in self.trades) / len(self.trades),
            "total_fees": sum(t.fees for t in self.trades),
        }
