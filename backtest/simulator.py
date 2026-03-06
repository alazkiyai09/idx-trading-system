"""
IDX Market Simulator Module

Simulates IDX market conditions for realistic backtesting including:
- IDX fees (0.15% buy, 0.25% sell)
- ARA/ARB limits (±25% daily price limits)
- Slippage modeling
- Lot size rules (100 shares)
- Tick size rules
"""

import logging
from dataclasses import dataclass
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple
from enum import Enum

import numpy as np

from config.constants import IDX_LOT_SIZE, IDX_BUY_FEE, IDX_SELL_FEE
from config.settings import settings
from core.data.models import Bar, OrderSide, OrderType

logger = logging.getLogger(__name__)


@dataclass
class SimulationConfig:
    """Configuration for market simulation.

    Attributes:
        buy_fee_pct: Buy fee percentage.
        sell_fee_pct: Sell fee percentage (includes tax).
        slippage_market: Slippage for market orders.
        slippage_limit: Slippage for limit orders.
        ara_limit_pct: ARA (Auto Reject Atas) upper limit.
        arb_limit_pct: ARB (Auto Reject Bawah) lower limit.
        lot_size: Minimum lot size.
        tick_sizes: Tick size rules by price range.
    """

    buy_fee_pct: float = IDX_BUY_FEE  # 0.15%
    sell_fee_pct: float = IDX_SELL_FEE  # 0.25%
    slippage_market: float = 0.001  # 0.1%
    slippage_limit: float = 0.0  # No slippage for limit
    ara_limit_pct: float = 0.25  # +25%
    arb_limit_pct: float = 0.25  # -25%
    lot_size: int = IDX_LOT_SIZE  # 100 shares
    tick_sizes: Dict[Tuple[float, float], int] = None

    def __post_init__(self):
        if self.tick_sizes is None:
            self.tick_sizes = {
                (0, 200): 1,
                (200, 500): 2,
                (500, 2000): 5,
                (2000, 5000): 10,
                (5000, 10000): 25,
                (10000, float('inf')): 50,
            }


@dataclass
class ExecutionResult:
    """Result of simulated execution.

    Attributes:
        success: Whether execution was successful.
        filled_price: Actual fill price.
        filled_quantity: Quantity filled.
        fees: Total fees.
        slippage: Applied slippage.
        message: Status message.
    """

    success: bool
    filled_price: float = 0.0
    filled_quantity: int = 0
    fees: float = 0.0
    slippage: float = 0.0
    message: str = ""


class IDXSimulator:
    """Simulates IDX market conditions.

    Provides realistic market simulation for backtesting including
    fees, price limits, slippage, and lot/tick size rules.

    Example:
        simulator = IDXSimulator()
        result = simulator.execute_buy(
            symbol="BBCA",
            quantity=100,
            price=9000.0,
            order_type=OrderType.MARKET,
            reference_price=9000.0,
        )
    """

    def __init__(self, config: Optional[SimulationConfig] = None) -> None:
        """Initialize simulator.

        Args:
            config: Simulation configuration.
        """
        self.config = config or SimulationConfig()
        self.random_state = np.random.RandomState(42)

    def execute_buy(
        self,
        symbol: str,
        quantity: int,
        price: Optional[float],
        order_type: OrderType,
        reference_price: float,
        day_high: Optional[float] = None,
        day_low: Optional[float] = None,
    ) -> ExecutionResult:
        """Simulate buy order execution.

        Args:
            symbol: Stock symbol.
            quantity: Number of shares.
            price: Limit price (for limit orders).
            order_type: Market or limit.
            reference_price: Reference price (previous close or current).
            day_high: Today's high (for limit checks).
            day_low: Today's low (for limit checks).

        Returns:
            ExecutionResult with execution details.
        """
        # Validate quantity
        if quantity < self.config.lot_size or quantity % self.config.lot_size != 0:
            return ExecutionResult(
                success=False,
                message=f"Invalid quantity {quantity}. Must be multiple of {self.config.lot_size}",
            )

        # Calculate execution price
        if order_type == OrderType.MARKET:
            # Market order uses reference price with slippage
            slippage = self._calculate_slippage(is_buy=True)
            fill_price = reference_price * (1 + slippage)
        else:
            # Limit order
            if price is None:
                return ExecutionResult(
                    success=False,
                    message="Limit order requires price",
                )
            fill_price = price
            slippage = 0.0

        # Round to tick size
        fill_price = self._round_to_tick(fill_price)

        # Check price limits (ARA/ARB)
        if day_high is not None and fill_price > day_high:
            return ExecutionResult(
                success=False,
                message=f"Fill price {fill_price:.0f} exceeds day high {day_high:.0f}",
            )

        # Calculate fees
        gross_value = fill_price * quantity
        fees = gross_value * self.config.buy_fee_pct

        return ExecutionResult(
            success=True,
            filled_price=fill_price,
            filled_quantity=quantity,
            fees=fees,
            slippage=slippage,
            message=f"Filled {quantity} @ {fill_price:.0f}",
        )

    def execute_sell(
        self,
        symbol: str,
        quantity: int,
        price: Optional[float],
        order_type: OrderType,
        reference_price: float,
        day_high: Optional[float] = None,
        day_low: Optional[float] = None,
    ) -> ExecutionResult:
        """Simulate sell order execution.

        Args:
            symbol: Stock symbol.
            quantity: Number of shares.
            price: Limit price (for limit orders).
            order_type: Market or limit.
            reference_price: Reference price.
            day_high: Today's high.
            day_low: Today's low.

        Returns:
            ExecutionResult with execution details.
        """
        # Validate quantity
        if quantity < self.config.lot_size or quantity % self.config.lot_size != 0:
            return ExecutionResult(
                success=False,
                message=f"Invalid quantity {quantity}. Must be multiple of {self.config.lot_size}",
            )

        # Calculate execution price
        if order_type == OrderType.MARKET:
            slippage = self._calculate_slippage(is_buy=False)
            fill_price = reference_price * (1 - slippage)  # Slippage against seller
        else:
            if price is None:
                return ExecutionResult(
                    success=False,
                    message="Limit order requires price",
                )
            fill_price = price
            slippage = 0.0

        # Round to tick size
        fill_price = self._round_to_tick(fill_price)

        # Check price limits
        if day_low is not None and fill_price < day_low:
            return ExecutionResult(
                success=False,
                message=f"Fill price {fill_price:.0f} below day low {day_low:.0f}",
            )

        # Calculate fees
        gross_value = fill_price * quantity
        fees = gross_value * self.config.sell_fee_pct

        return ExecutionResult(
            success=True,
            filled_price=fill_price,
            filled_quantity=quantity,
            fees=fees,
            slippage=slippage,
            message=f"Filled {quantity} @ {fill_price:.0f}",
        )

    def calculate_total_cost(
        self,
        price: float,
        quantity: int,
        side: OrderSide,
    ) -> Tuple[float, float, float]:
        """Calculate total cost including fees.

        Args:
            price: Execution price.
            quantity: Number of shares.
            side: Buy or sell.

        Returns:
            Tuple of (gross_value, fees, net_value).
        """
        gross_value = price * quantity
        fee_rate = self.config.buy_fee_pct if side == OrderSide.BUY else self.config.sell_fee_pct
        fees = gross_value * fee_rate

        if side == OrderSide.BUY:
            net_value = gross_value + fees
        else:
            net_value = gross_value - fees

        return gross_value, fees, net_value

    def check_price_limits(
        self,
        price: float,
        prev_close: float,
    ) -> Tuple[bool, str]:
        """Check if price is within ARA/ARB limits.

        Args:
            price: Price to check.
            prev_close: Previous closing price.

        Returns:
            Tuple of (is_valid, message).
        """
        ara = prev_close * (1 + self.config.ara_limit_pct)
        arb = prev_close * (1 - self.config.arb_limit_pct)

        if price > ara:
            return False, f"Price {price:.0f} exceeds ARA ({ara:.0f})"
        if price < arb:
            return False, f"Price {price:.0f} below ARB ({arb:.0f})"

        return True, "Within limits"

    def get_price_limits(
        self,
        prev_close: float,
    ) -> Tuple[float, float]:
        """Get ARA and ARB prices.

        Args:
            prev_close: Previous closing price.

        Returns:
            Tuple of (ara_price, arb_price).
        """
        ara = prev_close * (1 + self.config.ara_limit_pct)
        arb = prev_close * (1 - self.config.arb_limit_pct)
        return ara, arb

    def round_to_lot(self, shares: int) -> int:
        """Round shares to nearest lot.

        Args:
            shares: Number of shares.

        Returns:
            Rounded number of shares.
        """
        return (shares // self.config.lot_size) * self.config.lot_size

    def _calculate_slippage(self, is_buy: bool) -> float:
        """Calculate slippage for market order.

        Args:
            is_buy: Whether buy order.

        Returns:
            Slippage as decimal.
        """
        # Base slippage with some randomness
        base = self.config.slippage_market

        # Add some randomness (±50% of base)
        randomness = self.random_state.uniform(-0.5, 0.5) * base

        return max(0, base + randomness)

    def _round_to_tick(self, price: float) -> float:
        """Round price to valid tick size.

        Args:
            price: Price to round.

        Returns:
            Rounded price.
        """
        for (low, high), tick in self.config.tick_sizes.items():
            if low <= price < high:
                return round(price / tick) * tick

        # Default tick for very high prices
        return round(price / 50) * 50

    def simulate_gapped_open(
        self,
        prev_close: float,
        volatility: float = 0.02,
    ) -> float:
        """Simulate gapped opening price.

        IDX stocks often gap at open due to overnight news.

        Args:
            prev_close: Previous close price.
            volatility: Expected volatility.

        Returns:
            Simulated open price.
        """
        # Generate gap with some randomness
        gap = self.random_state.normal(0, volatility)

        # Limit to ARA/ARB bounds
        gap = max(-self.config.arb_limit_pct, min(self.config.ara_limit_pct, gap))

        open_price = prev_close * (1 + gap)
        return self._round_to_tick(open_price)

    def simulate_intraday_volatility(
        self,
        open_price: float,
        close_price: float,
        num_bars: int = 78,  # ~5 min bars in trading day
    ) -> List[Dict[str, float]]:
        """Simulate intraday price bars.

        Args:
            open_price: Opening price.
            close_price: Closing price.
            num_bars: Number of bars to generate.

        Returns:
            List of bar dictionaries with OHLC.
        """
        if num_bars < 2:
            return []

        # Simple random walk from open to close
        prices = [open_price]
        trend = (close_price - open_price) / num_bars

        for i in range(1, num_bars):
            # Add trend plus noise
            noise = self.random_state.normal(0, open_price * 0.002)
            next_price = prices[-1] + trend + noise
            next_price = max(next_price, open_price * 0.9)  # Limit downside
            prices.append(next_price)

        # Generate bars
        bars = []
        for i in range(num_bars):
            price = prices[i]

            # Generate high/low with some range
            high = price * (1 + abs(self.random_state.normal(0, 0.001)))
            low = price * (1 - abs(self.random_state.normal(0, 0.001)))

            bars.append({
                "open": self._round_to_tick(price),
                "high": self._round_to_tick(high),
                "low": self._round_to_tick(low),
                "close": self._round_to_tick(prices[min(i + 1, num_bars - 1)]),
            })

        return bars


def simulate_trade_execution(
    entry_price: float,
    exit_price: float,
    quantity: int,
    config: Optional[SimulationConfig] = None,
) -> Dict[str, float]:
    """Simulate complete trade execution with costs.

    Args:
        entry_price: Entry price.
        exit_price: Exit price.
        quantity: Number of shares.
        config: Simulation configuration.

    Returns:
        Dictionary with P&L breakdown.
    """
    config = config or SimulationConfig()

    # Buy side
    buy_gross = entry_price * quantity
    buy_fees = buy_gross * config.buy_fee_pct
    buy_total = buy_gross + buy_fees

    # Sell side
    sell_gross = exit_price * quantity
    sell_fees = sell_gross * config.sell_fee_pct
    sell_total = sell_gross - sell_fees

    # P&L
    gross_pnl = sell_gross - buy_gross
    total_fees = buy_fees + sell_fees
    net_pnl = sell_total - buy_total
    return_pct = (net_pnl / buy_total) * 100

    return {
        "buy_gross": buy_gross,
        "buy_fees": buy_fees,
        "buy_total": buy_total,
        "sell_gross": sell_gross,
        "sell_fees": sell_fees,
        "sell_total": sell_total,
        "gross_pnl": gross_pnl,
        "total_fees": total_fees,
        "net_pnl": net_pnl,
        "return_pct": return_pct,
    }
