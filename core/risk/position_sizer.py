"""
Position Sizer Module

Calculates position sizes based on risk parameters.
Uses fixed percentage risk method with lot size adjustment.
"""

import logging
from dataclasses import dataclass
from typing import Optional

from config.settings import settings
from config.trading_modes import ModeConfig
from config.constants import IDX_LOT_SIZE

logger = logging.getLogger(__name__)


@dataclass
class PositionSize:
    """Position size calculation result.

    Attributes:
        shares: Number of shares to trade.
        lots: Number of lots (100 shares each).
        value: Total position value in IDR.
        risk_amount: Amount at risk in IDR.
        risk_pct: Risk as percentage of capital.
        entry_price: Entry price per share.
        stop_loss: Stop loss price per share.
    """

    shares: int
    lots: int
    value: float
    risk_amount: float
    risk_pct: float
    entry_price: float
    stop_loss: float


class PositionSizer:
    """Calculates position sizes based on risk management rules.

    Uses a fixed percentage risk model where the position size
    is determined by the distance to stop loss and the maximum
    acceptable risk per trade.

    Example:
        sizer = PositionSizer(capital=100_000_000, config=swing_config)
        size = sizer.calculate(
            entry_price=9000,
            stop_loss=8550,
            signal_score=75
        )
        print(f"Buy {size.shares} shares ({size.lots} lots)")
    """

    def __init__(
        self,
        capital: float,
        config: ModeConfig,
    ) -> None:
        """Initialize position sizer.

        Args:
            capital: Trading capital in IDR.
            config: Trading mode configuration.
        """
        self.capital = capital
        self.config = config
        self.lot_size = IDX_LOT_SIZE

    def calculate(
        self,
        entry_price: float,
        stop_loss: float,
        signal_score: float = 50.0,
        position_multiplier: float = 1.0,
    ) -> PositionSize:
        """Calculate position size.

        Args:
            entry_price: Entry price per share.
            stop_loss: Stop loss price per share.
            signal_score: Signal quality score (0-100), affects size.
            position_multiplier: Additional size adjustment (0-1).

        Returns:
            PositionSize with calculated values.

        Raises:
            ValueError: If entry_price or stop_loss is invalid.
        """
        # Validate inputs
        if entry_price <= 0:
            raise ValueError(f"Invalid entry price: {entry_price}")
        if stop_loss <= 0:
            raise ValueError(f"Invalid stop loss: {stop_loss}")
        if stop_loss >= entry_price:
            raise ValueError(
                f"Stop loss ({stop_loss}) must be below entry ({entry_price})"
            )

        # Calculate risk per share
        risk_per_share = entry_price - stop_loss
        risk_pct_per_share = risk_per_share / entry_price

        # Determine risk percentage based on signal quality
        base_risk_pct = self.config.max_risk_per_trade

        # Adjust risk based on signal score
        if signal_score >= 80:
            # High quality signal - use full risk
            risk_multiplier = 1.0
        elif signal_score >= 70:
            # Good signal - 80% of max risk
            risk_multiplier = 0.8
        elif signal_score >= 60:
            # Moderate signal - 60% of max risk
            risk_multiplier = 0.6
        else:
            # Lower quality signal - 40% of max risk
            risk_multiplier = 0.4

        # Apply position multiplier (for portfolio-level adjustments)
        risk_multiplier *= position_multiplier

        # Calculate risk amount in IDR
        risk_pct = base_risk_pct * risk_multiplier
        risk_amount = self.capital * risk_pct

        # Calculate shares based on risk
        shares = int(risk_amount / risk_per_share)

        # Round down to nearest lot
        lots = shares // self.lot_size
        shares = lots * self.lot_size

        # Ensure minimum of 1 lot
        if shares < self.lot_size:
            shares = self.lot_size
            lots = 1

        # Check maximum position size
        max_position_value = self.capital * self.config.max_position_pct
        position_value = shares * entry_price

        if position_value > max_position_value:
            # Reduce to max position
            shares = int(max_position_value / entry_price)
            lots = shares // self.lot_size
            shares = lots * self.lot_size
            position_value = shares * entry_price

            # Recalculate risk
            risk_amount = shares * risk_per_share
            risk_pct = risk_amount / self.capital

        # Final calculations
        position_value = shares * entry_price
        risk_amount = shares * risk_per_share
        actual_risk_pct = risk_amount / self.capital

        logger.debug(
            f"Position size: {shares} shares ({lots} lots), "
            f"Value: {position_value:,.0f} IDR, "
            f"Risk: {actual_risk_pct:.2%}"
        )

        return PositionSize(
            shares=shares,
            lots=lots,
            value=position_value,
            risk_amount=risk_amount,
            risk_pct=actual_risk_pct,
            entry_price=entry_price,
            stop_loss=stop_loss,
        )

    def calculate_for_target_risk(
        self,
        entry_price: float,
        stop_loss: float,
        target_risk_pct: float,
    ) -> PositionSize:
        """Calculate position size for a specific risk percentage.

        Args:
            entry_price: Entry price per share.
            stop_loss: Stop loss price per share.
            target_risk_pct: Target risk as decimal (e.g., 0.01 for 1%).

        Returns:
            PositionSize with calculated values.
        """
        # Validate inputs
        if entry_price <= 0:
            raise ValueError(f"Invalid entry price: {entry_price}")
        if stop_loss <= 0:
            raise ValueError(f"Invalid stop loss: {stop_loss}")

        risk_per_share = entry_price - stop_loss
        if risk_per_share <= 0:
            raise ValueError("Stop loss must be below entry price")

        risk_amount = self.capital * target_risk_pct
        shares = int(risk_amount / risk_per_share)
        lots = shares // self.lot_size
        shares = lots * self.lot_size

        if shares < self.lot_size:
            shares = self.lot_size
            lots = 1

        position_value = shares * entry_price
        actual_risk = shares * risk_per_share
        actual_risk_pct = actual_risk / self.capital

        return PositionSize(
            shares=shares,
            lots=lots,
            value=position_value,
            risk_amount=actual_risk,
            risk_pct=actual_risk_pct,
            entry_price=entry_price,
            stop_loss=stop_loss,
        )

    def get_max_shares(self, price: float) -> int:
        """Get maximum number of shares for a price.

        Based on maximum position percentage.

        Args:
            price: Price per share.

        Returns:
            Maximum number of shares (rounded to lot size).
        """
        max_value = self.capital * self.config.max_position_pct
        max_shares = int(max_value / price)
        lots = max_shares // self.lot_size
        return lots * self.lot_size

    def calculate_kelly_size(
        self,
        entry_price: float,
        stop_loss: float,
        win_rate: float,
        avg_win_loss_ratio: float,
    ) -> PositionSize:
        """Calculate position size using Kelly Criterion.

        Kelly formula: f = (p * b - q) / b
        where p = win rate, q = loss rate, b = win/loss ratio

        Args:
            entry_price: Entry price per share.
            stop_loss: Stop loss price per share.
            win_rate: Historical win rate (0-1).
            avg_win_loss_ratio: Average win divided by average loss.

        Returns:
            PositionSize using Kelly fraction (capped at 25% of capital).
        """
        # Kelly calculation
        loss_rate = 1 - win_rate
        kelly_fraction = (win_rate * avg_win_loss_ratio - loss_rate) / avg_win_loss_ratio

        # Cap Kelly at 25% and use half-Kelly for safety
        kelly_fraction = min(kelly_fraction, 0.25)
        half_kelly = kelly_fraction / 2

        # Calculate risk amount
        risk_pct = min(half_kelly, self.config.max_risk_per_trade * 2)
        risk_amount = self.capital * risk_pct

        # Calculate shares
        risk_per_share = entry_price - stop_loss
        shares = int(risk_amount / risk_per_share)
        lots = shares // self.lot_size
        shares = lots * self.lot_size

        if shares < self.lot_size:
            shares = self.lot_size
            lots = 1

        position_value = shares * entry_price
        actual_risk = shares * risk_per_share
        actual_risk_pct = actual_risk / self.capital

        return PositionSize(
            shares=shares,
            lots=lots,
            value=position_value,
            risk_amount=actual_risk,
            risk_pct=actual_risk_pct,
            entry_price=entry_price,
            stop_loss=stop_loss,
        )

    def update_capital(self, new_capital: float) -> None:
        """Update the capital amount.

        Args:
            new_capital: New capital amount in IDR.
        """
        self.capital = new_capital
        logger.info(f"Capital updated to {new_capital:,.0f} IDR")
