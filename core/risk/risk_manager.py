"""
Risk Manager Module

THE MOST CRITICAL MODULE IN THE SYSTEM.
Has VETO power over all trades.

Responsibilities:
1. Validate all trades against risk rules
2. Calculate position sizes
3. Monitor portfolio risk
4. VETO any trade that violates rules
"""

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import List, Optional

from config.settings import settings
from config.trading_modes import ModeConfig, TradingMode, get_mode_config
from core.data.models import Signal, Position, PortfolioState
from core.risk.position_sizer import PositionSizer, PositionSize

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of trade validation.

    Contains approval status, reasons, and position details if approved.

    Attributes:
        approved: Whether the trade is approved.
        veto_reason: Reason for rejection if not approved.
        warnings: List of warning messages.
        position_size: Number of shares if approved.
        position_value: Position value in IDR if approved.
        risk_amount: Amount at risk in IDR if approved.
        adjusted_stop: Adjusted stop loss price.
        kelly_fraction: Kelly fraction if calculated.
    """

    approved: bool
    veto_reason: Optional[str] = None
    warnings: List[str] = field(default_factory=list)

    # If approved, position details
    position_size: Optional[int] = None
    position_value: Optional[float] = None
    risk_amount: Optional[float] = None
    adjusted_stop: Optional[float] = None
    kelly_fraction: Optional[float] = None

    def __post_init__(self) -> None:
        """Initialize warnings list if None."""
        if self.warnings is None:
            self.warnings = []


class RiskManager:
    """Central risk management.

    This class has VETO POWER over all trades. It enforces hard
    limits that cannot be overridden and soft limits that may
    trigger warnings or size reductions.

    Hard limits (automatic VETO):
    - Daily loss >= 2% of capital
    - Drawdown >= 10%
    - Max positions reached
    - Already have position in symbol
    - Score below minimum
    - Invalid stop loss

    Soft limits (warnings, may reduce size):
    - Drawdown > 5%: reduce position 50%
    - Sector concentration > 50%
    - Correlated positions

    Example:
        manager = RiskManager(mode=TradingMode.SWING)
        result = manager.validate_entry(signal, portfolio)
        if result.approved:
            print(f"Approved: {result.position_size} shares")
        else:
            print(f"Rejected: {result.veto_reason}")
    """

    def __init__(
        self,
        mode: TradingMode = TradingMode.SWING,
        capital: Optional[float] = None,
    ) -> None:
        """Initialize risk manager.

        Args:
            mode: Trading mode (default: SWING).
            capital: Initial capital (default: from settings).
        """
        self.mode = mode
        self.config: ModeConfig = get_mode_config(mode)
        self.capital = capital or settings.initial_capital
        self.position_sizer = PositionSizer(self.capital, self.config)

        # Track daily P&L
        self._daily_start_capital = self.capital

    def validate_entry(
        self,
        signal: Signal,
        portfolio: PortfolioState,
    ) -> ValidationResult:
        """Validate a potential entry.

        This is the MAIN validation function. All trades must
        pass through this function before execution.

        Args:
            signal: Trading signal to validate.
            portfolio: Current portfolio state.

        Returns:
            ValidationResult with approved/rejected status and reasons.
        """
        warnings: List[str] = []

        logger.info(
            f"Validating entry for {signal.symbol}: "
            f"score={signal.composite_score:.1f}, "
            f"entry={signal.entry_price:.0f}"
        )

        # ============== HARD LIMITS (VETO) ==============

        # Check daily loss limit
        daily_loss_pct = abs(portfolio.daily_pnl_pct) if portfolio.daily_pnl_pct < 0 else 0
        if daily_loss_pct >= settings.max_daily_loss:
            reason = f"Daily loss limit reached: {portfolio.daily_pnl_pct:.1%}"
            logger.warning(f"VETO: {reason}")
            return ValidationResult(approved=False, veto_reason=reason)

        # Check drawdown limit
        if portfolio.drawdown_pct >= settings.max_drawdown:
            reason = f"Max drawdown reached: {portfolio.drawdown_pct:.1%}"
            logger.warning(f"VETO: {reason}")
            return ValidationResult(approved=False, veto_reason=reason)

        # Check max positions
        if portfolio.open_positions >= settings.max_positions:
            reason = f"Max positions reached: {portfolio.open_positions}/{settings.max_positions}"
            logger.warning(f"VETO: {reason}")
            return ValidationResult(approved=False, veto_reason=reason)

        # Check if already have position
        existing = [p for p in portfolio.positions if p.symbol == signal.symbol]
        if existing:
            reason = f"Already have position in {signal.symbol}"
            logger.warning(f"VETO: {reason}")
            return ValidationResult(approved=False, veto_reason=reason)

        # Check minimum score
        if signal.composite_score < self.config.min_score:
            reason = (
                f"Score {signal.composite_score:.1f} below minimum {self.config.min_score}"
            )
            logger.warning(f"VETO: {reason}")
            return ValidationResult(approved=False, veto_reason=reason)

        # Validate stop loss
        if signal.stop_loss >= signal.entry_price:
            reason = f"Invalid stop loss: {signal.stop_loss:.0f} >= entry {signal.entry_price:.0f}"
            logger.warning(f"VETO: {reason}")
            return ValidationResult(approved=False, veto_reason=reason)

        # Check stop loss distance (not too tight)
        stop_distance_pct = (signal.entry_price - signal.stop_loss) / signal.entry_price
        if stop_distance_pct < 0.01:  # Less than 1%
            reason = f"Stop loss too tight: {stop_distance_pct:.2%}"
            logger.warning(f"VETO: {reason}")
            return ValidationResult(approved=False, veto_reason=reason)

        # ============== SOFT LIMITS (WARNINGS) ==============

        position_multiplier = 1.0

        # Check drawdown warning
        if portfolio.drawdown_pct > 0.05:  # 5%
            position_multiplier *= 0.5
            warning = f"Drawdown {portfolio.drawdown_pct:.1%} - reducing position 50%"
            warnings.append(warning)
            logger.info(warning)

        # Check high daily loss warning
        if portfolio.daily_pnl_pct < -0.01:  # -1%
            position_multiplier *= 0.75
            warning = f"Daily loss {portfolio.daily_pnl_pct:.1%} - reducing position 25%"
            warnings.append(warning)
            logger.info(warning)

        # ============== CALCULATE POSITION SIZE ==============

        try:
            size_result = self.position_sizer.calculate(
                entry_price=signal.entry_price,
                stop_loss=signal.stop_loss,
                signal_score=signal.composite_score,
                position_multiplier=position_multiplier,
            )
        except Exception as e:
            reason = f"Position sizing error: {e}"
            logger.error(f"VETO: {reason}")
            return ValidationResult(approved=False, veto_reason=reason)

        # Check minimum position size
        if size_result.shares < settings.lot_size:
            reason = f"Position size {size_result.shares} below minimum lot ({settings.lot_size})"
            logger.warning(f"VETO: {reason}")
            return ValidationResult(approved=False, veto_reason=reason)

        # Check maximum position percentage
        position_pct = size_result.value / portfolio.total_value
        if position_pct > settings.max_position_pct:
            reason = (
                f"Position {position_pct:.1%} exceeds max {settings.max_position_pct:.1%}"
            )
            logger.warning(f"VETO: {reason}")
            return ValidationResult(approved=False, veto_reason=reason)

        # ============== APPROVED ==============

        logger.info(
            f"APPROVED: {signal.symbol} - "
            f"{size_result.shares} shares, "
            f"risk {size_result.risk_pct:.2%}"
        )

        return ValidationResult(
            approved=True,
            warnings=warnings,
            position_size=size_result.shares,
            position_value=size_result.value,
            risk_amount=size_result.risk_amount,
            adjusted_stop=signal.stop_loss,
        )

    def validate_exit(
        self,
        position: Position,
        proposed_exit_price: float,
        exit_reason: str,
    ) -> ValidationResult:
        """Validate a proposed exit.

        Args:
            position: Position to exit.
            proposed_exit_price: Proposed exit price.
            exit_reason: Reason for exit.

        Returns:
            ValidationResult with approval status.
        """
        logger.info(
            f"Validating exit for {position.symbol}: "
            f"reason={exit_reason}, "
            f"price={proposed_exit_price:.0f}"
        )

        # Time stop - always allow
        if exit_reason == "time_stop":
            return ValidationResult(approved=True)

        # Stop loss hit - always allow
        if exit_reason == "stop_loss":
            return ValidationResult(approved=True)

        # Target hit - always allow
        if exit_reason in ["target_1", "target_2", "target_3"]:
            return ValidationResult(approved=True)

        # Signal reversal - allow
        if exit_reason == "signal_reversal":
            return ValidationResult(approved=True)

        # Manual exit - validate
        current_pnl_pct = (
            proposed_exit_price - position.entry_price
        ) / position.entry_price

        # Don't exit profitable position too early (< 2 days, > 5% profit)
        if position.days_held < 2 and current_pnl_pct > 0.05:
            reason = "Don't exit profitable position too early"
            warning = f"Position is +{current_pnl_pct:.1%} after only {position.days_held} days"
            logger.warning(f"VETO: {reason} - {warning}")
            return ValidationResult(
                approved=False,
                veto_reason=reason,
                warnings=[warning],
            )

        # Warn about exiting too early
        if position.days_held < self.config.min_hold_days:
            warning = f"Exiting before min hold days ({self.config.min_hold_days})"
            logger.info(warning)
            return ValidationResult(approved=True, warnings=[warning])

        return ValidationResult(approved=True)

    def check_portfolio_risk(self, portfolio: PortfolioState) -> List[str]:
        """Check overall portfolio risk.

        Args:
            portfolio: Current portfolio state.

        Returns:
            List of risk warnings.
        """
        warnings: List[str] = []

        # Check drawdown
        if portfolio.drawdown_pct > 0.05:
            warnings.append(f"Drawdown at {portfolio.drawdown_pct:.1%}")

        # Check daily loss
        if portfolio.daily_pnl_pct < -0.01:
            warnings.append(f"Daily loss at {portfolio.daily_pnl_pct:.1%}")

        # Check position count
        if portfolio.open_positions >= settings.max_positions - 1:
            warnings.append(
                f"Near max positions: {portfolio.open_positions}/{settings.max_positions}"
            )

        # Check position concentration
        if portfolio.positions:
            for pos in portfolio.positions:
                position_pct = (pos.current_price * pos.quantity) / portfolio.total_value
                if position_pct > 0.30:
                    warnings.append(
                        f"Large position in {pos.symbol}: {position_pct:.1%}"
                    )

        # Check if near cash limits
        cash_pct = portfolio.cash / portfolio.total_value
        if cash_pct < 0.10:
            warnings.append(f"Low cash: {cash_pct:.1%}")

        return warnings

    def should_halt_trading(self, portfolio: PortfolioState) -> tuple:
        """Check if trading should be halted.

        Args:
            portfolio: Current portfolio state.

        Returns:
            Tuple of (should_halt, reason).
        """
        # Critical drawdown
        if portfolio.drawdown_pct >= settings.max_drawdown:
            return True, f"Critical drawdown: {portfolio.drawdown_pct:.1%}"

        # Critical daily loss
        if portfolio.daily_pnl_pct <= -settings.max_daily_loss:
            return True, f"Critical daily loss: {portfolio.daily_pnl_pct:.1%}"

        return False, ""

    def update_capital(self, new_capital: float) -> None:
        """Update capital amount.

        Args:
            new_capital: New capital amount.
        """
        self.capital = new_capital
        self.position_sizer.update_capital(new_capital)
        logger.info(f"Risk manager capital updated: {new_capital:,.0f}")

    def reset_daily_pnl(self) -> None:
        """Reset daily P&L tracking at start of new day."""
        self._daily_start_capital = self.capital
        logger.debug("Daily P&L tracking reset")

    def get_risk_report(self, portfolio: PortfolioState) -> str:
        """Generate a risk report.

        Args:
            portfolio: Current portfolio state.

        Returns:
            Formatted risk report string.
        """
        lines = [
            "=" * 50,
            "RISK REPORT",
            "=" * 50,
            f"Total Value: {portfolio.total_value:,.0f} IDR",
            f"Cash: {portfolio.cash:,.0f} IDR ({portfolio.cash/portfolio.total_value:.1%})",
            f"Open Positions: {portfolio.open_positions}",
            "",
            f"Daily P&L: {portfolio.daily_pnl:,.0f} IDR ({portfolio.daily_pnl_pct:.2%})",
            f"Total P&L: {portfolio.total_pnl:,.0f} IDR ({portfolio.total_pnl_pct:.2%})",
            "",
            f"Peak Value: {portfolio.peak_value:,.0f} IDR",
            f"Drawdown: {portfolio.drawdown:,.0f} IDR ({portfolio.drawdown_pct:.2%})",
            "",
        ]

        # Add risk warnings
        risk_warnings = self.check_portfolio_risk(portfolio)
        if risk_warnings:
            lines.append("RISK WARNINGS:")
            for warning in risk_warnings:
                lines.append(f"  - {warning}")
        else:
            lines.append("No risk warnings")

        # Check for halt
        should_halt, halt_reason = self.should_halt_trading(portfolio)
        if should_halt:
            lines.append("")
            lines.append(f"TRADING HALT: {halt_reason}")

        lines.append("=" * 50)

        return "\n".join(lines)
