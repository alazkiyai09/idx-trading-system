"""Forecast-Enhanced Risk Manager Module

Extends the base RiskManager to incorporate TimesFM forecast data
into risk validation and position sizing decisions.
"""

import logging
from typing import List, Optional, Tuple

from config.settings import settings, timesfm_settings
from config.trading_modes import ModeConfig, TradingMode, get_mode_config
from core.data.models import PortfolioState
from core.risk.risk_manager import RiskManager, ValidationResult
from core.risk.position_sizer import PositionSize
from core.signals.forecast_enhanced_generator import EnhancedSignal

logger = logging.getLogger(__name__)


class ForecastEnhancedRiskManager(RiskManager):
    """Risk manager enhanced with TimesFM forecast validation.

    Extends the base RiskManager to incorporate forecast-based
    validation rules and position sizing adjustments.

    Additional validation:
    - Forecast R:R ratio check (minimum threshold)
    - Forecast uncertainty check (maximum threshold)
    - Position size adjustment based on forecast confidence

    Example:
        manager = ForecastEnhancedRiskManager(mode=TradingMode.SWING)
        result = manager.validate_entry(enhanced_signal, portfolio)
        if result.approved:
            print(f"Position size: {result.position_size}")
    """

    def __init__(
        self,
        mode: TradingMode = TradingMode.SWING,
        capital: Optional[float] = None,
        min_risk_reward: float = 1.5,
        max_uncertainty: float = 0.20,
    ) -> None:
        """Initialize forecast-enhanced risk manager.

        Args:
            mode: Trading mode.
            capital: Initial capital (default from settings).
            min_risk_reward: Minimum forecast R:R ratio.
            max_uncertainty: Maximum forecast uncertainty (as decimal).
        """
        super().__init__(mode, capital)
        self.min_risk_reward = min_risk_reward
        self.max_uncertainty = max_uncertainty

    def validate_entry(
        self,
        signal: EnhancedSignal,
        portfolio: PortfolioState,
    ) -> ValidationResult:
        """Validate entry with forecast-based checks.

        Extends base validation with:
        - Forecast R:R ratio validation
        - Forecast uncertainty validation
        - Position size adjustment for confidence

        Args:
            signal: EnhancedSignal to validate.
            portfolio: Current portfolio state.

        Returns:
            ValidationResult with approval status and position details.
        """
        # First run base validation
        base_result = super().validate_entry(signal, portfolio)

        if not base_result.approved:
            return base_result

        # If no forecast, return base result
        if signal.price_forecast is None:
            logger.debug(f"No forecast for {signal.symbol}, using base validation")
            return base_result

        # Additional forecast-based validation
        warnings = list(base_result.warnings)
        position_multiplier = 1.0

        # Check forecast R:R ratio
        rr_valid, rr_warning = self._validate_forecast_risk_reward(signal)
        if not rr_valid:
            warnings.append(rr_warning)
            position_multiplier *= 0.5  # Reduce position

        # Check forecast uncertainty
        uncertainty_valid, uncertainty_warning = self._validate_forecast_uncertainty(signal)
        if not uncertainty_valid:
            warnings.append(uncertainty_warning)
            position_multiplier *= 0.5  # Reduce position

        # Adjust position for confidence
        confidence_multiplier = self._get_confidence_multiplier(signal.forecast_confidence)
        position_multiplier *= confidence_multiplier

        if position_multiplier < 1.0:
            logger.info(
                f"Position reduced to {position_multiplier:.0%} for {signal.symbol} "
                f"due to forecast factors"
            )

        # Recalculate position size with multiplier
        adjusted_size = int(base_result.position_size * position_multiplier)
        adjusted_size = max(0, adjusted_size // 100 * 100)  # Round to lot size

        if adjusted_size < settings.lot_size:
            return ValidationResult(
                approved=False,
                veto_reason=f"Position size {adjusted_size} below minimum after forecast adjustment",
                warnings=warnings,
            )

        return ValidationResult(
            approved=True,
            warnings=warnings,
            position_size=adjusted_size,
            position_value=adjusted_size * signal.entry_price,
            risk_amount=base_result.risk_amount * position_multiplier,
            adjusted_stop=base_result.adjusted_stop,
        )

    def _validate_forecast_risk_reward(
        self,
        signal: EnhancedSignal,
    ) -> Tuple[bool, Optional[str]]:
        """Validate forecast R:R ratio.

        Args:
            signal: EnhancedSignal with forecast.

        Returns:
            Tuple of (is_valid, warning_message).
        """
        if signal.price_forecast is None:
            return True, None

        forecast = signal.price_forecast

        # Calculate R:R from forecast levels
        risk = signal.entry_price - forecast.stop_loss_price
        if risk <= 0:
            return False, "Invalid forecast stop loss (above entry)"

        reward = forecast.target_1_price - signal.entry_price
        rr_ratio = reward / risk

        if rr_ratio < self.min_risk_reward:
            return False, f"Forecast R:R ({rr_ratio:.1f}) below minimum ({self.min_risk_reward})"

        return True, None

    def _validate_forecast_uncertainty(
        self,
        signal: EnhancedSignal,
    ) -> Tuple[bool, Optional[str]]:
        """Validate forecast uncertainty.

        Args:
            signal: EnhancedSignal with forecast.

        Returns:
            Tuple of (is_valid, warning_message).
        """
        if signal.price_forecast is None:
            return True, None

        uncertainty = signal.price_forecast.uncertainty_pct

        if uncertainty > self.max_uncertainty:
            return False, f"Forecast uncertainty ({uncertainty:.1%}) exceeds max ({self.max_uncertainty:.1%})"

        return True, None

    def _get_confidence_multiplier(self, confidence: Optional[str]) -> float:
        """Get position multiplier based on forecast confidence.

        Args:
            confidence: "high", "medium", "low", or None.

        Returns:
            Position size multiplier.
        """
        if confidence is None:
            return 1.0

        multipliers = {
            "high": 1.0,     # Full position
            "medium": 0.8,   # 80% position
            "low": 0.6,      # 60% position
        }
        return multipliers.get(confidence, 1.0)
