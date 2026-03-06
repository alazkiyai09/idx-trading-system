"""
Forecast-Enhanced Signal Generator Module

Generates trading signals enhanced with TimesFM price forecasts.
Extends the base SignalGenerator to incorporate forecast-derived
stop loss and target levels.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

from config.trading_modes import ModeConfig
from core.data.models import (
    OHLCV,
    Signal,
    SignalType,
    SetupType,
    FlowSignal,
)
from core.data.foreign_flow import FlowAnalysis
from core.forecasting.timesfm_forecaster import (
    TimesFMForecaster,
    PriceForecast,
    ForecastScorer,
)
from core.signals.signal_generator import SignalGenerator

logger = logging.getLogger(__name__)


@dataclass
class EnhancedSignal(Signal):
    """Trading signal enhanced with TimesFM forecast.

    Extends the base Signal with forecast-derived metrics for
    data-driven stop loss and target levels.

    Attributes:
        price_forecast: PriceForecast from TimesFM (if available).
        forecast_score: 0-100 score from ForecastScorer.
        forecast_confidence: "high", "medium", "low", or None.
        forecast_stop_loss: Stop loss from forecast quantile.
        forecast_target_1: Target 1 from forecast quantile.
        forecast_target_2: Target 2 from forecast quantile.
        forecast_target_3: Target 3 from forecast quantile.
    """

    price_forecast: Optional[PriceForecast] = None
    forecast_score: Optional[float] = None
    forecast_confidence: Optional[str] = None  # "high", "medium", "low"
    forecast_stop_loss: Optional[float] = None
    forecast_target_1: Optional[float] = None
    forecast_target_2: Optional[float] = None
    forecast_target_3: Optional[float] = None


class ForecastEnhancedSignalGenerator(SignalGenerator):
    """Signal generator enhanced with TimesFM forecasts.

    Extends the base SignalGenerator to incorporate price forecasts
    into signal generation, providing data-driven stop loss and
    target levels based on forecast quantiles.

    The generator uses forecast quantiles to derive trading levels:
    - Stop loss: q10 (10th percentile - worst case)
    - Target 1: q50 (median)
    - Target 2: q70
    - Target 3: q90 (90th percentile - best case)

    The composite score is adjusted by blending the original score
    with the forecast score based on the forecast_weight parameter.

    Example:
        generator = ForecastEnhancedSignalGenerator(mode_config)
        signal = generator.generate_enhanced(
            symbol="BBCA",
            ohlcv_data=prices,
            flow_analysis=flow
        )
        if signal and signal.forecast_confidence == "high":
            print(f"High confidence forecast for {signal.symbol}")
    """

    # Confidence thresholds
    HIGH_UNCERTAINTY_THRESHOLD: float = 0.05  # 5%
    MEDIUM_UNCERTAINTY_THRESHOLD: float = 0.10  # 10%
    MIN_EXPECTED_RETURN_FOR_HIGH: float = 0.03  # 3%

    def __init__(
        self,
        config: ModeConfig,
        forecast_weight: float = 0.25,
    ) -> None:
        """Initialize forecast-enhanced signal generator.

        Args:
            config: Trading mode configuration.
            forecast_weight: Weight for forecast score in composite (0-1).
                Original score weight is (1 - forecast_weight).
                Default 0.25 means 25% forecast, 75% original.
        """
        super().__init__(config)
        self.forecaster = TimesFMForecaster()
        self.scorer = ForecastScorer()
        self.forecast_weight = forecast_weight

    def generate_enhanced(
        self,
        symbol: str,
        ohlcv_data: List[OHLCV],
        flow_analysis: Optional[FlowAnalysis] = None,
        fundamental_score: Optional[float] = None,
    ) -> Optional[EnhancedSignal]:
        """Generate enhanced signal with forecast data.

        Args:
            symbol: Stock symbol.
            ohlcv_data: Historical price data.
            flow_analysis: Foreign flow analysis (optional).
            fundamental_score: Fundamental analysis score (optional).

        Returns:
            EnhancedSignal if conditions are met, None otherwise.
            Returns EnhancedSignal with None forecast fields if
            forecasting is unavailable.
        """
        # Generate base signal using parent class
        base_signal = self.generate(
            symbol=symbol,
            ohlcv_data=ohlcv_data,
            flow_analysis=flow_analysis,
            fundamental_score=fundamental_score,
        )

        if base_signal is None:
            return None

        # Try to generate forecast
        price_forecast = self._generate_forecast(symbol, ohlcv_data)

        if price_forecast is None:
            # Return enhanced signal without forecast data
            logger.debug(f"Forecast unavailable for {symbol}, using base signal")
            return self._create_enhanced_without_forecast(base_signal)

        # Adjust signal with forecast-derived levels
        return self._adjust_with_forecast(base_signal, price_forecast)

    def _generate_forecast(
        self,
        symbol: str,
        ohlcv_data: List[OHLCV],
    ) -> Optional[PriceForecast]:
        """Generate price forecast from OHLCV data.

        Args:
            symbol: Stock symbol.
            ohlcv_data: Historical price data.

        Returns:
            PriceForecast if successful, None otherwise.
        """
        if len(ohlcv_data) < 10:
            logger.warning(f"Insufficient OHLCV data for forecast: {symbol}")
            return None

        # Extract close prices for forecasting
        prices = [float(bar.close) for bar in ohlcv_data]

        try:
            forecast = self.forecaster.forecast_price(
                symbol=symbol,
                prices=prices,
            )
            return forecast
        except Exception as e:
            logger.error(f"Error generating forecast for {symbol}: {e}")
            return None

    def _create_enhanced_without_forecast(
        self,
        signal: Signal,
    ) -> EnhancedSignal:
        """Create EnhancedSignal when forecast is unavailable.

        Args:
            signal: Base signal from parent generator.

        Returns:
            EnhancedSignal with None forecast fields.
        """
        return EnhancedSignal(
            # Base Signal fields
            symbol=signal.symbol,
            timestamp=signal.timestamp,
            signal_type=signal.signal_type,
            composite_score=signal.composite_score,
            technical_score=signal.technical_score,
            flow_score=signal.flow_score,
            fundamental_score=signal.fundamental_score,
            setup_type=signal.setup_type,
            flow_signal=signal.flow_signal,
            entry_price=signal.entry_price,
            stop_loss=signal.stop_loss,
            target_1=signal.target_1,
            target_2=signal.target_2,
            target_3=signal.target_3,
            risk_pct=signal.risk_pct,
            position_size=signal.position_size,
            position_value=signal.position_value,
            key_factors=signal.key_factors,
            risks=signal.risks,
            # Enhanced fields (all None)
            price_forecast=None,
            forecast_score=None,
            forecast_confidence=None,
            forecast_stop_loss=None,
            forecast_target_1=None,
            forecast_target_2=None,
            forecast_target_3=None,
        )

    def _adjust_with_forecast(
        self,
        signal: Signal,
        forecast: PriceForecast,
    ) -> EnhancedSignal:
        """Adjust signal with forecast-derived levels.

        Args:
            signal: Base signal from parent generator.
            forecast: Price forecast from TimesFM.

        Returns:
            EnhancedSignal with forecast-derived levels and adjusted score.
        """
        # Score the forecast
        forecast_score = self.scorer.score_forecast(forecast)

        # Determine confidence level
        confidence = self._get_forecast_confidence(forecast)

        # Adjust composite score with forecast
        adjusted_score = self._adjust_score(
            original_score=signal.composite_score,
            forecast_score=forecast_score,
        )

        # Build enhanced key factors
        enhanced_factors = list(signal.key_factors)
        enhanced_factors.extend(self._get_forecast_factors(forecast, confidence))

        # Build enhanced risks
        enhanced_risks = list(signal.risks)
        enhanced_risks.extend(self._get_forecast_risks(forecast, confidence))

        return EnhancedSignal(
            # Base Signal fields
            symbol=signal.symbol,
            timestamp=signal.timestamp,
            signal_type=signal.signal_type,
            composite_score=adjusted_score,
            technical_score=signal.technical_score,
            flow_score=signal.flow_score,
            fundamental_score=signal.fundamental_score,
            setup_type=signal.setup_type,
            flow_signal=signal.flow_signal,
            entry_price=signal.entry_price,
            stop_loss=signal.stop_loss,
            target_1=signal.target_1,
            target_2=signal.target_2,
            target_3=signal.target_3,
            risk_pct=signal.risk_pct,
            position_size=signal.position_size,
            position_value=signal.position_value,
            key_factors=enhanced_factors,
            risks=enhanced_risks,
            # Enhanced fields
            price_forecast=forecast,
            forecast_score=forecast_score,
            forecast_confidence=confidence,
            forecast_stop_loss=forecast.stop_loss_price,
            forecast_target_1=forecast.target_1_price,
            forecast_target_2=forecast.target_2_price,
            forecast_target_3=forecast.target_3_price,
        )

    def _adjust_score(
        self,
        original_score: float,
        forecast_score: float,
    ) -> float:
        """Adjust composite score with forecast score.

        Args:
            original_score: Original composite score (0-100).
            forecast_score: Forecast score from ForecastScorer (0-100).

        Returns:
            Adjusted score blending original and forecast.
        """
        weight = self.forecast_weight
        adjusted = original_score * (1 - weight) + forecast_score * weight
        return max(0.0, min(100.0, adjusted))

    def _get_forecast_confidence(self, forecast: PriceForecast) -> str:
        """Determine forecast confidence level.

        Confidence levels:
        - High: uncertainty < 5% AND expected_return > 3%
        - Medium: uncertainty < 10%
        - Low: otherwise

        Args:
            forecast: Price forecast.

        Returns:
            Confidence level as string: "high", "medium", or "low".
        """
        uncertainty = forecast.uncertainty_pct
        expected_return = forecast.expected_return

        if (
            uncertainty < self.HIGH_UNCERTAINTY_THRESHOLD
            and expected_return > self.MIN_EXPECTED_RETURN_FOR_HIGH
        ):
            return "high"
        elif uncertainty < self.MEDIUM_UNCERTAINTY_THRESHOLD:
            return "medium"
        else:
            return "low"

    def _get_forecast_factors(
        self,
        forecast: PriceForecast,
        confidence: str,
    ) -> List[str]:
        """Get forecast-derived key factors.

        Args:
            forecast: Price forecast.
            confidence: Confidence level.

        Returns:
            List of key factor strings.
        """
        factors = []

        # Expected return
        if forecast.expected_return > 0:
            factors.append(f"Forecast expects +{forecast.expected_return:.1%} return")
        else:
            factors.append(f"Forecast expects {forecast.expected_return:.1%} return")

        # Confidence
        factors.append(f"Forecast confidence: {confidence}")

        # Upside/downside
        if forecast.upside_pct > 0:
            factors.append(f"Forecast upside potential: +{forecast.upside_pct:.1%}")
        if forecast.downside_pct < 0:
            factors.append(f"Forecast downside risk: {forecast.downside_pct:.1%}")

        # Low uncertainty bonus
        if confidence == "high":
            factors.append(f"Low forecast uncertainty ({forecast.uncertainty_pct:.1%})")

        return factors

    def _get_forecast_risks(
        self,
        forecast: PriceForecast,
        confidence: str,
    ) -> List[str]:
        """Get forecast-derived risk factors.

        Args:
            forecast: Price forecast.
            confidence: Confidence level.

        Returns:
            List of risk factor strings.
        """
        risks = []

        # High uncertainty
        if confidence == "low":
            risks.append(
                f"High forecast uncertainty ({forecast.uncertainty_pct:.1%})"
            )

        # Negative expected return
        if forecast.expected_return < 0:
            risks.append(f"Forecast predicts negative return ({forecast.expected_return:.1%})")

        # High downside risk
        if forecast.downside_pct < -0.05:  # More than 5% downside
            risks.append(f"Significant forecast downside ({forecast.downside_pct:.1%})")

        # Narrow upside
        if forecast.upside_pct < 0.02:  # Less than 2% upside
            risks.append(f"Limited forecast upside ({forecast.upside_pct:.1%})")

        return risks


def generate_enhanced_signals_for_universe(
    universe: List[str],
    ohlcv_data: dict[str, List[OHLCV]],
    flow_analyses: dict[str, FlowAnalysis],
    config: ModeConfig,
    forecast_weight: float = 0.25,
) -> List[EnhancedSignal]:
    """Generate enhanced signals for multiple symbols.

    Args:
        universe: List of symbols to analyze.
        ohlcv_data: Dictionary of symbol to OHLCV data.
        flow_analyses: Dictionary of symbol to flow analysis.
        config: Trading mode configuration.
        forecast_weight: Weight for forecast score (0-1).

    Returns:
        List of generated EnhancedSignals, sorted by composite score.
    """
    generator = ForecastEnhancedSignalGenerator(
        config=config,
        forecast_weight=forecast_weight,
    )
    signals: List[EnhancedSignal] = []

    for symbol in universe:
        if symbol not in ohlcv_data:
            continue

        flow = flow_analyses.get(symbol)

        try:
            signal = generator.generate_enhanced(
                symbol=symbol,
                ohlcv_data=ohlcv_data[symbol],
                flow_analysis=flow,
            )

            if signal and signal.signal_type == SignalType.BUY:
                signals.append(signal)

        except Exception as e:
            logger.error(f"Error generating enhanced signal for {symbol}: {e}")

    # Sort by composite score (highest first)
    signals.sort(key=lambda s: s.composite_score, reverse=True)

    return signals
