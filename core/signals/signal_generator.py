"""
Signal Generator Module

Generates trading signals based on technical and flow analysis.
Uses mode-specific configurations for signal generation.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, date
from typing import List, Optional, Tuple, Dict

from config.trading_modes import ModeConfig, TradingMode
from core.data.models import (
    OHLCV,
    Signal,
    SignalType,
    SetupType,
    FlowSignal,
    TechnicalIndicators,
)
from core.data.foreign_flow import FlowAnalysis
from core.analysis.technical import TechnicalAnalyzer, TechnicalScore

logger = logging.getLogger(__name__)


@dataclass
class CompositeScore:
    """Composite score combining multiple analysis sources.

    Attributes:
        total: Total composite score (0-100).
        technical: Technical analysis score.
        flow: Foreign flow score.
        fundamental: Fundamental analysis score (if available).
        technical_weight: Weight applied to technical score.
        flow_weight: Weight applied to flow score.
        fundamental_weight: Weight applied to fundamental score.
    """

    total: float
    technical: float
    flow: float
    fundamental: Optional[float]
    technical_weight: float
    flow_weight: float
    fundamental_weight: float


class CompositeScorer:
    """Combines scores from different analysis sources.

    Uses mode-specific weights to calculate a composite score
    that determines signal strength.
    """

    def __init__(self, config: ModeConfig) -> None:
        """Initialize composite scorer.

        Args:
            config: Trading mode configuration with weights.
        """
        self.config = config

    def calculate(
        self,
        technical_score: float,
        flow_score: float,
        fundamental_score: Optional[float] = None,
    ) -> CompositeScore:
        """Calculate composite score.

        Args:
            technical_score: Technical analysis score (0-100).
            flow_score: Foreign flow score (0-100).
            fundamental_score: Fundamental analysis score (0-100), optional.

        Returns:
            CompositeScore with total and component breakdown.
        """
        weights = self.config
        tech_weight = weights.technical_weight
        flow_weight = weights.flow_weight
        fund_weight = weights.fundamental_weight

        # Adjust weights if no fundamental score
        if fundamental_score is None:
            total_weight = tech_weight + flow_weight
            tech_weight = tech_weight / total_weight
            flow_weight = flow_weight / total_weight
            fund_weight = 0.0

        # Calculate weighted total
        total = technical_score * tech_weight + flow_score * flow_weight

        if fundamental_score is not None:
            total += fundamental_score * fund_weight

        return CompositeScore(
            total=total,
            technical=technical_score,
            flow=flow_score,
            fundamental=fundamental_score,
            technical_weight=tech_weight,
            flow_weight=flow_weight,
            fundamental_weight=fund_weight,
        )


class SignalGenerator:
    """Generates trading signals.

    This is the main signal generation class that combines
    technical analysis, foreign flow analysis, and fundamental
    analysis to produce actionable trading signals.

    Example:
        generator = SignalGenerator(mode_config)
        signal = generator.generate(
            symbol="BBCA",
            ohlcv_data=prices,
            flow_analysis=flow
        )
        if signal and signal.signal_type == SignalType.BUY:
            print(f"Buy signal for {symbol} at {signal.entry_price}")
    """

    def __init__(self, config: ModeConfig) -> None:
        """Initialize signal generator.

        Args:
            config: Trading mode configuration.
        """
        self.config = config
        self.technical_analyzer = TechnicalAnalyzer()
        self.composite_scorer = CompositeScorer(config)

    def generate(
        self,
        symbol: str,
        ohlcv_data: List[OHLCV],
        flow_analysis: Optional[FlowAnalysis] = None,
        fundamental_score: Optional[float] = None,
    ) -> Optional[Signal]:
        """Generate trading signal.

        Args:
            symbol: Stock symbol.
            ohlcv_data: Historical price data.
            flow_analysis: Foreign flow analysis (optional).
            fundamental_score: Fundamental score (optional).

        Returns:
            Signal if conditions are met, None otherwise.
        """
        # Calculate technical indicators
        indicators = self.technical_analyzer.calculate(ohlcv_data)
        if not indicators:
            logger.warning(f"Insufficient data for signal generation: {symbol}")
            return None

        latest = indicators[-1]
        tech_score = self.technical_analyzer.calculate_score(latest)

        # Get flow score
        flow_score = 50.0  # Default neutral
        if flow_analysis:
            flow_score = flow_analysis.signal_score

        # Calculate composite score
        composite = self.composite_scorer.calculate(
            technical_score=tech_score.score,
            flow_score=flow_score,
            fundamental_score=fundamental_score,
        )

        # Check minimum score threshold
        if composite.total < self.config.min_score:
            logger.debug(
                f"Score {composite.total:.1f} below minimum {self.config.min_score}"
            )
            return None

        # Determine signal type
        signal_type = self._determine_signal_type(
            composite.total, latest, flow_analysis
        )

        if signal_type == SignalType.HOLD:
            return None

        # Get entry price from the last OHLCV bar
        entry_price = ohlcv_data[-1].close

        # Determine setup type
        setup_type = self._determine_setup_type(latest, flow_analysis, tech_score, entry_price)

        # Calculate entry, stop, and targets
        stop_loss, targets = self._calculate_levels(
            entry_price=entry_price,
            indicators=latest,
            setup_type=setup_type,
        )

        # Calculate risk percentage
        risk_pct = abs(entry_price - stop_loss) / entry_price

        # Generate signal
        signal = Signal(
            symbol=symbol,
            timestamp=datetime.now(),
            signal_type=signal_type,
            composite_score=composite.total,
            technical_score=composite.technical,
            flow_score=composite.flow,
            fundamental_score=composite.fundamental,
            setup_type=setup_type,
            flow_signal=flow_analysis.signal if flow_analysis else FlowSignal.NEUTRAL,
            entry_price=entry_price,
            stop_loss=stop_loss,
            target_1=targets[0],
            target_2=targets[1],
            target_3=targets[2],
            risk_pct=risk_pct,
            key_factors=self._get_key_factors(latest, flow_analysis, tech_score),
            risks=self._get_risks(latest, flow_analysis, entry_price),
        )

        return signal

    def _determine_signal_type(
        self,
        composite_score: float,
        indicators: TechnicalIndicators,
        flow_analysis: Optional[FlowAnalysis],
    ) -> SignalType:
        """Determine the type of signal.

        Args:
            composite_score: Composite analysis score.
            indicators: Technical indicators.
            flow_analysis: Foreign flow analysis.

        Returns:
            SignalType enum value.
        """
        # Check for strong sell conditions
        if indicators.trend == "downtrend" and indicators.rsi > 70:
            return SignalType.SELL

        if flow_analysis and flow_analysis.signal in [
            FlowSignal.SELL,
            FlowSignal.STRONG_SELL,
        ]:
            if composite_score < 40:
                return SignalType.SELL

        # Check for buy conditions
        if composite_score >= self.config.min_score:
            # Check flow signal requirement
            if self.config.min_flow_signal == "buy":
                if flow_analysis and flow_analysis.signal not in [
                    FlowSignal.BUY,
                    FlowSignal.STRONG_BUY,
                ]:
                    return SignalType.HOLD

            # Check RSI conditions
            if indicators.rsi > self.config.rsi_overbought:
                return SignalType.HOLD

            # Check volume
            if indicators.volume_ratio < self.config.min_volume_ratio:
                return SignalType.WATCH

            return SignalType.BUY

        return SignalType.HOLD

    def _determine_setup_type(
        self,
        indicators: TechnicalIndicators,
        flow_analysis: Optional[FlowAnalysis],
        tech_score: TechnicalScore,
        entry_price: float,
    ) -> SetupType:
        """Determine the type of trading setup.

        Args:
            indicators: Technical indicators.
            flow_analysis: Foreign flow analysis.
            tech_score: Technical score.
            entry_price: Current entry price.

        Returns:
            SetupType enum value.
        """
        # Check for oversold bounce
        if indicators.rsi < self.config.rsi_oversold:
            if entry_price < indicators.ema_20:
                return SetupType.OVERSOLD_BOUNCE

        # Check for foreign accumulation
        if flow_analysis and flow_analysis.signal in [
            FlowSignal.BUY,
            FlowSignal.STRONG_BUY,
        ]:
            if flow_analysis.consecutive_buy_days >= 3:
                return SetupType.FOREIGN_ACCUMULATION

        # Check for pullback to MA
        if indicators.trend == "uptrend":
            if entry_price <= indicators.ema_20 * 1.02:  # Within 2% of EMA20
                return SetupType.PULLBACK_TO_MA

        # Check for breakout
        if entry_price > indicators.bb_upper:
            if indicators.volume_ratio > 1.5:
                return SetupType.BREAKOUT

        # Check for trend continuation
        if indicators.trend in ["uptrend", "downtrend"]:
            if indicators.macd_hist > 0:
                return SetupType.TREND_CONTINUATION

        # Default to momentum
        return SetupType.MOMENTUM

    def _calculate_levels(
        self,
        entry_price: float,
        indicators: TechnicalIndicators,
        setup_type: SetupType,
    ) -> Tuple[float, List[float]]:
        """Calculate stop loss and target levels.

        Args:
            entry_price: Entry price.
            indicators: Technical indicators.
            setup_type: Type of trading setup.

        Returns:
            Tuple of (stop_loss, [target_1, target_2, target_3]).
        """
        # Use ATR for stop calculation
        atr = indicators.atr if indicators.atr else entry_price * 0.02

        # Stop loss based on setup type
        if setup_type == SetupType.OVERSOLD_BOUNCE:
            stop_distance = atr * 1.5
        elif setup_type == SetupType.BREAKOUT:
            stop_distance = atr * 2.0
        else:
            stop_distance = atr * 2.0

        stop_loss = entry_price - stop_distance

        # Ensure stop is not below support
        if indicators.support and stop_loss < indicators.support:
            stop_loss = indicators.support

        # Calculate targets based on R-multiples
        risk = entry_price - stop_loss
        target_1 = entry_price + (risk * self.config.target_1_ratio)
        target_2 = entry_price + (risk * self.config.target_2_ratio)
        target_3 = entry_price + (risk * self.config.target_3_ratio)

        return stop_loss, [target_1, target_2, target_3]

    def _get_key_factors(
        self,
        indicators: TechnicalIndicators,
        flow_analysis: Optional[FlowAnalysis],
        tech_score: TechnicalScore,
    ) -> List[str]:
        """Get key factors for the signal.

        Args:
            indicators: Technical indicators.
            flow_analysis: Foreign flow analysis.
            tech_score: Technical score.

        Returns:
            List of key factor strings.
        """
        factors = []

        # Trend
        if indicators.trend == "uptrend":
            factors.append(f"Uptrend confirmed (EMA20 > EMA50)")
        elif indicators.trend == "downtrend":
            factors.append(f"Price in downtrend")

        # RSI
        if indicators.rsi < 30:
            factors.append(f"RSI oversold ({indicators.rsi:.1f})")
        elif indicators.rsi < 40:
            factors.append(f"RSI approaching oversold ({indicators.rsi:.1f})")

        # MACD
        if indicators.macd_hist > 0:
            factors.append("MACD histogram positive")

        # Volume
        if indicators.volume_ratio > 1.5:
            factors.append(f"High volume ({indicators.volume_ratio:.1f}x avg)")

        # Foreign flow
        if flow_analysis:
            if flow_analysis.signal in [FlowSignal.BUY, FlowSignal.STRONG_BUY]:
                factors.append(
                    f"Foreign buying ({flow_analysis.consecutive_buy_days} days)"
                )
            net_flow_b = flow_analysis.five_day_net / 1_000_000_000
            factors.append(f"5-day net flow: {net_flow_b:.1f}B IDR")

        return factors

    def _get_risks(
        self,
        indicators: TechnicalIndicators,
        flow_analysis: Optional[FlowAnalysis],
        entry_price: float,
    ) -> List[str]:
        """Get risk factors for the signal.

        Args:
            indicators: Technical indicators.
            flow_analysis: Foreign flow analysis.
            entry_price: Current entry price.

        Returns:
            List of risk factor strings.
        """
        risks = []

        # High RSI
        if indicators.rsi > 70:
            risks.append(f"RSI overbought ({indicators.rsi:.1f})")

        # High volatility
        if indicators.atr_pct > 4.0:
            risks.append(f"High volatility (ATR: {indicators.atr_pct:.1f}%)")

        # Near resistance
        if indicators.resistance:
            dist_to_resistance = (indicators.resistance - entry_price) / entry_price * 100
            if dist_to_resistance < 3:
                risks.append(f"Near resistance ({dist_to_resistance:.1f}% away)")

        # Foreign selling
        if flow_analysis:
            if flow_analysis.signal in [FlowSignal.SELL, FlowSignal.STRONG_SELL]:
                risks.append("Foreign selling pressure")

        # Sideways market
        if indicators.trend == "sideways":
            risks.append("Market in consolidation")

        return risks


def generate_signals_for_universe(
    universe: List[str],
    ohlcv_data: Dict[str, List[OHLCV]],
    flow_analyses: Dict[str, FlowAnalysis],
    config: ModeConfig,
) -> List[Signal]:
    """Generate signals for multiple symbols.

    Args:
        universe: List of symbols to analyze.
        ohlcv_data: Dictionary of symbol to OHLCV data.
        flow_analyses: Dictionary of symbol to flow analysis.
        config: Trading mode configuration.

    Returns:
        List of generated signals, sorted by composite score.
    """
    generator = SignalGenerator(config)
    signals: List[Signal] = []

    for symbol in universe:
        if symbol not in ohlcv_data:
            continue

        flow = flow_analyses.get(symbol)

        try:
            signal = generator.generate(
                symbol=symbol,
                ohlcv_data=ohlcv_data[symbol],
                flow_analysis=flow,
            )

            if signal and signal.signal_type == SignalType.BUY:
                signals.append(signal)

        except Exception as e:
            logger.error(f"Error generating signal for {symbol}: {e}")

    # Sort by composite score (highest first)
    signals.sort(key=lambda s: s.composite_score, reverse=True)

    return signals
