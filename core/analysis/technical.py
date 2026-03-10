"""
Technical Analysis Module

Calculates technical indicators for IDX stocks.
Uses pandas-ta for most indicators.
"""

import logging
from dataclasses import dataclass
from datetime import date
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from core.data.models import OHLCV, TechnicalIndicators

logger = logging.getLogger(__name__)


@dataclass
class TechnicalScore:
    """Technical analysis score.

    Attributes:
        score: Overall technical score (0-100).
        trend_score: Trend direction score.
        momentum_score: Momentum indicator score.
        volume_score: Volume analysis score.
        volatility_score: Volatility analysis score.
        trend: Current trend direction.
        signal: Overall signal (bullish, bearish, neutral).
    """

    score: float
    trend_score: float
    momentum_score: float
    volume_score: float
    volatility_score: float
    trend: str
    signal: str


class TechnicalAnalyzer:
    """Calculates technical indicators.

    This class computes various technical indicators used for
    trading signal generation.

    Indicators calculated:
    - Moving Averages: EMA20, EMA50, SMA200
    - Momentum: RSI, MACD
    - Volatility: ATR, Bollinger Bands
    - Volume: Volume SMA, Volume Ratio
    - Trend: Based on MA alignment
    - Support/Resistance: Rolling highs/lows

    Example:
        analyzer = TechnicalAnalyzer()
        indicators = analyzer.calculate("BBCA", ohlcv_list)
        latest = indicators[-1]
        print(f"RSI: {latest.rsi}, Trend: {latest.trend}")
    """

    def __init__(self) -> None:
        """Initialize technical analyzer."""
        pass

    def calculate(self, symbol: str, ohlcv_list: List[OHLCV]) -> List[TechnicalIndicators]:
        """Calculate all technical indicators.

        Args:
            symbol: Stock symbol.
            ohlcv_list: List of OHLCV data sorted by date ascending.

        Returns:
            List of TechnicalIndicators for each bar with sufficient data.
        """
        if len(ohlcv_list) < 50:
            logger.warning(
                f"Insufficient data for technical analysis: {len(ohlcv_list)} bars"
            )
            return []

        # Convert to DataFrame
        df = self._to_dataframe(ohlcv_list)

        # Calculate indicators
        df = self._calculate_moving_averages(df)
        df = self._calculate_momentum(df)
        df = self._calculate_volatility(df)
        df = self._calculate_volume(df)
        df = self._determine_trend(df)
        df = self._find_support_resistance(df)

        # Convert back to dataclass
        return self._to_indicators(df, symbol)

    def _to_dataframe(self, ohlcv_list: List[OHLCV]) -> pd.DataFrame:
        """Convert OHLCV list to DataFrame.

        Args:
            ohlcv_list: List of OHLCV data.

        Returns:
            DataFrame with OHLCV data indexed by date.
        """
        data = {
            "date": [o.timestamp for o in ohlcv_list],
            "open": [o.open for o in ohlcv_list],
            "high": [o.high for o in ohlcv_list],
            "low": [o.low for o in ohlcv_list],
            "close": [o.close for o in ohlcv_list],
            "volume": [o.volume for o in ohlcv_list],
        }
        df = pd.DataFrame(data)
        df.set_index("date", inplace=True)
        return df

    def _calculate_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate moving averages.

        Computes EMA20, EMA50, and SMA200.

        Args:
            df: DataFrame with price data.

        Returns:
            DataFrame with moving average columns added.
        """
        # EMA calculations using pandas
        df["ema_20"] = df["close"].ewm(span=20, adjust=False).mean()
        df["ema_50"] = df["close"].ewm(span=50, adjust=False).mean()
        df["sma_200"] = df["close"].rolling(window=200).mean()
        return df

    def _calculate_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate momentum indicators.

        Computes RSI and MACD indicators.

        Args:
            df: DataFrame with price data.

        Returns:
            DataFrame with momentum columns added.
        """
        # RSI (14-period)
        df["rsi"] = self._calculate_rsi(df["close"], period=14)

        # MACD (12, 26, 9)
        macd_result = self._calculate_macd(df["close"], fast=12, slow=26, signal=9)
        df["macd"] = macd_result["macd"]
        df["macd_signal"] = macd_result["signal"]
        df["macd_hist"] = macd_result["histogram"]

        return df

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index.

        Args:
            prices: Series of closing prices.
            period: RSI period (default 14).

        Returns:
            Series with RSI values.
        """
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)

        avg_gain = gain.rolling(window=period, min_periods=period).mean()
        avg_loss = loss.rolling(window=period, min_periods=period).mean()

        # Use exponential smoothing for subsequent values
        avg_gain = avg_gain.combine_first(gain.ewm(alpha=1 / period, adjust=False).mean())
        avg_loss = avg_loss.combine_first(loss.ewm(alpha=1 / period, adjust=False).mean())

        rs = avg_gain / avg_loss.replace(0, np.inf)
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def _calculate_macd(
        self,
        prices: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> dict:
        """Calculate MACD indicator.

        Args:
            prices: Series of closing prices.
            fast: Fast EMA period.
            slow: Slow EMA period.
            signal: Signal line period.

        Returns:
            Dictionary with MACD, signal, and histogram values.
        """
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line

        return {
            "macd": macd_line,
            "signal": signal_line,
            "histogram": histogram,
        }

    def _calculate_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volatility indicators.

        Computes ATR and Bollinger Bands.

        Args:
            df: DataFrame with OHLC data.

        Returns:
            DataFrame with volatility columns added.
        """
        # ATR (14-period)
        df["atr"] = self._calculate_atr(df, period=14)
        df["atr_pct"] = (df["atr"] / df["close"]) * 100

        # Bollinger Bands (20, 2)
        bb_result = self._calculate_bollinger_bands(df["close"], period=20, std_dev=2)
        df["bb_upper"] = bb_result["upper"]
        df["bb_middle"] = bb_result["middle"]
        df["bb_lower"] = bb_result["lower"]

        return df

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range.

        Args:
            df: DataFrame with high, low, close columns.
            period: ATR period.

        Returns:
            Series with ATR values.
        """
        high = df["high"]
        low = df["low"]
        close = df["close"]

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))

        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=period, min_periods=period).mean()

        return atr

    def _calculate_bollinger_bands(
        self,
        prices: pd.Series,
        period: int = 20,
        std_dev: float = 2.0,
    ) -> dict:
        """Calculate Bollinger Bands.

        Args:
            prices: Series of closing prices.
            period: Moving average period.
            std_dev: Standard deviation multiplier.

        Returns:
            Dictionary with upper, middle, lower band values.
        """
        middle = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()

        return {
            "upper": middle + (std * std_dev),
            "middle": middle,
            "lower": middle - (std * std_dev),
        }

    def _calculate_volume(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volume indicators.

        Computes volume SMA and volume ratio.

        Args:
            df: DataFrame with volume data.

        Returns:
            DataFrame with volume columns added.
        """
        df["volume_sma_20"] = df["volume"].rolling(window=20).mean()
        df["volume_ratio"] = df["volume"] / df["volume_sma_20"].replace(0, np.inf)
        return df

    def _determine_trend(self, df: pd.DataFrame) -> pd.DataFrame:
        """Determine trend based on MA alignment.

        Args:
            df: DataFrame with price and MA data.

        Returns:
            DataFrame with trend column added.
        """

        def get_trend(row: pd.Series) -> str:
            if pd.isna(row["ema_20"]) or pd.isna(row["ema_50"]):
                return "sideways"

            price = row["close"]
            ema20 = row["ema_20"]
            ema50 = row["ema_50"]

            if price > ema20 > ema50:
                return "uptrend"
            elif price < ema20 < ema50:
                return "downtrend"
            else:
                return "sideways"

        df["trend"] = df.apply(get_trend, axis=1)
        return df

    def _find_support_resistance(
        self,
        df: pd.DataFrame,
        lookback: int = 20,
    ) -> pd.DataFrame:
        """Find support and resistance levels.

        Uses rolling window to find local highs and lows.

        Args:
            df: DataFrame with high/low data.
            lookback: Window size for finding levels.

        Returns:
            DataFrame with support/resistance columns added.
        """
        df["support"] = df["low"].rolling(window=lookback).min()
        df["resistance"] = df["high"].rolling(window=lookback).max()
        return df

    def _to_indicators(
        self,
        df: pd.DataFrame,
        symbol: str,
    ) -> List[TechnicalIndicators]:
        """Convert DataFrame to list of TechnicalIndicators.

        Args:
            df: DataFrame with all calculated indicators.
            symbol: Stock symbol.

        Returns:
            List of TechnicalIndicators objects.
        """
        result: List[TechnicalIndicators] = []

        for date_idx, row in df.iterrows():
            # Skip rows with insufficient data
            if pd.isna(row["ema_20"]):
                continue

            timestamp = date_idx.to_pydatetime() if hasattr(date_idx, "to_pydatetime") else date_idx
            indicators = TechnicalIndicators(
                timestamp=timestamp,
                symbol=symbol,
                close=float(row["close"]),
                volume=int(row["volume"]),
                ema_20=float(row["ema_20"]),
                ema_50=float(row["ema_50"]),
                sma_200=float(row["sma_200"]) if pd.notna(row.get("sma_200")) else None,
                rsi=float(row["rsi"]),
                macd=float(row["macd"]),
                macd_signal=float(row["macd_signal"]),
                macd_hist=float(row["macd_hist"]),
                atr=float(row["atr"]),
                atr_pct=float(row["atr_pct"]),
                bb_upper=float(row["bb_upper"]),
                bb_middle=float(row["bb_middle"]),
                bb_lower=float(row["bb_lower"]),
                volume_sma_20=float(row["volume_sma_20"]),
                volume_ratio=float(row["volume_ratio"]),
                trend=str(row["trend"]),
                support=float(row["support"]) if pd.notna(row.get("support")) else None,
                resistance=float(row["resistance"]) if pd.notna(row.get("resistance")) else None,
            )
            result.append(indicators)

        return result

    def get_current_indicators(
        self, symbol: str, ohlcv_list: List[OHLCV]
    ) -> Optional[TechnicalIndicators]:
        """Get indicators for the most recent bar.

        Args:
            symbol: Stock symbol.
            ohlcv_list: List of OHLCV data.

        Returns:
            TechnicalIndicators for the latest bar, or None if insufficient data.
        """
        indicators = self.calculate(symbol, ohlcv_list)
        return indicators[-1] if indicators else None

    def calculate_score(
        self,
        indicators: TechnicalIndicators,
    ) -> TechnicalScore:
        """Calculate overall technical score.

        Computes a composite score based on trend, momentum,
        volume, and volatility analysis.

        Args:
            indicators: Technical indicators for analysis.

        Returns:
            TechnicalScore with breakdown by category.
        """
        # Trend score (0-100)
        if indicators.trend == "uptrend":
            trend_score = 80.0
        elif indicators.trend == "downtrend":
            trend_score = 20.0
        else:
            trend_score = 50.0

        # Momentum score based on RSI (0-100)
        if indicators.rsi < 30:
            # Oversold - potential buy
            momentum_score = 70.0
        elif indicators.rsi > 70:
            # Overbought - potential sell
            momentum_score = 30.0
        else:
            # Neutral zone - higher is more bullish
            momentum_score = indicators.rsi

        # MACD contribution
        if indicators.macd_hist > 0:
            momentum_score = min(100, momentum_score + 10)
        elif indicators.macd_hist < 0:
            momentum_score = max(0, momentum_score - 10)

        # Volume score (0-100)
        if indicators.volume_ratio > 1.5:
            # High volume - confirms move
            volume_score = 75.0
        elif indicators.volume_ratio > 1.0:
            volume_score = 60.0
        else:
            volume_score = 40.0

        # Volatility score (0-100) - moderate volatility preferred
        if 1.5 <= indicators.atr_pct <= 3.0:
            volatility_score = 70.0
        elif indicators.atr_pct < 1.5:
            volatility_score = 50.0  # Too low
        else:
            volatility_score = 40.0  # Too high

        # Calculate overall score
        overall_score = (
            trend_score * 0.30
            + momentum_score * 0.35
            + volume_score * 0.20
            + volatility_score * 0.15
        )

        # Determine overall signal
        if overall_score >= 65:
            signal = "bullish"
        elif overall_score <= 35:
            signal = "bearish"
        else:
            signal = "neutral"

        return TechnicalScore(
            score=overall_score,
            trend_score=trend_score,
            momentum_score=momentum_score,
            volume_score=volume_score,
            volatility_score=volatility_score,
            trend=indicators.trend,
            signal=signal,
        )
