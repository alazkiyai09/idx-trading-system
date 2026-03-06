"""
TimesFM Forecaster Module

Uses Google's TimesFM foundation model for time series forecasting.
Provides probabilistic price forecasts with quantile predictions.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ForecastConfig:
    """Configuration for TimesFM forecaster.

    Attributes:
        model_name: Name of the TimesFM model to use.
        max_context_len: Maximum context length for the model.
        horizon_len: Forecast horizon (number of steps to predict).
        device: Device to run inference on ("auto", "cpu", "cuda").
        use_log_returns: Whether to use log returns for forecasting.
    """

    model_name: str = "google/timesfm-2.5-200m"
    max_context_len: int = 512
    horizon_len: int = 32
    device: str = "auto"
    use_log_returns: bool = True


@dataclass
class PriceForecast:
    """Price forecast with quantile predictions.

    Contains probabilistic forecasts at different quantiles,
    along with derived metrics for trading decisions.

    Attributes:
        symbol: Stock symbol.
        timestamp: When the forecast was generated.
        current_price: Current price of the stock.
        forecast_horizon: Number of steps forecasted.
        q10: 10th percentile forecast (worst case).
        q30: 30th percentile forecast.
        q50: 50th percentile (median/point forecast).
        q70: 70th percentile forecast.
        q90: 90th percentile forecast (best case).
        point_forecast: Point forecast (q50).
        expected_return: Expected return as decimal.
        upside_pct: Upside potential to q90 as percentage.
        downside_pct: Downside risk to q10 as percentage.
        uncertainty_pct: Uncertainty (q90 - q10) / current_price.
        stop_loss_price: Suggested stop loss (q10).
        target_1_price: First target (q50).
        target_2_price: Second target (q70).
        target_3_price: Third target (q90).
    """

    symbol: str
    timestamp: datetime
    current_price: float
    forecast_horizon: int

    # Quantile forecasts
    q10: float
    q30: float
    q50: float
    q70: float
    q90: float

    # Derived metrics (computed in __post_init__)
    point_forecast: float = field(init=False)
    expected_return: float = field(init=False)
    upside_pct: float = field(init=False)
    downside_pct: float = field(init=False)
    uncertainty_pct: float = field(init=False)
    stop_loss_price: float = field(init=False)
    target_1_price: float = field(init=False)
    target_2_price: float = field(init=False)
    target_3_price: float = field(init=False)

    def __post_init__(self) -> None:
        """Compute derived metrics from quantile forecasts."""
        # Point forecast is the median
        self.point_forecast = self.q50

        # Expected return (as decimal)
        self.expected_return = (self.q50 - self.current_price) / self.current_price

        # Upside to q90 (best case)
        self.upside_pct = (self.q90 - self.current_price) / self.current_price

        # Downside to q10 (worst case)
        self.downside_pct = (self.q10 - self.current_price) / self.current_price

        # Uncertainty (spread between q90 and q10)
        self.uncertainty_pct = (self.q90 - self.q10) / self.current_price

        # Trading levels
        self.stop_loss_price = self.q10
        self.target_1_price = self.q50
        self.target_2_price = self.q70
        self.target_3_price = self.q90


class TimesFMForecaster:
    """TimesFM-based price forecaster.

    Uses Google's TimesFM foundation model for probabilistic
    time series forecasting. Supports lazy model loading and
    graceful degradation when TimesFM is unavailable.

    Example:
        forecaster = TimesFMForecaster(ForecastConfig())
        forecast = forecaster.forecast_price(
            symbol="BBCA",
            prices=[1000, 1010, 1020, ...],
            horizon=16
        )
        if forecast:
            print(f"Expected return: {forecast.expected_return:.2%}")
    """

    # Quantiles to forecast
    QUANTILES = [0.1, 0.3, 0.5, 0.7, 0.9]

    def __init__(self, config: Optional[ForecastConfig] = None) -> None:
        """Initialize TimesFM forecaster.

        Args:
            config: Forecast configuration. Uses defaults if None.
        """
        self.config = config or ForecastConfig()
        self._model: Optional[object] = None
        self._model_loaded = False
        self._load_attempted = False

    def forecast_price(
        self,
        symbol: str,
        prices: List[float],
        horizon: Optional[int] = None,
    ) -> Optional[PriceForecast]:
        """Generate price forecast for a single symbol.

        Args:
            symbol: Stock symbol.
            prices: Historical price series (most recent last).
            horizon: Forecast horizon. Uses config default if None.

        Returns:
            PriceForecast if successful, None if unavailable.
        """
        horizon = horizon or self.config.horizon_len

        if len(prices) < 10:
            logger.warning(f"Insufficient data for forecast: {symbol} ({len(prices)} points)")
            return None

        current_price = float(prices[-1])

        try:
            # Load model if not already loaded
            if not self._model_loaded:
                self._load_model()
                if not self._model_loaded:
                    return None

            # Convert to numpy array
            price_array = np.array(prices, dtype=np.float64)

            # Prepare input data
            if self.config.use_log_returns:
                input_data = self._prices_to_log_returns(price_array)
                # Normalize returns (zero mean, unit variance)
                input_data = self._normalize(input_data)
            else:
                # Normalize prices
                input_data = self._normalize(price_array)

            # Truncate to max context length
            if len(input_data) > self.config.max_context_len:
                input_data = input_data[-self.config.max_context_len :]

            # Generate forecast
            forecast_result = self._forecast(input_data, horizon)

            if forecast_result is None:
                return None

            # Denormalize and convert back to prices
            if self.config.use_log_returns:
                # Forecast returns, then convert to prices
                forecast_returns = self._denormalize(forecast_result, input_data)
                forecast_prices = self._returns_to_prices(forecast_returns, current_price)
            else:
                # Forecast prices directly
                forecast_prices = self._denormalize(forecast_result, input_data)

            # Extract quantile forecasts at horizon
            q10 = float(forecast_prices[0, -1])  # 10th percentile
            q30 = float(forecast_prices[1, -1])  # 30th percentile
            q50 = float(forecast_prices[2, -1])  # 50th percentile (median)
            q70 = float(forecast_prices[3, -1])  # 70th percentile
            q90 = float(forecast_prices[4, -1])  # 90th percentile

            # Validate forecasts
            if not all(q > 0 for q in [q10, q30, q50, q70, q90]):
                logger.warning(f"Invalid forecast values for {symbol}")
                return None

            return PriceForecast(
                symbol=symbol,
                timestamp=datetime.now(),
                current_price=current_price,
                forecast_horizon=horizon,
                q10=q10,
                q30=q30,
                q50=q50,
                q70=q70,
                q90=q90,
            )

        except Exception as e:
            logger.error(f"Error forecasting price for {symbol}: {e}")
            return None

    def forecast_batch(
        self,
        symbols: List[str],
        price_data: Dict[str, List[float]],
        horizon: Optional[int] = None,
    ) -> Dict[str, PriceForecast]:
        """Generate price forecasts for multiple symbols.

        Args:
            symbols: List of stock symbols.
            price_data: Dictionary mapping symbols to price series.
            horizon: Forecast horizon. Uses config default if None.

        Returns:
            Dictionary mapping symbols to PriceForecast objects.
        """
        forecasts: Dict[str, PriceForecast] = {}

        for symbol in symbols:
            if symbol not in price_data:
                logger.warning(f"No price data for {symbol}")
                continue

            forecast = self.forecast_price(symbol, price_data[symbol], horizon)
            if forecast:
                forecasts[symbol] = forecast

        logger.info(f"Generated {len(forecasts)}/{len(symbols)} forecasts")
        return forecasts

    def _prices_to_log_returns(self, prices: np.ndarray) -> np.ndarray:
        """Convert prices to log returns.

        Args:
            prices: Array of prices.

        Returns:
            Array of log returns (one shorter than input).
        """
        return np.log(prices[1:] / prices[:-1])

    def _returns_to_prices(
        self, log_returns: np.ndarray, last_price: float
    ) -> np.ndarray:
        """Convert log returns back to prices.

        Args:
            log_returns: Array of log returns.
            last_price: Last known price (anchor point).

        Returns:
            Array of forecasted prices.
        """
        prices = np.empty(len(log_returns) + 1, dtype=np.float64)
        prices[0] = last_price

        for i in range(len(log_returns)):
            prices[i + 1] = prices[i] * np.exp(log_returns[i])

        return prices

    def _normalize(self, data: np.ndarray) -> np.ndarray:
        """Normalize data to zero mean and unit variance.

        Args:
            data: Input data array.

        Returns:
            Normalized data array.
        """
        if len(data) == 0:
            return data

        mean = np.mean(data)
        std = np.std(data)

        if std == 0:
            return data - mean

        return (data - mean) / std

    def _denormalize(
        self, normalized_data: np.ndarray, reference_data: np.ndarray
    ) -> np.ndarray:
        """Denormalize data using reference statistics.

        Args:
            normalized_data: Normalized data.
            reference_data: Original data for statistics.

        Returns:
            Denormalized data array.
        """
        if len(reference_data) == 0:
            return normalized_data

        mean = np.mean(reference_data)
        std = np.std(reference_data)

        if std == 0:
            return normalized_data + mean

        return normalized_data * std + mean

    def _forecast(
        self, input_data: np.ndarray, horizon: int
    ) -> Optional[np.ndarray]:
        """Generate forecast using TimesFM model.

        Args:
            input_data: Normalized input time series.
            horizon: Forecast horizon.

        Returns:
            Array of quantile forecasts (n_quantiles, horizon) or None.
        """
        if self._model is None:
            return None

        try:
            # Try calling TimesFM forecast
            # TimesFM expects shape (batch, seq_len) or (seq_len,)
            if input_data.ndim == 1:
                input_data = input_data.reshape(1, -1)

            # Call the model's forecast method
            result = self._model.forecast(
                input_data,
                freq=[0],  # Default frequency
                horizon=horizon,
                quantiles=self.QUANTILES,
            )

            # Result shape varies by TimesFM version
            # Expected: (batch, horizon, quantiles) or similar
            if result is None or len(result) == 0:
                return None

            # Extract quantile forecasts
            # Reshape to (n_quantiles, horizon)
            forecast_array = np.array(result)

            if forecast_array.ndim == 3:
                # (batch, horizon, quantiles) -> (quantiles, horizon)
                forecast_array = forecast_array[0].T
            elif forecast_array.ndim == 2:
                if forecast_array.shape[0] == len(self.QUANTILES):
                    # Already (quantiles, horizon)
                    pass
                else:
                    # (horizon, quantiles) -> (quantiles, horizon)
                    forecast_array = forecast_array.T

            return forecast_array

        except Exception as e:
            logger.error(f"TimesFM forecast error: {e}")
            return None

    def _load_model(self) -> None:
        """Lazy load TimesFM model.

        Model is only loaded when first needed to avoid startup delay.
        """
        if self._load_attempted:
            return

        self._load_attempted = True

        try:
            from timesfm import TimesFm

            logger.info(f"Loading TimesFM model: {self.config.model_name}")

            # Determine device
            device = self.config.device
            if device == "auto":
                try:
                    import torch

                    device = "cuda" if torch.cuda.is_available() else "cpu"
                except ImportError:
                    device = "cpu"

            # Initialize TimesFM
            self._model = TimesFm(
                h=self.config.max_context_len,
                horizon=self.config.horizon_len,
                num_patches=-1,
                backend=device,
            )

            # Load pre-trained weights
            self._model.load_from_checkpoint(repo_id=self.config.model_name)

            self._model_loaded = True
            logger.info(f"TimesFM model loaded successfully on {device}")

        except ImportError as e:
            logger.warning(f"TimesFM not available: {e}")
            logger.info("Forecasting will be disabled. Install with: pip install timesfm")
        except Exception as e:
            logger.error(f"Failed to load TimesFM model: {e}")
            logger.info("Forecasting will be disabled due to model loading error")

    def is_available(self) -> bool:
        """Check if TimesFM is available.

        Returns:
            True if TimesFM is loaded and available.
        """
        if not self._load_attempted:
            self._load_model()
        return self._model_loaded


class ForecastScorer:
    """Converts price forecasts to 0-100 scores.

    Scores forecasts based on expected return, uncertainty,
    and directional conviction. Higher scores indicate better
    risk-adjusted opportunities.

    Example:
        scorer = ForecastScorer()
        forecast = forecaster.forecast_price("BBCA", prices)
        if forecast:
            score = scorer.score_forecast(forecast)
            print(f"Forecast score: {score:.1f}/100")
    """

    # Scoring weights
    RETURN_WEIGHT = 0.5
    UNCERTAINTY_PENALTY_WEIGHT = 0.3
    DIRECTION_WEIGHT = 0.2

    # Score ranges
    MIN_EXPECTED_RETURN = -0.10  # -10%
    MAX_EXPECTED_RETURN = 0.20  # +20%
    MAX_ACCEPTABLE_UNCERTAINTY = 0.15  # 15%

    def score_forecast(self, forecast: PriceForecast) -> float:
        """Score a forecast from 0-100.

        Args:
            forecast: Price forecast to score.

        Returns:
            Score from 0-100, where higher is better.
        """
        # Score based on expected return (0-50 points)
        return_score = self._score_return(forecast.expected_return)

        # Score based on uncertainty (0-30 points, inverted)
        uncertainty_score = self._score_uncertainty(forecast.uncertainty_pct)

        # Score based on directional conviction (0-20 points)
        direction_score = self._score_direction(forecast)

        total_score = (
            return_score * self.RETURN_WEIGHT
            + uncertainty_score * self.UNCERTAINTY_PENALTY_WEIGHT
            + direction_score * self.DIRECTION_WEIGHT
        )

        return max(0.0, min(100.0, total_score))

    def _score_return(self, expected_return: float) -> float:
        """Score expected return component.

        Args:
            expected_return: Expected return as decimal.

        Returns:
            Score from 0-100.
        """
        # Normalize return to 0-100 range
        range_size = self.MAX_EXPECTED_RETURN - self.MIN_EXPECTED_RETURN
        normalized = (expected_return - self.MIN_EXPECTED_RETURN) / range_size
        return max(0.0, min(100.0, normalized * 100))

    def _score_uncertainty(self, uncertainty_pct: float) -> float:
        """Score uncertainty component (lower is better).

        Args:
            uncertainty_pct: Uncertainty as percentage (q90 - q10) / price.

        Returns:
            Score from 0-100 (inverted - high uncertainty = low score).
        """
        # Invert: low uncertainty = high score
        normalized = uncertainty_pct / self.MAX_ACCEPTABLE_UNCERTAINTY
        score = 100 * (1 - normalized)
        return max(0.0, min(100.0, score))

    def _score_direction(self, forecast: PriceForecast) -> float:
        """Score directional conviction.

        Args:
            forecast: Price forecast.

        Returns:
            Score from 0-100 based on conviction.
        """
        # Measure conviction by how much q50 differs from current
        # relative to the uncertainty band
        conviction = abs(forecast.expected_return) / forecast.uncertainty_pct

        # Normalize: conviction of 1+ means good signal
        score = min(100.0, conviction * 50)

        # Bonus for bullish bias (upside >> downside)
        upside_downside_ratio = abs(forecast.upside_pct / forecast.downside_pct)
        if forecast.upside_pct > 0 and forecast.downside_pct < 0:
            if upside_downside_ratio > 2:
                score = min(100.0, score * 1.2)

        return score


def create_forecaster(config: Optional[ForecastConfig] = None) -> TimesFMForecaster:
    """Factory function to create a TimesFM forecaster.

    Args:
        config: Optional forecast configuration.

    Returns:
        Configured TimesFMForecaster instance.
    """
    return TimesFMForecaster(config)


def create_scorer() -> ForecastScorer:
    """Factory function to create a forecast scorer.

    Returns:
        Configured ForecastScorer instance.
    """
    return ForecastScorer()
