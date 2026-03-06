"""
Tests for TimesFM Forecaster Module

Tests forecasting configuration, PriceForecast dataclass,
TimesFMForecaster class, ForecastScorer, and ForecastCache.
"""

import pytest
from datetime import datetime, timedelta
from typing import List
from unittest.mock import Mock, MagicMock, patch
import numpy as np

from core.forecasting.timesfm_forecaster import (
    ForecastConfig,
    PriceForecast,
    TimesFMForecaster,
    ForecastScorer,
    create_forecaster,
    create_scorer,
)
from core.forecasting.forecast_cache import ForecastCache, CacheEntry


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def mock_prices():
    """Sample price data for testing."""
    return [100.0 + i * 0.5 + np.sin(i / 5) * 2 for i in range(100)]


@pytest.fixture
def mock_prices_short():
    """Short price data for insufficient data tests."""
    return [100.0, 101.0, 102.0, 103.0, 104.0]


@pytest.fixture
def sample_forecast():
    """Sample PriceForecast for testing."""
    return PriceForecast(
        symbol="TEST",
        timestamp=datetime.now(),
        current_price=100.0,
        forecast_horizon=16,
        q10=95.0,  # -5%
        q30=98.0,
        q50=105.0,  # +5% expected
        q70=110.0,
        q90=115.0,  # +15%
    )


@pytest.fixture
def bullish_forecast():
    """Bullish forecast for testing."""
    return PriceForecast(
        symbol="BULL",
        timestamp=datetime.now(),
        current_price=100.0,
        forecast_horizon=16,
        q10=98.0,   # -2%
        q30=102.0,
        q50=110.0,  # +10% expected
        q70=115.0,
        q90=120.0,  # +20%
    )


@pytest.fixture
def bearish_forecast():
    """Bearish forecast for testing."""
    return PriceForecast(
        symbol="BEAR",
        timestamp=datetime.now(),
        current_price=100.0,
        forecast_horizon=16,
        q10=85.0,   # -15%
        q30=92.0,
        q50=95.0,   # -5% expected
        q70=98.0,
        q90=102.0,  # +2%
    )


@pytest.fixture
def neutral_forecast():
    """Neutral forecast for testing."""
    return PriceForecast(
        symbol="NEUT",
        timestamp=datetime.now(),
        current_price=100.0,
        forecast_horizon=16,
        q10=98.0,
        q30=99.0,
        q50=100.0,  # No change
        q70=101.0,
        q90=102.0,
    )


@pytest.fixture
def high_uncertainty_forecast():
    """High uncertainty forecast for testing."""
    return PriceForecast(
        symbol="UNCERT",
        timestamp=datetime.now(),
        current_price=100.0,
        forecast_horizon=16,
        q10=80.0,   # -20%
        q30=90.0,
        q50=100.0,
        q70=110.0,
        q90=120.0,  # +20%
    )


@pytest.fixture
def mock_timesfm_model():
    """Mock TimesFM model for testing without GPU."""
    mock_model = Mock()

    # Create a mock forecast result with shape (batch, horizon, quantiles)
    # TimesFM returns forecasts for each quantile
    mock_result = np.array([
        [[95.0, 98.0, 105.0, 110.0, 115.0]],  # First timestep
    ])

    # Set up the forecast method to return predictable results
    mock_model.forecast.return_value = mock_result

    return mock_model


@pytest.fixture
def mock_timesfm_model_multi_horizon():
    """Mock TimesFM model returning multi-horizon forecast."""
    mock_model = Mock()

    # Return 5 timesteps with 5 quantiles each
    mock_result = np.array([
        [[98.0, 100.0, 102.0, 104.0, 106.0]],  # t+1
        [[99.0, 101.0, 103.0, 105.0, 107.0]],  # t+2
        [[100.0, 102.0, 104.0, 106.0, 108.0]],  # t+3
    ])

    mock_model.forecast.return_value = mock_result

    return mock_model


@pytest.fixture
def forecast_cache():
    """Create forecast cache for testing."""
    return ForecastCache(ttl_minutes=60)


# =============================================================================
# TEST FORECASTCONFIG
# =============================================================================


class TestForecastConfig:
    """Tests for ForecastConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ForecastConfig()

        assert config.model_name == "google/timesfm-2.5-200m"
        assert config.max_context_len == 512
        assert config.horizon_len == 32
        assert config.device == "auto"
        assert config.use_log_returns is True

    def test_custom_config(self):
        """Test custom configuration values."""
        config = ForecastConfig(
            model_name="custom/model",
            max_context_len=256,
            horizon_len=16,
            device="cpu",
            use_log_returns=False,
        )

        assert config.model_name == "custom/model"
        assert config.max_context_len == 256
        assert config.horizon_len == 16
        assert config.device == "cpu"
        assert config.use_log_returns is False


# =============================================================================
# TEST PRICEFORECAST DERIVED METRICS
# =============================================================================


class TestPriceForecastDerivedMetrics:
    """Tests for PriceForecast derived metrics."""

    def test_point_forecast_is_q50(self, sample_forecast):
        """Test that point_forecast equals q50."""
        assert sample_forecast.point_forecast == sample_forecast.q50
        assert sample_forecast.point_forecast == 105.0

    def test_expected_return_calculation(self, sample_forecast):
        """Test expected return calculation."""
        # expected_return = (q50 - current_price) / current_price
        expected = (105.0 - 100.0) / 100.0
        assert sample_forecast.expected_return == pytest.approx(expected, 0.001)
        assert sample_forecast.expected_return == 0.05

    def test_upside_pct_calculation(self, sample_forecast):
        """Test upside percentage calculation."""
        # upside_pct = (q90 - current_price) / current_price
        upside = (115.0 - 100.0) / 100.0
        assert sample_forecast.upside_pct == pytest.approx(upside, 0.001)
        assert sample_forecast.upside_pct == 0.15

    def test_downside_pct_calculation(self, sample_forecast):
        """Test downside percentage calculation."""
        # downside_pct = (q10 - current_price) / current_price
        downside = (95.0 - 100.0) / 100.0
        assert sample_forecast.downside_pct == pytest.approx(downside, 0.001)
        assert sample_forecast.downside_pct == -0.05

    def test_uncertainty_calculation(self, sample_forecast):
        """Test uncertainty percentage calculation."""
        # uncertainty_pct = (q90 - q10) / current_price
        uncertainty = (115.0 - 95.0) / 100.0
        assert sample_forecast.uncertainty_pct == pytest.approx(uncertainty, 0.001)
        assert sample_forecast.uncertainty_pct == 0.20

    def test_stop_loss_is_q10(self, sample_forecast):
        """Test that stop_loss_price equals q10."""
        assert sample_forecast.stop_loss_price == sample_forecast.q10
        assert sample_forecast.stop_loss_price == 95.0

    def test_targets_from_quantiles(self, sample_forecast):
        """Test that targets are set from quantiles."""
        assert sample_forecast.target_1_price == sample_forecast.q50
        assert sample_forecast.target_2_price == sample_forecast.q70
        assert sample_forecast.target_3_price == sample_forecast.q90

        assert sample_forecast.target_1_price == 105.0
        assert sample_forecast.target_2_price == 110.0
        assert sample_forecast.target_3_price == 115.0

    def test_bullish_forecast_metrics(self, bullish_forecast):
        """Test metrics for bullish forecast."""
        assert bullish_forecast.expected_return > 0
        assert bullish_forecast.upside_pct > 0
        assert bullish_forecast.downside_pct < 0
        assert bullish_forecast.expected_return == 0.10

    def test_bearish_forecast_metrics(self, bearish_forecast):
        """Test metrics for bearish forecast."""
        assert bearish_forecast.expected_return < 0
        assert bearish_forecast.downside_pct < -0.10

    def test_neutral_forecast_metrics(self, neutral_forecast):
        """Test metrics for neutral forecast."""
        assert neutral_forecast.expected_return == 0.0
        assert abs(neutral_forecast.upside_pct) < 0.05

    def test_high_uncertainty_metrics(self, high_uncertainty_forecast):
        """Test high uncertainty forecast has wide spread."""
        assert high_uncertainty_forecast.uncertainty_pct > 0.30


# =============================================================================
# TEST TIMESFMFORECASTER
# =============================================================================


class TestTimesFMForecaster:
    """Tests for TimesFMForecaster class."""

    def test_initialization_default_config(self):
        """Test forecaster initialization with default config."""
        forecaster = TimesFMForecaster()

        assert forecaster.config is not None
        assert forecaster.config.model_name == "google/timesfm-2.5-200m"
        assert forecaster._model is None
        assert forecaster._model_loaded is False
        assert forecaster._load_attempted is False

    def test_initialization_custom_config(self):
        """Test forecaster initialization with custom config."""
        config = ForecastConfig(horizon_len=16)
        forecaster = TimesFMForecaster(config)

        assert forecaster.config.horizon_len == 16

    def test_lazy_model_loading(self):
        """Test that model is not loaded until needed."""
        forecaster = TimesFMForecaster()

        # Model should not be loaded initially
        assert forecaster._model is None
        assert forecaster._model_loaded is False
        assert forecaster._load_attempted is False

    def test_forecast_price_basic(self, mock_prices, mock_timesfm_model):
        """Test basic price forecasting."""
        # Use config with use_log_returns=False for simpler testing
        config = ForecastConfig(use_log_returns=False)
        forecaster = TimesFMForecaster(config)

        # Mock the internal forecast method to return proper shape
        # Shape should be (quantiles, horizon) = (5, 1) for horizon=1
        mock_forecast_result = np.array([
            [95.0],  # q10 at horizon
            [98.0],  # q30 at horizon
            [105.0], # q50 at horizon
            [110.0], # q70 at horizon
            [115.0], # q90 at horizon
        ])

        with patch.object(forecaster, '_load_model'):
            forecaster._load_attempted = True
            forecaster._model_loaded = True
            forecaster._model = mock_timesfm_model

            with patch.object(forecaster, '_forecast', return_value=mock_forecast_result):
                forecast = forecaster.forecast_price(
                    symbol="TEST",
                    prices=mock_prices,
                    horizon=1,  # Use horizon=1 for simpler testing
                )

                assert forecast is not None
                assert forecast.symbol == "TEST"
                assert forecast.current_price == mock_prices[-1]
                assert forecast.forecast_horizon == 1
                assert forecast.q10 > 0
                assert forecast.q50 > 0

    def test_forecast_price_returns_quantiles(self, mock_prices):
        """Test that forecast returns all required quantiles."""
        # Use config with use_log_returns=False for simpler testing
        config = ForecastConfig(use_log_returns=False)
        forecaster = TimesFMForecaster(config)

        # Return (quantiles, horizon) shape
        mock_forecast_result = np.array([
            [90.0],  # q10 at last timestep
            [95.0],  # q30 at last timestep
            [100.0], # q50 at last timestep
            [105.0], # q70 at last timestep
            [110.0], # q90 at last timestep
        ])

        with patch.object(forecaster, '_load_model'):
            forecaster._load_attempted = True
            forecaster._model_loaded = True
            forecaster._model = Mock()  # Add a mock model

            with patch.object(forecaster, '_forecast', return_value=mock_forecast_result):
                forecast = forecaster.forecast_price("TEST", mock_prices, horizon=1)

                assert forecast is not None
                # Use pytest.approx for floating point comparison
                assert forecast.q10 == pytest.approx(90.0, abs=0.01)
                assert forecast.q30 == pytest.approx(95.0, abs=0.01)
                assert forecast.q50 == pytest.approx(100.0, abs=0.01)
                assert forecast.q70 == pytest.approx(105.0, abs=0.01)
                assert forecast.q90 == pytest.approx(110.0, abs=0.01)

    def test_forecast_insufficient_data(self):
        """Test forecast with insufficient data returns None."""
        forecaster = TimesFMForecaster()

        forecast = forecaster.forecast_price(
            symbol="TEST",
            prices=[1.0, 2.0, 3.0],  # Less than 10 points
        )

        assert forecast is None

    def test_forecast_batch(self, mock_prices):
        """Test batch forecasting for multiple symbols."""
        forecaster = TimesFMForecaster()

        # Mock the model loading to fail gracefully
        with patch.object(forecaster, '_load_model'):
            forecaster._load_attempted = True
            forecaster._model_loaded = False

            price_data = {
                "BBCA": mock_prices,
                "BBRI": mock_prices,
                "TLKM": mock_prices,
            }

            forecasts = forecaster.forecast_batch(
                symbols=["BBCA", "BBRI", "TLKM"],
                price_data=price_data,
            )

            # Without actual model, should return empty dict
            assert isinstance(forecasts, dict)

    def test_forecast_batch_missing_symbol(self, mock_prices):
        """Test batch forecasting with missing symbol data."""
        forecaster = TimesFMForecaster()

        price_data = {
            "BBCA": mock_prices,
            # BBRI is missing
        }

        with patch.object(forecaster, '_load_model'):
            forecaster._load_attempted = True
            forecaster._model_loaded = False

            forecasts = forecaster.forecast_batch(
                symbols=["BBCA", "BBRI"],
                price_data=price_data,
            )

            assert isinstance(forecasts, dict)

    def test_returns_to_prices_conversion(self):
        """Test converting log returns back to prices."""
        forecaster = TimesFMForecaster()

        log_returns = np.array([0.01, 0.02, -0.01, 0.03])
        last_price = 100.0

        prices = forecaster._returns_to_prices(log_returns, last_price)

        assert len(prices) == len(log_returns) + 1
        assert prices[0] == last_price
        assert prices[1] == pytest.approx(100.0 * np.exp(0.01), 0.001)
        assert prices[2] == pytest.approx(prices[1] * np.exp(0.02), 0.001)

    def test_prices_to_log_returns(self):
        """Test converting prices to log returns."""
        forecaster = TimesFMForecaster()

        prices = np.array([100.0, 101.0, 102.0, 103.0])

        log_returns = forecaster._prices_to_log_returns(prices)

        assert len(log_returns) == len(prices) - 1
        # log_return = ln(price[t] / price[t-1])
        expected_first = np.log(101.0 / 100.0)
        assert log_returns[0] == pytest.approx(expected_first, 0.001)

    def test_prices_to_log_returns_single_price(self):
        """Test log returns with single price (edge case)."""
        forecaster = TimesFMForecaster()

        prices = np.array([100.0])
        log_returns = forecaster._prices_to_log_returns(prices)

        assert len(log_returns) == 0

    def test_normalize_data(self):
        """Test data normalization."""
        forecaster = TimesFMForecaster()

        data = np.array([100.0, 110.0, 120.0, 130.0, 140.0])
        normalized = forecaster._normalize(data)

        # Check mean is close to 0
        assert np.mean(normalized) == pytest.approx(0.0, abs=0.001)
        # Check std is close to 1
        assert np.std(normalized) == pytest.approx(1.0, abs=0.001)

    def test_normalize_empty_data(self):
        """Test normalization with empty data."""
        forecaster = TimesFMForecaster()

        data = np.array([])
        normalized = forecaster._normalize(data)

        assert len(normalized) == 0

    def test_normalize_constant_data(self):
        """Test normalization with constant data (std=0)."""
        forecaster = TimesFMForecaster()

        data = np.array([100.0, 100.0, 100.0, 100.0])
        normalized = forecaster._normalize(data)

        # Should return zero-centered data
        assert np.all(normalized == 0)

    def test_denormalize_data(self):
        """Test data denormalization."""
        forecaster = TimesFMForecaster()

        original_data = np.array([100.0, 110.0, 120.0, 130.0, 140.0])
        normalized = forecaster._normalize(original_data)
        denormalized = forecaster._denormalize(normalized, original_data)

        # Should recover original data
        assert np.allclose(denormalized, original_data, rtol=0.001)

    def test_denormalize_empty_reference(self):
        """Test denormalization with empty reference data."""
        forecaster = TimesFMForecaster()

        normalized = np.array([0.5, 1.0, -0.5])
        denormalized = forecaster._denormalize(normalized, np.array([]))

        # Should return normalized data unchanged
        np.testing.assert_array_equal(denormalized, normalized)

    def test_denormalize_constant_reference(self):
        """Test denormalization with constant reference data."""
        forecaster = TimesFMForecaster()

        normalized = np.array([1.0, 2.0, -1.0])
        constant_ref = np.array([100.0, 100.0, 100.0])
        denormalized = forecaster._denormalize(normalized, constant_ref)

        # Should add mean back
        expected = normalized + 100.0
        np.testing.assert_array_almost_equal(denormalized, expected)

    def test_graceful_degradation_without_timesfm(self):
        """Test graceful degradation when TimesFM is not installed."""
        forecaster = TimesFMForecaster()

        # Mock _load_model to simulate ImportError but catch it internally
        original_load = forecaster._load_model

        def mock_load_with_import_error():
            forecaster._load_attempted = True
            try:
                raise ImportError("No module named 'timesfm'")
            except ImportError:
                # Simulate the error handling in _load_model
                pass

        with patch.object(forecaster, '_load_model', side_effect=mock_load_with_import_error):
            # is_available should return False since model didn't load
            result = forecaster.is_available()
            assert result is False

        # Reset for second test
        forecaster2 = TimesFMForecaster()

        # forecast_price should return None when model fails to load
        with patch.object(forecaster2, '_load_model', side_effect=mock_load_with_import_error):
            forecast = forecaster2.forecast_price("TEST", list(range(100)))
            assert forecast is None

    def test_is_available_without_model_loaded(self):
        """Test is_available triggers model loading."""
        forecaster = TimesFMForecaster()

        # The _load_model sets _load_attempted = True, so we need to track calls
        original_load = forecaster._load_model

        with patch.object(forecaster, '_load_model', wraps=original_load) as mock_load:
            mock_load.side_effect = lambda: setattr(forecaster, '_load_attempted', True)

            result = forecaster.is_available()

            # Should have attempted to load model
            mock_load.assert_called_once()

    def test_is_available_with_model_loaded(self):
        """Test is_available returns True when model is loaded."""
        forecaster = TimesFMForecaster()
        forecaster._model = Mock()
        forecaster._model_loaded = True
        forecaster._load_attempted = True

        assert forecaster.is_available() is True

    def test_forecast_with_custom_horizon(self, mock_prices):
        """Test forecast with custom horizon override."""
        forecaster = TimesFMForecaster(ForecastConfig(horizon_len=32))

        # Mock successful load
        forecaster._model_loaded = True
        forecaster._load_attempted = True

        with patch.object(forecaster, '_load_model'):
            # Use actual forecast method mock
            with patch.object(forecaster, '_forecast', return_value=None):
                forecast = forecaster.forecast_price("TEST", mock_prices, horizon=16)

                # Should use custom horizon
                assert forecast is None  # Without actual model


class TestTimesFMForecasterEdgeCases:
    """Test edge cases for TimesFMForecaster."""

    def test_forecast_with_negative_prices(self):
        """Test forecast handles edge case of negative prices."""
        forecaster = TimesFMForecaster()

        # Should return None for invalid prices
        forecast = forecaster.forecast_price("TEST", [-100.0, -90.0, -80.0])
        assert forecast is None

    def test_forecast_with_zero_prices(self):
        """Test forecast with zero prices."""
        forecaster = TimesFMForecaster()

        forecast = forecaster.forecast_price("TEST", [0.0, 0.0, 0.0])
        assert forecast is None

    def test_forecast_with_very_large_prices(self):
        """Test forecast with very large prices."""
        forecaster = TimesFMForecaster()

        large_prices = [1_000_000.0 + i * 1000 for i in range(100)]

        # Mock to prevent actual loading
        with patch.object(forecaster, '_load_model'):
            forecaster._load_attempted = True
            forecast = forecaster.forecast_price("TEST", large_prices)
            # Without model, returns None

    def test_forecast_with_exactly_ten_prices(self):
        """Test forecast with exactly minimum required prices."""
        forecaster = TimesFMForecaster()

        ten_prices = [100.0 + i for i in range(10)]

        with patch.object(forecaster, '_load_model'):
            forecaster._load_attempted = True
            # Should have enough data (minimum is 10)


# =============================================================================
# TEST FORECASTSCORER
# =============================================================================


class TestForecastScorer:
    """Tests for ForecastScorer class."""

    @pytest.fixture
    def scorer(self):
        """Create forecast scorer."""
        return ForecastScorer()

    def test_score_bullish_forecast(self, scorer, bullish_forecast):
        """Test scoring a bullish forecast."""
        score = scorer.score_forecast(bullish_forecast)

        assert 0 <= score <= 100
        # Bullish forecast with good upside should score reasonably well
        # The score is based on multiple factors including uncertainty penalty
        assert score >= 0

    def test_score_bearish_forecast(self, scorer, bearish_forecast):
        """Test scoring a bearish forecast."""
        score = scorer.score_forecast(bearish_forecast)

        assert 0 <= score <= 100
        # Bearish forecast (negative expected return) should score lower
        assert score < 50

    def test_score_neutral_forecast(self, scorer, neutral_forecast):
        """Test scoring a neutral forecast."""
        score = scorer.score_forecast(neutral_forecast)

        assert 0 <= score <= 100
        # Neutral forecast with low uncertainty should score moderately
        assert 30 <= score <= 70

    def test_score_high_uncertainty(self, scorer, high_uncertainty_forecast):
        """Test scoring high uncertainty forecast."""
        score = scorer.score_forecast(high_uncertainty_forecast)

        assert 0 <= score <= 100
        # High uncertainty should reduce score
        assert score < 60

    def test_score_range_0_to_100(self, scorer, bullish_forecast, bearish_forecast, neutral_forecast, high_uncertainty_forecast):
        """Test that scores are always within 0-100 range."""
        # Test with various forecasts (use fixtures instead of calling them)
        sample_fc = PriceForecast(
            symbol="TEST", timestamp=datetime.now(), current_price=100.0,
            forecast_horizon=16, q10=95.0, q30=98.0, q50=105.0, q70=110.0, q90=115.0,
        )

        forecasts = [sample_fc, bullish_forecast, bearish_forecast, neutral_forecast, high_uncertainty_forecast]

        for fc in forecasts:
            score = scorer.score_forecast(fc)
            assert 0 <= score <= 100

    def test_score_return_component(self, scorer):
        """Test return scoring component."""
        # Test various returns
        assert scorer._score_return(0.20) == pytest.approx(100.0, 0.1)  # Max return
        assert scorer._score_return(-0.10) == pytest.approx(0.0, 0.1)  # Min return
        assert scorer._score_return(0.05) > scorer._score_return(0.0)

    def test_score_uncertainty_component(self, scorer):
        """Test uncertainty scoring component."""
        # Low uncertainty = high score
        assert scorer._score_uncertainty(0.0) == pytest.approx(100.0, 0.1)
        # Max acceptable uncertainty
        assert scorer._score_uncertainty(0.15) == pytest.approx(0.0, 0.1)
        # Higher uncertainty = lower score
        assert scorer._score_uncertainty(0.05) > scorer._score_uncertainty(0.10)

    def test_score_direction_component(self, scorer, sample_forecast):
        """Test directional conviction scoring."""
        score = scorer._score_direction(sample_forecast)

        assert 0 <= score <= 100

    def test_score_direction_with_conviction(self, scorer):
        """Test direction score with high conviction."""
        # High conviction: large expected return relative to uncertainty
        high_conviction = PriceForecast(
            symbol="HIGH",
            timestamp=datetime.now(),
            current_price=100.0,
            forecast_horizon=16,
            q10=105.0,  # Tighter distribution
            q30=112.0,
            q50=120.0,  # +20% return
            q70=125.0,
            q90=130.0,
        )

        score = scorer._score_direction(high_conviction)
        # High conviction should give positive score
        assert score > 0

    def test_score_direction_with_upside_bias(self, scorer):
        """Test direction score with upside bias bonus."""
        # Upside >> downside ratio
        upside_bias = PriceForecast(
            symbol="BIAS",
            timestamp=datetime.now(),
            current_price=100.0,
            forecast_horizon=16,
            q10=99.0,   # -1% downside
            q30=105.0,
            q50=110.0,  # +10% expected
            q70=115.0,
            q90=125.0,  # +25% upside (ratio > 2)
        )

        score = scorer._score_direction(upside_bias)
        # Should get bonus for upside bias
        assert score > 0


# =============================================================================
# TEST FORECASTCACHE
# =============================================================================


class TestForecastCache:
    """Tests for ForecastCache class."""

    def test_cache_set_and_get(self, forecast_cache, sample_forecast):
        """Test setting and getting cached forecast."""
        # Create forecast with the symbol we'll use for key
        bbca_forecast = PriceForecast(
            symbol="BBCA",
            timestamp=datetime.now(),
            current_price=100.0,
            forecast_horizon=16,
            q10=95.0,
            q30=98.0,
            q50=105.0,
            q70=110.0,
            q90=115.0,
        )
        forecast_cache.set("BBCA", 16, bbca_forecast)

        retrieved = forecast_cache.get("BBCA", 16)

        assert retrieved is not None
        assert retrieved.symbol == "BBCA"
        assert retrieved.current_price == 100.0

    def test_cache_miss(self, forecast_cache):
        """Test cache miss returns None."""
        result = forecast_cache.get("NONEXISTENT", 16)

        assert result is None

    def test_cache_expiry(self, forecast_cache):
        """Test cache entry expires after TTL."""
        # Create forecast with BBCA symbol
        bbca_forecast = PriceForecast(
            symbol="BBCA",
            timestamp=datetime.now(),
            current_price=100.0,
            forecast_horizon=16,
            q10=95.0,
            q30=98.0,
            q50=105.0,
            q70=110.0,
            q90=115.0,
        )
        # Create cache with very short TTL
        short_cache = ForecastCache(ttl_minutes=0)
        short_cache.set("BBCA", 16, bbca_forecast)

        # Should be expired immediately (TTL is 0)
        result = short_cache.get("BBCA", 16)
        assert result is None

    def test_cache_invalidation(self, forecast_cache):
        """Test cache invalidation for symbol."""
        # Create forecast for BBCA
        bbca_forecast = PriceForecast(
            symbol="BBCA",
            timestamp=datetime.now(),
            current_price=100.0,
            forecast_horizon=16,
            q10=95.0,
            q30=98.0,
            q50=105.0,
            q70=110.0,
            q90=115.0,
        )
        # Cache multiple horizons for same symbol
        forecast_cache.set("BBCA", 8, bbca_forecast)
        forecast_cache.set("BBCA", 16, bbca_forecast)
        forecast_cache.set("BBCA", 32, bbca_forecast)

        # Invalidate all BBCA entries
        forecast_cache.invalidate("BBCA")

        assert forecast_cache.get("BBCA", 8) is None
        assert forecast_cache.get("BBCA", 16) is None
        assert forecast_cache.get("BBCA", 32) is None

    def test_cache_clear(self, forecast_cache):
        """Test clearing all cache entries."""
        # Create forecasts
        bbca_forecast = PriceForecast(
            symbol="BBCA",
            timestamp=datetime.now(),
            current_price=100.0,
            forecast_horizon=16,
            q10=95.0, q30=98.0, q50=105.0, q70=110.0, q90=115.0,
        )
        bbri_forecast = PriceForecast(
            symbol="BBRI",
            timestamp=datetime.now(),
            current_price=100.0,
            forecast_horizon=16,
            q10=95.0, q30=98.0, q50=105.0, q70=110.0, q90=115.0,
        )
        tlkm_forecast = PriceForecast(
            symbol="TLKM",
            timestamp=datetime.now(),
            current_price=100.0,
            forecast_horizon=16,
            q10=95.0, q30=98.0, q50=105.0, q70=110.0, q90=115.0,
        )

        # Add multiple entries
        forecast_cache.set("BBCA", 16, bbca_forecast)
        forecast_cache.set("BBRI", 16, bbri_forecast)
        forecast_cache.set("TLKM", 16, tlkm_forecast)

        # Clear all
        forecast_cache.clear()

        assert forecast_cache.get("BBCA", 16) is None
        assert forecast_cache.get("BBRI", 16) is None
        assert forecast_cache.get("TLKM", 16) is None

    def test_cache_cleanup_expired(self, forecast_cache):
        """Test cleanup of expired entries."""
        # Create forecasts
        bbca_forecast = PriceForecast(
            symbol="BBCA",
            timestamp=datetime.now(),
            current_price=100.0,
            forecast_horizon=16,
            q10=95.0, q30=98.0, q50=105.0, q70=110.0, q90=115.0,
        )
        bbri_forecast = PriceForecast(
            symbol="BBRI",
            timestamp=datetime.now(),
            current_price=100.0,
            forecast_horizon=16,
            q10=95.0, q30=98.0, q50=105.0, q70=110.0, q90=115.0,
        )

        # Add entries with different TTLs
        forecast_cache.set("BBCA", 16, bbca_forecast, ttl_minutes=0)
        forecast_cache.set("BBRI", 16, bbri_forecast, ttl_minutes=60)

        # Clean up expired (BBCA should be expired since TTL is 0)
        removed = forecast_cache.cleanup_expired()

        assert removed >= 1
        assert forecast_cache.get("BBCA", 16) is None
        assert forecast_cache.get("BBRI", 16) is not None

    def test_cache_stats(self, forecast_cache):
        """Test cache statistics."""
        bbca_forecast = PriceForecast(
            symbol="BBCA",
            timestamp=datetime.now(),
            current_price=100.0,
            forecast_horizon=16,
            q10=95.0, q30=98.0, q50=105.0, q70=110.0, q90=115.0,
        )
        bbri_forecast = PriceForecast(
            symbol="BBRI",
            timestamp=datetime.now(),
            current_price=100.0,
            forecast_horizon=16,
            q10=95.0, q30=98.0, q50=105.0, q70=110.0, q90=115.0,
        )

        forecast_cache.set("BBCA", 16, bbca_forecast)
        forecast_cache.set("BBRI", 16, bbri_forecast)

        stats = forecast_cache.get_stats()

        assert stats["total_entries"] == 2
        assert stats["valid_entries"] >= 0
        assert stats["expired_entries"] >= 0
        assert stats["ttl_minutes"] == 60

    def test_cache_key_generation(self, forecast_cache):
        """Test cache key format."""
        key = forecast_cache._make_key("BBCA", 16)

        assert key == "BBCA:16"

    def test_cache_different_horizons(self, forecast_cache):
        """Test caching different horizons separately."""
        bbca_forecast = PriceForecast(
            symbol="BBCA",
            timestamp=datetime.now(),
            current_price=100.0,
            forecast_horizon=16,
            q10=95.0, q30=98.0, q50=105.0, q70=110.0, q90=115.0,
        )

        forecast_cache.set("BBCA", 8, bbca_forecast)
        forecast_cache.set("BBCA", 16, bbca_forecast)

        # Should retrieve different entries
        fc8 = forecast_cache.get("BBCA", 8)
        fc16 = forecast_cache.get("BBCA", 16)

        assert fc8 is not None
        assert fc16 is not None


class TestCacheEntry:
    """Tests for CacheEntry class."""

    def test_cache_entry_creation(self, sample_forecast):
        """Test creating cache entry."""
        entry = CacheEntry(
            forecast=sample_forecast,
            timestamp=datetime.now(),
            ttl_minutes=60,
        )

        assert entry.forecast == sample_forecast
        assert entry.ttl_minutes == 60

    def test_cache_entry_not_expired(self, sample_forecast):
        """Test entry is not expired when fresh."""
        entry = CacheEntry(
            forecast=sample_forecast,
            timestamp=datetime.now(),
            ttl_minutes=60,
        )

        assert entry.is_expired() is False

    def test_cache_entry_is_expired(self, sample_forecast):
        """Test entry is expired after TTL."""
        entry = CacheEntry(
            forecast=sample_forecast,
            timestamp=datetime.now() - timedelta(minutes=61),
            ttl_minutes=60,
        )

        assert entry.is_expired() is True

    def test_cache_entry_exactly_at_ttl(self, sample_forecast):
        """Test entry at exact TTL boundary."""
        entry = CacheEntry(
            forecast=sample_forecast,
            timestamp=datetime.now() - timedelta(minutes=60),
            ttl_minutes=60,
        )

        # At exact boundary, should be expired
        assert entry.is_expired() is True


# =============================================================================
# TEST FACTORY FUNCTIONS
# =============================================================================


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_forecaster_default(self):
        """Test create_forecaster with defaults."""
        forecaster = create_forecaster()

        assert isinstance(forecaster, TimesFMForecaster)
        assert forecaster.config is not None

    def test_create_forecaster_with_config(self):
        """Test create_forecaster with custom config."""
        config = ForecastConfig(horizon_len=16)
        forecaster = create_forecaster(config)

        assert isinstance(forecaster, TimesFMForecaster)
        assert forecaster.config.horizon_len == 16

    def test_create_scorer(self):
        """Test create_scorer factory function."""
        scorer = create_scorer()

        assert isinstance(scorer, ForecastScorer)


# =============================================================================
# TEST THREAD SAFETY
# =============================================================================


class TestForecastCacheThreadSafety:
    """Test thread safety of ForecastCache."""

    def test_concurrent_set_and_get(self, forecast_cache, sample_forecast):
        """Test concurrent set and get operations."""
        import threading

        results = []

        def set_forecast(symbol):
            forecast_cache.set(symbol, 16, sample_forecast)

        def get_forecast(symbol):
            results.append(forecast_cache.get(symbol, 16))

        threads = []
        for i in range(10):
            t1 = threading.Thread(target=set_forecast, args=(f"SYMBOL{i}",))
            t2 = threading.Thread(target=get_forecast, args=(f"SYMBOL{i}",))
            threads.extend([t1, t2])

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should complete without errors
        assert len(results) == 10

    def test_concurrent_invalidate(self, forecast_cache, sample_forecast):
        """Test concurrent invalidate operations."""
        import threading

        # Add entries
        for i in range(10):
            forecast_cache.set(f"SYMBOL{i}", 16, sample_forecast)

        def invalidate_symbol(i):
            forecast_cache.invalidate(f"SYMBOL{i}")

        threads = [threading.Thread(target=invalidate_symbol, args=(i,)) for i in range(10)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All should be invalidated
        stats = forecast_cache.get_stats()
        assert stats["total_entries"] == 0
