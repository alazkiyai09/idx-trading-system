"""Forecasting package for IDX Trading System.

This package provides time series forecasting capabilities using
the TimesFM model for price prediction.
"""

from core.forecasting.forecast_cache import CacheEntry, ForecastCache
from core.forecasting.timesfm_forecaster import (
    ForecastConfig,
    PriceForecast,
    TimesFMForecaster,
    ForecastScorer,
    create_forecaster,
    create_scorer,
)

__all__ = [
    "CacheEntry",
    "ForecastCache",
    "ForecastConfig",
    "PriceForecast",
    "TimesFMForecaster",
    "ForecastScorer",
    "create_forecaster",
    "create_scorer",
]
