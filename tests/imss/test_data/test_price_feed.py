"""Test price feed data ingestion."""

import pytest
import pandas as pd

from imss.data.price_feed import fetch_idx_prices, validate_price_data, IDX_TICKER_MAP


def test_idx_ticker_map():
    """BBRI maps to BBRI.JK."""
    assert IDX_TICKER_MAP["BBRI"] == "BBRI.JK"


def test_validate_price_data_removes_zero_volume():
    """Rows with zero volume are removed."""
    df = pd.DataFrame({
        "Open": [5100, 5200],
        "High": [5150, 5250],
        "Low": [5050, 5150],
        "Close": [5125, 5225],
        "Volume": [0, 85000000],
        "Adj Close": [5125, 5225],
    }, index=pd.to_datetime(["2024-07-01", "2024-07-02"]))
    result = validate_price_data(df)
    assert len(result) == 1
    assert result.iloc[0]["Volume"] == 85000000


def test_validate_price_data_drops_nan():
    """Rows with NaN close are removed."""
    df = pd.DataFrame({
        "Open": [5100, float("nan")],
        "High": [5150, 5250],
        "Low": [5050, 5150],
        "Close": [5125, float("nan")],
        "Volume": [85000000, 90000000],
        "Adj Close": [5125, float("nan")],
    }, index=pd.to_datetime(["2024-07-01", "2024-07-02"]))
    result = validate_price_data(df)
    assert len(result) == 1
