"""Dashboard testing fixtures."""
import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


@pytest.fixture
def mock_api_response():
    """Mock API responses for testing."""
    return {
        "stocks": [
            {"symbol": "BBCA", "name": "Bank Central Asia", "sector": "Financials", "sub_sector": "Banking", "is_lq45": True, "market_cap": 1000000000000},
            {"symbol": "TLKM", "name": "Telkom Indonesia", "sector": "Infrastructure", "sub_sector": "Telecom", "is_lq45": True, "market_cap": 500000000000},
            {"symbol": "BBRI", "name": "Bank Rakyat Indonesia", "sector": "Financials", "sub_sector": "Banking", "is_lq45": True, "market_cap": 800000000000},
        ],
        "price_history": [
            {"date": "2024-01-01", "open": 9000, "high": 9100, "low": 8950, "close": 9050, "volume": 1000000},
            {"date": "2024-01-02", "open": 9050, "high": 9200, "low": 9000, "close": 9150, "volume": 1200000},
            {"date": "2024-01-03", "open": 9150, "high": 9300, "low": 9100, "close": 9250, "volume": 1100000},
        ],
        "signals": [
            {"symbol": "BBCA", "type": "BUY", "score": 75, "entry_price": 9050, "stop_loss": 8800, "targets": [9500, 9800]},
            {"symbol": "TLKM", "type": "HOLD", "score": 50, "message": "No clear signal"},
        ],
    }


@pytest.fixture
def mock_requests_get():
    """Mock requests.get for API calls."""
    def _mock_get(url, *args, **kwargs):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"data": "mock"}
        mock_resp.raise_for_status = MagicMock()
        return mock_resp
    return _mock_get


@pytest.fixture
def sample_stock_data():
    """Sample stock DataFrame for testing."""
    return pd.DataFrame({
        "symbol": ["BBCA", "BBRI", "TLKM"],
        "name": ["Bank Central Asia", "Bank Rakyat", "Telkom Indonesia"],
        "sector": ["Financials", "Financials", "Infrastructure"],
        "sub_sector": ["Banking", "Banking", "Telecom"],
        "market_cap": [1e12, 5e11, 3e11],
        "change_pct": [1.5, -0.5, 2.0],
        "is_lq45": [True, True, True],
        "is_idx30": [True, False, True],
    })


@pytest.fixture
def sample_ohlcv_data():
    """Sample OHLCV DataFrame for chart testing."""
    dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
    np.random.seed(42)
    base_price = 9000
    returns = np.random.randn(100) * 0.02
    prices = base_price * (1 + returns).cumprod()

    return pd.DataFrame({
        "date": dates,
        "open": prices * (1 + np.random.randn(100) * 0.005),
        "high": prices * (1 + abs(np.random.randn(100)) * 0.01),
        "low": prices * (1 - abs(np.random.randn(100)) * 0.01),
        "close": prices,
        "volume": np.random.randint(100000, 1000000, 100),
    })


@pytest.fixture
def sample_signal_data_buy():
    """Sample BUY signal data for testing."""
    return {
        "signal": "BUY",
        "type": "BUY",
        "setup": "Breakout",
        "score": 75.5,
        "entry_price": 9050,
        "stop_loss": 8800,
        "targets": [9500, 9800],
        "risk_reward": 1.8,
    }


@pytest.fixture
def sample_signal_data_sell():
    """Sample SELL signal data for testing."""
    return {
        "signal": "SELL",
        "type": "SELL",
        "setup": "Breakdown",
        "score": 70.0,
        "entry_price": 9050,
        "stop_loss": 9300,
        "targets": [8700, 8500],
        "risk_reward": 1.4,
    }


@pytest.fixture
def sample_signal_data_none():
    """Sample no-signal data for testing."""
    return {
        "signal": "None",
        "message": "No signal generated.",
    }


@pytest.fixture
def sample_stock_details():
    """Sample stock details for info card testing."""
    return {
        "symbol": "BBCA",
        "name": "Bank Central Asia",
        "sector": "Financials",
        "sub_sector": "Banking",
        "board": "Main",
        "market_cap": 1000000000000,
        "is_lq45": True,
        "is_idx30": True,
        "latest_price": {"close": 9050, "date": "2024-01-15"},
    }


@pytest.fixture
def sample_prediction_data():
    """Sample prediction data for chart testing."""
    dates = pd.date_range(start="2024-01-16", periods=7, freq="D")
    return {
        "predictions": [
            {"date": str(d.date()), "predicted_price": 9100 + i * 50, "upper_band": 9200 + i * 50, "lower_band": 9000 + i * 50}
            for i, d in enumerate(dates)
        ],
        "is_mock": False,
    }


@pytest.fixture
def sample_sentiment_data():
    """Sample sentiment data for testing."""
    return {
        "overall_score": 65.5,
        "sector_scores": {
            "Financials": 70.0,
            "Infrastructure": 60.0,
            "Consumer": 55.0,
        },
        "news_count": 25,
        "last_updated": "2024-01-15T10:00:00",
    }


@pytest.fixture
def sample_portfolio_state():
    """Sample portfolio state for trading tests."""
    return {
        "capital": 100000000,
        "positions": [
            {"symbol": "BBCA", "quantity": 1000, "avg_price": 9000, "current_price": 9050},
        ],
        "total_pnl": 50000,
        "win_rate": 0.65,
        "total_trades": 20,
    }


@pytest.fixture
def mock_streamlit():
    """Mock Streamlit for component testing."""
    with patch("streamlit.sidebar") as mock_sidebar, \
         patch("streamlit.columns") as mock_columns, \
         patch("streamlit.metric") as mock_metric, \
         patch("streamlit.markdown") as mock_markdown, \
         patch("streamlit.info") as mock_info, \
         patch("streamlit.error") as mock_error, \
         patch("streamlit.success") as mock_success, \
         patch("streamlit.warning") as mock_warning:
        yield {
            "sidebar": mock_sidebar,
            "columns": mock_columns,
            "metric": mock_metric,
            "markdown": mock_markdown,
            "info": mock_info,
            "error": mock_error,
            "success": mock_success,
            "warning": mock_warning,
        }
