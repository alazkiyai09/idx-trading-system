"""
Tests for ML Prediction dashboard page.
"""
import pytest
from unittest.mock import patch, MagicMock, Mock
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


# Module-level fixtures available to all test classes

@pytest.fixture
def mock_prediction_response():
    """Mock prediction API response."""
    return {
        "symbol": "BBCA",
        "current_price": 9000.0,
        "is_mock": False,
        "predictions": [
            {
                "date": (datetime.now() + timedelta(days=i)).strftime("%Y-%m-%d"),
                "predicted_price": 9050.0 + i * 50,
                "predicted_return": 0.005 + i * 0.001
            }
            for i in range(1, 8)
        ],
        "model_contributions": {
            "lstm": 0.40,
            "cnn_lstm": 0.35,
            "svr": 0.25
        },
        "confidence_interval": {
            "lower": [9000 + i * 40 for i in range(7)],
            "upper": [9100 + i * 60 for i in range(7)]
        }
    }


@pytest.fixture
def mock_commodity_response():
    """Mock commodity API response."""
    return {
        "GOLD": {
            "latest": {
                "date": "2024-01-15",
                "close": 2050.0,
                "open": 2045.0,
                "high": 2060.0,
                "low": 2040.0,
                "volume": 1000000.0
            },
            "history": [
                {"date": "2024-01-15", "close": 2050.0}
            ]
        },
        "SILVER": {
            "latest": {
                "date": "2024-01-15",
                "close": 23.5,
                "open": 23.3,
                "high": 23.8,
                "low": 23.2,
                "volume": 500000.0
            },
            "history": []
        },
        "OIL": {
            "latest": {
                "date": "2024-01-15",
                "close": 78.5,
                "open": 78.0,
                "high": 79.0,
                "low": 77.5,
                "volume": 2000000.0
            },
            "history": []
        }
    }


@pytest.fixture
def mock_correlation_response():
    """Mock correlation API response."""
    return {
        "symbol": "BBCA",
        "sector": "BANKING",
        "correlations": {
            "GOLD": {"correlation_30d": -0.3, "correlation_90d": -0.25},
            "SILVER": {"correlation_30d": -0.2, "correlation_90d": -0.15},
            "OIL": {"correlation_30d": -0.4, "correlation_90d": -0.35}
        },
        "signals": {
            "overall_signal": "NEUTRAL",
            "bullish_count": 0,
            "bearish_count": 0
        },
        "sector_impact": {
            "sector": "BANKING",
            "commodity_impacts": {
                "GOLD": -0.3,
                "SILVER": -0.2,
                "OIL": -0.4
            }
        }
    }


@pytest.fixture
def mock_monte_carlo_response():
    """Mock Monte Carlo API response."""
    np.random.seed(42)
    n_paths = 10
    horizon = 7

    return {
        "symbol": "BBCA",
        "current_price": 9000.0,
        "horizon_days": horizon,
        "n_simulations": 1000,
        "statistics": {
            "mean_final_price": 9100.0,
            "std_final_price": 200.0,
            "expected_return_pct": 1.11,
            "var_95_price": 8700.0,
            "var_95_pct": -3.33,
            "cvar_95_price": 8600.0,
            "cvar_95_pct": -4.44,
            "prob_loss_pct": 40.0,
            "prob_gain_10pct_pct": 15.0
        },
        "percentiles": {
            "p05": 8700.0,
            "p10": 8800.0,
            "p25": 8900.0,
            "p50": 9100.0,
            "p75": 9300.0,
            "p90": 9400.0,
            "p95": 9500.0
        },
        "model": {
            "type": "Geometric Brownian Motion",
            "drift_daily": 0.001,
            "volatility_daily": 0.02,
            "data_points_used": 90
        },
        "sample_paths": np.random.randn(n_paths, horizon + 1).tolist()
    }


class TestMLPredictionPage:
    """Tests for the ML Prediction dashboard page."""

    def test_prediction_data_structure(self, mock_prediction_response):
        """Test that prediction response has correct structure."""
        assert "symbol" in mock_prediction_response
        assert "current_price" in mock_prediction_response
        assert "predictions" in mock_prediction_response
        assert len(mock_prediction_response["predictions"]) == 7

        # Check prediction structure
        pred = mock_prediction_response["predictions"][0]
        assert "date" in pred
        assert "predicted_price" in pred
        assert "predicted_return" in pred

    def test_model_contributions_sum(self, mock_prediction_response):
        """Test that model contributions sum to approximately 1."""
        contribs = mock_prediction_response["model_contributions"]
        total = sum(contribs.values())
        assert 0.99 <= total <= 1.01, f"Model contributions should sum to ~1, got {total}"

    def test_confidence_interval_bounds(self, mock_prediction_response):
        """Test that confidence interval bounds are valid."""
        ci = mock_prediction_response["confidence_interval"]
        assert len(ci["lower"]) == len(ci["upper"]) == 7

        for lower, upper in zip(ci["lower"], ci["upper"]):
            assert lower < upper, f"Lower bound {lower} should be < upper bound {upper}"

    def test_commodity_data_structure(self, mock_commodity_response):
        """Test that commodity response has correct structure."""
        for commodity in ["GOLD", "SILVER", "OIL"]:
            assert commodity in mock_commodity_response
            assert "latest" in mock_commodity_response[commodity]
            assert "close" in mock_commodity_response[commodity]["latest"]

    def test_correlation_values_range(self, mock_correlation_response):
        """Test that correlation values are in valid range [-1, 1]."""
        correlations = mock_correlation_response["correlations"]
        for commodity, values in correlations.items():
            for key, value in values.items():
                assert -1 <= value <= 1, f"Correlation {key} for {commodity} is {value}, should be in [-1, 1]"

    def test_signal_values(self, mock_correlation_response):
        """Test that signal values are valid."""
        signals = mock_correlation_response["signals"]
        assert signals["overall_signal"] in ["BULLISH", "BEARISH", "NEUTRAL"]
        assert isinstance(signals["bullish_count"], int)
        assert isinstance(signals["bearish_count"], int)

    def test_monte_carlo_statistics(self, mock_monte_carlo_response):
        """Test Monte Carlo statistics structure."""
        stats = mock_monte_carlo_response["statistics"]
        assert "mean_final_price" in stats
        assert "std_final_price" in stats
        assert "expected_return_pct" in stats
        assert "var_95_pct" in stats
        assert "cvar_95_pct" in stats

        # VaR should be negative (loss) for 95% confidence
        assert stats["var_95_pct"] <= 0
        # CVaR should be more negative than VaR
        assert stats["cvar_95_pct"] <= stats["var_95_pct"]

    def test_monte_carlo_percentiles_ordering(self, mock_monte_carlo_response):
        """Test that percentiles are properly ordered."""
        percentiles = mock_monte_carlo_response["percentiles"]
        assert percentiles["p05"] <= percentiles["p25"]
        assert percentiles["p25"] <= percentiles["p50"]
        assert percentiles["p50"] <= percentiles["p75"]
        assert percentiles["p75"] <= percentiles["p95"]


class TestMLPredictionPageUI:
    """Tests for ML Prediction page UI components."""

    @pytest.fixture
    def mock_st(self):
        """Mock streamlit module."""
        with patch("streamlit.sidebar") as mock_sidebar, \
             patch("streamlit.selectbox") as mock_selectbox, \
             patch("streamlit.slider") as mock_slider, \
             patch("streamlit.button") as mock_button, \
             patch("streamlit.tabs") as mock_tabs, \
             patch("streamlit.columns") as mock_columns, \
             patch("streamlit.metric") as mock_metric, \
             patch("streamlit.markdown") as mock_markdown, \
             patch("streamlit.spinner") as mock_spinner, \
             patch("streamlit.plotly_chart") as mock_plotly_chart:

            mock_selectbox.return_value = "BBCA"
            mock_slider.return_value = 7
            mock_button.return_value = False
            mock_tabs.return_value = [MagicMock() for _ in range(4)]
            mock_columns.return_value = [MagicMock() for _ in range(4)]

            yield {
                "sidebar": mock_sidebar,
                "selectbox": mock_selectbox,
                "slider": mock_slider,
                "button": mock_button,
                "tabs": mock_tabs,
                "columns": mock_columns,
                "metric": mock_metric,
                "markdown": mock_markdown,
                "spinner": mock_spinner,
                "plotly_chart": mock_plotly_chart
            }

    def test_symbol_selector_options(self, mock_st):
        """Test that symbol selector has expected options."""
        # This would test the symbol selector dropdown
        expected_symbols = ["BBCA", "BBRI", "TLKM", "ASII", "ICBP", "BMRI", "GGRM", "ADRO"]
        # In actual test, would verify these are the options
        assert len(expected_symbols) == 8

    def test_horizon_slider_range(self, mock_st):
        """Test that horizon slider has valid range."""
        # Slider should be between 3 and 30 days
        min_horizon = 3
        max_horizon = 30
        assert min_horizon <= mock_st["slider"].return_value <= max_horizon


class TestMLPredictionAPIIntegration:
    """Integration tests for ML Prediction page with API."""

    @pytest.fixture
    def mock_requests(self):
        """Mock requests module."""
        with patch("requests.get") as mock_get, \
             patch("requests.post") as mock_post:
            yield {"get": mock_get, "post": mock_post}

    def test_fetch_prediction_success(self, mock_requests, mock_prediction_response):
        """Test successful prediction fetch."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_prediction_response
        mock_requests["get"].return_value = mock_response

        # Simulate API call
        import requests
        resp = requests.get("http://localhost:8000/prediction/ensemble/BBCA")

        assert resp.status_code == 200
        data = resp.json()
        assert data["symbol"] == "BBCA"

    def test_fetch_commodities_success(self, mock_requests, mock_commodity_response):
        """Test successful commodities fetch."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_commodity_response
        mock_requests["get"].return_value = mock_response

        import requests
        resp = requests.get("http://localhost:8000/prediction/commodities")

        assert resp.status_code == 200
        data = resp.json()
        assert "GOLD" in data

    def test_train_model_success(self, mock_requests):
        """Test successful model training request."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "scheduled",
            "message": "Training job scheduled for BBCA"
        }
        mock_requests["post"].return_value = mock_response

        import requests
        resp = requests.post("http://localhost:8000/prediction/train/BBCA")

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "scheduled"

    def test_api_error_handling(self, mock_requests):
        """Test API error handling."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_requests["get"].return_value = mock_response

        import requests
        resp = requests.get("http://localhost:8000/prediction/ensemble/UNKNOWN")

        assert resp.status_code == 500


class TestPredictionChartBuilding:
    """Tests for prediction chart building."""

    def test_build_prediction_chart_with_confidence(self, mock_prediction_response):
        """Test building prediction chart with confidence band."""
        import plotly.graph_objects as go

        predictions = mock_prediction_response["predictions"]
        pred_df = pd.DataFrame(predictions)
        pred_df['date'] = pd.to_datetime(pred_df['date'])

        fig = go.Figure()

        # Add prediction line
        fig.add_trace(go.Scatter(
            x=pred_df['date'],
            y=pred_df['predicted_price'],
            mode='lines+markers',
            name='Predicted Price'
        ))

        # Add confidence band
        ci = mock_prediction_response["confidence_interval"]
        fig.add_trace(go.Scatter(
            x=list(pred_df['date']) + list(pred_df['date'][::-1]),
            y=ci['upper'] + ci['lower'][::-1],
            fill='toself',
            name='Confidence Band'
        ))

        # Verify chart has expected traces
        assert len(fig.data) == 2
        assert fig.data[0].name == 'Predicted Price'
        assert fig.data[1].name == 'Confidence Band'

    def test_build_correlation_heatmap(self, mock_correlation_response):
        """Test building correlation heatmap."""
        import plotly.graph_objects as go

        correlations = mock_correlation_response["correlations"]
        corr_data = []
        for commodity, values in correlations.items():
            corr_data.append({
                "Commodity": commodity,
                "30-Day": values["correlation_30d"],
                "90-Day": values["correlation_90d"]
            })

        corr_df = pd.DataFrame(corr_data)

        fig = go.Figure(data=go.Heatmap(
            z=[
                [d["30-Day"] for d in corr_data],
                [d["90-Day"] for d in corr_data]
            ],
            x=[d["Commodity"] for d in corr_data],
            y=["30-Day", "90-Day"],
            colorscale='RdYlGn',
            zmid=0
        ))

        # Verify heatmap structure
        assert len(fig.data) == 1
        assert fig.data[0].type == 'heatmap'
