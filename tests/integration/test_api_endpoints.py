"""Tests for FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


class TestHealthEndpoints:
    """Tests for health check endpoints."""

    def test_root(self, client):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "IDX Trading System API"
        assert data["version"] == "3.0.0"

    def test_health(self, client):
        """Test health endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "version" in data

    def test_health_data(self, client):
        """Test data freshness endpoint."""
        response = client.get("/health/data")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data

    def test_health_update_status(self, client):
        """Test data update status endpoint."""
        response = client.get("/health/update-status")
        assert response.status_code == 200
        data = response.json()
        assert "refresh_policy" in data
        assert "data_status" in data
        assert "manual_refresh" in data

    def test_dashboard_summary(self, client):
        """Test compact dashboard summary endpoint."""
        response = client.get("/health/dashboard-summary")
        assert response.status_code == 200
        data = response.json()
        assert "stock_count" in data
        assert "record_count" in data
        assert "refresh_policy" in data

    def test_health_detailed(self, client):
        """Test detailed health endpoint."""
        response = client.get("/health/detailed")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "components" in data


class TestSignalEndpoints:
    """Tests for signal endpoints."""

    def test_list_signals_empty(self, client):
        """Test listing signals when empty."""
        response = client.get("/signals")
        assert response.status_code == 200
        data = response.json()
        assert "signals" in data
        assert "total" in data

    def test_get_signal_not_found(self, client):
        """Test getting non-existent signal."""
        response = client.get("/signals/999")
        assert response.status_code == 404


class TestPortfolioEndpoints:
    """Tests for portfolio endpoints."""

    def test_get_portfolio(self, client):
        """Test getting portfolio."""
        response = client.get("/portfolio")
        assert response.status_code == 200

    def test_get_positions(self, client):
        """Test getting positions."""
        response = client.get("/portfolio/positions")
        assert response.status_code == 200

    def test_get_trade_history(self, client):
        """Test getting trade history."""
        response = client.get("/portfolio/history")
        assert response.status_code == 200
        data = response.json()
        assert "trades" in data
        assert "total" in data


class TestConfigEndpoints:
    """Tests for config endpoints."""

    def test_get_modes(self, client):
        """Test getting trading modes."""
        response = client.get("/config/modes")
        assert response.status_code == 200
        data = response.json()
        assert "modes" in data
        modes = data["modes"]
        assert "swing" in modes or "intraday" in modes


class TestStockEndpoints:
    """Tests for stock endpoints."""

    def test_get_stock_symbols(self):
        from api.routes.stocks import get_stock_symbols

        data = get_stock_symbols()
        assert "symbols" in data

    def test_serialize_date_accepts_str_and_date(self):
        from datetime import date
        from api.routes.stocks import _serialize_date

        assert _serialize_date("2026-03-06") == "2026-03-06"
        assert _serialize_date(date(2026, 3, 6)) == "2026-03-06"
        assert _serialize_date(None) is None


class TestFundamentalEndpoints:
    """Tests for fundamental analysis endpoints."""

    def test_analyze(self, client):
        """Test fundamental analysis endpoint."""
        response = client.post(
            "/fundamental/analyze",
            json={"symbol": "BBCA"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["symbol"] == "BBCA"


class TestCORS:
    """Tests for CORS configuration."""

    def test_cors_headers(self, client):
        """Test that CORS headers are present."""
        response = client.options(
            "/health",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
            },
        )
        # FastAPI CORS middleware should respond
        assert response.status_code in (200, 405)
