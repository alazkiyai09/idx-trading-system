"""
Tests for Virtual Trading API endpoints (/simulation/*).

These tests verify the simulation/paper trading API endpoints work correctly.
"""
import pytest
import requests
from unittest.mock import patch, MagicMock
import json

# API base URL
API_URL = "http://localhost:8000"


class TestSimulationAPIEndpoints:
    """Test suite for /simulation/* API endpoints."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup for each test."""
        self.created_sessions = []

    def teardown_method(self):
        """Cleanup created sessions after each test."""
        # Note: In a real implementation, we would delete created sessions
        pass

    # -------------------------------------------------------------------------
    # Test 1: Page loads without errors (via API health check)
    # -------------------------------------------------------------------------
    def test_api_is_running(self):
        """Test that the API server is running and responding."""
        try:
            response = requests.get(f"{API_URL}/docs", timeout=5)
            assert response.status_code == 200, f"API returned {response.status_code}"
        except requests.exceptions.ConnectionError:
            pytest.skip("API server not running on localhost:8000")

    # -------------------------------------------------------------------------
    # Test 2: Portfolio summary displays (via API)
    # -------------------------------------------------------------------------
    def test_list_simulations_endpoint(self):
        """Test GET /simulation/ returns list of sessions."""
        response = requests.get(f"{API_URL}/simulation/", timeout=10)

        assert response.status_code == 200, f"Expected 200, got {response.status_code}"

        data = response.json()
        assert isinstance(data, list), "Response should be a list"

        # If there are sessions, verify structure
        if len(data) > 0:
            session = data[0]
            required_fields = ["session_id", "name", "mode", "trading_mode", "status"]
            for field in required_fields:
                assert field in session, f"Missing field: {field}"

    def test_create_simulation_endpoint(self):
        """Test POST /simulation/create creates a new session."""
        payload = {
            "name": "Test Session API",
            "mode": "live",
            "trading_mode": "swing",
            "initial_capital": 100_000_000
        }

        response = requests.post(
            f"{API_URL}/simulation/create",
            json=payload,
            timeout=10
        )

        assert response.status_code == 200, f"Expected 200, got {response.status_code}"

        data = response.json()
        assert data.get("status") == "success", "Status should be 'success'"
        assert "session_id" in data, "Response should contain session_id"
        assert data["session_id"].startswith("sim_"), "Session ID should start with 'sim_'"

        self.created_sessions.append(data["session_id"])

    def test_get_portfolio_endpoint(self):
        """Test GET /simulation/{session_id}/portfolio returns portfolio data."""
        # First create a session
        create_response = requests.post(
            f"{API_URL}/simulation/create",
            json={"name": "Portfolio Test", "mode": "live", "trading_mode": "swing"},
            timeout=10
        )
        session_id = create_response.json()["session_id"]

        # Get portfolio
        response = requests.get(
            f"{API_URL}/simulation/{session_id}/portfolio",
            timeout=10
        )

        assert response.status_code == 200, f"Expected 200, got {response.status_code}"

        data = response.json()
        required_fields = ["capital", "pnl", "positions"]
        for field in required_fields:
            assert field in data, f"Missing field: {field}"

        assert isinstance(data["positions"], list), "Positions should be a list"

    # -------------------------------------------------------------------------
    # Test 3: Order entry form works (BUY/SELL buttons via API)
    # -------------------------------------------------------------------------
    def test_execute_buy_order(self):
        """Test POST /simulation/{session_id}/order with BUY order."""
        # Create session
        create_response = requests.post(
            f"{API_URL}/simulation/create",
            json={"name": "Buy Order Test", "mode": "live", "trading_mode": "swing"},
            timeout=10
        )
        session_id = create_response.json()["session_id"]

        # Execute BUY order
        order_payload = {
            "symbol": "BBCA",
            "side": "BUY",
            "quantity": 1000,
            "order_type": "MARKET",
            "price": 9000,
            "targets": []
        }

        response = requests.post(
            f"{API_URL}/simulation/{session_id}/order",
            json=order_payload,
            timeout=10
        )

        assert response.status_code == 200, f"Expected 200, got {response.status_code}"

        data = response.json()
        assert data.get("status") == "success", "Order status should be 'success'"
        assert "order_id" in data, "Response should contain order_id"
        assert "BUY" in data.get("message", ""), "Message should confirm BUY"

    def test_execute_sell_order(self):
        """Test POST /simulation/{session_id}/order with SELL order."""
        # Create session
        create_response = requests.post(
            f"{API_URL}/simulation/create",
            json={"name": "Sell Order Test", "mode": "live", "trading_mode": "swing"},
            timeout=10
        )
        session_id = create_response.json()["session_id"]

        # Execute SELL order
        order_payload = {
            "symbol": "TLKM",
            "side": "SELL",
            "quantity": 500,
            "order_type": "MARKET",
            "price": 3800,
            "targets": []
        }

        response = requests.post(
            f"{API_URL}/simulation/{session_id}/order",
            json=order_payload,
            timeout=10
        )

        assert response.status_code == 200, f"Expected 200, got {response.status_code}"

        data = response.json()
        assert data.get("status") == "success", "Order status should be 'success'"
        assert "SELL" in data.get("message", ""), "Message should confirm SELL"

    def test_order_with_limit_type(self):
        """Test POST /simulation/{session_id}/order with LIMIT order type."""
        # Create session
        create_response = requests.post(
            f"{API_URL}/simulation/create",
            json={"name": "Limit Order Test", "mode": "live", "trading_mode": "swing"},
            timeout=10
        )
        session_id = create_response.json()["session_id"]

        # Execute LIMIT order
        order_payload = {
            "symbol": "BBRI",
            "side": "BUY",
            "quantity": 2000,
            "order_type": "LIMIT",
            "price": 4500,
            "targets": [4800, 5000]
        }

        response = requests.post(
            f"{API_URL}/simulation/{session_id}/order",
            json=order_payload,
            timeout=10
        )

        assert response.status_code == 200, f"Expected 200, got {response.status_code}"

        data = response.json()
        assert data.get("status") == "success", "Order status should be 'success'"

    # -------------------------------------------------------------------------
    # Test 4: Positions table displays correctly (via API)
    # -------------------------------------------------------------------------
    def test_positions_in_portfolio(self):
        """Test that positions are returned with expected structure."""
        # Create session
        create_response = requests.post(
            f"{API_URL}/simulation/create",
            json={"name": "Positions Test", "mode": "live", "trading_mode": "swing"},
            timeout=10
        )
        session_id = create_response.json()["session_id"]

        # Get portfolio
        response = requests.get(
            f"{API_URL}/simulation/{session_id}/portfolio",
            timeout=10
        )

        data = response.json()
        positions = data.get("positions", [])

        # If there are positions, verify structure
        if len(positions) > 0:
            position = positions[0]
            expected_fields = ["symbol", "quantity", "entry_price"]
            for field in expected_fields:
                assert field in position, f"Position missing field: {field}"

    # -------------------------------------------------------------------------
    # Test 5: Trade history shows executed trades
    # -------------------------------------------------------------------------
    def test_trade_history_endpoint(self):
        """Test GET /simulation/{session_id}/history returns trade history."""
        # Create session
        create_response = requests.post(
            f"{API_URL}/simulation/create",
            json={"name": "History Test", "mode": "live", "trading_mode": "swing"},
            timeout=10
        )
        session_id = create_response.json()["session_id"]

        # Get trade history
        response = requests.get(
            f"{API_URL}/simulation/{session_id}/history",
            timeout=10
        )

        assert response.status_code == 200, f"Expected 200, got {response.status_code}"

        data = response.json()
        assert isinstance(data, list), "History should be a list"

        # If there are trades, verify structure
        if len(data) > 0:
            trade = data[0]
            # Check for common trade fields
            assert "symbol" in trade, "Trade should have symbol"

    # -------------------------------------------------------------------------
    # Test 6: Risk metrics display (Kelly, Sharpe, drawdown)
    # -------------------------------------------------------------------------
    def test_metrics_endpoint(self):
        """Test GET /simulation/{session_id}/metrics returns risk metrics."""
        # Create session
        create_response = requests.post(
            f"{API_URL}/simulation/create",
            json={"name": "Metrics Test", "mode": "live", "trading_mode": "swing"},
            timeout=10
        )
        session_id = create_response.json()["session_id"]

        # Get metrics
        response = requests.get(
            f"{API_URL}/simulation/{session_id}/metrics",
            timeout=10
        )

        assert response.status_code == 200, f"Expected 200, got {response.status_code}"

        data = response.json()

        # Verify required risk metrics
        required_metrics = [
            "current_capital",
            "total_pnl",
            "win_rate",
            "total_trades",
            "max_drawdown",
            "sharpe_ratio"
        ]

        for metric in required_metrics:
            assert metric in data, f"Missing metric: {metric}"

        # Verify types
        assert isinstance(data["win_rate"], (int, float)), "win_rate should be numeric"
        assert isinstance(data["max_drawdown"], (int, float)), "max_drawdown should be numeric"

    def test_equity_curve_endpoint(self):
        """Test GET /simulation/{session_id}/equity-curve returns equity data."""
        # Create session
        create_response = requests.post(
            f"{API_URL}/simulation/create",
            json={"name": "Equity Test", "mode": "live", "trading_mode": "swing"},
            timeout=10
        )
        session_id = create_response.json()["session_id"]

        # Get equity curve
        response = requests.get(
            f"{API_URL}/simulation/{session_id}/equity-curve",
            timeout=10
        )

        assert response.status_code == 200, f"Expected 200, got {response.status_code}"

        data = response.json()
        assert isinstance(data, list), "Equity curve should be a list"

        if len(data) > 0:
            point = data[0]
            assert "date" in point, "Equity point should have date"
            assert "value" in point, "Equity point should have value"

    # -------------------------------------------------------------------------
    # Test 7: Replay step functionality
    # -------------------------------------------------------------------------
    def test_replay_step_endpoint(self):
        """Test POST /simulation/{session_id}/step advances replay."""
        # Create replay session
        create_response = requests.post(
            f"{API_URL}/simulation/create",
            json={"name": "Replay Test", "mode": "replay", "trading_mode": "swing"},
            timeout=10
        )
        session_id = create_response.json()["session_id"]

        # Advance replay
        response = requests.post(
            f"{API_URL}/simulation/{session_id}/step",
            timeout=10
        )

        assert response.status_code == 200, f"Expected 200, got {response.status_code}"

        data = response.json()
        assert data.get("status") == "success", "Step status should be 'success'"
        assert "current_date" in data, "Response should contain current_date"

    # -------------------------------------------------------------------------
    # Test 8: Error handling
    # -------------------------------------------------------------------------
    def test_metrics_for_nonexistent_session(self):
        """Test that metrics endpoint handles non-existent session."""
        response = requests.get(
            f"{API_URL}/simulation/nonexistent_session_12345/metrics",
            timeout=10
        )

        # Should return 404 or appropriate error
        assert response.status_code in [404, 400, 500], \
            f"Expected error status, got {response.status_code}"


class TestSimulationAPIValidation:
    """Test validation and edge cases for simulation API."""

    def test_create_session_with_minimal_data(self):
        """Test creating session with minimal required data."""
        response = requests.post(
            f"{API_URL}/simulation/create",
            json={},
            timeout=10
        )

        # Should succeed with defaults
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"

    def test_create_session_with_all_parameters(self):
        """Test creating session with all parameters."""
        payload = {
            "name": "Full Parameter Test",
            "mode": "replay",
            "trading_mode": "position",
            "initial_capital": 500_000_000
        }

        response = requests.post(
            f"{API_URL}/simulation/create",
            json=payload,
            timeout=10
        )

        assert response.status_code == 200, f"Expected 200, got {response.status_code}"

        data = response.json()
        assert data["data"]["name"] == "Full Parameter Test"
        assert data["data"]["mode"] == "replay"
        assert data["data"]["trading_mode"] == "position"

    def test_order_with_zero_quantity(self):
        """Test order validation with zero quantity."""
        # Create session
        create_response = requests.post(
            f"{API_URL}/simulation/create",
            json={"name": "Zero Qty Test", "mode": "live"},
            timeout=10
        )
        session_id = create_response.json()["session_id"]

        # Execute order with zero quantity
        order_payload = {
            "symbol": "BBCA",
            "side": "BUY",
            "quantity": 0,
            "order_type": "MARKET",
            "price": 9000
        }

        response = requests.post(
            f"{API_URL}/simulation/{session_id}/order",
            json=order_payload,
            timeout=10
        )

        # API should accept it (validation happens at business logic level)
        # or reject it with appropriate error
        assert response.status_code in [200, 400, 422], \
            f"Unexpected status: {response.status_code}"

    def test_order_with_invalid_symbol_format(self):
        """Test order with invalid symbol format."""
        # Create session
        create_response = requests.post(
            f"{API_URL}/simulation/create",
            json={"name": "Invalid Symbol Test", "mode": "live"},
            timeout=10
        )
        session_id = create_response.json()["session_id"]

        # Execute order with invalid symbol
        order_payload = {
            "symbol": "INVALID_SYMBOL_TOO_LONG",
            "side": "BUY",
            "quantity": 100,
            "order_type": "MARKET",
            "price": 1000
        }

        response = requests.post(
            f"{API_URL}/simulation/{session_id}/order",
            json=order_payload,
            timeout=10
        )

        # API may accept it (mock) or reject it
        # Just verify it doesn't crash
        assert response.status_code in [200, 400, 422], \
            f"Unexpected status: {response.status_code}"
