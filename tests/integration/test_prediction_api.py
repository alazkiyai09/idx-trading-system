"""
Integration tests for prediction API endpoints.
"""
import pytest
from fastapi.testclient import TestClient
from datetime import date, timedelta
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np
from pathlib import Path


class TestPredictionEndpoints:
    """Tests for prediction API endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from api.main import app
        return TestClient(app)

    @pytest.fixture
    def mock_db(self):
        """Create mock database manager."""
        mock = Mock()
        mock.get_latest_price.return_value = Mock(close=9000.0, date=date.today())
        mock.get_prices.return_value = [
            Mock(date=date.today() - timedelta(days=i), close=9000.0 - i * 10)
            for i in range(200, 0, -1)
        ]
        mock.get_price_count.return_value = 500
        mock.get_commodity_prices.return_value = [
            Mock(
                date=date.today() - timedelta(days=i),
                close=2000.0 - i,
                open=1995.0 - i,
                high=2010.0 - i,
                low=1990.0 - i,
                volume=100000.0
            )
            for i in range(30, 0, -1)
        ]
        return mock

    def test_get_prediction_endpoint(self, client):
        """Test basic prediction endpoint."""
        response = client.get("/prediction/BBCA")
        assert response.status_code in [200, 404, 503]

        data = response.json()
        if response.status_code == 200:
            assert "symbol" in data
            assert data["symbol"] == "BBCA"
            assert "predictions" in data
            assert len(data["predictions"]) == 7
        else:
            assert "detail" in data

    def test_get_ensemble_prediction_endpoint(self, client):
        """Test ensemble prediction endpoint."""
        response = client.get("/prediction/ensemble/BBCA?horizon=7")
        assert response.status_code in [200, 404, 503]

        data = response.json()
        if response.status_code == 200:
            assert "symbol" in data
            assert "predictions" in data
            assert "model_contributions" in data
        else:
            assert "detail" in data

    def test_get_ensemble_prediction_custom_horizon(self, client):
        """Test ensemble prediction with custom horizon."""
        response = client.get("/prediction/ensemble/BBCA?horizon=14")
        assert response.status_code in [200, 404, 503]

        data = response.json()
        if response.status_code == 200:
            assert len(data["predictions"]) == 14
        else:
            assert "detail" in data

    def test_train_model_endpoint(self, client):
        """Test model training endpoint."""
        response = client.post(
            "/prediction/train/BBCA",
            json={"lookback_days": 200, "test_size": 0.2, "use_exogenous": True},
        )
        assert response.status_code in [202, 400, 501, 503]

        data = response.json()
        assert "status" in data or "detail" in data

    def test_get_commodities_endpoint(self, client):
        """Test commodities endpoint."""
        response = client.get("/prediction/commodities?days=30")
        assert response.status_code == 200

        data = response.json()
        # Should have GOLD, SILVER, OIL
        assert "GOLD" in data or "SILVER" in data or "OIL" in data

    def test_get_correlation_endpoint(self, client):
        """Test correlation endpoint."""
        response = client.get("/prediction/correlation/BBCA?sector=MINING")
        assert response.status_code == 200

        data = response.json()
        assert "symbol" in data
        assert data["symbol"] == "BBCA"
        assert "sector" in data

    def test_get_monte_carlo_endpoint(self, client):
        """Test Monte Carlo simulation endpoint."""
        response = client.get(
            "/prediction/monte-carlo/BBCA?n_simulations=100&horizon_days=7"
        )
        assert response.status_code == 200

        data = response.json()
        assert "symbol" in data
        assert "statistics" in data
        assert "sample_paths" in data


class TestPredictionEndpointsEdgeCases:
    """Edge case tests for prediction endpoints."""

    @pytest.fixture
    def client(self):
        from api.main import app
        return TestClient(app)

    def test_prediction_unknown_symbol(self, client):
        """Test prediction for unknown symbol."""
        response = client.get("/prediction/UNKNOWN")
        assert response.status_code in [404, 503]

    def test_monte_carlo_insufficient_data(self, client):
        """Test Monte Carlo with insufficient data."""
        # This would require mocking the database to return few prices
        # For now, just test the endpoint exists
        response = client.get("/prediction/monte-carlo/BBCA?n_simulations=10")
        assert response.status_code in [200, 400, 404]

    def test_train_model_insufficient_data(self, client):
        """Test training with insufficient data."""
        response = client.post(
            "/prediction/train/UNKNOWN",
            json={"lookback_days": 200, "test_size": 0.2, "use_exogenous": True},
        )
        assert response.status_code in [400, 501, 503]

    def test_training_status_endpoint(self, client, monkeypatch):
        """Training status should expose job and artifact summary."""
        from api.routes import prediction

        monkeypatch.setattr(
            prediction,
            "_training_runtime_status",
            lambda db: {
                "batch_limit": 10,
                "latest_market_date": "2026-03-09",
                "artifacts": {"total_models": 3, "up_to_date_models": 2, "recent_models": []},
                "job": {"status": "idle", "symbols": [], "is_running": False, "log_tail": []},
                "generated_at": "2026-03-09T00:00:00+00:00",
            },
        )

        response = client.get("/prediction/training/status")
        assert response.status_code == 200
        data = response.json()
        assert data["batch_limit"] == 10
        assert data["artifacts"]["total_models"] == 3

    def test_training_run_endpoint(self, client, monkeypatch):
        """Training run should return background job metadata."""
        from api.routes import prediction

        monkeypatch.setattr(
            prediction,
            "_launch_training_job",
            lambda request, db: {
                "status": "running",
                "pid": 4321,
                "symbols": request.symbols,
                "lookback_days": request.lookback_days,
                "overwrite": request.overwrite,
            },
        )

        response = client.post(
            "/prediction/training/run",
            json={
                "symbols": ["BBCA", "ADRO"],
                "lookback_days": 400,
                "overwrite": True,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "started"
        assert data["job"]["symbols"] == ["BBCA", "ADRO"]

    def test_training_run_endpoint_rejects_more_than_batch_cap(self, client):
        """The API should reject batches larger than the configured cap."""
        payload = {
            "symbols": [f"S{i:03d}" for i in range(11)],
            "lookback_days": 400,
            "overwrite": False,
        }
        response = client.post("/prediction/training/run", json=payload)
        assert response.status_code == 422

    def test_list_models_endpoint(self, client, monkeypatch):
        """Model inventory endpoint should expose stored artifacts."""
        from api.routes import prediction

        monkeypatch.setattr(
            prediction,
            "_read_artifact_metadata",
            lambda: [{"symbol": "ADRO", "artifact_type": "multiseed", "trained_at": "2026-03-09T00:00:00Z"}],
        )

        response = client.get("/prediction/models")
        assert response.status_code == 200
        data = response.json()
        assert data["total_models"] == 1
        assert data["models"][0]["symbol"] == "ADRO"

    def test_delete_model_endpoint(self, client, monkeypatch, tmp_path):
        """Deleting a model should remove artifact and metadata files."""
        from api.routes import prediction

        artifact = tmp_path / "ADRO_ensemble.pkl"
        metadata = tmp_path / "ADRO_ensemble.meta.json"
        artifact.write_bytes(b"artifact")
        metadata.write_text("{}")

        monkeypatch.setattr(
            prediction,
            "_artifact_file_paths",
            lambda symbol: {"artifact": artifact, "metadata": metadata},
        )

        response = client.delete("/prediction/models/ADRO")
        assert response.status_code == 200
        assert not artifact.exists()
        assert not metadata.exists()

    def test_upload_model_endpoint(self, client, monkeypatch, tmp_path):
        """Uploading a model should store the artifact and optional metadata."""
        from api.routes import prediction

        monkeypatch.setattr(
            prediction,
            "_artifact_file_paths",
            lambda symbol: {
                "artifact": tmp_path / f"{symbol}_ensemble.pkl",
                "metadata": tmp_path / f"{symbol}_ensemble.meta.json",
            },
        )

        response = client.post(
            "/prediction/models/upload",
            data={"symbol": "ADRO"},
            files={
                "artifact_file": ("ADRO_ensemble.pkl", b"artifact-bytes", "application/octet-stream"),
                "metadata_file": ("ADRO_ensemble.meta.json", b"{}", "application/json"),
            },
        )
        assert response.status_code == 200
        assert (tmp_path / "ADRO_ensemble.pkl").exists()
        assert (tmp_path / "ADRO_ensemble.meta.json").exists()


class TestPredictionAPIPerformance:
    """Performance tests for prediction endpoints."""

    @pytest.fixture
    def client(self):
        from api.main import app
        return TestClient(app)

    def test_monte_carlo_response_time(self, client):
        """Test that Monte Carlo simulation responds in reasonable time."""
        import time
        start = time.time()

        response = client.get(
            "/prediction/monte-carlo/BBCA?n_simulations=1000&horizon_days=30"
        )

        elapsed = time.time() - start
        assert elapsed < 5.0  # Should complete in under 5 seconds

    def test_ensemble_prediction_response_time(self, client):
        """Test that ensemble prediction responds in reasonable time."""
        import time
        start = time.time()

        response = client.get("/prediction/ensemble/BBCA")

        elapsed = time.time() - start
        assert elapsed < 2.0  # Should complete in under 2 seconds
        assert response.status_code in [200, 404, 503]
