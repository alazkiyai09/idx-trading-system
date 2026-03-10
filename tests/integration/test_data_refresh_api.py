from fastapi.testclient import TestClient

from api.main import app


def test_trigger_data_update_starts_job(monkeypatch):
    from api.routes import health

    monkeypatch.setattr(
        health,
        "_launch_manual_refresh",
        lambda: {
            "status": "running",
            "pid": 12345,
            "started_at": "2026-03-09T00:00:00+00:00",
            "finished_at": None,
            "command": "python scripts/ingest_all_stocks.py",
        },
    )

    client = TestClient(app)
    response = client.post("/health/update-data")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "started"
    assert data["job"]["status"] == "running"
