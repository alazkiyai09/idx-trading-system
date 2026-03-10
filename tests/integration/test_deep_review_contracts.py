import asyncio
from datetime import date, timedelta
from unittest.mock import Mock


def _seed_prices(db, symbol: str, days: int = 30, start: date | None = None, close: float = 1000.0):
    start = start or (date.today() - timedelta(days=days))
    rows = []
    for idx in range(days):
        trade_date = start + timedelta(days=idx)
        price = close + idx
        rows.append(
            {
                "symbol": symbol,
                "date": trade_date,
                "open": price,
                "high": price + 5,
                "low": price - 5,
                "close": price,
                "volume": 1000000,
            }
        )
    db.save_prices(rows)


def test_replay_session_requires_and_persists_start_date(temp_db, monkeypatch):
    from api.routes import simulation
    from core.data.database import DatabaseManager
    from fastapi import HTTPException

    monkeypatch.setattr(simulation, "DatabaseManager", lambda: DatabaseManager(temp_db.database_url))
    _seed_prices(temp_db, "BBCA", days=10, start=date(2024, 1, 1))

    try:
        simulation.create_simulation(
            simulation.CreateSimulationRequest(
                name="Replay", mode="replay", trading_mode="swing", initial_capital=100000000
            )
        )
        assert False, "Expected replay mode without start_date to raise"
    except HTTPException as exc:
        assert exc.status_code == 400
        assert "start_date" in exc.detail

    payload = simulation.create_simulation(
        simulation.CreateSimulationRequest(
            name="Replay",
            mode="replay",
            trading_mode="swing",
            initial_capital=100000000,
            start_date=date(2024, 1, 2),
        )
    )
    assert payload["feature_state"] == "beta"
    assert payload["data"]["start_date"] == date(2024, 1, 2)
    assert payload["data"]["current_date"] == "2024-01-02"


def test_simulation_order_flow_is_stateful(temp_db, monkeypatch):
    from api.routes import simulation
    from core.data.database import DatabaseManager

    monkeypatch.setattr(simulation, "DatabaseManager", lambda: DatabaseManager(temp_db.database_url))
    _seed_prices(temp_db, "BBCA", days=5, start=date.today() - timedelta(days=5), close=1000.0)

    create_resp = simulation.create_simulation(
        simulation.CreateSimulationRequest(
            name="Stateful", mode="live", trading_mode="swing", initial_capital=100000000
        )
    )
    session_id = create_resp["session_id"]

    buy_resp = simulation.execute_order(
        session_id,
        simulation.OrderRequest(symbol="BBCA", side="BUY", quantity=100, order_type="MARKET", price=0),
    )
    assert buy_resp["status"] == "success"
    assert buy_resp["feature_state"] == "beta"

    portfolio = simulation.get_portfolio(session_id)
    assert portfolio["feature_state"] == "beta"
    assert len(portfolio["positions"]) == 1
    assert portfolio["positions"][0]["symbol"] == "BBCA"

    sell_resp = simulation.execute_order(
        session_id,
        simulation.OrderRequest(symbol="BBCA", side="SELL", quantity=100, order_type="MARKET", price=0),
    )
    assert sell_resp["status"] == "success"

    history = simulation.get_trade_history(session_id)
    assert len(history) == 1
    assert history[0]["symbol"] == "BBCA"

    metrics = simulation.get_session_metrics(session_id)
    assert metrics["feature_state"] == "beta"
    assert metrics["total_trades"] == 1


def test_ensemble_prediction_rejects_unsupported_horizon(monkeypatch):
    from api.routes import prediction
    from fastapi import HTTPException

    class StubService:
        def has_model(self, symbol: str) -> bool:
            return True

        def predict(self, symbol: str, n_days: int = 7):
            if n_days != 7:
                raise ValueError(
                    f"Unsupported horizon {n_days} for {symbol}. Model artifact is trained for 7 business days."
                )
            return {
                "symbol": symbol,
                "current_price": 9000.0,
                "predictions": [{"date": "2026-03-10", "predicted_price": 9100.0, "predicted_return": 0.01}],
                "artifact_metadata": {"trained_horizon": 7, "trained_at": "2026-03-09T00:00:00Z"},
            }

    monkeypatch.setattr(prediction, "_prediction_service_or_raise", lambda db: StubService())

    try:
        asyncio.run(prediction.get_ensemble_prediction("BBCA", horizon=14, db=Mock()))
        assert False, "Expected unsupported horizon to raise"
    except HTTPException as exc:
        assert exc.status_code == 400
        assert "Unsupported horizon 14" in exc.detail

    payload = asyncio.run(prediction.get_ensemble_prediction("BBCA", horizon=7, db=Mock()))
    assert payload["feature_state"] == "live"
    assert payload["uncertainty"]["status"] == "not_available"
    assert payload["artifact_metadata"]["trained_horizon"] == 7


def test_training_endpoint_is_explicitly_offline_only(temp_db):
    from api.routes import prediction

    prediction._training_requests.clear()
    _seed_prices(temp_db, "BBCA", days=250, start=date.today() - timedelta(days=300), close=9000.0)
    payload = asyncio.run(
        prediction.train_prediction_model(
            "BBCA",
            prediction.TrainingRequest(lookback_days=200, test_size=0.2, use_exogenous=True),
            db=temp_db,
        )
    )

    assert payload.status_code == 501
    assert payload.body
    assert b'"status":"unavailable"' in payload.body
    assert b'"feature_state":"offline-only"' in payload.body
