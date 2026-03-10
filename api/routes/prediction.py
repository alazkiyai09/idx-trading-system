"""
ML Prediction API endpoints.

Provides endpoints for ensemble predictions, commodity data,
and macro correlation analysis.

IMPORTANT: Mock predictions are for demonstration purposes only.
Do NOT use mock predictions for actual trading decisions.
"""
from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Form
from fastapi.responses import JSONResponse
from typing import Dict, Any, List, Optional
import logging
import os
import json
import subprocess
import sys
from datetime import datetime, date, timedelta, timezone
import numpy as np
import time
from pathlib import Path
from pydantic import BaseModel, Field
from sqlalchemy import text

from core.data.database import DatabaseManager
from core.analysis.macro_correlation import MacroCorrelationAnalyzer
from core.data.commodity_scraper import CommodityScraper
from config.settings import settings, MLConfig

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/prediction", tags=["ML Prediction"])
ENABLE_REAL_PREDICTION = os.getenv("ENABLE_REAL_PREDICTION", "false").lower() == "true"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAINING_DIR = settings.get_data_path("training_jobs")
TRAINING_STATUS_FILE = TRAINING_DIR / "model_training_status.json"
TRAINING_HISTORY_FILE = TRAINING_DIR / "model_training_history.json"
TRAINING_LOG_FILE = TRAINING_DIR / "model_training.log"


# Rate limiting storage (in-memory, resets on server restart)
_training_requests: Dict[str, float] = {}
TRAINING_COOLDOWN_SECONDS = 300  # 5 minutes between training requests per symbol


async def get_db() -> DatabaseManager:
    """Dependency injection for database manager."""
    return DatabaseManager()


# Input validation constants
MIN_SIMULATIONS = 100
MAX_SIMULATIONS = 10000
MIN_HORIZON = 1
MAX_HORIZON = 90
MIN_DAYS = 1
MAX_DAYS = 365


class TrainingRequest(BaseModel):
    """Validated training request payload for the ML workflow."""

    lookback_days: int = Field(default=200, ge=100, le=500)
    test_size: float = Field(default=0.20, gt=0.0, lt=0.5)
    use_exogenous: bool = True


class BatchTrainingRequest(BaseModel):
    """Validated payload for offline batch training launch."""

    symbols: List[str] = Field(min_length=1, max_length=settings.model_training_batch_limit)
    lookback_days: int = Field(default=400, ge=200, le=1000)
    overwrite: bool = False


def _artifact_file_paths(symbol: str) -> Dict[str, Path]:
    artifacts_dir = settings.get_data_path("ml_artifacts")
    return {
        "artifact": artifacts_dir / f"{symbol}_ensemble.pkl",
        "metadata": artifacts_dir / f"{symbol}_ensemble.meta.json",
    }


def _training_readiness_payload(
    db: DatabaseManager,
    symbol: str,
    lookback_days: int,
) -> Dict[str, Any]:
    symbol = symbol.strip().upper()
    price_count = db.get_price_count(symbol)
    minimum_rows = MLConfig.MIN_TOTAL_ROWS
    recommended_rows = max(minimum_rows + 100, lookback_days)

    if price_count < minimum_rows:
        status = "not_ready"
        message = f"Not enough history for ML training yet: {price_count} rows available, need at least {minimum_rows}."
    elif price_count < 300:
        status = "limited"
        message = (
            f"Training can run, but history is still short ({price_count} rows). "
            "Expect lower trust until the stock has a longer listing history."
        )
    else:
        status = "ready"
        message = f"History is sufficient for ML training ({price_count} rows available)."

    return {
        "symbol": symbol,
        "status": status,
        "data_rows": price_count,
        "minimum_rows": minimum_rows,
        "recommended_rows": recommended_rows,
        "lookback_days": lookback_days,
        "message": message,
        "generated_at": _utc_now_iso(),
    }


def _feature_metadata(
    *,
    feature_state: str,
    is_demo: bool,
    data_source: str,
    detail: Optional[str] = None,
) -> Dict[str, Any]:
    """Build consistent feature-state metadata for transparent API responses."""
    payload = {
        "feature_state": feature_state,
        "is_demo": is_demo,
        "data_source": data_source,
        "last_computed_at": datetime.now(timezone.utc).isoformat(),
    }
    if detail:
        payload["feature_detail"] = detail
    return payload


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _process_running(pid: Optional[int]) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _read_training_status() -> Dict[str, Any]:
    if not TRAINING_STATUS_FILE.exists():
        return {}
    try:
        return json.loads(TRAINING_STATUS_FILE.read_text())
    except Exception:
        return {}


def _write_training_status(payload: Dict[str, Any]) -> None:
    TRAINING_STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    TRAINING_STATUS_FILE.write_text(json.dumps(payload, indent=2, sort_keys=True))


def _read_training_history() -> List[Dict[str, Any]]:
    if not TRAINING_HISTORY_FILE.exists():
        return []
    try:
        payload = json.loads(TRAINING_HISTORY_FILE.read_text())
        return payload if isinstance(payload, list) else []
    except Exception:
        return []


def _tail_training_log(lines: int = 30) -> list[str]:
    if not TRAINING_LOG_FILE.exists():
        return []
    try:
        return TRAINING_LOG_FILE.read_text(errors="ignore").splitlines()[-lines:]
    except Exception:
        return []


def _read_artifact_metadata() -> List[Dict[str, Any]]:
    artifacts_dir = settings.get_data_path("ml_artifacts")
    records: List[Dict[str, Any]] = []
    seen_symbols = set()
    for meta_file in sorted(artifacts_dir.glob("*_ensemble.meta.json")):
        try:
            payload = json.loads(meta_file.read_text())
            payload["metadata_path"] = str(meta_file)
            records.append(payload)
            if payload.get("symbol"):
                seen_symbols.add(payload["symbol"])
        except Exception:
            continue
    for artifact_file in sorted(artifacts_dir.glob("*_ensemble.pkl")):
        symbol = artifact_file.name.replace("_ensemble.pkl", "")
        if symbol in seen_symbols:
            continue
        records.append(
            {
                "symbol": symbol,
                "artifact_path": str(artifact_file),
                "artifact_size_bytes": artifact_file.stat().st_size,
                "trained_at": datetime.utcfromtimestamp(artifact_file.stat().st_mtime).isoformat() + "Z",
                "source_latest_date": None,
                "source_row_count": None,
                "trained_horizon": None,
                "artifact_type": "unknown",
                "status": "artifact_only",
            }
        )
    records.sort(key=lambda item: item.get("trained_at") or "", reverse=True)
    return records


def _latest_price_history_date(db: DatabaseManager) -> Optional[str]:
    with db.get_session() as session:
        result = session.execute(text("SELECT MAX(date) FROM price_history")).scalar()
    if result is None:
        return None
    return str(result)


def _training_runtime_status(db: DatabaseManager) -> Dict[str, Any]:
    status_file = _read_training_status()
    pid = status_file.get("pid")
    running = _process_running(pid)
    if status_file and status_file.get("status") == "running" and not running:
        status_file["status"] = "finished"
        status_file.setdefault("finished_at", _utc_now_iso())
        _write_training_status(status_file)

    artifacts = _read_artifact_metadata()
    latest_market_date = _latest_price_history_date(db)
    up_to_date = 0
    for item in artifacts:
        if item.get("source_latest_date") == latest_market_date:
            up_to_date += 1

    total_symbols = len(status_file.get("symbols", []))
    completed_count = len(status_file.get("completed", []))
    failed_count = len(status_file.get("failed", []))
    skipped_count = len(status_file.get("skipped", []))
    finished_count = completed_count + failed_count + skipped_count
    progress_pct = round((finished_count / total_symbols) * 100, 1) if total_symbols else 0.0

    return {
        "batch_limit": settings.model_training_batch_limit,
        "latest_market_date": latest_market_date,
        "artifacts": {
            "total_models": len(artifacts),
            "up_to_date_models": up_to_date,
            "recent_models": artifacts[:10],
        },
        "job": {
            "job_id": status_file.get("job_id"),
            "status": status_file.get("status", "idle"),
            "pid": pid,
            "started_at": status_file.get("started_at"),
            "finished_at": status_file.get("finished_at"),
            "symbols": status_file.get("symbols", []),
            "lookback_days": status_file.get("lookback_days"),
            "overwrite": status_file.get("overwrite", False),
            "completed": status_file.get("completed", []),
            "failed": status_file.get("failed", []),
            "skipped": status_file.get("skipped", []),
            "current_symbol": status_file.get("current_symbol"),
            "symbol_statuses": status_file.get("symbol_statuses", {}),
            "total_symbols": total_symbols,
            "completed_count": completed_count,
            "failed_count": failed_count,
            "skipped_count": skipped_count,
            "finished_count": finished_count,
            "progress_pct": progress_pct,
            "log_path": str(TRAINING_LOG_FILE),
            "log_tail": _tail_training_log(),
            "is_running": running,
        },
        "history": _read_training_history()[-10:][::-1],
        "generated_at": _utc_now_iso(),
    }


def _launch_training_job(request: BatchTrainingRequest, db: DatabaseManager) -> Dict[str, Any]:
    current = _training_runtime_status(db)
    if current["job"]["is_running"]:
        raise HTTPException(status_code=409, detail="A model training job is already running.")

    symbols = [symbol.strip().upper() for symbol in request.symbols if symbol.strip()]
    if not symbols:
        raise HTTPException(status_code=400, detail="At least one symbol is required.")
    if len(symbols) > settings.model_training_batch_limit:
        raise HTTPException(
            status_code=400,
            detail=f"Batch limit exceeded: {len(symbols)} > {settings.model_training_batch_limit}",
        )

    for symbol in symbols:
        price_count = db.get_price_count(symbol)
        if price_count < MLConfig.MIN_TOTAL_ROWS:
            raise HTTPException(
                status_code=400,
                detail=f"Insufficient data for {symbol}: {price_count} rows available.",
            )

    TRAINING_DIR.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "scripts/train_models.py",
        "--symbols",
        *symbols,
        "--lookback-days",
        str(request.lookback_days),
        "--status-file",
        str(TRAINING_STATUS_FILE),
        "--history-file",
        str(TRAINING_HISTORY_FILE),
    ]
    if request.overwrite:
        command.append("--overwrite")

    with TRAINING_LOG_FILE.open("ab") as log_handle:
        process = subprocess.Popen(
            command,
            cwd=str(PROJECT_ROOT),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    payload = {
        "job_id": datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S"),
        "status": "running",
        "pid": process.pid,
        "started_at": _utc_now_iso(),
        "finished_at": None,
        "symbols": symbols,
        "lookback_days": request.lookback_days,
        "overwrite": request.overwrite,
        "completed": [],
        "failed": [],
        "skipped": [],
        "current_symbol": None,
        "symbol_statuses": {symbol: {"status": "pending"} for symbol in symbols},
        "command": " ".join(command),
    }
    _write_training_status(payload)
    return payload


def _validate_simulation_params(n_simulations: int, horizon_days: int) -> None:
    """Validate Monte Carlo simulation parameters."""
    if n_simulations < MIN_SIMULATIONS or n_simulations > MAX_SIMULATIONS:
        raise HTTPException(
            status_code=400,
            detail=f"n_simulations must be between {MIN_SIMULATIONS} and {MAX_SIMULATIONS}"
        )
    if horizon_days < MIN_HORIZON or horizon_days > MAX_HORIZON:
        raise HTTPException(
            status_code=400,
            detail=f"horizon_days must be between {MIN_HORIZON} and {MAX_HORIZON}"
        )


def _check_training_rate_limit(symbol: str) -> None:
    """Check if training request is rate-limited."""
    now = time.time()
    last_request = _training_requests.get(symbol, 0)

    if now - last_request < TRAINING_COOLDOWN_SECONDS:
        remaining = int(TRAINING_COOLDOWN_SECONDS - (now - last_request))
        raise HTTPException(
            status_code=429,
            detail=f"Training rate limit exceeded. Please wait {remaining} seconds before training {symbol} again."
        )

    _training_requests[symbol] = now


def _prediction_service_or_raise(db: DatabaseManager):
    """Load the real prediction service or fail truthfully."""
    if not ENABLE_REAL_PREDICTION:
        raise HTTPException(
            status_code=503,
            detail="Real prediction service is disabled. Set ENABLE_REAL_PREDICTION=true to enable model inference."
        )

    try:
        from core.ml_prediction.service import PredictionService
    except ImportError as exc:
        logger.error(f"Prediction service import failed: {exc}")
        raise HTTPException(
            status_code=503,
            detail="ML prediction service is unavailable in this deployment."
        ) from exc

    return PredictionService(db)


# ─── Commodity Endpoints (must come before /{symbol}) ─────────────────────

@router.get("/commodities")
async def get_commodity_prices(
    days: int = 30,
    db: DatabaseManager = Depends(get_db)
):
    """
    Get latest commodity prices (Gold, Silver, Oil).

    Args:
        days: Number of days of history (default 30, max 365)
    """
    days = min(max(days, MIN_DAYS), MAX_DAYS)  # Clamp to valid range

    commodities = ["GOLD", "SILVER", "OIL"]
    result = {}
    start_date = date.today() - timedelta(days=days)

    for commodity in commodities:
        try:
            prices = db.get_commodity_prices(commodity, start_date=start_date)
            if prices:
                # Access ORM attributes
                latest = prices[-1]
                result[commodity] = {
                    "latest": {
                        "date": str(latest.date),
                        "close": float(latest.close),
                        "open": float(latest.open),
                        "high": float(latest.high),
                        "low": float(latest.low),
                        "volume": float(latest.volume) if latest.volume else 0,
                    },
                    "history": [
                        {
                            "date": str(p.date),
                            "close": float(p.close),
                            "open": float(p.open),
                            "high": float(p.high),
                            "low": float(p.low),
                        } for p in prices
                    ]
                }
        except Exception as e:
            logger.warning(f"Error fetching {commodity} prices: {e}")

    if not result:
        # Return mock data if no commodity data available
        mock_data = _mock_commodity_prices()
        mock_data["_warning"] = "⚠️ DEMO DATA - No real commodity data available. Run POST /prediction/commodities/refresh first."
        mock_data["last_updated"] = datetime.now(timezone.utc).isoformat()
        return JSONResponse(content=mock_data)

    # Add last updated timestamp
    result["last_updated"] = datetime.now(timezone.utc).isoformat()
    return result


@router.post("/commodities/refresh")
async def refresh_commodity_data(
    days: int = 365,
    db: DatabaseManager = Depends(get_db)
):
    """
    Fetch latest commodity data from Yahoo Finance.

    Args:
        days: Number of days of history to fetch (default 365, max 365)
    """
    days = min(max(days, MIN_DAYS), MAX_DAYS)

    try:
        scraper = CommodityScraper(db_manager=db)
        total_inserted = 0
        results = {}

        for commodity in ["GOLD", "SILVER", "OIL"]:
            try:
                data = scraper.fetch_commodity_data(commodity, days=days)
                if data:
                    inserted = scraper.save_to_database(data)
                    total_inserted += inserted
                    results[commodity] = {"records": len(data), "inserted": inserted}
            except Exception as e:
                logger.error(f"Error fetching {commodity}: {e}")
                results[commodity] = {"error": str(e)}

        return {
            "status": "success",
            "message": f"Refreshed commodity data: {total_inserted} total records inserted",
            "commodities": results
        }
    except Exception as e:
        logger.error(f"Error refreshing commodity data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/training/status")
async def training_status(db: DatabaseManager = Depends(get_db)):
    """Get offline model training job state and artifact summary."""
    return _training_runtime_status(db)


@router.get("/models")
async def list_models():
    """List available trained model artifacts."""
    models = _read_artifact_metadata()
    return {
        "models": models,
        "total_models": len(models),
        "generated_at": _utc_now_iso(),
    }


@router.delete("/models/{symbol}")
async def delete_model(symbol: str):
    """Delete a trained model artifact and its metadata sidecar."""
    symbol = symbol.strip().upper()
    paths = _artifact_file_paths(symbol)
    removed = []
    for path in paths.values():
        if path.exists():
            path.unlink()
            removed.append(str(path))

    if not removed:
        raise HTTPException(status_code=404, detail=f"No stored model files found for {symbol}")

    return {
        "status": "deleted",
        "symbol": symbol,
        "removed_files": removed,
        "generated_at": _utc_now_iso(),
    }


@router.post("/models/upload")
async def upload_model(
    artifact_file: UploadFile = File(...),
    metadata_file: UploadFile | None = File(default=None),
    symbol: str | None = Form(default=None),
):
    """Upload a trained model artifact and optional metadata sidecar."""
    inferred_symbol = symbol or artifact_file.filename.replace("_ensemble.pkl", "").split(".")[0]
    target_symbol = inferred_symbol.strip().upper()
    if not target_symbol:
        raise HTTPException(status_code=400, detail="A symbol is required for upload.")
    if not artifact_file.filename.endswith(".pkl"):
        raise HTTPException(status_code=400, detail="Artifact upload must be a .pkl file.")
    if metadata_file and not metadata_file.filename.endswith(".json"):
        raise HTTPException(status_code=400, detail="Metadata upload must be a .json file.")

    paths = _artifact_file_paths(target_symbol)
    paths["artifact"].parent.mkdir(parents=True, exist_ok=True)

    artifact_bytes = await artifact_file.read()
    paths["artifact"].write_bytes(artifact_bytes)

    metadata_path = None
    if metadata_file:
        metadata_bytes = await metadata_file.read()
        paths["metadata"].write_bytes(metadata_bytes)
        metadata_path = str(paths["metadata"])

    return {
        "status": "uploaded",
        "symbol": target_symbol,
        "artifact_path": str(paths["artifact"]),
        "metadata_path": metadata_path,
        "generated_at": _utc_now_iso(),
    }


@router.post("/training/run")
async def run_training_batch(
    request: BatchTrainingRequest,
    db: DatabaseManager = Depends(get_db),
):
    """Launch a capped offline training batch in the background."""
    job = _launch_training_job(request, db)
    return {
        "status": "started",
        "message": "Offline model training started in the background.",
        "job": job,
        "training_status_endpoint": "/prediction/training/status",
    }


# ─── Symbol-Specific Prediction Endpoints ─────────────────────────────────


@router.get("/training/readiness/{symbol}")
async def training_readiness(
    symbol: str,
    lookback_days: int = 400,
    db: DatabaseManager = Depends(get_db),
):
    """Return ML training readiness for a symbol based on available history."""
    return _training_readiness_payload(db, symbol, lookback_days)

@router.get("/{symbol}")
async def get_prediction(symbol: str, db: DatabaseManager = Depends(get_db)):
    """
    Get 7-day ML ensemble prediction for a specific stock.
    Returns 404 if the model isn't trained for this stock.
    """
    try:
        service = _prediction_service_or_raise(db)
        if not service.has_model(symbol):
            raise HTTPException(status_code=404, detail=f"No trained model available for {symbol}")

        result = service.predict(symbol)
        result.update(_feature_metadata(
            feature_state="live",
            is_demo=False,
            data_source="ml_artifact",
        ))
        result["is_mock"] = False
        result["model_status"] = "trained"
        result["disclaimer"] = "Real model prediction. Past performance does not guarantee future results."
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error for {symbol}: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"Prediction service failed for {symbol}. Check model artifacts and feature pipeline."
        ) from e


@router.get("/ensemble/{symbol}")
async def get_ensemble_prediction(
    symbol: str,
    horizon: int = 7,
    db: DatabaseManager = Depends(get_db)
):
    """
    Get multi-model ensemble prediction for the artifact's trained horizon.

    Returns predictions from the stored ensemble artifact. Unsupported horizons
    are rejected explicitly instead of being approximated.
    """
    # Validate horizon
    if horizon < 1 or horizon > 30:
        raise HTTPException(status_code=400, detail="horizon must be between 1 and 30 days")

    try:
        service = _prediction_service_or_raise(db)
        if not service.has_model(symbol):
            raise HTTPException(status_code=404, detail=f"No trained model available for {symbol}")

        result = service.predict(symbol, n_days=horizon)
        result.update(_feature_metadata(
            feature_state="live",
            is_demo=False,
            data_source="ml_artifact",
        ))
        result["is_mock"] = False
        result["model_status"] = "trained"
        result["uncertainty"] = {
            "status": "not_available",
            "method": None,
            "message": "Uncertainty bands are not returned because calibrated interval artifacts are not available in this build.",
        }
        result["disclaimer"] = "Real model prediction. Past performance does not guarantee future results."
        return result
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as e:
        logger.error(f"Ensemble prediction error for {symbol}: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"Ensemble prediction service failed for {symbol}. Check model artifacts and feature pipeline."
        ) from e


@router.post("/train/{symbol}")
async def train_prediction_model(
    symbol: str,
    request: TrainingRequest,
    db: DatabaseManager = Depends(get_db)
):
    """
    Trigger training for a specific stock.

    Args:
        symbol: Stock symbol to train
    Note: Training is rate-limited to one request per symbol every 5 minutes.
    """
    readiness = _training_readiness_payload(db, symbol, request.lookback_days)
    if readiness["status"] == "not_ready":
        raise HTTPException(status_code=400, detail=readiness["message"])

    # Check rate limit
    _check_training_rate_limit(symbol)

    # Check if we have enough data. Lookback is a max recent window, not a hard requirement.
    price_count = db.get_price_count(symbol)
    minimum_rows = MLConfig.MIN_TOTAL_ROWS
    if price_count < minimum_rows:
        raise HTTPException(
            status_code=400,
            detail=f"Insufficient data for {symbol}: {price_count} rows (need {minimum_rows}+)"
        )

    return JSONResponse(
        status_code=501,
        content={
            "status": "unavailable",
            "message": "Training is offline-only in this runtime. The dashboard action is informational and does not enqueue a job.",
            "symbol": symbol,
            "requested_config": request.model_dump(),
            "data_rows": price_count,
            "next_step": "Use Settings -> Model Ops to launch an offline training batch or run scripts/train_models.py directly, then publish the updated artifact before requesting live predictions.",
            **_feature_metadata(
                feature_state="offline-only",
                is_demo=False,
                data_source="offline_pipeline",
            ),
        }
    )


@router.get("/correlation/{symbol}")
async def get_macro_correlation(
    symbol: str,
    sector: Optional[str] = None,
    db: DatabaseManager = Depends(get_db)
):
    """
    Get stock-commodity correlation analysis.

    Args:
        symbol: Stock symbol
        sector: Optional sector for impact analysis
    """
    try:
        analyzer = MacroCorrelationAnalyzer(db)

        # Get correlations
        correlations = analyzer.compute_correlations(symbol)

        # Get signals
        signals = analyzer.generate_signals(symbol, sector)

        # Get sector impact if sector provided
        sector_impact = None
        if sector:
            sector_impact = analyzer.get_commodity_impact(sector)

        return {
            "symbol": symbol,
            "sector": sector,
            "correlations": correlations,
            "signals": signals,
            "sector_impact": sector_impact
        }
    except Exception as e:
        logger.error(f"Error computing correlations for {symbol}: {e}")
        return {
            "symbol": symbol,
            "sector": sector,
            "correlations": {},
            "signals": [],
            "sector_impact": None,
            "is_mock": True,
            "warning": "Correlation engine unavailable. Returning empty demo response.",
        }


@router.get("/monte-carlo/{symbol}")
async def get_monte_carlo_simulation(
    symbol: str,
    n_simulations: int = 1000,
    horizon_days: int = 30,
    db: DatabaseManager = Depends(get_db)
):
    """
    Run Monte Carlo simulation for price distribution.

    Uses Geometric Brownian Motion model with historical parameters.

    Args:
        symbol: Stock symbol
        n_simulations: Number of simulation paths (100-10000)
        horizon_days: Simulation horizon in days (1-90)

    Returns:
        Dict with statistics, percentiles, and sample paths for visualization
    """
    # Validate parameters
    _validate_simulation_params(n_simulations, horizon_days)

    # Get historical prices for parameter estimation
    latest = db.get_latest_price(symbol)
    if not latest:
        return _mock_monte_carlo(symbol, horizon_days, n_simulations)

    current_price = float(latest.close)

    # Get historical volatility (need at least 30 days)
    start_date = date.today() - timedelta(days=180)  # Look back 6 months
    prices = db.get_prices(symbol, start_date, date.today())
    if len(prices) < 30:
        return _mock_monte_carlo(symbol, horizon_days, n_simulations)

    # Calculate daily returns and volatility
    # Use at least 180 days (6 months) for better volatility estimation
    n_days_data = min(180, len(prices))
    closes = [float(p.close) for p in prices[-n_days_data:]]
    returns = np.diff(np.log(closes))
    mu = np.mean(returns)
    sigma = np.std(returns)

    # Handle zero volatility edge case
    if sigma == 0 or np.isnan(sigma):
        logger.warning(f"Zero or NaN volatility detected for {symbol}, using minimum fallback")
        sigma = 0.001  # 0.1% daily vol as minimum fallback
    if np.isnan(mu):
        mu = 0.0  # Zero drift if NaN

    # Run Monte Carlo simulation (Geometric Brownian Motion)
    # NOTE: No fixed seed - each simulation should be statistically independent
    dt = 1  # Daily time step
    paths = np.zeros((n_simulations, horizon_days + 1))
    paths[:, 0] = current_price

    for t in range(1, horizon_days + 1):
        z = np.random.standard_normal(n_simulations)
        paths[:, t] = paths[:, t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)

    # Calculate statistics on final prices
    final_prices = paths[:, -1]
    mean_price = float(np.mean(final_prices))
    std_price = float(np.std(final_prices))
    median_price = float(np.median(final_prices))

    # Calculate VaR and CVaR (95% confidence)
    # VaR is the 5th percentile of final prices (worst 5% scenarios)
    var_95_price = float(np.percentile(final_prices, 5))
    # CVaR (Expected Shortfall) - use at least 5% of simulations for tail
    # This ensures we capture the tail even with small sample sizes
    tail_size = max(5, int(np.ceil(0.05 * n_simulations)))
    sorted_prices = np.sort(final_prices)
    worst_tail = sorted_prices[:tail_size]
    cvar_95_price = float(np.mean(worst_tail))

    # Calculate probabilities
    prob_loss = float(np.mean(final_prices < current_price))
    prob_gain_10 = float(np.mean(final_prices > current_price * 1.10))

    # Calculate percentiles
    percentiles = {
        "p05": float(np.percentile(final_prices, 5)),
        "p10": float(np.percentile(final_prices, 10)),
        "p25": float(np.percentile(final_prices, 25)),
        "p50": float(np.percentile(final_prices, 50)),
        "p75": float(np.percentile(final_prices, 75)),
        "p90": float(np.percentile(final_prices, 90)),
        "p95": float(np.percentile(final_prices, 95)),
    }

    return {
        "symbol": symbol,
        "current_price": current_price,
        "horizon_days": horizon_days,
        "n_simulations": n_simulations,
        "model": {
            "type": "Geometric Brownian Motion",
            "drift_daily": float(mu),
            "volatility_daily": float(sigma),
            "data_points_used": len(closes),
        },
        "statistics": {
            "mean_final_price": mean_price,
            "median_final_price": median_price,
            "std_final_price": std_price,
            "expected_return_pct": (mean_price / current_price - 1) * 100,
            "median_return_pct": (median_price / current_price - 1) * 100,
            "var_95_price": var_95_price,
            "var_95_pct": (var_95_price / current_price - 1) * 100,
            "cvar_95_price": cvar_95_price,
            "cvar_95_pct": (cvar_95_price / current_price - 1) * 100,
            "prob_loss_pct": prob_loss * 100,
            "prob_gain_10pct_pct": prob_gain_10 * 100,
        },
        "percentiles": percentiles,
        "sample_paths": paths[:min(100, n_simulations), :].tolist(),  # Sample paths for visualization
        "final_prices_sample": final_prices[:min(500, n_simulations)].tolist(),  # For histogram
        "disclaimer": "Monte Carlo simulation uses historical parameters and assumes GBM. "
                      "Actual returns may differ significantly. Not financial advice."
    }


def _mock_prediction(symbol: str, db: DatabaseManager) -> Dict[str, Any]:
    """Provide a mocked prediction response for UI demonstration.

    ⚠️ DO NOT USE FOR TRADING DECISIONS - This is random data.
    """
    import pandas as pd

    # Get last price from database
    session = db.get_session()
    try:
        from sqlalchemy import text
        result = session.execute(
            text("SELECT close FROM price_history WHERE symbol = :symbol ORDER BY date DESC LIMIT 1"),
            {"symbol": symbol}
        ).scalar()

        if not result:
            raise HTTPException(status_code=404, detail=f"No data for {symbol}")

        current_price = float(result)
    finally:
        session.close()

    # Generate 7 days of random walk (NO positive drift - fair demo)
    np.random.seed(hash(symbol) % 10000)
    daily_vol = 0.025
    returns = np.random.normal(0.0, daily_vol, 7)  # Zero drift for fair demo

    prices = [current_price]
    for r in returns:
        prices.append(prices[-1] * (1 + r))
    prices = prices[1:]

    dates = pd.date_range(start=datetime.now() + pd.Timedelta(days=1), periods=7, freq='B')

    return {
        "symbol": symbol,
        "current_price": current_price,
        "is_mock": True,
        "warning": "⚠️ DEMO DATA - This is a random walk for UI demonstration. "
                   "Do NOT use for trading decisions. Train a real model first.",
        **_feature_metadata(
            feature_state="demo",
            is_demo=True,
            data_source="synthetic_random_walk",
        ),
        "predictions": [
            {
                "date": d.strftime("%Y-%m-%d"),
                "predicted_price": round(float(p), 0),
                "predicted_return": round(float(r), 4)
            } for d, p, r in zip(dates, prices, returns)
        ]
    }


def _mock_ensemble_prediction(symbol: str, db: DatabaseManager, horizon: int) -> Dict[str, Any]:
    """Provide mocked ensemble prediction with confidence bounds.

    ⚠️ DO NOT USE FOR TRADING DECISIONS - This is random data.
    """
    import pandas as pd

    base = _mock_prediction(symbol, db)

    # Generate predictions for the requested horizon
    current_price = base["current_price"]
    np.random.seed(hash(symbol + str(horizon)) % 10000)
    daily_vol = 0.025
    returns = np.random.normal(0.0, daily_vol, horizon)

    prices = [current_price]
    for r in returns:
        prices.append(prices[-1] * (1 + r))
    prices = prices[1:]

    dates = pd.date_range(start=datetime.now() + pd.Timedelta(days=1), periods=horizon, freq='B')

    predictions = [
        {
            "date": d.strftime("%Y-%m-%d"),
            "predicted_price": round(float(p), 0),
            "predicted_return": round(float(r), 4)
        } for d, p, r in zip(dates, prices, returns)
    ]

    return {
        "symbol": symbol,
        "current_price": current_price,
        "is_mock": True,
        "warning": "⚠️ DEMO DATA - This is a random walk for UI demonstration. "
                   "Do NOT use for trading decisions. Train a real model first.",
        "predictions": predictions,
        "model_contributions": {
            "lstm": 0.40,
            "cnn_lstm": 0.35,
            "svr": 0.25,
            "note": "Mock weights - no real model exists"
        },
        "uncertainty": {
            "status": "demo_only",
            "method": "fixed_percentage_band",
            "message": "This demo response uses a placeholder band and is not statistically valid.",
        },
        **_feature_metadata(
            feature_state="demo",
            is_demo=True,
            data_source="synthetic_random_walk",
        ),
    }


def _mock_commodity_prices() -> Dict[str, Any]:
    """Provide mock commodity prices when no data available.

    ⚠️ DO NOT USE FOR TRADING DECISIONS - This is fake data.
    """
    import pandas as pd

    np.random.seed(42)

    base_prices = {"GOLD": 2000.0, "SILVER": 25.0, "OIL": 75.0}
    result = {}

    for commodity, base_price in base_prices.items():
        dates = pd.date_range(end=datetime.now(), periods=30, freq='D')

        # Generate price movements (zero drift)
        returns = np.random.normal(0.0, 0.015, 30)
        prices = [base_price]
        for r in returns:
            prices.append(prices[-1] * (1 + r))
        prices = prices[1:]

        result[commodity] = {
            "latest": {
                "date": str(dates[-1].date()),
                "close": float(prices[-1]),
                "open": float(prices[-2]) if len(prices) > 1 else float(prices[-1]),
                "high": float(max(prices[-5:])) if len(prices) > 5 else float(prices[-1]),
                "low": float(min(prices[-5:])) if len(prices) > 5 else float(prices[-1]),
                "volume": 1000000,
            },
            "history": [
                {
                    "date": str(d.date()),
                    "close": float(p),
                    "open": float(prices[max(0, i-1)]) if i > 0 else float(p),
                    "high": float(p * 1.01),
                    "low": float(p * 0.99),
                } for i, (d, p) in enumerate(zip(dates, prices))
            ]
        }

    return result


def _mock_monte_carlo(symbol: str, horizon_days: int, n_simulations: int) -> Dict[str, Any]:
    """Provide a lightweight Monte Carlo response when real data is unavailable."""
    current_price = 10000.0
    np.random.seed(hash((symbol, horizon_days, n_simulations)) % 10000)
    returns = np.random.normal(0.0, 0.02, size=n_simulations)
    final_prices = current_price * (1 + returns)
    sample_paths = np.tile(
        np.linspace(current_price, float(np.mean(final_prices)), horizon_days + 1),
        (min(100, n_simulations), 1),
    )

    return {
        "symbol": symbol,
        "current_price": current_price,
        "horizon_days": horizon_days,
        "n_simulations": n_simulations,
        "model": {
            "type": "Mock GBM",
            "drift_daily": 0.0,
            "volatility_daily": 0.02,
            "data_points_used": 0,
        },
        "statistics": {
            "mean_final_price": float(np.mean(final_prices)),
            "median_final_price": float(np.median(final_prices)),
            "std_final_price": float(np.std(final_prices)),
            "expected_return_pct": float((np.mean(final_prices) / current_price - 1) * 100),
            "median_return_pct": float((np.median(final_prices) / current_price - 1) * 100),
            "var_95_price": float(np.percentile(final_prices, 5)),
            "var_95_pct": float((np.percentile(final_prices, 5) / current_price - 1) * 100),
            "cvar_95_price": float(np.mean(np.sort(final_prices)[:max(5, int(np.ceil(0.05 * n_simulations)))])),
            "cvar_95_pct": 0.0,
            "prob_loss_pct": float(np.mean(final_prices < current_price) * 100),
            "prob_gain_10pct_pct": float(np.mean(final_prices > current_price * 1.10) * 100),
        },
        "percentiles": {
            "p05": float(np.percentile(final_prices, 5)),
            "p10": float(np.percentile(final_prices, 10)),
            "p25": float(np.percentile(final_prices, 25)),
            "p50": float(np.percentile(final_prices, 50)),
            "p75": float(np.percentile(final_prices, 75)),
            "p90": float(np.percentile(final_prices, 90)),
            "p95": float(np.percentile(final_prices, 95)),
        },
        "sample_paths": sample_paths.tolist(),
        "final_prices_sample": final_prices[:min(500, n_simulations)].tolist(),
        "is_mock": True,
        **_feature_metadata(
            feature_state="demo",
            is_demo=True,
            data_source="mock_gbm",
        ),
        "disclaimer": "Mock Monte Carlo response. Real historical data unavailable.",
    }
