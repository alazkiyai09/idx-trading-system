"""Health check routes."""

import json
import os
import signal
import subprocess
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Optional, Any

from fastapi import APIRouter
from fastapi import HTTPException

from api.cache import api_cache
from api.schemas.common import HealthResponse, DataFreshnessResponse
from config.settings import settings

router = APIRouter(tags=["health"])

PROJECT_ROOT = Path(__file__).resolve().parents[2]
UPDATE_DIR = settings.get_data_path("update_jobs")
PRICE_REFRESH_STATUS_FILE = UPDATE_DIR / "price_refresh_status.json"
PRICE_REFRESH_LOG_FILE = UPDATE_DIR / "price_refresh.log"


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


def _read_status_file() -> Dict[str, Any]:
    if not PRICE_REFRESH_STATUS_FILE.exists():
        return {}
    try:
        return json.loads(PRICE_REFRESH_STATUS_FILE.read_text())
    except Exception:
        return {}


def _write_status_file(payload: Dict[str, Any]) -> None:
    PRICE_REFRESH_STATUS_FILE.write_text(json.dumps(payload, indent=2, sort_keys=True))


def _tail_refresh_log(lines: int = 20) -> list[str]:
    if not PRICE_REFRESH_LOG_FILE.exists():
        return []
    try:
        content = PRICE_REFRESH_LOG_FILE.read_text(errors="ignore").splitlines()
        return content[-lines:]
    except Exception:
        return []


def _get_freshness_snapshot() -> Dict[str, Any]:
    warnings = []
    price_data_age_hours: Optional[float] = None
    flow_data_age_hours: Optional[float] = None
    last_price_update: Optional[datetime] = None
    last_flow_update: Optional[datetime] = None
    price_record_count = 0
    flow_record_count = 0

    try:
        from core.data.database import DatabaseManager
        from sqlalchemy import text

        db = DatabaseManager()
        session = db.get_session()
        try:
            result = session.execute(text("SELECT MAX(date) as max_date FROM price_history")).fetchone()
            price_record_count = session.execute(text("SELECT COUNT(*) FROM price_history")).scalar() or 0

            if result and result.max_date:
                max_date = result.max_date
                if isinstance(max_date, str):
                    from datetime import datetime as dt
                    max_date = dt.strptime(max_date, "%Y-%m-%d").date()
                last_price_update = datetime.combine(max_date, datetime.min.time(), tzinfo=timezone.utc)
                age = datetime.now(timezone.utc) - last_price_update
                price_data_age_hours = age.total_seconds() / 3600

                if price_data_age_hours > 24:
                    warnings.append(f"Price data is {price_data_age_hours:.1f} hours old (may be stale)")
                elif price_data_age_hours > 6:
                    warnings.append(f"Price data is {price_data_age_hours:.1f} hours old")

            try:
                result = session.execute(text("SELECT MAX(date) as max_date FROM foreign_flow_history")).fetchone()
                flow_record_count = session.execute(text("SELECT COUNT(*) FROM foreign_flow_history")).scalar() or 0
                if result and result.max_date:
                    max_date = result.max_date
                    if isinstance(max_date, str):
                        from datetime import datetime as dt
                        max_date = dt.strptime(max_date, "%Y-%m-%d").date()
                    last_flow_update = datetime.combine(max_date, datetime.min.time(), tzinfo=timezone.utc)
                    age = datetime.now(timezone.utc) - last_flow_update
                    flow_data_age_hours = age.total_seconds() / 3600
            except Exception:
                pass
        finally:
            session.close()
    except Exception as e:
        warnings.append(f"Could not check data freshness: {str(e)}")

    if price_data_age_hours is None:
        status = "unknown"
    elif price_data_age_hours > 24:
        status = "stale"
    else:
        status = "fresh"

    return {
        "status": status,
        "price_data_age_hours": price_data_age_hours,
        "flow_data_age_hours": flow_data_age_hours,
        "last_price_update": last_price_update,
        "last_flow_update": last_flow_update,
        "warnings": warnings,
        "price_record_count": price_record_count,
        "flow_record_count": flow_record_count,
    }


def _manual_refresh_status() -> Dict[str, Any]:
    freshness = _get_freshness_snapshot()
    status_file = _read_status_file()
    pid = status_file.get("pid")
    running = _process_running(pid)

    if status_file and status_file.get("status") == "running" and not running:
        status_file["status"] = "finished"
        status_file["finished_at"] = _utc_now_iso()
        _write_status_file(status_file)
        api_cache.invalidate("stocks:")
        api_cache.invalidate("dashboard:")

    return {
        "refresh_policy": {
            "expected_frequency": "daily",
            "expected_window": settings.daily_refresh_policy,
            "timezone": "Asia/Jakarta",
            "manual_refresh_available": True,
        },
        "data_status": {
            "price_status": freshness["status"],
            "last_price_update": freshness["last_price_update"].isoformat() if freshness["last_price_update"] else None,
            "last_flow_update": freshness["last_flow_update"].isoformat() if freshness["last_flow_update"] else None,
            "price_record_count": freshness["price_record_count"],
            "flow_record_count": freshness["flow_record_count"],
            "warnings": freshness["warnings"],
        },
        "manual_refresh": {
            "job_status": status_file.get("status", "idle"),
            "pid": pid,
            "started_at": status_file.get("started_at"),
            "finished_at": status_file.get("finished_at"),
            "command": status_file.get("command"),
            "log_path": str(PRICE_REFRESH_LOG_FILE),
            "log_tail": _tail_refresh_log(),
            "is_running": running,
            "stock_list_file": settings.stock_list_file,
        },
        "generated_at": _utc_now_iso(),
    }


def _launch_manual_refresh() -> Dict[str, Any]:
    current = _manual_refresh_status()
    if current["manual_refresh"]["is_running"]:
        raise HTTPException(status_code=409, detail="A price refresh job is already running.")

    stock_list_path = Path(settings.stock_list_file)
    if not stock_list_path.exists():
        raise HTTPException(
            status_code=503,
            detail=f"Stock list file not found: {settings.stock_list_file}. Update Settings.stock_list_file before running refresh.",
        )

    UPDATE_DIR.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "scripts/ingest_all_stocks.py",
        "--stock-file",
        settings.stock_list_file,
        "--database",
        settings.database_url,
        "--batch-size",
        "20",
        "--delay",
        "5",
        "--resume",
    ]

    with PRICE_REFRESH_LOG_FILE.open("ab") as log_handle:
        process = subprocess.Popen(
            command,
            cwd=str(PROJECT_ROOT),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    payload = {
        "status": "running",
        "pid": process.pid,
        "started_at": _utc_now_iso(),
        "finished_at": None,
        "command": " ".join(command),
    }
    _write_status_file(payload)
    return payload


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Basic liveness check.

    Returns system status and version.
    """
    return HealthResponse(
        status="ok",
        version="3.0.0",
        timestamp=datetime.now(),
        components={
            "api": "ok",
            "database": "ok",
        },
    )


@router.get("/health/data", response_model=DataFreshnessResponse)
async def data_freshness_check():
    """Check data freshness.

    Returns data age and freshness warnings.
    """
    freshness = _get_freshness_snapshot()

    return DataFreshnessResponse(
        status=freshness["status"],
        price_data_age_hours=freshness["price_data_age_hours"],
        flow_data_age_hours=freshness["flow_data_age_hours"],
        last_price_update=freshness["last_price_update"],
        last_flow_update=freshness["last_flow_update"],
        warnings=freshness["warnings"],
    )


@router.get("/health/update-status")
async def data_update_status():
    """Get current data refresh state and manual job status."""
    return _manual_refresh_status()


@router.post("/health/update-data")
async def trigger_data_update():
    """Launch a background price refresh job using the existing ingestion script."""
    job = _launch_manual_refresh()
    return {
        "status": "started",
        "message": "Price refresh job started in the background.",
        "job": job,
        "update_status_endpoint": "/health/update-status",
    }


@router.get("/health/detailed")
async def detailed_health():
    """Detailed system health check.

    Returns component-level health information.
    """
    components: Dict[str, Dict] = {
        "api": {"status": "ok", "latency_ms": 0},
        "database": {"status": "ok"},
        "data_freshness": {"status": "unknown"},
        "llm": {"status": "unknown"},
        "notifications": {"status": "unknown"},
    }

    # Check database and get real stats
    try:
        from core.data.database import DatabaseManager
        from sqlalchemy import text

        db = DatabaseManager()
        session = db.get_session()

        # Test database connection
        session.execute(text("SELECT 1"))
        components["database"]["status"] = "ok"

        # Get stock count
        try:
            result = session.execute(text("SELECT COUNT(DISTINCT symbol) FROM price_history")).scalar()
            components["database"]["stock_count"] = result or 0
        except Exception:
            pass

        # Get total record count
        try:
            result = session.execute(text("SELECT COUNT(*) FROM price_history")).scalar()
            components["database"]["record_count"] = result or 0
        except Exception:
            pass

        # Get latest price date
        try:
            result = session.execute(text("SELECT MAX(date) FROM price_history")).scalar()
            if result:
                components["data_freshness"]["last_price_date"] = str(result)
                # Calculate age
                last_date = datetime.combine(result, datetime.min.time(), tzinfo=timezone.utc)
                age_hours = (datetime.now(timezone.utc) - last_date).total_seconds() / 3600
                components["data_freshness"]["price_age_hours"] = round(age_hours, 1)
                components["data_freshness"]["status"] = "fresh" if age_hours < 24 else "stale"
        except Exception:
            pass

        session.close()

    except Exception as e:
        components["database"]["status"] = "error"
        components["database"]["error"] = str(e)

    # Check LLM configuration
    try:
        from config.settings import settings
        if settings.anthropic_api_key or settings.glm_api_key:
            components["llm"]["status"] = "configured"
            components["llm"]["provider"] = "claude" if settings.anthropic_api_key else "glm"
        else:
            components["llm"]["status"] = "not_configured"
    except Exception:
        pass

    # Check notifications
    try:
        from config.settings import settings
        if settings.telegram_bot_token:
            components["notifications"]["status"] = "configured"
        else:
            components["notifications"]["status"] = "not_configured"
    except Exception:
        pass

    overall = "ok" if all(
        c.get("status") in ("ok", "configured", "unknown", "not_configured", "fresh")
        for k, c in components.items()
        if k in ("api", "database")
    ) else "degraded"

    return {
        "status": overall,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "components": components,
    }


@router.get("/health/dashboard-summary")
async def dashboard_summary():
    """Return a compact cached summary payload for the dashboard home page."""
    def build_summary() -> Dict[str, Any]:
        from core.data.database import DatabaseManager
        from sqlalchemy import text

        freshness = _get_freshness_snapshot()
        status = _manual_refresh_status()
        db = DatabaseManager()
        session = db.get_session()
        try:
            stock_count = session.execute(text("SELECT COUNT(*) FROM stock_metadata")).scalar() or 0
            signal_count = 0
            try:
                signal_count = session.execute(text("SELECT COUNT(*) FROM signals WHERE status = 'active'")).scalar() or 0
            except Exception:
                pass
        finally:
            session.close()

        return {
            "stock_count": stock_count,
            "signal_count": signal_count,
            "record_count": freshness["price_record_count"],
            "freshness_status": freshness["status"],
            "price_data_age_hours": freshness["price_data_age_hours"],
            "last_price_update": freshness["last_price_update"].isoformat() if freshness["last_price_update"] else None,
            "warnings": freshness["warnings"],
            "update_status": status["manual_refresh"],
            "refresh_policy": status["refresh_policy"],
        }

    return api_cache.get_or_set("dashboard:summary", ttl_seconds=30, builder=build_summary)
