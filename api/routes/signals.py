"""Signal routes."""

import logging
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query

from api.mappers import map_signal_dict
from api.schemas.signal_schemas import ScanRequest, ScanResponse, SignalResponse, SignalListResponse
from api.dependencies import get_coordinator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/signals", tags=["signals"])

# In-memory signal store (would be replaced by database in production)
_recent_signals: List[dict] = []


@router.post("/scan", response_model=ScanResponse)
async def run_scan(request: ScanRequest):
    """Run a daily scan for trading signals.

    Scans the configured symbol universe for the specified trading mode.
    """
    try:
        from config.trading_modes import TradingMode

        coordinator = get_coordinator(request.mode)
        mode = TradingMode(request.mode)

        report = coordinator.run_daily_scan(
            symbols=request.symbols,
            mode=mode,
        )

        report_dict = report.to_dict()

        # Convert signals to response format
        signals = []
        for sig in report_dict.get("signals", []):
            payload = map_signal_dict(sig)
            signals.append(SignalResponse(
                symbol=payload["symbol"],
                signal_type=payload["signal_type"],
                setup_type=payload["setup_type"],
                entry_price=payload["entry_price"],
                stop_loss=payload["stop_loss"],
                targets=payload["targets"],
                composite_score=payload["composite_score"],
                key_factors=payload["key_factors"],
                risks=payload["risks"],
                mode=request.mode,
                timestamp=datetime.now(),
            ))

        # Store for retrieval
        for sig in signals:
            _recent_signals.append(sig.model_dump())

        return ScanResponse(
            scan_date=report_dict.get("scan_date", datetime.now().date()),
            mode=request.mode,
            result=report_dict.get("result", "unknown"),
            signals_generated=report_dict.get("signals_generated", 0),
            signals_approved=report_dict.get("signals_approved", 0),
            symbols_scanned=report_dict.get("symbols_scanned", 0),
            execution_time_seconds=report_dict.get("execution_time_seconds", 0),
            signals=signals,
            errors=report_dict.get("errors", []),
        )

    except Exception as e:
        logger.exception("Scan failed")
        raise HTTPException(status_code=500, detail=f"Signal scan failed: {e}")


@router.get("", response_model=SignalListResponse)
async def list_signals(
    limit: int = Query(default=20, ge=1, le=100),
    mode: Optional[str] = None,
):
    """Get recent trading signals.

    Args:
        limit: Maximum number of signals to return.
        mode: Filter by trading mode.
    """
    filtered = _recent_signals
    if mode:
        filtered = [s for s in filtered if s.get("mode") == mode]

    signals = filtered[-limit:]
    return SignalListResponse(
        signals=[SignalResponse(**s) for s in signals],
        total=len(signals),
    )


@router.get("/active")
async def get_active_signals():
    """Get currently active trading signals."""
    return _recent_signals[-20:] if _recent_signals else []


@router.get("/{signal_index}", response_model=SignalResponse)
async def get_signal(signal_index: int):
    """Get a specific signal by index.

    Args:
        signal_index: Signal index (0-based).
    """
    if signal_index < 0 or signal_index >= len(_recent_signals):
        raise HTTPException(status_code=404, detail="Signal not found")

    return SignalResponse(**_recent_signals[signal_index])
