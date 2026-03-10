import uuid
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from core.data.database import (
    DatabaseManager,
    OpenPositions,
    PriceHistory,
    SimulationSession,
    SimulationTrade,
    TradeHistory,
)

router = APIRouter(prefix="/simulation", tags=["Simulation"])

IDX_LOT_SIZE = 100


class CreateSimulationRequest(BaseModel):
    name: str = "My Simulation"
    mode: str = "live"
    trading_mode: str = "swing"
    initial_capital: float = Field(default=100000000, gt=0)
    start_date: Optional[date] = None
    end_date: Optional[date] = None


class OrderRequest(BaseModel):
    symbol: str
    side: str
    quantity: int = Field(..., gt=0)
    order_type: str = "MARKET"
    price: float = Field(default=0.0, ge=0.0)
    targets: List[float] = Field(default_factory=list)


def _feature_metadata(
    *,
    feature_state: str,
    is_demo: bool,
    data_source: str,
    detail: Optional[str] = None,
) -> Dict[str, Any]:
    payload = {
        "feature_state": feature_state,
        "is_demo": is_demo,
        "data_source": data_source,
        "last_computed_at": datetime.now(timezone.utc).isoformat(),
    }
    if detail:
        payload["feature_detail"] = detail
    return payload


def _get_session_or_404(db: DatabaseManager, session_id: str) -> SimulationSession:
    session = db.get_session()
    try:
        record = session.query(SimulationSession).filter_by(session_id=session_id).first()
        if not record:
            raise HTTPException(status_code=404, detail="Session not found")
        session.expunge(record)
        return record
    finally:
        session.close()


def _save_session_updates(db: DatabaseManager, session_obj: SimulationSession, **updates: Any) -> None:
    payload = {
        "session_id": session_obj.session_id,
        "name": session_obj.name,
        "mode": session_obj.mode,
        "trading_mode": session_obj.trading_mode,
        "start_date": session_obj.start_date,
        "end_date": session_obj.end_date,
        "initial_capital": session_obj.initial_capital,
        "current_capital": session_obj.current_capital,
        "status": session_obj.status,
        "total_trades": session_obj.total_trades,
        "win_rate": session_obj.win_rate,
        "total_pnl": session_obj.total_pnl,
        "max_drawdown": session_obj.max_drawdown,
        "sharpe_ratio": session_obj.sharpe_ratio,
    }
    payload.update(updates)
    db.save_simulation_session(payload)


def _validate_replay_dates(db: DatabaseManager, start_date: date, end_date: Optional[date]) -> date:
    effective_end = end_date or start_date
    if effective_end < start_date:
        raise HTTPException(status_code=400, detail="end_date must be on or after start_date")

    session = db.get_session()
    try:
        first_available = (
            session.query(PriceHistory.date)
            .filter(PriceHistory.date >= start_date)
            .order_by(PriceHistory.date.asc())
            .first()
        )
    finally:
        session.close()

    if not first_available:
        raise HTTPException(status_code=400, detail="Replay mode requires historical market data on or after start_date")
    return effective_end


def _session_positions(db: DatabaseManager, session_id: str) -> List[OpenPositions]:
    session = db.get_session()
    try:
        records = (
            session.query(OpenPositions)
            .filter(OpenPositions.position_id.like(f"{session_id}:%"))
            .order_by(OpenPositions.entry_date.asc(), OpenPositions.id.asc())
            .all()
        )
        for record in records:
            session.expunge(record)
        return records
    finally:
        session.close()


def _session_trades(db: DatabaseManager, session_id: str) -> List[TradeHistory]:
    session = db.get_session()
    try:
        records = (
            session.query(TradeHistory)
            .join(SimulationTrade, SimulationTrade.trade_id == TradeHistory.trade_id)
            .filter(SimulationTrade.session_id == session_id)
            .order_by(TradeHistory.exit_date.asc(), TradeHistory.entry_date.asc(), TradeHistory.id.asc())
            .all()
        )
        for record in records:
            session.expunge(record)
        return records
    finally:
        session.close()


def _price_on_or_before(db: DatabaseManager, symbol: str, as_of: date) -> Optional[float]:
    session = db.get_session()
    try:
        row = (
            session.query(PriceHistory)
            .filter(PriceHistory.symbol == symbol, PriceHistory.date <= as_of)
            .order_by(PriceHistory.date.desc())
            .first()
        )
        return float(row.close) if row else None
    finally:
        session.close()


def _execution_price(db: DatabaseManager, session_obj: SimulationSession, req: OrderRequest) -> float:
    if req.order_type.upper() == "LIMIT":
        if req.price <= 0:
            raise HTTPException(status_code=400, detail="Limit orders require a positive price")
        return float(req.price)

    if session_obj.mode == "replay":
        cursor = session_obj.end_date or session_obj.start_date
        if not cursor:
            raise HTTPException(status_code=400, detail="Replay session is missing a current_date cursor")
        replay_price = _price_on_or_before(db, req.symbol, cursor)
        if replay_price is None:
            raise HTTPException(status_code=400, detail=f"No replay price available for {req.symbol} on or before {cursor}")
        return replay_price

    latest = db.get_latest_price(req.symbol)
    if latest:
        return float(latest.close)
    if req.price > 0:
        return float(req.price)
    raise HTTPException(status_code=400, detail=f"No market price available for {req.symbol}")


def _portfolio_payload(db: DatabaseManager, session_obj: SimulationSession) -> Dict[str, Any]:
    cash = float(session_obj.current_capital or session_obj.initial_capital or 0.0)
    positions = _session_positions(db, session_obj.session_id)
    current_date = session_obj.end_date or session_obj.start_date or date.today()
    position_rows: List[Dict[str, Any]] = []
    market_value = 0.0

    for position in positions:
        current_price = _price_on_or_before(db, position.symbol, current_date) or float(position.entry_price)
        position_value = current_price * position.quantity
        unrealized_pnl = (current_price - float(position.entry_price)) * position.quantity
        market_value += position_value
        position_rows.append(
            {
                "symbol": position.symbol,
                "side": "LONG",
                "quantity": int(position.quantity),
                "entry_date": position.entry_date.isoformat(),
                "entry_price": float(position.entry_price),
                "current_price": float(current_price),
                "market_value": float(position_value),
                "unrealized_pnl": float(unrealized_pnl),
            }
        )

    total_equity = cash + market_value
    return {
        "capital": float(total_equity),
        "cash": float(cash),
        "market_value": float(market_value),
        "pnl": float(total_equity - float(session_obj.initial_capital or 0.0)),
        "positions": position_rows,
        "current_date": current_date.isoformat(),
        **_feature_metadata(
            feature_state="beta",
            is_demo=False,
            data_source="session_ledger",
            detail="Portfolio values are derived from persisted session cash and open positions.",
        ),
    }


def _equity_curve_payload(db: DatabaseManager, session_obj: SimulationSession) -> List[Dict[str, Any]]:
    trades = _session_trades(db, session_obj.session_id)
    portfolio = _portfolio_payload(db, session_obj)
    anchor_date = session_obj.start_date or session_obj.created_at.date() or date.today()
    running_value = float(session_obj.initial_capital or 0.0)
    points: List[Dict[str, Any]] = [{"date": anchor_date.isoformat(), "value": running_value}]

    for trade in trades:
        trade_date = trade.exit_date or trade.entry_date
        running_value += float(trade.net_pnl or 0.0)
        if points and points[-1]["date"] == trade_date.isoformat():
            points[-1]["value"] = running_value
        else:
            points.append({"date": trade_date.isoformat(), "value": running_value})

    current_date = portfolio["current_date"]
    if points[-1]["date"] != current_date:
        points.append({"date": current_date, "value": float(portfolio["capital"])})
    else:
        points[-1]["value"] = float(portfolio["capital"])
    return points


def _metrics_payload(db: DatabaseManager, session_obj: SimulationSession) -> Dict[str, Any]:
    portfolio = _portfolio_payload(db, session_obj)
    trades = _session_trades(db, session_obj.session_id)
    curve = _equity_curve_payload(db, session_obj)
    values = np.array([point["value"] for point in curve], dtype=float)

    max_drawdown = 0.0
    sharpe_ratio = None
    if len(values) > 1:
        cumulative_peak = np.maximum.accumulate(values)
        drawdowns = (values - cumulative_peak) / cumulative_peak
        max_drawdown = float(abs(drawdowns.min()) * 100)

        returns = np.diff(values) / values[:-1]
        if len(returns) > 1 and float(np.std(returns)) > 0:
            sharpe_ratio = float((np.mean(returns) / np.std(returns)) * np.sqrt(252))

    wins = [float(t.net_pnl or 0.0) for t in trades if float(t.net_pnl or 0.0) > 0]
    losses = [float(t.net_pnl or 0.0) for t in trades if float(t.net_pnl or 0.0) < 0]
    profit_factor = None
    if losses:
        profit_factor = float(sum(wins) / abs(sum(losses))) if wins else 0.0

    total_trades = len(trades)
    win_rate = float(len(wins) / total_trades) if total_trades else 0.0
    total_pnl = float(portfolio["capital"] - float(session_obj.initial_capital or 0.0))

    _save_session_updates(
        db,
        session_obj,
        total_trades=total_trades,
        win_rate=win_rate,
        total_pnl=total_pnl,
        max_drawdown=max_drawdown,
        sharpe_ratio=sharpe_ratio,
    )

    return {
        "current_capital": float(portfolio["capital"]),
        "cash": float(portfolio["cash"]),
        "market_value": float(portfolio["market_value"]),
        "total_pnl": total_pnl,
        "win_rate": win_rate,
        "total_trades": total_trades,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "profit_factor": profit_factor,
        **_feature_metadata(
            feature_state="beta",
            is_demo=False,
            data_source="session_ledger",
            detail="Metrics are derived from persisted trades and marked-to-market open positions.",
        ),
    }


@router.post("/create")
def create_simulation(req: CreateSimulationRequest):
    """Create a new virtual trading session."""
    db = DatabaseManager()
    mode = req.mode.lower()
    if mode not in {"live", "replay"}:
        raise HTTPException(status_code=400, detail="mode must be either 'live' or 'replay'")

    replay_cursor = None
    if mode == "replay":
        if not req.start_date:
            raise HTTPException(status_code=400, detail="Replay mode requires start_date")
        replay_cursor = _validate_replay_dates(db, req.start_date, req.end_date)

    session_id = f"sim_{uuid.uuid4().hex[:8]}"
    session_data = {
        "session_id": session_id,
        "name": req.name,
        "mode": mode,
        "trading_mode": req.trading_mode,
        "start_date": req.start_date if mode == "replay" else None,
        "end_date": replay_cursor if mode == "replay" else None,
        "initial_capital": req.initial_capital,
        "current_capital": req.initial_capital,
        "status": "active",
        "total_trades": 0,
        "win_rate": 0.0,
        "total_pnl": 0.0,
        "max_drawdown": 0.0,
        "sharpe_ratio": None,
    }
    db.save_simulation_session(session_data)
    return {
        "status": "success",
        "session_id": session_id,
        "data": {
            **session_data,
            "current_date": replay_cursor.isoformat() if replay_cursor else None,
        },
        **_feature_metadata(
            feature_state="beta",
            is_demo=False,
            data_source="simulation_sessions",
            detail="Session state is persisted, but advanced risk analytics are not implemented in this build.",
        ),
    }


@router.get("/")
def list_simulations():
    """List all simulation sessions."""
    db = DatabaseManager()
    sessions = db.get_simulation_sessions()
    return [
        {
            "session_id": s.session_id,
            "name": s.name,
            "mode": s.mode,
            "trading_mode": s.trading_mode,
            "status": s.status,
            "start_date": s.start_date.isoformat() if s.start_date else None,
            "current_date": (s.end_date or s.start_date).isoformat() if (s.end_date or s.start_date) else None,
            "win_rate": float(s.win_rate or 0.0),
            "total_pnl": float(s.total_pnl or 0.0),
            "current_capital": float(s.current_capital or 0.0),
            "total_trades": int(getattr(s, "total_trades", 0) or 0),
            "max_drawdown": float(getattr(s, "max_drawdown", 0) or 0.0),
            "sharpe_ratio": getattr(s, "sharpe_ratio", None),
            **_feature_metadata(
                feature_state="beta",
                is_demo=False,
                data_source="simulation_sessions",
            ),
        }
        for s in sessions
    ]


@router.get("/{session_id}/portfolio")
def get_portfolio(session_id: str):
    """Get the current portfolio state for a simulation session."""
    db = DatabaseManager()
    session_obj = _get_session_or_404(db, session_id)
    return _portfolio_payload(db, session_obj)


@router.get("/{session_id}/history")
def get_trade_history(session_id: str):
    """Get trade history for the session."""
    db = DatabaseManager()
    _get_session_or_404(db, session_id)
    trades = _session_trades(db, session_id)
    return [
        {
            "trade_id": trade.trade_id,
            "date": datetime.combine(trade.exit_date or trade.entry_date, datetime.min.time()).strftime("%Y-%m-%d %H:%M:%S"),
            "symbol": trade.symbol,
            "side": trade.side,
            "quantity": int(trade.quantity or 0),
            "entry_price": float(trade.entry_price or 0.0),
            "exit_price": float(trade.exit_price or 0.0),
            "realized_pnl": float(trade.net_pnl or 0.0),
            "return_pct": float(trade.return_pct or 0.0),
        }
        for trade in trades
    ]


@router.post("/{session_id}/order")
def execute_order(session_id: str, req: OrderRequest):
    """Submit an order using persisted simulation session state."""
    db = DatabaseManager()
    session_obj = _get_session_or_404(db, session_id)
    side = req.side.upper()
    symbol = req.symbol.upper()
    order_type = req.order_type.upper()

    if side not in {"BUY", "SELL"}:
        raise HTTPException(status_code=400, detail="side must be BUY or SELL")
    if order_type not in {"MARKET", "LIMIT"}:
        raise HTTPException(status_code=400, detail="order_type must be MARKET or LIMIT")
    if req.quantity % IDX_LOT_SIZE != 0:
        raise HTTPException(status_code=400, detail=f"quantity must be a multiple of {IDX_LOT_SIZE} shares")

    execution_price = _execution_price(db, session_obj, req)
    execution_date = session_obj.end_date or session_obj.start_date or date.today()
    cash = float(session_obj.current_capital or session_obj.initial_capital or 0.0)

    if side == "BUY":
        total_cost = execution_price * req.quantity
        if total_cost > cash:
            raise HTTPException(status_code=400, detail="Insufficient cash for BUY order")

        position_id = f"{session_id}:{uuid.uuid4().hex[:10]}"
        db.save_position(
            {
                "position_id": position_id,
                "symbol": symbol,
                "entry_date": execution_date,
                "entry_price": execution_price,
                "quantity": req.quantity,
                "target_1": req.targets[0] if len(req.targets) > 0 else None,
                "target_2": req.targets[1] if len(req.targets) > 1 else None,
                "target_3": req.targets[2] if len(req.targets) > 2 else None,
                "highest_price": execution_price,
            }
        )
        _save_session_updates(db, session_obj, current_capital=cash - total_cost)

        return {
            "status": "success",
            "message": f"Order executed: BUY {req.quantity} shares of {symbol}",
            "order_id": f"ord_{uuid.uuid4().hex[:6]}",
            "execution_price": execution_price,
            "execution_date": execution_date.isoformat(),
            **_feature_metadata(
                feature_state="beta",
                is_demo=False,
                data_source="session_ledger",
                detail="BUY orders update session cash and open positions immediately.",
            ),
        }

    positions = [position for position in _session_positions(db, session_id) if position.symbol == symbol]
    available_quantity = sum(int(position.quantity or 0) for position in positions)
    if available_quantity < req.quantity:
        raise HTTPException(status_code=400, detail=f"Insufficient position to sell. Available: {available_quantity}")

    remaining = req.quantity
    realized_pnl = 0.0
    released_cash = execution_price * req.quantity
    db_session = db.get_session()
    try:
        ordered_positions = (
            db_session.query(OpenPositions)
            .filter(OpenPositions.position_id.like(f"{session_id}:%"), OpenPositions.symbol == symbol)
            .order_by(OpenPositions.entry_date.asc(), OpenPositions.id.asc())
            .all()
        )

        for position in ordered_positions:
            if remaining <= 0:
                break
            close_qty = min(remaining, int(position.quantity or 0))
            pnl = (execution_price - float(position.entry_price)) * close_qty
            realized_pnl += pnl

            trade_id = f"trade_{uuid.uuid4().hex[:10]}"
            db_session.add(
                TradeHistory(
                    trade_id=trade_id,
                    symbol=symbol,
                    entry_date=position.entry_date,
                    entry_price=position.entry_price,
                    exit_date=execution_date,
                    exit_price=execution_price,
                    quantity=close_qty,
                    side="LONG",
                    gross_pnl=pnl,
                    net_pnl=pnl,
                    return_pct=((execution_price / float(position.entry_price)) - 1) * 100 if position.entry_price else 0.0,
                    holding_days=max((execution_date - position.entry_date).days, 0),
                    win=pnl > 0,
                )
            )
            db_session.add(SimulationTrade(session_id=session_id, trade_id=trade_id))

            remaining -= close_qty
            leftover = int(position.quantity or 0) - close_qty
            if leftover > 0:
                position.quantity = leftover
            else:
                db_session.delete(position)

        db_session.commit()
    except Exception:
        db_session.rollback()
        raise
    finally:
        db_session.close()

    _save_session_updates(db, session_obj, current_capital=cash + released_cash)
    return {
        "status": "success",
        "message": f"Order executed: SELL {req.quantity} shares of {symbol}",
        "order_id": f"ord_{uuid.uuid4().hex[:6]}",
        "execution_price": execution_price,
        "execution_date": execution_date.isoformat(),
        "realized_pnl": realized_pnl,
        **_feature_metadata(
            feature_state="beta",
            is_demo=False,
            data_source="session_ledger",
            detail="SELL orders close open positions FIFO and realize P&L into session cash.",
        ),
    }


@router.post("/{session_id}/step")
def advance_replay_step(session_id: str):
    """Advance the replay simulation to the next available trading day."""
    db = DatabaseManager()
    session_obj = _get_session_or_404(db, session_id)
    if session_obj.mode != "replay":
        raise HTTPException(status_code=400, detail="Replay stepping is only available for replay sessions")

    cursor = session_obj.end_date or session_obj.start_date
    if not cursor:
        raise HTTPException(status_code=400, detail="Replay session is missing start_date")

    db_session = db.get_session()
    try:
        next_row = (
            db_session.query(PriceHistory.date)
            .filter(PriceHistory.date > cursor)
            .order_by(PriceHistory.date.asc())
            .first()
        )
    finally:
        db_session.close()

    if not next_row:
        raise HTTPException(status_code=400, detail="Replay has reached the end of available market data")

    next_date = next_row[0]
    _save_session_updates(db, session_obj, end_date=next_date)
    return {
        "status": "success",
        "message": "Advanced 1 trading day",
        "current_date": next_date.isoformat(),
        **_feature_metadata(
            feature_state="beta",
            is_demo=False,
            data_source="price_history",
            detail="Replay steps are driven by the next available historical market date in the database.",
        ),
    }


@router.get("/{session_id}/metrics")
def get_session_metrics(session_id: str):
    """Get computed performance metrics for the session."""
    db = DatabaseManager()
    session_obj = _get_session_or_404(db, session_id)
    return _metrics_payload(db, session_obj)


@router.get("/{session_id}/equity-curve")
def get_equity_curve(session_id: str):
    """Get the derived portfolio value series for the equity curve chart."""
    db = DatabaseManager()
    session_obj = _get_session_or_404(db, session_id)
    return _equity_curve_payload(db, session_obj)
