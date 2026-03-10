"""Backtest routes."""

import logging
import uuid
from datetime import date, datetime
from typing import Dict

from fastapi import APIRouter, HTTPException

from api.schemas.backtest_schemas import BacktestRequest, BacktestResponse, BacktestMetrics
from config.settings import settings
from core.data.database import DatabaseManager
from core.data.models import OHLCV

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/backtest", tags=["backtest"])

# In-memory backtest result store
_backtest_results: Dict[str, dict] = {}


@router.post("/run", response_model=BacktestResponse)
async def run_backtest(request: BacktestRequest):
    """Start a backtest.

    Runs a backtest with the specified parameters.
    """
    backtest_id = str(uuid.uuid4())[:8]

    try:
        from config.trading_modes import TradingMode
        from backtest.engine import BacktestEngine, BacktestConfig

        # Create proper BacktestConfig
        config = BacktestConfig(
            start_date=request.start_date,
            end_date=request.end_date,
            initial_capital=request.initial_capital,
            trading_mode=TradingMode(request.mode),
        )

        # Fetch price data from database
        db = DatabaseManager(settings.database_url)
        bars_by_symbol: Dict[str, list] = {}

        for symbol in request.symbols:
            price_records = db.get_prices(symbol, request.start_date, request.end_date)
            if price_records:
                # Convert database records to OHLCV objects
                bars_by_symbol[symbol] = [
                    OHLCV(
                        timestamp=datetime.combine(record.date, datetime.min.time()),
                        open=record.open,
                        high=record.high,
                        low=record.low,
                        close=record.close,
                        volume=int(record.volume) if record.volume else 0,
                    )
                    for record in price_records
                ]

        # Check if we have any data
        if not bars_by_symbol:
            response = BacktestResponse(
                id=backtest_id,
                status="failed",
                mode=request.mode,
                start_date=request.start_date,
                end_date=request.end_date,
                initial_capital=request.initial_capital,
                errors=["No price data found for the specified symbols and date range"],
            )
            _backtest_results[backtest_id] = response.model_dump()
            raise HTTPException(status_code=400, detail="No price data found")

        # Create engine and run backtest
        engine = BacktestEngine(config)
        result = engine.run(bars_by_symbol)

        # Convert result to dict for accessing metrics
        result_dict = result.to_dict()
        metrics_data = result_dict.get("metrics", {})

        # Map nested metrics to flat schema fields
        # The metrics dict has nested structure: trade.*, drawdown.*, risk_adjusted.*
        metrics = BacktestMetrics(
            total_return_pct=result_dict.get("total_return_pct", 0) or metrics_data.get("total_return_pct", 0),
            annualized_return_pct=metrics_data.get("risk_adjusted", {}).get("annual_return", 0) * 100,
            sharpe_ratio=metrics_data.get("risk_adjusted", {}).get("sharpe_ratio", 0),
            sortino_ratio=metrics_data.get("risk_adjusted", {}).get("sortino_ratio", 0),
            max_drawdown_pct=metrics_data.get("drawdown", {}).get("max_drawdown_pct", 0),
            win_rate=metrics_data.get("trade", {}).get("win_rate", 0) * 100,
            profit_factor=metrics_data.get("trade", {}).get("profit_factor", 0),
            total_trades=metrics_data.get("trade", {}).get("total_trades", 0),
            winning_trades=metrics_data.get("trade", {}).get("winning_trades", 0),
            losing_trades=metrics_data.get("trade", {}).get("losing_trades", 0),
            avg_win_pct=metrics_data.get("trade", {}).get("avg_win", 0),
            avg_loss_pct=metrics_data.get("trade", {}).get("avg_loss", 0),
            avg_holding_days=metrics_data.get("trade", {}).get("avg_holding_days", 0),
            calmar_ratio=metrics_data.get("risk_adjusted", {}).get("calmar_ratio", 0),
        )

        response = BacktestResponse(
            id=backtest_id,
            status="completed",
            mode=request.mode,
            start_date=request.start_date,
            end_date=request.end_date,
            initial_capital=request.initial_capital,
            final_capital=result.final_capital,
            metrics=metrics,
        )

        _backtest_results[backtest_id] = response.model_dump()
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        response = BacktestResponse(
            id=backtest_id,
            status="failed",
            mode=request.mode,
            start_date=request.start_date,
            end_date=request.end_date,
            initial_capital=request.initial_capital,
            errors=["Backtest execution failed"],
        )
        _backtest_results[backtest_id] = response.model_dump()
        raise HTTPException(status_code=500, detail="Backtest execution failed")


@router.get("/{backtest_id}", response_model=BacktestResponse)
async def get_backtest(backtest_id: str):
    """Get backtest results by ID."""
    if backtest_id not in _backtest_results:
        raise HTTPException(status_code=404, detail="Backtest not found")
    return BacktestResponse(**_backtest_results[backtest_id])
