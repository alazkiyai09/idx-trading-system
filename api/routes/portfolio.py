"""Portfolio routes."""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from api.mappers import map_position_dict
from api.schemas.portfolio_schemas import (
    PortfolioResponse,
    PositionResponse,
    TradeHistoryResponse,
    TradeHistoryItem,
)
from api.dependencies import get_coordinator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/portfolio", tags=["portfolio"])


@router.get("", response_model=PortfolioResponse)
async def get_portfolio():
    """Get current portfolio state.

    Returns portfolio summary with all open positions.
    """
    try:
        coordinator = get_coordinator()
        summary = coordinator.get_portfolio_summary()

        positions = []
        for pos in summary.get("positions", []):
            payload = map_position_dict(pos)
            positions.append(PositionResponse(
                symbol=payload["symbol"],
                entry_price=payload["entry_price"],
                current_price=payload["current_price"],
                shares=payload["shares"],
                entry_date=payload["entry_date"],
                unrealized_pnl=payload["unrealized_pnl"],
                unrealized_pnl_pct=payload["unrealized_pnl_pct"],
                stop_loss=payload["stop_loss"],
                target=payload["target"],
            ))

        return PortfolioResponse(
            total_value=summary.get("total_value", 0),
            cash=summary.get("cash", 0),
            invested=summary.get("invested", summary.get("positions_value", 0)),
            unrealized_pnl=summary.get("unrealized_pnl", 0),
            realized_pnl=summary.get("realized_pnl", 0),
            total_pnl=summary.get("total_pnl", 0),
            total_pnl_pct=summary.get("total_pnl_pct", 0),
            num_positions=len(positions),
            positions=positions,
        )

    except Exception as e:
        logger.error(f"Failed to get portfolio: {e}")
        raise HTTPException(status_code=500, detail="Failed to load portfolio")


@router.get("/positions")
async def get_positions():
    """Get open positions only."""
    try:
        coordinator = get_coordinator()
        summary = coordinator.get_portfolio_summary()
        return {"positions": summary.get("positions", []), "count": len(summary.get("positions", []))}
    except Exception as e:
        logger.error(f"Failed to get positions: {e}")
        raise HTTPException(status_code=500, detail="Failed to load positions")


@router.get("/history", response_model=TradeHistoryResponse)
async def get_trade_history(
    limit: int = Query(default=50, ge=1, le=500),
    symbol: Optional[str] = None,
):
    """Get trade history.

    Args:
        limit: Maximum number of trades to return.
        symbol: Filter by symbol.
    """
    # TODO: Integrate with database for historical trade retrieval
    return TradeHistoryResponse(
        trades=[],
        total=0,
        total_pnl=0.0,
        win_rate=0.0,
    )
