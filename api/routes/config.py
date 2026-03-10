"""Configuration routes."""

import logging

from fastapi import APIRouter

from config.trading_modes import TradingMode, get_mode_config

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/config", tags=["config"])


@router.get("/modes")
async def get_modes():
    """Get all trading mode configurations.

    Returns configuration for all 4 trading modes.
    """
    modes = {}
    for mode in TradingMode:
        config = get_mode_config(mode)
        modes[mode.value] = {
            "name": config.name,
            "min_hold_days": config.min_hold_days,
            "max_hold_days": config.max_hold_days,
            "max_risk_per_trade": config.max_risk_per_trade,
            "max_position_pct": config.max_position_pct,
            "technical_weight": config.technical_weight,
            "flow_weight": config.flow_weight,
            "fundamental_weight": config.fundamental_weight,
            "min_score": config.min_score,
            "default_stop_pct": config.default_stop_pct,
        }

    return {"modes": modes, "default": "swing"}
