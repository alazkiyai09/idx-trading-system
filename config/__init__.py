"""
Configuration package for IDX Trading System.

This package contains all configuration modules:
- settings: Global application settings
- trading_modes: Trading mode configurations
- constants: IDX-specific constants
- logging_config: Logging configuration
"""

from config.settings import Settings, settings
from config.trading_modes import (
    TradingMode,
    ModeConfig,
    MODE_CONFIGS,
    get_mode_config,
    get_mode_from_string,
    get_all_modes,
)
from config.constants import (
    IDX_LOT_SIZE,
    IDX_BUY_FEE,
    IDX_SELL_FEE,
    ARA_LIMIT,
    ARB_LIMIT,
    IDX_TICK_SIZES,
    TRADING_HOURS,
    get_tick_size,
    round_to_tick,
    is_trading_hours,
    YAHOO_FINANCE_SUFFIX,
)

__all__ = [
    # Settings
    "Settings",
    "settings",
    # Trading modes
    "TradingMode",
    "ModeConfig",
    "MODE_CONFIGS",
    "get_mode_config",
    "get_mode_from_string",
    "get_all_modes",
    # Constants
    "IDX_LOT_SIZE",
    "IDX_BUY_FEE",
    "IDX_SELL_FEE",
    "ARA_LIMIT",
    "ARB_LIMIT",
    "IDX_TICK_SIZES",
    "TRADING_HOURS",
    "get_tick_size",
    "round_to_tick",
    "is_trading_hours",
    "YAHOO_FINANCE_SUFFIX",
]
