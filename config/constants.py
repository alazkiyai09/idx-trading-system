"""
IDX-specific constants and trading rules.

This module contains constants specific to the Indonesia Stock Exchange (IDX),
including lot sizes, fees, tick sizes, and trading hours.
"""

from typing import Dict, Tuple
from datetime import time


# =============================================================================
# LOT SIZE AND SHARES
# =============================================================================

IDX_LOT_SIZE: int = 100
"""Number of shares per lot on IDX."""


# =============================================================================
# FEES AND CHARGES
# =============================================================================

IDX_BUY_FEE: float = 0.0015
"""Buy fee percentage (0.15%) - includes broker fee."""

IDX_SELL_FEE: float = 0.0025
"""Sell fee percentage (0.25%) - includes broker fee and transaction tax."""


# =============================================================================
# AUTO REJECTION LIMITS
# =============================================================================

ARA_LIMIT: float = 0.25
"""Auto Reject Above limit - maximum price increase (25%) from previous close."""

ARB_LIMIT: float = 0.25
"""Auto Reject Below limit - maximum price decrease (25%) from previous close."""


# =============================================================================
# TICK SIZES (LOT-BASED)
# =============================================================================

# Price ranges and their corresponding tick sizes
# Format: (min_price, max_price) -> tick_size
IDX_TICK_SIZES: Dict[Tuple[float, float], int] = {
    (0, 50): 1,
    (50, 100): 2,
    (100, 200): 5,
    (200, 500): 10,
    (500, 1000): 25,
    (1000, 2000): 50,
    (2000, 5000): 100,
    (5000, 10000): 250,
    (10000, float("inf")): 500,
}
"""Tick sizes based on price ranges for IDX."""


def get_tick_size(price: float) -> int:
    """Get the appropriate tick size for a given price.

    Args:
        price: Current price in IDR.

    Returns:
        Tick size in IDR for the given price level.
    """
    for (min_price, max_price), tick_size in IDX_TICK_SIZES.items():
        if min_price <= price < max_price:
            return tick_size
    return 500  # Default for very high prices


def round_to_tick(price: float, direction: str = "down") -> float:
    """Round price to nearest valid tick.

    Args:
        price: Price to round.
        direction: 'up' for ceiling, 'down' for floor, 'nearest' for closest.

    Returns:
        Price rounded to valid tick size.
    """
    tick_size = get_tick_size(price)

    if direction == "up":
        return (price // tick_size + 1) * tick_size
    elif direction == "down":
        return (price // tick_size) * tick_size
    else:  # nearest
        return round(price / tick_size) * tick_size


# =============================================================================
# TRADING HOURS (WIB - Western Indonesia Time, UTC+7)
# =============================================================================

TRADING_HOURS = {
    "pre_open": (time(8, 45), time(8, 55)),
    "pre_close": (time(8, 55), time(9, 0)),
    "session_1": (time(9, 0), time(11, 30)),
    "lunch_break": (time(11, 30), time(13, 30)),
    "session_2": (time(13, 30), time(16, 15)),
    "post_close": (time(16, 15), time(16, 30)),
}
"""Trading schedule for IDX."""


def is_trading_hours(check_time: time = None) -> bool:
    """Check if current time is during trading hours.

    Args:
        check_time: Time to check. If None, uses current time.

    Returns:
        True if within trading hours (Session 1 or Session 2).
    """
    from datetime import datetime

    if check_time is None:
        check_time = datetime.now().time()

    session_1 = TRADING_HOURS["session_1"]
    session_2 = TRADING_HOURS["session_2"]

    return (session_1[0] <= check_time <= session_1[1]) or (
        session_2[0] <= check_time <= session_2[1]
    )


def get_next_session(check_time: time = None) -> str:
    """Get the next trading session.

    Args:
        check_time: Time to check. If None, uses current time.

    Returns:
        Name of the next trading session.
    """
    from datetime import datetime

    if check_time is None:
        check_time = datetime.now().time()

    if check_time < TRADING_HOURS["pre_open"][0]:
        return "pre_open"
    elif check_time < TRADING_HOURS["session_1"][0]:
        return "session_1"
    elif check_time < TRADING_HOURS["lunch_break"][0]:
        return "lunch_break"
    elif check_time < TRADING_HOURS["session_2"][0]:
        return "session_2"
    elif check_time < TRADING_HOURS["post_close"][0]:
        return "post_close"
    else:
        return "next_day"


# =============================================================================
# TRADING DAYS
# =============================================================================

IDX_TRADING_DAYS: Tuple[int, ...] = (0, 1, 2, 3, 4)
"""Trading days of the week (Monday=0 to Friday=4)."""


def is_trading_day(date=None) -> bool:
    """Check if a date is a trading day.

    Args:
        date: Date to check. If None, uses today.

    Returns:
        True if the date is a weekday (Monday-Friday).
    """
    from datetime import datetime

    if date is None:
        date = datetime.now().date()

    return date.weekday() in IDX_TRADING_DAYS


# =============================================================================
# INDICES AND SYMBOLS
# =============================================================================

IDX_MAJOR_INDICES: Dict[str, str] = {
    "JKSE": "Jakarta Composite Index (IHSG)",
    "LQ45": "LQ45 Index",
    "KOMPAS100": "Kompas 100 Index",
    "IDX30": "IDX30 Index",
    "IDXGROWTH": "IDX Growth Index",
    "JII": "Jakarta Islamic Index",
}
"""Major IDX indices."""


# =============================================================================
# CURRENCY
# =============================================================================

DEFAULT_CURRENCY: str = "IDR"
"""Default currency for the system (Indonesian Rupiah)."""

CURRENCY_SYMBOL: str = "Rp"
"""Currency symbol for display."""


# =============================================================================
# RISK CONSTANTS
# =============================================================================

MAX_PRICE_CHANGE_PCT: float = 0.25
"""Maximum single-day price change (25%)."""

MIN_TRADE_VALUE_IDR: float = 100_000
"""Minimum trade value in IDR (for valid trades)."""

# =============================================================================
# DATA CONSTANTS
# =============================================================================

YAHOO_FINANCE_SUFFIX: str = ".JK"
"""Suffix to add to IDX symbols for Yahoo Finance API."""

SYMBOL_PATTERN: str = r"^[A-Z]{2,4}$"
"""Regex pattern for valid IDX stock symbols."""

MAX_SYMBOL_LENGTH: int = 4
"""Maximum length of IDX stock symbol."""
