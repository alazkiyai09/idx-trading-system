"""
Dependencies Module

FastAPI dependency injection for shared resources.
"""

import logging
from typing import Optional

from config.settings import settings
from config.trading_modes import TradingMode

logger = logging.getLogger(__name__)

# Cached coordinator instance
_coordinator = None


def get_coordinator(mode: Optional[str] = None):
    """Get or create the Coordinator instance.

    Args:
        mode: Trading mode. Defaults to settings.default_mode.

    Returns:
        Coordinator instance.
    """
    global _coordinator

    if _coordinator is None:
        from agents.coordinator import Coordinator

        trading_mode = TradingMode(mode or settings.default_mode)
        _coordinator = Coordinator(mode=trading_mode, dry_run=True)
        logger.info(f"Initialized coordinator with mode: {trading_mode.value}")

    return _coordinator


def get_settings():
    """Get application settings.

    Returns:
        Settings instance.
    """
    return settings


def reset_coordinator():
    """Reset the coordinator instance. Used for testing."""
    global _coordinator
    _coordinator = None
