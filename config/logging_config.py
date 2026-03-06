"""
Logging configuration for IDX Trading System.

This module sets up structured logging using loguru with:
- Console output for development
- File output for production
- Rotation and retention policies
- Different log levels for different modules
"""

import sys
from pathlib import Path
from typing import Optional

from loguru import logger

from config.settings import settings


def setup_logging(
    log_level: Optional[str] = None,
    log_dir: Optional[Path] = None,
    console_output: bool = True,
) -> None:
    """Configure logging for the application.

    Sets up loguru logger with console and file handlers based on settings.

    Args:
        log_level: Override log level from settings.
        log_dir: Override log directory from settings.
        console_output: Whether to output to console.
    """
    # Remove default handler
    logger.remove()

    # Get settings
    level = log_level or settings.log_level
    log_path = log_dir or settings.logs_dir

    # Ensure log directory exists
    log_path.mkdir(parents=True, exist_ok=True)

    # Console format (more readable)
    console_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )

    # File format (more structured)
    file_format = (
        "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
        "{level: <8} | "
        "{name}:{function}:{line} | "
        "{message}"
    )

    # Add console handler
    if console_output:
        logger.add(
            sys.stderr,
            format=console_format,
            level=level,
            colorize=True,
            backtrace=True,
            diagnose=True,
        )

    # Add main log file (all levels)
    logger.add(
        log_path / "trading_{time:YYYY-MM-DD}.log",
        format=file_format,
        level=level,
        rotation="00:00",  # Rotate at midnight
        retention="30 days",  # Keep 30 days
        compression="gz",  # Compress old logs
        backtrace=True,
        diagnose=True,
    )

    # Add error log file (errors only)
    logger.add(
        log_path / "errors_{time:YYYY-MM-DD}.log",
        format=file_format,
        level="ERROR",
        rotation="00:00",
        retention="90 days",  # Keep errors longer
        compression="gz",
        backtrace=True,
        diagnose=True,
    )

    # Add trade log (for trade-related messages)
    logger.add(
        log_path / "trades_{time:YYYY-MM-DD}.log",
        format=file_format,
        level="INFO",
        filter=lambda record: "trade" in record["extra"],
        rotation="00:00",
        retention="365 days",  # Keep trade logs for a year
        compression="gz",
    )

    logger.info(f"Logging configured with level={level}")


def get_trade_logger():
    """Get a logger bound to trade context.

    Returns:
        Logger instance bound with trade context for filtering.
    """
    return logger.bind(trade=True)


def get_logger(name: str):
    """Get a logger with module name context.

    Args:
        name: Module name for context.

    Returns:
        Logger instance with module context.
    """
    return logger.bind(module=name)


class LoggingContext:
    """Context manager for temporary log level changes.

    Useful for debugging specific sections of code.

    Example:
        with LoggingContext("DEBUG"):
            # Debug logging enabled here
            logger.debug("This will be shown")
    """

    def __init__(self, level: str):
        """Initialize context with specified log level.

        Args:
            level: Log level to use within context.
        """
        self.level = level
        self.original_handlers = []

    def __enter__(self):
        """Enter context and change log level."""
        self.original_handlers = logger._core.handlers.copy()
        logger.remove()
        logger.add(
            sys.stderr,
            level=self.level,
            colorize=True,
            format="<green>{time}</green> | <level>{level}</level> | <level>{message}</level>",
        )
        return logger

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context and restore original handlers."""
        logger.remove()
        for handler_id, handler in self.original_handlers.items():
            logger._core.handlers[handler_id] = handler
        return False


# Initialize logging on import if settings are available
try:
    setup_logging()
except Exception:
    # Settings may not be configured yet during import
    pass
