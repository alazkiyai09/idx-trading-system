"""
IDX Trading System REST API Package.

Provides FastAPI-based REST endpoints for the trading system.

Usage:
    uvicorn api.main:app --reload --port 8000
"""

from api.main import app

__all__ = ["app"]
