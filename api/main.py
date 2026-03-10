"""
IDX Trading System REST API

FastAPI application providing REST endpoints for the trading system.

Usage:
    uvicorn api.main:app --reload --port 8000
"""

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from api.routes import health, signals, portfolio, backtest, fundamental, config
from api.routes import stocks, sentiment, simulation, prediction, analysis

logger = logging.getLogger(__name__)

from config.settings import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lightweight lifespan hook without deprecated on_event handlers."""
    logger.info("IDX Trading System API starting up...")
    settings.ensure_directories()
    yield
    logger.info("IDX Trading System API shutting down...")

app = FastAPI(
    title="IDX Trading System API",
    description=(
        "REST API for the IDX Trading System — an institutional-grade "
        "trading platform for the Indonesia Stock Exchange (IDX)."
    ),
    version="3.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware - restricted for production
import os
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "http://localhost:8501,http://localhost:3000,http://127.0.0.1:8501").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle uncaught exceptions."""
    logger.error(f"Unhandled error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
        },
    )


# Include routers
app.include_router(health.router)
app.include_router(signals.router)
app.include_router(portfolio.router)
app.include_router(backtest.router)
app.include_router(fundamental.router)
app.include_router(config.router)
app.include_router(stocks.router)
app.include_router(sentiment.router)
app.include_router(simulation.router)
app.include_router(prediction.router)
app.include_router(analysis.router)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "IDX Trading System API",
        "version": "3.0.0",
        "docs": "/docs",
        "health": "/health",
    }
