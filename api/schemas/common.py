"""Common API schemas shared across routes."""

import re
from datetime import date, datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


# IDX stock symbols are 4 letters uppercase (e.g., BBCA, TLKM, BBRI)
SYMBOL_PATTERN = re.compile(r"^[A-Z]{4}$")


def validate_idx_symbol(symbol: str) -> str:
    """Validate and normalize an IDX stock symbol.

    Args:
        symbol: Stock symbol to validate

    Returns:
        Normalized uppercase symbol

    Raises:
        ValueError: If symbol format is invalid
    """
    symbol = symbol.upper().strip()
    if not SYMBOL_PATTERN.match(symbol):
        raise ValueError(
            f"Invalid IDX symbol '{symbol}'. Must be 4 uppercase letters (e.g., BBCA, TLKM)"
        )
    return symbol


class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str = Field(..., min_length=1, description="Error type")
    detail: Optional[str] = Field(default=None, description="Detailed error message")
    status_code: int = Field(default=500, ge=100, le=599, description="HTTP status code")


class PaginationParams(BaseModel):
    """Pagination parameters."""
    page: int = Field(default=1, ge=1, description="Page number (1-indexed)")
    per_page: int = Field(default=20, ge=1, le=100, description="Items per page")


class PaginatedResponse(BaseModel):
    """Paginated response wrapper."""
    items: List[Any] = Field(default_factory=list, description="List of items")
    total: int = Field(..., ge=0, description="Total number of items")
    page: int = Field(..., ge=1, description="Current page number")
    per_page: int = Field(..., ge=1, le=100, description="Items per page")
    pages: int = Field(..., ge=0, description="Total number of pages")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(default="ok", pattern="^(ok|degraded|error)$", description="Health status")
    version: str = Field(default="3.0.0", description="API version")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")
    components: Optional[Dict[str, str]] = Field(default=None, description="Component status map")


class DataFreshnessResponse(BaseModel):
    """Data freshness check response."""
    status: str = Field(..., pattern="^(fresh|stale|unknown)$", description="Data freshness status")
    price_data_age_hours: Optional[float] = Field(default=None, ge=0, description="Age of price data in hours")
    flow_data_age_hours: Optional[float] = Field(default=None, ge=0, description="Age of flow data in hours")
    last_price_update: Optional[datetime] = Field(default=None, description="Last price update timestamp")
    last_flow_update: Optional[datetime] = Field(default=None, description="Last flow update timestamp")
    warnings: List[str] = Field(default_factory=list, description="Freshness warnings")


class SuccessResponse(BaseModel):
    """Generic success response."""
    success: bool = Field(default=True, description="Operation success flag")
    message: str = Field(default="Operation completed successfully", description="Success message")
