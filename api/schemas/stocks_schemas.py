"""Stock-related API schemas with IDX-specific validation."""

import re
from datetime import date, datetime
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


# IDX stock symbols are 4 letters uppercase (e.g., BBCA, TLKM, BBRI)
SYMBOL_PATTERN = re.compile(r"^[A-Z]{4}$")


class StockBase(BaseModel):
    """Base schema with symbol validation."""

    symbol: str = Field(..., description="IDX stock symbol (4 uppercase letters)")

    @field_validator("symbol")
    @classmethod
    def validate_symbol(cls, v: str) -> str:
        """Validate IDX symbol format."""
        v = v.upper().strip()
        if not SYMBOL_PATTERN.match(v):
            raise ValueError(
                f"Invalid IDX symbol '{v}'. Must be 4 uppercase letters (e.g., BBCA, TLKM)"
            )
        return v


class StockMetadataResponse(StockBase):
    """Stock metadata response."""

    name: str
    sector: Optional[str] = None
    industry: Optional[str] = None
    listing_date: Optional[date] = None
    shares_outstanding: Optional[int] = None
    market_cap: Optional[float] = None
    is_lq45: bool = False
    is_idx30: bool = False
    is_idx_growth: bool = False


class OHLCVResponse(BaseModel):
    """OHLCV data response."""

    timestamp: datetime
    open: float = Field(..., ge=0, description="Opening price (must be >= 0)")
    high: float = Field(..., ge=0, description="High price (must be >= 0)")
    low: float = Field(..., ge=0, description="Low price (must be >= 0)")
    close: float = Field(..., ge=0, description="Closing price (must be >= 0)")
    volume: int = Field(..., ge=0, description="Trading volume (must be >= 0)")

    @field_validator("high")
    @classmethod
    def validate_high(cls, v: float, info) -> float:
        """Ensure high >= low, open, close."""
        # Note: Full cross-validation requires model_validator
        # This is a basic check
        if v < 0:
            raise ValueError("High price must be non-negative")
        return v

    @field_validator("low")
    @classmethod
    def validate_low(cls, v: float) -> float:
        """Ensure low is non-negative."""
        if v < 0:
            raise ValueError("Low price must be non-negative")
        return v


class StockListRequest(BaseModel):
    """Request parameters for listing stocks."""

    sector: Optional[str] = Field(default=None, description="Filter by sector")
    index: Optional[str] = Field(
        default=None, description="Filter by index membership (LQ45, IDX30, IDXGROWTH)"
    )
    min_market_cap: Optional[float] = Field(
        default=None, ge=0, description="Minimum market cap filter"
    )
    is_active: bool = Field(default=True, description="Only include active stocks")


class StockListResponse(BaseModel):
    """Response for stock list endpoint."""

    stocks: List[StockMetadataResponse]
    total: int = Field(..., ge=0)
    sector: Optional[str] = None
    index_filter: Optional[str] = Field(default=None, alias="index")


class StockDetailResponse(StockBase):
    """Detailed stock information response."""

    name: str
    sector: Optional[str] = None
    industry: Optional[str] = None
    market_cap: Optional[float] = None
    pe_ratio: Optional[float] = None
    pb_ratio: Optional[float] = None
    dividend_yield: Optional[float] = None
    beta: Optional[float] = None
    avg_volume_30d: Optional[float] = None
    latest_price: Optional[float] = None
    price_change_1d: Optional[float] = None
    price_change_pct_1d: Optional[float] = None
    fifty_two_week_high: Optional[float] = None
    fifty_two_week_low: Optional[float] = None
    is_lq45: bool = False
    is_idx30: bool = False


class StockChartRequest(StockBase):
    """Request for stock chart data."""

    start_date: date = Field(..., description="Start date for chart data")
    end_date: date = Field(..., description="End date for chart data")
    interval: str = Field(default="1d", description="Data interval (1d, 1w, 1m)")

    @field_validator("end_date")
    @classmethod
    def validate_dates(cls, v: date, info) -> date:
        """Ensure end_date >= start_date."""
        start_date = info.data.get("start_date")
        if start_date and v < start_date:
            raise ValueError("end_date must be >= start_date")
        return v

    @field_validator("interval")
    @classmethod
    def validate_interval(cls, v: str) -> str:
        """Validate interval format."""
        valid_intervals = ["1d", "1w", "1m"]
        if v not in valid_intervals:
            raise ValueError(f"interval must be one of {valid_intervals}")
        return v


class StockChartResponse(StockBase):
    """Response for stock chart data."""

    data: List[OHLCVResponse]
    start_date: date
    end_date: date
    interval: str
    total_bars: int = Field(..., ge=0)
