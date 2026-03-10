"""Signal-related API schemas with validation."""

import re
from datetime import date, datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# Valid trading modes
VALID_TRADING_MODES = ["intraday", "swing", "position", "investor"]

# Valid signal types
VALID_SIGNAL_TYPES = ["BUY", "SELL", "HOLD"]

# IDX stock symbols are 4 letters uppercase
SYMBOL_PATTERN = re.compile(r"^[A-Z]{4}$")


class ScanRequest(BaseModel):
    """Request to run a daily scan."""
    mode: str = Field(
        default="swing",
        description="Trading mode: intraday, swing, position, investor"
    )
    symbols: Optional[List[str]] = Field(
        default=None,
        description="Symbols to scan. If None, uses LQ45"
    )
    dry_run: bool = Field(default=True, description="If True, no trades are executed")

    @field_validator("mode")
    @classmethod
    def validate_mode(cls, v: str) -> str:
        """Validate trading mode."""
        v = v.lower().strip()
        if v not in VALID_TRADING_MODES:
            raise ValueError(f"mode must be one of {VALID_TRADING_MODES}")
        return v

    @field_validator("symbols")
    @classmethod
    def validate_symbols(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        """Validate and normalize symbol list."""
        if v is None:
            return v
        normalized = []
        for symbol in v:
            symbol = symbol.upper().strip()
            if not SYMBOL_PATTERN.match(symbol):
                raise ValueError(
                    f"Invalid symbol '{symbol}'. Must be 4 uppercase letters"
                )
            normalized.append(symbol)
        return normalized


class SignalResponse(BaseModel):
    """A single trading signal."""
    symbol: str = Field(..., description="IDX stock symbol")
    signal_type: str = Field(..., description="Signal type: BUY, SELL, HOLD")
    setup_type: str = Field(..., description="Setup classification")
    entry_price: float = Field(..., gt=0, description="Entry price (must be > 0)")
    stop_loss: float = Field(..., gt=0, description="Stop loss price (must be > 0)")
    targets: List[float] = Field(..., min_length=1, description="Target prices")
    composite_score: float = Field(
        ...,
        ge=0,
        le=100,
        description="Composite score (0-100)"
    )
    key_factors: List[str] = Field(default_factory=list, description="Key factors")
    risks: List[str] = Field(default_factory=list, description="Risk factors")
    mode: str = Field(default="", description="Trading mode")
    timestamp: Optional[datetime] = Field(default=None, description="Signal timestamp")

    @field_validator("symbol")
    @classmethod
    def validate_symbol(cls, v: str) -> str:
        """Validate and normalize symbol."""
        v = v.upper().strip()
        if not SYMBOL_PATTERN.match(v):
            raise ValueError(f"Invalid symbol '{v}'. Must be 4 uppercase letters")
        return v

    @field_validator("signal_type")
    @classmethod
    def validate_signal_type(cls, v: str) -> str:
        """Validate signal type."""
        v = v.upper().strip()
        if v not in VALID_SIGNAL_TYPES:
            raise ValueError(f"signal_type must be one of {VALID_SIGNAL_TYPES}")
        return v

    @field_validator("targets")
    @classmethod
    def validate_targets(cls, v: List[float]) -> List[float]:
        """Validate target prices."""
        for t in v:
            if t <= 0:
                raise ValueError("All target prices must be > 0")
        return v

    @model_validator(mode="after")
    def validate_buy_signal_prices(self) -> "SignalResponse":
        """For BUY signals, ensure targets > entry > stop_loss."""
        if self.signal_type == "BUY":
            if self.entry_price <= self.stop_loss:
                raise ValueError(
                    f"For BUY signals, entry_price ({self.entry_price}) must be > "
                    f"stop_loss ({self.stop_loss})"
                )
            for target in self.targets:
                if target <= self.entry_price:
                    raise ValueError(
                        f"For BUY signals, all targets must be > entry_price ({self.entry_price})"
                    )
        return self


class ScanResponse(BaseModel):
    """Response from a daily scan."""
    scan_date: date = Field(..., description="Date of the scan")
    mode: str = Field(..., description="Trading mode used")
    result: str = Field(..., description="Scan result summary")
    signals_generated: int = Field(default=0, ge=0, description="Number of signals generated")
    signals_approved: int = Field(default=0, ge=0, description="Number of signals approved by risk manager")
    symbols_scanned: int = Field(default=0, ge=0, description="Number of symbols scanned")
    execution_time_seconds: float = Field(default=0.0, ge=0, description="Execution time in seconds")
    signals: List[SignalResponse] = Field(default_factory=list, description="Generated signals")
    errors: List[str] = Field(default_factory=list, description="Any errors encountered")


class SignalListResponse(BaseModel):
    """List of recent signals."""
    signals: List[SignalResponse] = Field(default_factory=list, description="List of signals")
    total: int = Field(..., ge=0, description="Total number of signals")
