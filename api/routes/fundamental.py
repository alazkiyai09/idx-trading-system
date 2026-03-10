"""Fundamental analysis routes."""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/fundamental", tags=["fundamental"])


class FundamentalRequest(BaseModel):
    """Request for fundamental analysis."""
    symbol: str = Field(description="Stock symbol to analyze")
    pdf_path: Optional[str] = Field(default=None, description="Path to financial report PDF")


class FundamentalResponse(BaseModel):
    """Fundamental analysis response."""
    symbol: str
    overall_score: float = 0.0
    recommendation: str = ""
    confidence: float = 0.0
    agent_reports: dict = {}
    veto_triggered: bool = False
    veto_reason: Optional[str] = None


@router.post("/analyze", response_model=FundamentalResponse)
async def analyze_fundamental(request: FundamentalRequest):
    """Run fundamental analysis on a symbol.

    Runs the multi-agent analysis pipeline on the given symbol.
    """
    try:
        # TODO: Integrate with full fundamental analysis pipeline
        return FundamentalResponse(
            symbol=request.symbol,
            overall_score=0.0,
            recommendation="Analysis not yet integrated with API",
            confidence=0.0,
        )
    except Exception as e:
        logger.error(f"Fundamental analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
