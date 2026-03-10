from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any, List, Optional
import logging
import pandas as pd
from datetime import datetime

from pydantic import BaseModel

from sqlalchemy.orm import Session
from core.data.database import DatabaseManager, get_db
from core.data.models import OHLCV
from core.analysis.technical import TechnicalAnalyzer, TechnicalScore
from core.signals.signal_generator import SignalGenerator
from core.risk.risk_manager import RiskManager
from config.trading_modes import TradingMode, get_mode_config

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/analysis", tags=["Market Analysis"])

class AnalysisRequest(BaseModel):
    mode: str = "swing"
    capital: float = 100000000.0

def _get_ohlcv_data(symbol: str, session: Session, limit: int = 250) -> List[OHLCV]:
    """Helper to fetch OHLCV objects for analysis engines."""
    from sqlalchemy import text
    query = """
        SELECT date as date, open, high, low, close, volume
        FROM price_history
        WHERE symbol = :symbol
        ORDER BY date DESC
        LIMIT :limit
    """
    rows = session.execute(text(query), {"symbol": symbol, "limit": limit}).fetchall()
    
    if not rows:
        raise HTTPException(status_code=404, detail=f"No data for {symbol}")
        
    # Reverse to get chronological order (oldest first)
    rows = list(reversed(rows))
    
    return [OHLCV(
        timestamp=datetime.strptime(row.date, "%Y-%m-%d"),
        open=float(row.open),
            high=float(row.high),
            low=float(row.low),
            close=float(row.close),
            volume=int(row.volume)
        ) for row in rows]

@router.post("/technical/{symbol}")
def run_technical_analysis(symbol: str, db: Session = Depends(get_db)):
    """
    Run TechnicalAnalyzer on a specific stock.
    Returns the latest technical indicators and scoring.
    """
    ohlcv_list = _get_ohlcv_data(symbol, db)
    
    analyzer = TechnicalAnalyzer()
    indicators_list = analyzer.calculate(symbol, ohlcv_list)
    
    if not indicators_list:
        raise HTTPException(status_code=400, detail="Not enough data to calculate indicators")
        
    latest = indicators_list[-1]
    score = analyzer.calculate_score(latest)
    
    return {
        "symbol": symbol,
        "date": latest.timestamp.strftime("%Y-%m-%d"),
        "score": {
            "total": score.score,
            "trend_score": score.trend_score,
            "momentum_score": score.momentum_score,
            "volume_score": score.volume_score,
            "volatility_score": score.volatility_score,
            "trend": score.trend,
            "signal": score.signal
        },
        "indicators": {
            "close": latest.close,
            "ema20": latest.ema20,
            "ema50": latest.ema50,
            "sma200": latest.sma200,
            "rsi": latest.rsi,
            "macd": latest.macd,
            "macd_signal": latest.macd_signal,
            "atr": latest.atr,
            "bb_upper": latest.bb_upper,
            "bb_lower": latest.bb_lower,
            "support": latest.support,
            "resistance": latest.resistance
        }
    }

@router.post("/signal/{symbol}")
def generate_trading_signal(symbol: str, request: AnalysisRequest, db: Session = Depends(get_db)):
    """
    Run SignalGenerator to find actionable setups.
    """
    ohlcv_list = _get_ohlcv_data(symbol, db)
    
    try:
        mode = TradingMode(request.mode.lower())
    except ValueError:
        mode = TradingMode.SWING
        
    config = get_mode_config(mode)
    generator = SignalGenerator(config)
    
    signal = generator.generate(symbol, ohlcv_list)
    
    if not signal:
        return {"symbol": symbol, "signal": "None", "message": "No actionable setup found."}
        
    return {
        "symbol": symbol,
        "type": signal.signal_type.value,
        "setup": signal.setup_type.value,
        "score": signal.composite_score,
        "entry_price": signal.entry_price,
        "stop_loss": signal.stop_loss,
        "targets": signal.targets,
        "risk_reward": signal.risk_reward_ratio,
        "factors": signal.key_factors,
        "risks": signal.risks
    }

@router.post("/risk-check/{symbol}")
def check_risk_validation(symbol: str, request: AnalysisRequest, db: Session = Depends(get_db)):
    """
    Run the RiskManager validation against a generated signal.
    """
    # 1. Generate signal first
    ohlcv_list = _get_ohlcv_data(symbol, db)
    
    try:
        mode = TradingMode(request.mode.lower())
    except ValueError:
        mode = TradingMode.SWING
        
    config = get_mode_config(mode)
    generator = SignalGenerator(config)
    signal = generator.generate(symbol, ohlcv_list)
    
    if not signal:
        return {"approved": False, "reasons": ["No actionable signal generated to validate."]}
        
    # 2. Mock a portfolio state
    from core.data.models import PortfolioState
    portfolio = PortfolioState(
        total_equity=request.capital,
        cash=request.capital,
        positions=[],
        daily_pnl=0.0,
        unrealized_pnl=0.0
    )
    
    # 3. Validate
    risk_manager = RiskManager(mode, request.capital)
    result = risk_manager.validate_entry(signal, portfolio)
    
    return {
        "approved": result.approved,
        "reasons": [result.veto_reason] if result.veto_reason else [],
        "warnings": result.warnings,
        "position_size": result.position_size,
        "position_shares": result.position_size,
        "kelly_fraction": result.kelly_fraction,
        "risk_amount": result.risk_amount
    }
