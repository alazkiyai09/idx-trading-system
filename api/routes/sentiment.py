from fastapi import APIRouter, BackgroundTasks, Query
from typing import Dict, Any, List, Optional

from core.data.database import DatabaseManager

router = APIRouter(prefix="/sentiment", tags=["Sentiment"])


@router.get("/latest")
def get_latest_sentiment(symbol: Optional[str] = Query(None)):
    """Get the latest sentiment articles, optionally filtered by symbol."""
    db = DatabaseManager()
    try:
        session = db.get_session()
        from core.data.database import SentimentRecord
        query = session.query(SentimentRecord).order_by(SentimentRecord.analyzed_at.desc())
        if symbol:
            query = query.filter(SentimentRecord.symbol == symbol.upper())
        records = query.limit(20).all()
        articles = [
            {
                "symbol": r.symbol,
                "article_title": r.article_title,
                "source": r.source,
                "url": r.url,
                "published_at": str(r.published_at) if r.published_at else None,
                "analyzed_at": str(r.analyzed_at) if r.analyzed_at else None,
                "sentiment_score": r.sentiment_score,
                "confidence": r.confidence,
                "key_topics": r.key_topics,
                "themes": r.themes,
                "sector": r.sector,
            }
            for r in records
        ]
        session.close()
        return {"articles": articles}
    except Exception as e:
        return {"articles": [], "error": str(e)}


@router.get("/sector")
def get_sector_sentiment():
    """Get aggregated sentiment by sector from the database."""
    db = DatabaseManager()
    try:
        session = db.get_session()
        from core.data.database import SentimentSector
        records = session.query(SentimentSector).order_by(SentimentSector.date.desc()).limit(20).all()
        result = []
        seen_sectors = set()
        for r in records:
            if r.sector not in seen_sectors:
                seen_sectors.add(r.sector)
                result.append({
                    "sector": r.sector,
                    "avg_score": r.avg_score,
                    "article_count": r.article_count,
                    "dominant_themes": r.dominant_themes,
                    "signal": r.signal,
                    "date": str(r.date),
                })
        session.close()
        if result:
            return result
        # Return empty list if no data - the frontend has fallback demo data
        return []
    except Exception:
        return []


@router.get("/themes")
def get_themes():
    """Get trending themes with sector mappings."""
    db = DatabaseManager()
    try:
        session = db.get_session()
        from core.data.database import ThemeSectorMapping
        mappings = session.query(ThemeSectorMapping).all()
        result = [
            {
                "theme": m.theme,
                "sector": m.sector,
                "sub_sector": m.sub_sector,
                "impact_direction": m.impact_direction,
            }
            for m in mappings
        ]
        session.close()
        if result:
            return result
        # Return common default themes
        return [
            {"theme": "Oil Prices", "sector": "Energy", "sub_sector": "Oil & Gas", "impact_direction": "positive"},
            {"theme": "Interest Rates", "sector": "Financials", "sub_sector": "Banking", "impact_direction": "positive"},
            {"theme": "Inflation", "sector": "Consumer Non-Cyclicals", "sub_sector": "Food & Beverage", "impact_direction": "negative"},
            {"theme": "Nickel Prices", "sector": "Basic Materials", "sub_sector": "Mining", "impact_direction": "positive"},
            {"theme": "Rupiah Weakening", "sector": "Industrials", "sub_sector": "Import-heavy", "impact_direction": "negative"},
            {"theme": "Government Spending", "sector": "Infrastructure", "sub_sector": "Construction", "impact_direction": "positive"},
        ]
    except Exception:
        return [
            {"theme": "Oil Prices", "sector": "Energy", "sub_sector": "Oil & Gas", "impact_direction": "positive"},
            {"theme": "Interest Rates", "sector": "Financials", "sub_sector": "Banking", "impact_direction": "positive"},
            {"theme": "Inflation", "sector": "Consumer Non-Cyclicals", "sub_sector": "Food & Beverage", "impact_direction": "negative"},
        ]


@router.post("/fetch/{symbol}")
def fetch_sentiment(symbol: str, background_tasks: BackgroundTasks):
    """Trigger sentiment fetch for a specific stock."""
    # TODO: Connect to SentimentScorer/NewsFetcher pipeline
    return {"status": "accepted", "message": f"Sentiment fetch started for {symbol}"}


@router.delete("/cleanup")
def trigger_cleanup(days: int = 30):
    """Trigger manual cleanup of old sentiment data."""
    db = DatabaseManager()
    try:
        results = db.clean_old_sentiment(days=days)
        return {"status": "success", "deleted": results}
    except Exception as e:
        return {"status": "error", "message": str(e)}
