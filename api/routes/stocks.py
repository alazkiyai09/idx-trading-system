from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional, Dict, Any
from datetime import date, datetime, timedelta
from pydantic import BaseModel

from api.cache import api_cache
from core.data.database import DatabaseManager

router = APIRouter(prefix="/stocks", tags=["Stocks"])

class StockMetadataModel(BaseModel):
    symbol: str
    name: Optional[str] = None
    sector: Optional[str] = None
    sub_sector: Optional[str] = None
    market_cap: Optional[float] = None
    is_lq45: bool = False


def _load_tradingview_name_overrides() -> Dict[str, str]:
    """Load curated TradingView-derived name overrides if available."""
    try:
        from scripts.populate_from_tradingview import IDX_STOCK_DATA

        return {
            symbol: values[0]
            for symbol, values in IDX_STOCK_DATA.items()
            if values and len(values) >= 1
        }
    except Exception:
        return {}


def _serialize_date(value: Any) -> Optional[str]:
    """Return a stable ISO-like string for SQL date/datetime or raw string values."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _build_stock_snapshot() -> Dict[str, Any]:
    """Build a cached stock universe snapshot for dashboard reads."""
    db = DatabaseManager()
    session = db.get_session()
    from sqlalchemy import text

    query = """
    WITH latest_prices AS (
        SELECT p.symbol, p.date, p.close, p.volume
        FROM price_history p
        INNER JOIN (
            SELECT symbol, MAX(date) AS max_date
            FROM price_history
            GROUP BY symbol
        ) latest
        ON p.symbol = latest.symbol AND p.date = latest.max_date
    ),
    prev_prices AS (
        SELECT p.symbol, p.close
        FROM price_history p
        INNER JOIN (
            SELECT ph.symbol, MAX(ph.date) AS prev_date
            FROM price_history ph
            INNER JOIN (
                SELECT symbol, MAX(date) AS max_date
                FROM price_history
                GROUP BY symbol
            ) latest
            ON ph.symbol = latest.symbol
            WHERE ph.date < latest.max_date
            GROUP BY ph.symbol
        ) prev
        ON p.symbol = prev.symbol AND p.date = prev.prev_date
    )
    SELECT
        m.symbol,
        m.name,
        m.sector,
        m.sub_sector,
        m.market_cap,
        m.is_lq45,
        m.last_updated,
        lp.date AS latest_date,
        lp.close AS latest_close,
        lp.volume AS latest_volume,
        pp.close AS prev_close
    FROM stock_metadata m
    LEFT JOIN latest_prices lp ON m.symbol = lp.symbol
    LEFT JOIN prev_prices pp ON m.symbol = pp.symbol
    ORDER BY m.symbol
    """

    try:
        rows = session.execute(text(query)).fetchall()
    finally:
        session.close()

    name_overrides = _load_tradingview_name_overrides()
    stocks = []
    latest_snapshot_date = None
    for row in rows:
        change_pct = 0.0
        if row.prev_close and row.latest_close and row.prev_close > 0:
            change_pct = ((row.latest_close - row.prev_close) / row.prev_close) * 100.0

        latest_date = _serialize_date(row.latest_date)
        if latest_date and (latest_snapshot_date is None or latest_date > latest_snapshot_date):
            latest_snapshot_date = latest_date

        stocks.append(
            {
                "symbol": row.symbol,
                "name": name_overrides.get(row.symbol, row.name),
                "sector": row.sector,
                "sub_sector": row.sub_sector,
                "market_cap": row.market_cap,
                "is_lq45": bool(row.is_lq45),
                "close": float(row.latest_close) if row.latest_close is not None else None,
                "volume": float(row.latest_volume) if row.latest_volume is not None else 0.0,
                "latest_date": latest_date,
                "change_pct": round(change_pct, 2),
            }
        )

    return {
        "stocks": stocks,
        "generated_at": datetime.now().isoformat(),
        "latest_snapshot_date": latest_snapshot_date,
        "total": len(stocks),
    }


def _get_stock_snapshot() -> Dict[str, Any]:
    return api_cache.get_or_set("stocks:snapshot", ttl_seconds=3600, builder=_build_stock_snapshot)


def _build_symbol_list() -> Dict[str, Any]:
    db = DatabaseManager()
    session = db.get_session()
    from sqlalchemy import text
    name_overrides = _load_tradingview_name_overrides()

    try:
        metadata_rows = [row for row in db.get_all_stock_metadata() if getattr(row, "symbol", None)]
        metadata_symbols = {row.symbol for row in metadata_rows}
        metadata_names = {
            row.symbol: row.name for row in metadata_rows if getattr(row, "name", None)
        }
        price_symbols = {
            row[0]
            for row in session.execute(
                text("SELECT DISTINCT symbol FROM price_history WHERE symbol IS NOT NULL ORDER BY symbol")
            ).fetchall()
            if row and row[0]
        }
    finally:
        session.close()

    symbols = sorted(metadata_symbols | price_symbols)
    labels = {}
    for symbol in symbols:
        name = name_overrides.get(symbol) or metadata_names.get(symbol)
        labels[symbol] = f"{symbol} - {name}" if name else symbol

    return {
        "symbols": symbols,
        "labels": labels,
        "total": len(symbols),
        "generated_at": datetime.now().isoformat(),
    }


def _get_symbol_list() -> Dict[str, Any]:
    return api_cache.get_or_set("stocks:symbols", ttl_seconds=3600, builder=_build_symbol_list)


@router.get("/symbols")
def get_stock_symbols():
    """Get a lightweight symbol list for selectors and navigation."""
    return _get_symbol_list()

@router.get("")
def get_stocks(
    sector: Optional[str] = None,
    is_lq45: Optional[bool] = None,
    min_market_cap: Optional[float] = None,
    skip: int = Query(0, ge=0, description="Pagination offset"),
    limit: int = Query(200, ge=1, le=500, description="Max results (use lower for faster response)")
):
    """Get stock metadata with cached latest-price snapshot and pagination."""
    snapshot = _get_stock_snapshot()
    result = []
    for row in snapshot["stocks"]:
        if sector and row.get("sector") != sector:
            continue
        if is_lq45 is not None and row.get("is_lq45") != is_lq45:
            continue
        market_cap = row.get("market_cap")
        if min_market_cap and (not market_cap or market_cap < min_market_cap):
            continue
        result.append(row)

    # Apply pagination
    total = len(result)
    paginated = result[skip:skip + limit]

    return {
        "total": total,
        "skip": skip,
        "limit": limit,
        "stocks": paginated,
        "latest_snapshot_date": snapshot.get("latest_snapshot_date"),
        "generated_at": snapshot.get("generated_at"),
    }

@router.get("/{symbol}")
def get_stock_detail(symbol: str):
    """Get detailed information for a specific stock including latest price."""
    snapshot = _get_stock_snapshot()
    db = DatabaseManager()

    meta = next((item for item in snapshot["stocks"] if item["symbol"] == symbol), None)
    if not meta:
        raise HTTPException(status_code=404, detail=f"Stock {symbol} not found")

    result = {
        "symbol": meta["symbol"],
        "name": meta.get("name"),
        "sector": meta.get("sector"),
        "sub_sector": meta.get("sub_sector"),
        "market_cap": meta.get("market_cap"),
        "is_lq45": meta.get("is_lq45", False),
        "latest_price": {
            "date": meta.get("latest_date"),
            "close": meta.get("close"),
            "volume": meta.get("volume", 0),
            "change_pct": meta.get("change_pct", 0),
        },
        "snapshot_generated_at": snapshot.get("generated_at"),
    }

    # Fallback if snapshot lacks a latest price.
    if not result["latest_price"]["date"]:
        latest_price = db.get_latest_price(symbol)
        if latest_price:
            result["latest_price"] = {
                "date": latest_price.date.isoformat(),
                "close": latest_price.close,
                "volume": latest_price.volume,
                "change_pct": meta.get("change_pct", 0),
            }

    return result

@router.get("/{symbol}/chart")
def get_stock_chart(symbol: str, days: int = Query(200, ge=10, le=1000)):
    """Get OHLCV price history for charting."""
    db = DatabaseManager()
    start_date = date.today() - timedelta(days=days)
    prices = db.get_prices(symbol, start_date)

    if not prices:
        raise HTTPException(status_code=404, detail=f"No price data found for {symbol}")

    return [
        {
            "date": p.date.isoformat(),
            "open": p.open,
            "high": p.high,
            "low": p.low,
            "close": p.close,
            "volume": p.volume
        }
        for p in prices
    ]


@router.get("/{symbol}/foreign-flow")
def get_stock_foreign_flow(symbol: str, days: int = Query(30, ge=7, le=365)):
    """Get foreign investor flow history for a stock."""
    db = DatabaseManager()
    session = db.get_session()
    from sqlalchemy import text

    start_date = date.today() - timedelta(days=days)

    try:
        query = text("""
            SELECT date, foreign_buy, foreign_sell, foreign_net
            FROM foreign_flow_history
            WHERE symbol = :symbol AND date >= :start_date
            ORDER BY date DESC
        """)
        rows = session.execute(query, {"symbol": symbol, "start_date": start_date}).fetchall()

        if not rows:
            # Return empty data with structure instead of 404
            return {"symbol": symbol, "data": [], "message": "No foreign flow data available"}

        return {
            "symbol": symbol,
            "data": [
                {
                    "date": r.date.isoformat() if r.date else None,
                    "foreign_buy": float(r.foreign_buy) if r.foreign_buy else 0,
                    "foreign_sell": float(r.foreign_sell) if r.foreign_sell else 0,
                    "foreign_net": float(r.foreign_net) if r.foreign_net else 0,
                }
                for r in rows
            ]
        }
    except Exception as e:
        # Table might not exist
        return {"symbol": symbol, "data": [], "message": f"Foreign flow data unavailable: {str(e)}"}
