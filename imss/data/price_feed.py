"""IDX price data ingestion via Yahoo Finance."""

from __future__ import annotations

import logging
from datetime import datetime

import pandas as pd
import yfinance as yf
from sqlalchemy.ext.asyncio import AsyncSession

from imss.db.models import StockOHLCV

logger = logging.getLogger(__name__)

IDX_TICKER_MAP: dict[str, str] = {
    "BBRI": "BBRI.JK",
    "BMRI": "BMRI.JK",
    "BBCA": "BBCA.JK",
    "TLKM": "TLKM.JK",
    "ASII": "ASII.JK",
    "IHSG": "^JKSE",
}


def fetch_idx_prices(
    symbols: list[str],
    start_date: str,
    end_date: str,
) -> dict[str, pd.DataFrame]:
    """Download daily OHLCV data for IDX symbols.

    Args:
        symbols: IDX ticker symbols (e.g., ["BBRI"]).
        start_date: ISO date string.
        end_date: ISO date string.

    Returns:
        Dict mapping symbol to validated DataFrame.
    """
    results: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        yf_ticker = IDX_TICKER_MAP.get(symbol, f"{symbol}.JK")
        logger.info("Downloading %s (%s) from %s to %s", symbol, yf_ticker, start_date, end_date)
        df = yf.download(yf_ticker, start=start_date, end=end_date, interval="1d", progress=False)
        if df.empty:
            logger.warning("No data returned for %s", symbol)
            continue
        # Flatten MultiIndex columns if present (yfinance >= 0.2.36)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = validate_price_data(df)
        results[symbol] = df
        logger.info("Downloaded %d rows for %s", len(df), symbol)
    return results


def validate_price_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and validate price data.

    - Remove rows with zero volume (non-trading days).
    - Remove rows with NaN close.
    - Log warnings for >25% daily price changes.
    """
    initial_len = len(df)

    # Drop NaN close
    df = df.dropna(subset=["Close"])

    # Drop zero volume
    df = df[df["Volume"] > 0]

    dropped = initial_len - len(df)
    if dropped > 0:
        logger.info("Removed %d invalid rows", dropped)

    # Warn on extreme price changes
    if len(df) > 1:
        pct_changes = df["Close"].pct_change().abs()
        extreme = pct_changes[pct_changes > 0.25]
        for idx in extreme.index:
            logger.warning(
                "Extreme price change on %s: %.1f%% (possible corporate action)",
                idx,
                pct_changes[idx] * 100,
            )

    return df


async def store_prices(
    session: AsyncSession,
    symbol: str,
    df: pd.DataFrame,
) -> int:
    """Store validated price data into stocks_ohlcv table.

    Returns number of rows inserted.
    """
    rows = []
    for idx, row in df.iterrows():
        ts = idx if isinstance(idx, datetime) else pd.Timestamp(idx).to_pydatetime()
        rows.append(
            StockOHLCV(
                symbol=symbol,
                timestamp=ts,
                open=float(row["Open"]),
                high=float(row["High"]),
                low=float(row["Low"]),
                close=float(row["Close"]),
                volume=int(row["Volume"]),
                adjusted_close=float(row.get("Adj Close", row["Close"])),
            )
        )
    session.add_all(rows)
    await session.flush()
    return len(rows)
