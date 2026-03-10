#!/usr/bin/env python3
"""Backfill missing stock metadata for traded IDX symbols.

This script uses the traded symbol universe from `price_history` and enriches any
missing `stock_metadata` rows using Yahoo Finance with `.JK` suffixes. It is safe
to rerun and only fills gaps or empty names unless `--overwrite` is passed.
"""

import argparse
import logging
import os
import sys
import time
from typing import Dict, List

import yfinance as yf
from sqlalchemy import text

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.data.database import DatabaseManager
from scripts.populate_metadata import map_yf_sector_to_idx


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def get_symbols_to_enrich(db: DatabaseManager, overwrite: bool) -> List[str]:
    """Return traded symbols that need metadata enrichment."""
    session = db.get_session()
    try:
        if overwrite:
            rows = session.execute(
                text("SELECT DISTINCT symbol FROM price_history WHERE symbol IS NOT NULL ORDER BY symbol")
            ).fetchall()
        else:
            rows = session.execute(
                text(
                    """
                    SELECT DISTINCT p.symbol
                    FROM price_history p
                    LEFT JOIN stock_metadata m ON m.symbol = p.symbol
                    WHERE p.symbol IS NOT NULL
                      AND (m.symbol IS NULL OR m.name IS NULL OR trim(m.name) = '')
                    ORDER BY p.symbol
                    """
                )
            ).fetchall()
        return [row[0] for row in rows if row and row[0]]
    finally:
        session.close()


def fetch_metadata(symbol: str) -> Dict:
    """Fetch symbol metadata from Yahoo Finance."""
    jk_symbol = f"{symbol}.JK"
    ticker = yf.Ticker(jk_symbol)
    info = ticker.info

    exchange = info.get("exchange", "")
    if exchange and exchange not in ("JKT", "Jakarta", "IDX", "JKS"):
        raise ValueError(f"{symbol} resolved to non-IDX exchange '{exchange}'")

    long_name = info.get("longName") or info.get("shortName") or symbol
    sector = map_yf_sector_to_idx(info.get("sector", ""))
    return {
        "symbol": symbol,
        "name": long_name,
        "sector": sector,
        "sub_sector": info.get("industry", "") or None,
        "market_cap": info.get("marketCap", 0) or None,
        "board": "Main",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Enrich missing stock metadata from Yahoo Finance")
    parser.add_argument("--overwrite", action="store_true", help="Refresh all traded symbols, not just missing rows")
    parser.add_argument("--limit", type=int, default=0, help="Only process the first N symbols")
    parser.add_argument("--sleep", type=float, default=0.3, help="Delay between requests in seconds")
    args = parser.parse_args()

    db = DatabaseManager()
    db.create_tables()

    symbols = get_symbols_to_enrich(db, overwrite=args.overwrite)
    if args.limit > 0:
        symbols = symbols[: args.limit]

    logger.info("Symbols requiring enrichment: %s", len(symbols))
    if not symbols:
        return 0

    batch: List[Dict] = []
    success = 0
    failures = 0

    for idx, symbol in enumerate(symbols, start=1):
        try:
            meta = fetch_metadata(symbol)
            batch.append(meta)
            success += 1
            logger.info("[%s/%s] %s -> %s", idx, len(symbols), symbol, meta["name"])
        except Exception as exc:
            failures += 1
            logger.warning("[%s/%s] %s failed: %s", idx, len(symbols), symbol, exc)

        if len(batch) >= 25:
            db.save_stock_metadata(batch)
            batch.clear()

        time.sleep(args.sleep)

    if batch:
        db.save_stock_metadata(batch)

    logger.info("Metadata enrichment complete. success=%s failures=%s", success, failures)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
