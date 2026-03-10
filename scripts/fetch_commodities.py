#!/usr/bin/env python
"""
Daily commodity data fetch script.

Fetches Gold, Silver, and Oil futures data from Yahoo Finance
and stores in the database.

Usage:
    python scripts/fetch_commodities.py --days 365
    python scripts/fetch_commodities.py --symbols GOLD,SILVER,OIL
"""
import argparse
import logging
import sys
from datetime import date

# Add project root to path
sys.path.insert(0, "/mnt/data/Project/idx-trading-system")

from core.data.database import DatabaseManager
from core.data.commodity_scraper import CommodityScraper

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Fetch commodity price data")
    parser.add_argument(
        "--days",
        type=int,
        default=365,
        help="Number of days of historical data to fetch"
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default="GOLD,SILVER,OIL",
        help="Comma-separated commodity symbols to fetch"
    )
    parser.add_argument(
        "--db-url",
        type=str,
        default="sqlite:///data/trading.db",
        help="Database URL"
    )
    args = parser.parse_args()

    # Parse symbols
    symbols = [s.strip().upper() for s in args.symbols.split(",")]

    # Initialize
    db = DatabaseManager(args.db_url)
    scraper = CommodityScraper(db)
    db.create_tables()

    logger.info(f"Fetching {args.days} days of data for: {symbols}")

    # Fetch each commodity
    total_inserted = 0
    for symbol in symbols:
        try:
            logger.info(f"Fetching {symbol}...")
            data = scraper.fetch_commodity_data(symbol, days=args.days)

            if data:
                inserted = scraper.save_to_database(data)
                total_inserted += inserted
                logger.info(f"Inserted {inserted} records for {symbol}")
            else:
                logger.warning(f"No data fetched for {symbol}")

        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")

    logger.info(f"Total records inserted: {total_inserted}")

    # Verify
    for symbol in symbols:
        prices = db.get_commodity_prices(symbol, days=30)
        logger.info(f"Verified {symbol}: {len(prices)} recent records")


if __name__ == "__main__":
    main()
