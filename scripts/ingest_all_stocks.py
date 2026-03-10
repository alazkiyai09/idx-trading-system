#!/usr/bin/env python3
"""
Batch Data Ingestion Script

Fetches all available daily OHLCV data from Yahoo Finance for all stocks
in the provided stock list, and stores it in the SQLite database.

Uses batching (20 stocks per batch, 5-second delay between batches) to
avoid Yahoo Finance rate limiting.

Usage:
    python scripts/ingest_all_stocks.py
    python scripts/ingest_all_stocks.py --stock-file /path/to/Stock_list.txt
    python scripts/ingest_all_stocks.py --batch-size 10 --delay 10
    python scripts/ingest_all_stocks.py --resume  # skip symbols already in DB
"""

import argparse
import logging
import sys
import time
from datetime import date, datetime
from pathlib import Path
from typing import List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.data.database import DatabaseManager
from core.data.scraper import IDXScraper

logger = logging.getLogger(__name__)

DEFAULT_STOCK_FILE = "/home/ubuntu/Downloads/Stock_list.txt"
DEFAULT_DB = "sqlite:///data/trading.db"


def read_stock_list(filepath: str) -> List[str]:
    """Read stock symbols from file.

    Args:
        filepath: Path to stock list file.

    Returns:
        List of stock symbols.
    """
    symbols = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip().replace("\r", "")
            if not line or line.startswith("#") or line.lower() == "stock_name":
                continue
            symbols.append(line.upper())

    # Deduplicate
    seen = set()
    unique = []
    for s in symbols:
        if s not in seen:
            seen.add(s)
            unique.append(s)
    return unique


def progress_bar(current: int, total: int, symbol: str, width: int = 40) -> str:
    """Generate progress bar string."""
    if total == 0:
        pct = 100
    else:
        pct = int(current / total * 100)
    filled = int(width * pct / 100)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {pct:3d}% | {current}/{total} | {symbol}"


def format_eta(seconds: float) -> str:
    """Format seconds into human-readable ETA."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}m"
    else:
        return f"{seconds / 3600:.1f}h"


def ingest_data(
    symbols: List[str],
    db_manager: DatabaseManager,
    batch_size: int = 20,
    delay: float = 5.0,
    resume: bool = True,
) -> dict:
    """Fetch and store all available data for symbols with batching.

    Args:
        symbols: List of stock symbols.
        db_manager: Database manager.
        batch_size: Stocks per batch.
        delay: Seconds to wait between batches.
        resume: If True, skip symbols already in DB.

    Returns:
        Statistics dictionary.
    """
    scraper = IDXScraper(delay_between_calls=0.3)

    stats = {
        "total": len(symbols),
        "success": 0,
        "failed": 0,
        "skipped": 0,
        "total_records": 0,
        "errors": [],
    }

    # Filter symbols if resuming
    if resume:
        to_fetch = []
        for symbol in symbols:
            count = db_manager.get_price_count(symbol)
            if count > 0:
                stats["skipped"] += 1
                logger.debug(f"Skipping {symbol} ({count} records already in DB)")
            else:
                to_fetch.append(symbol)
        if stats["skipped"] > 0:
            print(f"  Resuming: skipping {stats['skipped']} symbols already in DB")
        symbols = to_fetch

    if not symbols:
        print("  No symbols to fetch. All data already in database.")
        return stats

    total_to_fetch = len(symbols)
    batches = [
        symbols[i : i + batch_size] for i in range(0, len(symbols), batch_size)
    ]
    total_batches = len(batches)

    print(f"  Fetching {total_to_fetch} symbols in {total_batches} batches")
    print(f"  Batch size: {batch_size}, Delay: {delay}s")
    print("-" * 70)

    start_time = time.time()
    processed = 0

    for batch_idx, batch in enumerate(batches, 1):
        print(f"\n  Batch {batch_idx}/{total_batches}:")

        for symbol in batch:
            processed += 1
            elapsed = time.time() - start_time
            if processed > 1:
                avg_time = elapsed / (processed - 1)
                remaining = avg_time * (total_to_fetch - processed + 1)
                eta_str = f" | ETA: {format_eta(remaining)}"
            else:
                eta_str = ""

            print(
                f"\r    {progress_bar(processed, total_to_fetch, symbol)}{eta_str}   ",
                end="",
                flush=True,
            )

            try:
                # Fetch all available data
                ohlcv_data = scraper.fetch_historical_max(symbol)

                if not ohlcv_data:
                    logger.warning(f"No data for {symbol}")
                    stats["failed"] += 1
                    stats["errors"].append((symbol, "No data returned"))
                    continue

                # Convert to database format
                price_records = [
                    {
                        "symbol": d.symbol,
                        "date": d.date,
                        "open": d.open,
                        "high": d.high,
                        "low": d.low,
                        "close": d.close,
                        "volume": d.volume,
                        "value": d.value,
                    }
                    for d in ohlcv_data
                ]

                # Store in database
                inserted = db_manager.save_prices(price_records)

                stats["success"] += 1
                stats["total_records"] += inserted
                logger.debug(
                    f"  {symbol}: {inserted} new / {len(price_records)} total records"
                )

            except Exception as e:
                stats["failed"] += 1
                stats["errors"].append((symbol, str(e)))
                logger.error(f"Error fetching {symbol}: {e}")

            # Small delay between individual stocks
            time.sleep(0.3)

        # Delay between batches (not after the last batch)
        if batch_idx < total_batches:
            print(f"\n    Waiting {delay}s before next batch...", end="", flush=True)
            time.sleep(delay)

    print()  # Newline after progress
    return stats


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Fetch all available daily data from Yahoo Finance for IDX stocks.",
    )
    parser.add_argument(
        "--stock-file",
        default=DEFAULT_STOCK_FILE,
        help=f"Path to stock list file (default: {DEFAULT_STOCK_FILE})",
    )
    parser.add_argument(
        "--database",
        default=DEFAULT_DB,
        help=f"Database URL (default: {DEFAULT_DB})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=20,
        help="Stocks per batch (default: 20)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=5.0,
        help="Seconds between batches (default: 5)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Skip symbols already in database (default: True)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Re-fetch all symbols even if already in database",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Read stock list
    print("=" * 70)
    print("IDX TRADING SYSTEM — BATCH DATA INGESTION")
    print("=" * 70)

    stock_file = args.stock_file
    if not Path(stock_file).exists():
        print(f"\n  Error: Stock file not found: {stock_file}")
        return 1

    symbols = read_stock_list(stock_file)
    print(f"\n  Loaded {len(symbols)} symbols from {stock_file}")

    # Setup database
    db_path = Path("data")
    db_path.mkdir(exist_ok=True)
    db_manager = DatabaseManager(args.database)
    db_manager.create_tables()
    print(f"  Database: {args.database}")

    # Run ingestion
    resume = not args.no_resume
    print(f"  Resume mode: {'ON' if resume else 'OFF'}")
    print()

    start = time.time()
    stats = ingest_data(
        symbols=symbols,
        db_manager=db_manager,
        batch_size=args.batch_size,
        delay=args.delay,
        resume=resume,
    )
    elapsed = time.time() - start

    # Print summary
    print("\n" + "=" * 70)
    print("INGESTION SUMMARY")
    print("=" * 70)
    print(f"  Total symbols:        {stats['total']}")
    print(f"  Successful:           {stats['success']}")
    print(f"  Failed:               {stats['failed']}")
    print(f"  Skipped (in DB):      {stats['skipped']}")
    print(f"  New records stored:   {stats['total_records']:,}")
    print(f"  Elapsed time:         {format_eta(elapsed)}")

    if stats["errors"]:
        print(f"\n  Errors ({len(stats['errors'])}):")
        for symbol, error in stats["errors"][:10]:
            print(f"    - {symbol}: {error[:80]}")
        if len(stats["errors"]) > 10:
            print(f"    ... and {len(stats['errors']) - 10} more")

    print("=" * 70)
    return 0 if stats["failed"] < stats["total"] // 2 else 1


if __name__ == "__main__":
    sys.exit(main())
