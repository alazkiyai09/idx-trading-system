#!/usr/bin/env python3
"""
Script to fetch and store historical price data.

Fetches historical price data from Yahoo Finance and stores
it in the database for analysis.

Usage:
    python scripts/fetch_historical_data.py --symbols BBCA,TLKM,ASII
    python scripts/fetch_historical_data.py --file tickers.txt --days 365
    python scripts/fetch_historical_data.py --lq45 --days 365
    python scripts/fetch_historical_data.py --all --days 730

Ticker file format (tickers.txt):
    # This is a comment
    BBCA
    BBRI
    TLKM
    ASII
"""

import argparse
import logging
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import List, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.logging_config import setup_logging
from core.data.database import DatabaseManager
from core.data.scraper import IDXScraper

logger = logging.getLogger(__name__)


def read_tickers_from_file(filepath: str) -> List[str]:
    """Read stock tickers from a text file.

    Supports:
    - One ticker per line
    - Comments starting with #
    - Empty lines (ignored)
    - Comma-separated tickers on a line

    Args:
        filepath: Path to the ticker file.

    Returns:
        List of cleaned ticker symbols.

    Example file:
        # Indonesian Banks
        BBCA, BBRI, BBNI
        BMRI
        # Telco
        TLKM
    """
    tickers = []
    path = Path(filepath)

    if not path.exists():
        raise FileNotFoundError(f"Ticker file not found: {filepath}")

    with open(path, "r") as f:
        for line in f:
            line = line.strip()

            # Skip empty lines and comments
            if not line or line.startswith("#"):
                continue

            # Handle comma-separated tickers
            if "," in line:
                tickers.extend([t.strip() for t in line.split(",") if t.strip()])
            else:
                tickers.append(line.upper())

    # Remove duplicates while preserving order
    seen = set()
    unique_tickers = []
    for t in tickers:
        t = t.upper().strip()
        if t and t not in seen:
            seen.add(t)
            unique_tickers.append(t)

    return unique_tickers


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Fetch and store historical price data for IDX stocks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Fetch last year of data for specific stocks
    python scripts/fetch_historical_data.py --symbols BBCA,TLKM,ASII

    # Fetch tickers from a file
    python scripts/fetch_historical_data.py --file tickers.txt --days 365

    # Fetch last 2 years for all LQ45 stocks
    python scripts/fetch_historical_data.py --lq45 --days 730

    # Update last 30 days for all tracked stocks
    python scripts/fetch_historical_data.py --all --days 30

    # Dry run to see what would be fetched
    python scripts/fetch_historical_data.py --lq45 --dry-run

Ticker File Format:
    # Comments start with #
    # One ticker per line or comma-separated
    BBCA
    BBRI, BBNI, BMRI
    TLKM
        """,
    )

    # Symbol selection
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--symbols",
        type=str,
        help="Comma-separated list of symbols (e.g., BBCA,TLKM,ASII)",
    )
    group.add_argument(
        "--file",
        type=str,
        help="Path to text file containing ticker symbols (one per line)",
    )
    group.add_argument(
        "--lq45",
        action="store_true",
        help="Fetch all LQ45 stocks",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Fetch all tracked stocks (LQ45 + Kompas100)",
    )

    # Time range
    parser.add_argument(
        "--days",
        type=int,
        default=365,
        help="Number of days of historical data (default: 365)",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        help="Start date (YYYY-MM-DD format)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        help="End date (YYYY-MM-DD format)",
    )

    # Options
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be fetched without actually fetching",
    )
    parser.add_argument(
        "--database",
        type=str,
        default="sqlite:///data/trading.db",
        help="Database URL (default: sqlite:///data/trading.db)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        default=True,
        help="Continue fetching other symbols if one fails (default: True)",
    )

    return parser.parse_args()


def parse_date(date_str: str) -> date:
    """Parse date string to date object.

    Args:
        date_str: Date string in YYYY-MM-DD format.

    Returns:
        Date object.

    Raises:
        ValueError: If date string is invalid.
    """
    try:
        return date.fromisoformat(date_str)
    except ValueError:
        raise ValueError(f"Invalid date format: {date_str}. Use YYYY-MM-DD.")


def progress_bar(current: int, total: int, symbol: str, width: int = 40) -> str:
    """Generate a progress bar string.

    Args:
        current: Current progress value.
        total: Total value.
        symbol: Current symbol being processed.
        width: Width of progress bar.

    Returns:
        Progress bar string.
    """
    if total == 0:
        pct = 100
    else:
        pct = int(current / total * 100)

    filled = int(width * pct / 100)
    bar = "=" * filled + "-" * (width - filled)
    return f"[{bar}] {pct:3d}% | {current}/{total} | {symbol}"


def fetch_and_store(
    symbols: List[str],
    start_date: date,
    end_date: date,
    db_manager: DatabaseManager,
    dry_run: bool = False,
    continue_on_error: bool = True,
) -> dict:
    """Fetch and store historical data for given symbols.

    Args:
        symbols: List of stock symbols.
        start_date: Start date for data.
        end_date: End date for data.
        db_manager: Database manager instance.
        dry_run: If True, don't actually store data.
        continue_on_error: Continue on individual symbol errors.

    Returns:
        Dictionary with fetch statistics.
    """
    scraper = IDXScraper()
    stats = {
        "total": len(symbols),
        "success": 0,
        "failed": 0,
        "skipped": 0,
        "errors": [],
        "total_records": 0,
    }

    print(f"\nFetching data for {len(symbols)} symbols...")
    print(f"Date range: {start_date} to {end_date}")
    print("-" * 60)

    for i, symbol in enumerate(symbols, 1):
        # Show progress
        print(f"\r{progress_bar(i, len(symbols), symbol)}", end="", flush=True)

        try:
            # Fetch data
            ohlcv_data = scraper.fetch_historical(symbol, start_date, end_date)

            if not ohlcv_data:
                logger.warning(f"No data returned for {symbol}")
                stats["skipped"] += 1
                continue

            if dry_run:
                logger.info(f"[DRY RUN] Would store {len(ohlcv_data)} records for {symbol}")
                stats["success"] += 1
                stats["total_records"] += len(ohlcv_data)
                continue

            # Convert to database format
            price_records = []
            for ohlcv in ohlcv_data:
                price_records.append({
                    "symbol": ohlcv.symbol,
                    "date": ohlcv.date,
                    "open": ohlcv.open,
                    "high": ohlcv.high,
                    "low": ohlcv.low,
                    "close": ohlcv.close,
                    "volume": ohlcv.volume,
                    "value": ohlcv.value,
                })

            # Store in database
            db_manager.save_prices(price_records)

            stats["success"] += 1
            stats["total_records"] += len(price_records)
            logger.debug(f"Stored {len(price_records)} records for {symbol}")

        except Exception as e:
            error_msg = f"Error fetching {symbol}: {e}"
            logger.error(error_msg)
            stats["errors"].append((symbol, str(e)))

            if continue_on_error:
                stats["failed"] += 1
            else:
                print(f"\n\nError: {error_msg}")
                print("Stopping due to error (use --continue-on-error to continue)")
                break

    print()  # New line after progress bar
    return stats


def main() -> int:
    """Main entry point.

    Returns:
        Exit code (0 for success, 1 for error).
    """
    args = parse_args()

    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(log_level=log_level)

    # Determine symbols to fetch
    scraper = IDXScraper()

    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",")]
    elif args.file:
        symbols = read_tickers_from_file(args.file)
        print(f"Loaded {len(symbols)} tickers from {args.file}")
    elif args.lq45:
        symbols = scraper.LQ45_SYMBOLS
    else:  # --all
        symbols = scraper.get_universe(include_kompas100=True)

    # Determine date range
    end_date = date.today()
    start_date = end_date - timedelta(days=args.days)

    if args.start_date:
        start_date = parse_date(args.start_date)
    if args.end_date:
        end_date = parse_date(args.end_date)

    if start_date > end_date:
        print("Error: Start date must be before end date")
        return 1

    # Show dry run info
    if args.dry_run:
        print("\n" + "=" * 60)
        print("DRY RUN MODE - No data will be stored")
        print("=" * 60)

    # Initialize database
    if not args.dry_run:
        db_manager = DatabaseManager(args.database)
        db_manager.create_tables()
    else:
        db_manager = None

    # Fetch and store data
    stats = fetch_and_store(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        db_manager=db_manager,
        dry_run=args.dry_run,
        continue_on_error=args.continue_on_error,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("FETCH SUMMARY")
    print("=" * 60)
    print(f"Total symbols:     {stats['total']}")
    print(f"Successful:        {stats['success']}")
    print(f"Failed:            {stats['failed']}")
    print(f"Skipped (no data): {stats['skipped']}")
    print(f"Total records:     {stats['total_records']}")

    if stats["errors"]:
        print(f"\nErrors ({len(stats['errors'])}):")
        for symbol, error in stats["errors"][:5]:  # Show first 5 errors
            print(f"  - {symbol}: {error}")
        if len(stats["errors"]) > 5:
            print(f"  ... and {len(stats['errors']) - 5} more")

    if args.dry_run:
        print("\n[DRY RUN] No data was actually stored.")

    return 0 if stats["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
