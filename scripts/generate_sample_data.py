#!/usr/bin/env python3
"""
Script to generate and store sample market data.

Creates realistic simulated market data for testing and development.
Useful when Yahoo Finance API is not accessible.

Usage:
    python scripts/generate_sample_data.py
    python scripts/generate_sample_data.py --symbols BBCA,BBRI,TLKM --days 500
    python scripts/generate_sample_data.py --lq45 --days 252
"""

import argparse
import logging
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import List, Dict
import random

import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.logging_config import setup_logging
from core.data.database import DatabaseManager
from core.data.models import OHLCV

logger = logging.getLogger(__name__)


# Stock configurations with realistic parameters
STOCK_CONFIGS = {
    # Blue chip banks
    "BBCA": {"start_price": 9000, "trend": 0.0003, "volatility": 0.012, "sector": "banking"},
    "BBRI": {"start_price": 5000, "trend": 0.0004, "volatility": 0.015, "sector": "banking"},
    "BBNI": {"start_price": 4500, "trend": 0.0003, "volatility": 0.014, "sector": "banking"},
    "BMRI": {"start_price": 5500, "trend": 0.0003, "volatility": 0.013, "sector": "banking"},
    "BBTN": {"start_price": 1500, "trend": 0.0002, "volatility": 0.018, "sector": "banking"},

    # Telco
    "TLKM": {"start_price": 3500, "trend": 0.0001, "volatility": 0.010, "sector": "telco"},
    "EXCL": {"start_price": 2500, "trend": 0.0002, "volatility": 0.016, "sector": "telco"},
    "ISAT": {"start_price": 2000, "trend": 0.0001, "volatility": 0.015, "sector": "telco"},

    # Consumer
    "ICBP": {"start_price": 9500, "trend": 0.0002, "volatility": 0.012, "sector": "consumer"},
    "INDF": {"start_price": 5500, "trend": 0.0002, "volatility": 0.013, "sector": "consumer"},
    "UNVR": {"start_price": 4000, "trend": 0.0001, "volatility": 0.010, "sector": "consumer"},
    "CPIN": {"start_price": 3500, "trend": 0.0003, "volatility": 0.015, "sector": "consumer"},

    # Tobacco
    "GGRM": {"start_price": 25000, "trend": 0.0001, "volatility": 0.011, "sector": "tobacco"},
    "HMSP": {"start_price": 1500, "trend": 0.0000, "volatility": 0.014, "sector": "tobacco"},

    # Mining & Resources
    "ADRO": {"start_price": 2500, "trend": 0.0003, "volatility": 0.022, "sector": "mining"},
    "INCO": {"start_price": 4500, "trend": 0.0002, "volatility": 0.020, "sector": "mining"},
    "PTBA": {"start_price": 2000, "trend": 0.0003, "volatility": 0.025, "sector": "mining"},
    "MDKA": {"start_price": 3500, "trend": 0.0002, "volatility": 0.023, "sector": "mining"},

    # Auto
    "ASII": {"start_price": 5500, "trend": 0.0002, "volatility": 0.016, "sector": "auto"},
    "UNTR": {"start_price": 28000, "trend": 0.0002, "volatility": 0.014, "sector": "auto"},

    # Infrastructure
    "WIKA": {"start_price": 1200, "trend": 0.0001, "volatility": 0.020, "sector": "infra"},
    "PTPP": {"start_price": 1500, "trend": 0.0002, "volatility": 0.022, "sector": "infra"},
    "JSMR": {"start_price": 2500, "trend": 0.0002, "volatility": 0.018, "sector": "infra"},

    # Property
    "BCMA": {"start_price": 500, "trend": 0.0001, "volatility": 0.020, "sector": "property"},
    "PWON": {"start_price": 400, "trend": 0.0001, "volatility": 0.018, "sector": "property"},

    # Retail
    "AMRT": {"start_price": 2500, "trend": 0.0003, "volatility": 0.016, "sector": "retail"},
    "MAPI": {"start_price": 4000, "trend": 0.0004, "volatility": 0.018, "sector": "retail"},

    # Tech
    "GOTO": {"start_price": 80, "trend": 0.0000, "volatility": 0.030, "sector": "tech"},
    "BUKA": {"start_price": 150, "trend": 0.0001, "volatility": 0.028, "sector": "tech"},

    # Healthcare
    "MIKA": {"start_price": 15000, "trend": 0.0002, "volatility": 0.012, "sector": "healthcare"},

    # Chemical
    "TPIA": {"start_price": 3500, "trend": 0.0002, "volatility": 0.018, "sector": "chemical"},
    "BRPT": {"start_price": 8000, "trend": 0.0002, "volatility": 0.015, "sector": "chemical"},

    # Cement
    "SMGR": {"start_price": 9000, "trend": 0.0001, "volatility": 0.014, "sector": "cement"},
    "INTP": {"start_price": 8000, "trend": 0.0001, "volatility": 0.015, "sector": "cement"},
}

# LQ45 symbols
LQ45_SYMBOLS = list(STOCK_CONFIGS.keys())


def generate_ohlcv_data(
    symbol: str,
    start_date: date,
    end_date: date,
    config: dict,
) -> List[OHLCV]:
    """Generate realistic OHLCV data for a single stock.

    Uses geometric Brownian motion with sector-specific parameters.

    Args:
        symbol: Stock symbol.
        start_date: Start date.
        end_date: End date.
        config: Stock configuration dict.

    Returns:
        List of OHLCV data points.
    """
    # Set seed based on symbol for reproducibility
    np.random.seed(sum(ord(c) for c in symbol))
    random.seed(sum(ord(c) for c in symbol))

    data = []
    current_date = start_date
    current_price = config["start_price"]
    trend = config["trend"]
    volatility = config["volatility"]

    # Sector-specific volume
    base_volume = {
        "banking": 50_000_000,
        "telco": 100_000_000,
        "consumer": 30_000_000,
        "tobacco": 10_000_000,
        "mining": 40_000_000,
        "auto": 20_000_000,
        "infra": 25_000_000,
        "property": 15_000_000,
        "retail": 40_000_000,
        "tech": 200_000_000,
        "healthcare": 5_000_000,
        "chemical": 15_000_000,
        "cement": 10_000_000,
    }.get(config["sector"], 30_000_000)

    while current_date <= end_date:
        # Skip weekends
        if current_date.weekday() >= 5:
            current_date += timedelta(days=1)
            continue

        # Daily return with trend and volatility
        daily_return = np.random.normal(trend, volatility)

        # Add some autocorrelation (momentum)
        if len(data) > 0:
            prev_return = (data[-1].close - data[-1].open) / data[-1].open
            daily_return += prev_return * 0.1

        current_price = current_price * (1 + daily_return)

        # Generate OHLC
        intra_day_vol = volatility * 0.3
        open_price = current_price * (1 + np.random.normal(0, intra_day_vol))
        high_price = max(open_price, current_price) * (1 + abs(np.random.normal(0, intra_day_vol)))
        low_price = min(open_price, current_price) * (1 - abs(np.random.normal(0, intra_day_vol)))
        close_price = current_price

        # Generate volume
        volume = int(base_volume * (1 + np.random.uniform(-0.4, 0.6)))

        ohlcv = OHLCV(
            symbol=symbol,
            date=current_date,
            open=round(open_price, 0),
            high=round(high_price, 0),
            low=round(low_price, 0),
            close=round(close_price, 0),
            volume=volume,
            value=round(close_price * volume, 0),
        )
        data.append(ohlcv)

        current_date += timedelta(days=1)

    return data


def generate_foreign_flow_data(symbol: str, start_date: date, end_date: date) -> List[dict]:
    """Generate foreign flow data for a stock.

    Args:
        symbol: Stock symbol.
        start_date: Start date.
        end_date: End date.

    Returns:
        List of foreign flow records.
    """
    np.random.seed(sum(ord(c) for c in symbol) + 1000)

    data = []
    current_date = start_date

    # Base flow magnitude
    base_flow = 50_000_000_000  # 50 billion IDR

    while current_date <= end_date:
        if current_date.weekday() >= 5:
            current_date += timedelta(days=1)
            continue

        # Random flow with slight bias based on symbol
        symbol_bias = (sum(ord(c) for c in symbol) % 10 - 5) / 100

        foreign_buy = base_flow * (0.3 + np.random.random() * 0.4)
        foreign_sell = base_flow * (0.3 + np.random.random() * 0.4 + symbol_bias)
        foreign_sell = max(0, foreign_sell)

        data.append({
            "symbol": symbol,
            "date": current_date,
            "foreign_buy": round(foreign_buy, 0),
            "foreign_sell": round(foreign_sell, 0),
            "foreign_net": round(foreign_buy - foreign_sell, 0),
            "total_value": round(base_flow * (1 + np.random.random() * 0.5), 0),
        })

        current_date += timedelta(days=1)

    return data


def parse_args():
    parser = argparse.ArgumentParser(description="Generate sample market data")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--symbols", type=str, help="Comma-separated symbols")
    group.add_argument("--lq45", action="store_true", help="Generate LQ45 data")
    group.add_argument("--all", action="store_true", help="Generate all available stocks")

    parser.add_argument("--days", type=int, default=365, help="Number of days (default: 365)")
    parser.add_argument("--database", type=str, default="sqlite:///data/trading.db")
    parser.add_argument("--include-flow", action="store_true", help="Also generate foreign flow data")
    parser.add_argument("--verbose", "-v", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    setup_logging(log_level="DEBUG" if args.verbose else "INFO")

    # Determine symbols
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",")]
    elif args.lq45:
        symbols = LQ45_SYMBOLS
    else:
        symbols = list(STOCK_CONFIGS.keys())

    # Filter to available configs
    symbols = [s for s in symbols if s in STOCK_CONFIGS]

    if not symbols:
        print("No valid symbols found!")
        return 1

    # Date range
    end_date = date.today()
    start_date = end_date - timedelta(days=args.days)

    print(f"\n{'='*60}")
    print("SAMPLE DATA GENERATION")
    print(f"{'='*60}")
    print(f"Symbols: {len(symbols)}")
    print(f"Date range: {start_date} to {end_date}")
    print(f"Days: {args.days}")
    print(f"{'='*60}\n")

    # Initialize database
    db_manager = DatabaseManager(args.database)
    db_manager.create_tables()

    # Generate and store data
    total_prices = 0
    total_flows = 0

    for i, symbol in enumerate(symbols, 1):
        config = STOCK_CONFIGS.get(symbol)
        if not config:
            print(f"  [{i}/{len(symbols)}] {symbol}: SKIPPED (no config)")
            continue

        print(f"  [{i}/{len(symbols)}] {symbol}: ", end="", flush=True)

        try:
            # Generate OHLCV data
            ohlcv_data = generate_ohlcv_data(symbol, start_date, end_date, config)

            # Convert to records
            price_records = [{
                "symbol": o.symbol,
                "date": o.date,
                "open": o.open,
                "high": o.high,
                "low": o.low,
                "close": o.close,
                "volume": o.volume,
                "value": o.value,
            } for o in ohlcv_data]

            # Store in database
            db_manager.save_prices(price_records)
            total_prices += len(price_records)

            msg = f"{len(price_records)} prices"

            # Generate foreign flow if requested
            if args.include_flow:
                flow_data = generate_foreign_flow_data(symbol, start_date, end_date)
                db_manager.save_foreign_flow(flow_data)
                total_flows += len(flow_data)
                msg += f", {len(flow_data)} flows"

            print(f"{msg}")

        except Exception as e:
            print(f"ERROR: {e}")
            logger.error(f"Error generating data for {symbol}: {e}")

    # Summary
    print(f"\n{'='*60}")
    print("GENERATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total symbols:  {len(symbols)}")
    print(f"Price records:  {total_prices:,}")
    if args.include_flow:
        print(f"Flow records:   {total_flows:,}")
    print(f"Database:       {args.database}")
    print(f"{'='*60}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
