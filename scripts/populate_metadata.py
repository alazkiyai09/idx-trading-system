#!/usr/bin/env python3
"""
Populate Stock Metadata

Fetches static stock information (sector, industry, market cap)
from Yahoo Finance for all stocks in the local database and
saves it to the `stock_metadata` table.
"""

import logging
import os
import sys
import time
from datetime import datetime

import yfinance as yf

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.data.database import DatabaseManager
from core.data.scraper import IDXScraper

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def map_yf_sector_to_idx(yf_sector: str) -> str:
    """Map Yahoo Finance sector to IDX standard sector."""
    if not yf_sector:
        return "Unknown"

    sector_map = {
        "Energy": "Energy",
        "Basic Materials": "Basic Materials",
        "Industrials": "Industrials",
        "Consumer Cyclical": "Consumer Cyclicals",
        "Consumer Defensive": "Consumer Non-Cyclicals",
        "Healthcare": "Healthcare",
        "Financial Services": "Financials",
        "Real Estate": "Properties & Real Estate",
        "Technology": "Technology",
        "Communication Services": "Infrastructure",
        "Utilities": "Infrastructure",
    }
    return sector_map.get(yf_sector, yf_sector)


def populate_metadata():
    """Fetch and save metadata for all stocks."""
    db = DatabaseManager()
    db.create_tables()

    scraper = IDXScraper()
    symbols = scraper.get_universe("ALL")
    
    logger.info(f"Fetching metadata for {len(symbols)} stocks...")
    
    batch_size = 20
    metadata_list = []
    
    for i, symbol in enumerate(symbols):
        try:
            # yfinance needs .JK suffix for IDX (Jakarta Stock Exchange)
            jk_symbol = f"{symbol}.JK" if not symbol.endswith(".JK") else symbol
            ticker = yf.Ticker(jk_symbol)
            info = ticker.info
            
            # Validate we got IDX data, not a US ETF
            exchange = info.get("exchange", "")
            if exchange and exchange not in ("JKT", "Jakarta", "IDX", "JKS"):
                logger.warning(f"Skipping {symbol}: resolved to exchange '{exchange}' (not IDX)")
                continue
            
            if not info:
                logger.warning(f"No info found for {symbol}")
                continue
                
            yf_sector = info.get("sector", "")
            idx_sector = map_yf_sector_to_idx(yf_sector)
            
            meta = {
                "symbol": symbol.replace(".JK", ""),
                "name": info.get("longName", info.get("shortName", "")),
                "sector": idx_sector,
                "sub_sector": info.get("industry", ""),
                "market_cap": info.get("marketCap", 0),
                "is_lq45": symbol.replace(".JK", "") in scraper.get_universe("LQ45"),
                "is_idx30": False,  # Would need true IDX30 list
            }
            
            metadata_list.append(meta)
            logger.info(f"[{i+1}/{len(symbols)}] Fetched {symbol}: {idx_sector} / {meta['sub_sector']}")
            
            # Save in batches
            if len(metadata_list) >= batch_size:
                db.save_stock_metadata(metadata_list)
                logger.info(f"Saved batch of {len(metadata_list)} records")
                metadata_list = []
                # Sleep to avoid rate limits
                time.sleep(2)
                
        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")
            time.sleep(5)  # Longer backoff on error

    # Save remaining
    if metadata_list:
        db.save_stock_metadata(metadata_list)
        logger.info(f"Saved final batch of {len(metadata_list)} records")

    logger.info("Metadata population complete.")


if __name__ == "__main__":
    populate_metadata()
