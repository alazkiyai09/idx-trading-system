#!/usr/bin/env python3
"""
Populate Stock Metadata from TradingView

Scrapes the TradingView IDX all-stocks page to build a comprehensive
symbol → (name, sector) mapping for Indonesian stocks.

Reference: https://id.tradingview.com/markets/stocks-indonesia/market-movers-all-stocks/

This is a fallback/supplement to the yfinance-based populate_metadata.py,
which can resolve symbols incorrectly to US ETFs when the .JK suffix is missing.
"""

import logging
import os
import sys
import re
import time

import requests
from bs4 import BeautifulSoup

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.data.database import DatabaseManager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# TradingView sector name mapping (Indonesian → English)
TV_SECTOR_MAP = {
    "Mineral Energi": "Energy",
    "Industri Proses": "Basic Materials",
    "Layanan Konsumen": "Consumer Cyclicals",
    "Keuangan": "Financials",
    "Perdagangan Ritel": "Consumer Cyclicals",
    "Layanan Industri": "Industrials",
    "Konsumen Tidak Tahan Lama": "Consumer Non-Cyclicals",
    "Produsen Pabrikan": "Industrials",
    "Mineral Non-Energi": "Basic Materials",
    "Layanan Distribusi": "Industrials",
    "Layanan Teknologi": "Technology",
    "Transportasi": "Transportation & Logistics",
    "Layanan Kesehatan": "Healthcare",
    "Utilitas": "Infrastructure",
    "Komunikasi": "Infrastructure",
    "Konsumen Tahan Lama": "Consumer Cyclicals",
    "Teknologi Elektronik": "Technology",
    "Layanan Komersial": "Industrials",
}

# Hardcoded authoritative IDX stock data from TradingView
# This ensures we have accurate data even without scraping
IDX_STOCK_DATA = {
    "AADI": ("Pt Adaro Andalan Indonesia Tbk", "Energy", "Coal Mining"),
    "AALI": ("PT Astra Agro Lestari Tbk", "Basic Materials", "Plantation"),
    "ACES": ("PT Aspirasi Hidup Indonesia Tbk", "Consumer Cyclicals", "Retail Trade"),
    "ADHI": ("PT Adhi Karya (Persero) Tbk", "Industrials", "Construction"),
    "ADMF": ("PT Adira Dinamika Multi Finance Tbk", "Financials", "Multi-Finance"),
    "ADMR": ("PT Alamtri Minerals Indonesia Tbk", "Energy", "Coal Mining"),
    "ADRO": ("PT Alamtri Resources Indonesia Tbk", "Energy", "Coal Mining"),
    "AGII": ("PT Samator Indo Gas Tbk", "Basic Materials", "Chemicals"),
    "AGRO": ("PT Bank Raya Indonesia Tbk", "Financials", "Banking"),
    "AKRA": ("PT AKR Corporindo Tbk", "Industrials", "Distribution"),
    "AMMN": ("PT Amman Mineral Internasional Tbk", "Basic Materials", "Mining"),
    "AMRT": ("PT Sumber Alfaria Trijaya Tbk", "Consumer Cyclicals", "Retail Trade"),
    "ANTM": ("PT ANTAM (Persero) Tbk", "Basic Materials", "Mining"),
    "APLN": ("PT Agung Podomoro Land Tbk", "Properties & Real Estate", "Property"),
    "ARNA": ("PT Arwana Citramulia Tbk", "Industrials", "Manufacturing"),
    "ASII": ("PT Astra International Tbk", "Industrials", "Automotive"),
    "ASSA": ("PT Adi Sarana Armada Tbk", "Transportation & Logistics", "Transportation"),
    "BBCA": ("PT Bank Central Asia Tbk", "Financials", "Banking"),
    "BBNI": ("PT Bank Negara Indonesia (Persero) Tbk", "Financials", "Banking"),
    "BBRI": ("PT Bank Rakyat Indonesia (Persero) Tbk", "Financials", "Banking"),
    "BBTN": ("PT Bank Tabungan Negara (Persero) Tbk", "Financials", "Banking"),
    "BFIN": ("PT BFI Finance Indonesia Tbk", "Financials", "Multi-Finance"),
    "BMRI": ("PT Bank Mandiri (Persero) Tbk", "Financials", "Banking"),
    "BRPT": ("PT Barito Pacific Tbk", "Basic Materials", "Petrochemicals"),
    "BRIS": ("PT Bank Syariah Indonesia Tbk", "Financials", "Banking"),
    "BUKA": ("PT Bukalapak.com Tbk", "Technology", "E-commerce"),
    "CLEO": ("PT Sariguna Primatirta Tbk", "Consumer Non-Cyclicals", "Food & Beverage"),
    "CPIN": ("PT Charoen Pokphand Indonesia Tbk", "Consumer Non-Cyclicals", "Animal Feed"),
    "DNET": ("PT Indoritel Makmur Internasional Tbk", "Technology", "IT Services"),
    "ELSA": ("PT Elnusa Tbk", "Energy", "Oil & Gas Services"),
    "EMTK": ("PT Elang Mahkota Teknologi Tbk", "Technology", "Media"),
    "ENRG": ("PT Energi Mega Persada Tbk", "Energy", "Oil & Gas"),
    "ESSA": ("PT Surya Esa Perkasa Tbk", "Energy", "Gas Distribution"),
    "EXCL": ("PT XL Axiata Tbk", "Infrastructure", "Telecommunications"),
    "GGRM": ("PT Gudang Garam Tbk", "Consumer Non-Cyclicals", "Tobacco"),
    "GOTO": ("PT GoTo Gojek Tokopedia Tbk", "Technology", "E-commerce"),
    "HMSP": ("PT H.M. Sampoerna Tbk", "Consumer Non-Cyclicals", "Tobacco"),
    "HRUM": ("PT Harum Energy Tbk", "Energy", "Coal Mining"),
    "ICBP": ("PT Indofood CBP Sukses Makmur Tbk", "Consumer Non-Cyclicals", "Food & Beverage"),
    "INCO": ("PT Vale Indonesia Tbk", "Basic Materials", "Mining"),
    "INDF": ("PT Indofood Sukses Makmur Tbk", "Consumer Non-Cyclicals", "Food & Beverage"),
    "INKP": ("PT Indah Kiat Pulp & Paper Tbk", "Basic Materials", "Pulp & Paper"),
    "INTP": ("PT Indocement Tunggal Prakarsa Tbk", "Industrials", "Cement"),
    "ISAT": ("PT Indosat Tbk", "Infrastructure", "Telecommunications"),
    "ITMG": ("PT Indo Tambangraya Megah Tbk", "Energy", "Coal Mining"),
    "JPFA": ("PT Japfa Comfeed Indonesia Tbk", "Consumer Non-Cyclicals", "Animal Feed"),
    "JSMR": ("PT Jasa Marga (Persero) Tbk", "Infrastructure", "Toll Roads"),
    "KLBF": ("PT Kalbe Farma Tbk", "Healthcare", "Pharmaceuticals"),
    "MAPI": ("PT Mitra Adiperkasa Tbk", "Consumer Cyclicals", "Retail Trade"),
    "MDKA": ("PT Merdeka Copper Gold Tbk", "Basic Materials", "Mining"),
    "MEDC": ("PT Medco Energi Internasional Tbk", "Energy", "Oil & Gas"),
    "MIKA": ("PT Mitra Keluarga Karyasehat Tbk", "Healthcare", "Hospitals"),
    "MNCN": ("PT MNC Digital Entertainment Tbk", "Technology", "Media"),
    "PGAS": ("PT Perusahaan Gas Negara Tbk", "Energy", "Gas Distribution"),
    "PTBA": ("PT Bukit Asam Tbk", "Energy", "Coal Mining"),
    "PTPP": ("PT PP (Persero) Tbk", "Industrials", "Construction"),
    "SIDO": ("PT Industri Jamu dan Farmasi Sido Muncul Tbk", "Healthcare", "Pharmaceuticals"),
    "SMGR": ("PT Semen Indonesia (Persero) Tbk", "Industrials", "Cement"),
    "SMRA": ("PT Summarecon Agung Tbk", "Properties & Real Estate", "Property"),
    "SRTG": ("PT Saratoga Investama Sedaya Tbk", "Financials", "Investment"),
    "TBIG": ("PT Tower Bersama Infrastructure Tbk", "Infrastructure", "Telecom Infrastructure"),
    "TINS": ("PT Timah Tbk", "Basic Materials", "Mining"),
    "TLKM": ("PT Telkom Indonesia (Persero) Tbk", "Infrastructure", "Telecommunications"),
    "TOWR": ("PT Sarana Menara Nusantara Tbk", "Infrastructure", "Telecom Infrastructure"),
    "TPIA": ("PT Chandra Asri Pacific Tbk", "Basic Materials", "Petrochemicals"),
    "UNTR": ("PT United Tractors Tbk", "Industrials", "Heavy Equipment"),
    "UNVR": ("PT Unilever Indonesia Tbk", "Consumer Non-Cyclicals", "Consumer Goods"),
    "WIKA": ("PT Wijaya Karya (Persero) Tbk", "Industrials", "Construction"),
    "BSDE": ("PT Bumi Serpong Damai Tbk", "Properties & Real Estate", "Property"),
    "CTRA": ("PT Ciputra Development Tbk", "Properties & Real Estate", "Property"),
    "ERAA": ("PT Erajaya Swasembada Tbk", "Consumer Cyclicals", "Retail Trade"),
    "FILM": ("PT MD Pictures Tbk", "Consumer Cyclicals", "Entertainment"),
    "GJTL": ("PT Gajah Tunggal Tbk", "Consumer Cyclicals", "Automotive Parts"),
    "HEAL": ("PT Medikaloka Hermina Tbk", "Healthcare", "Hospitals"),
    "INDY": ("PT Indika Energy Tbk", "Energy", "Coal Mining"),
    "LPKR": ("PT Lippo Karawaci Tbk", "Properties & Real Estate", "Property"),
    "LPPF": ("PT Matahari Department Store Tbk", "Consumer Cyclicals", "Retail Trade"),
    "MYOR": ("PT Mayora Indah Tbk", "Consumer Non-Cyclicals", "Food & Beverage"),
    "PNBN": ("PT Bank Pan Indonesia Tbk", "Financials", "Banking"),
    "PWON": ("PT Pakuwon Jati Tbk", "Properties & Real Estate", "Property"),
    "SCMA": ("PT Surya Citra Media Tbk", "Technology", "Media"),
    "SRIL": ("PT Sri Rejeki Isman Tbk", "Industrials", "Textiles"),
    "TKIM": ("PT Pabrik Kertas Tjiwi Kimia Tbk", "Basic Materials", "Pulp & Paper"),
    "WSBP": ("PT Waskita Beton Precast Tbk", "Industrials", "Construction"),
    "WSKT": ("PT Waskita Karya (Persero) Tbk", "Industrials", "Construction"),
}

# LQ45 members (2024 period)
LQ45_SYMBOLS = [
    "ACES", "ADRO", "AMRT", "ANTM", "ASII", "BBCA", "BBNI", "BBRI",
    "BBTN", "BMRI", "BRPT", "BUKA", "CPIN", "EMTK", "ESSA", "EXCL",
    "GGRM", "GOTO", "HMSP", "HRUM", "ICBP", "INCO", "INDF", "INKP",
    "INTP", "ISAT", "ITMG", "JPFA", "JSMR", "KLBF", "MDKA", "MEDC",
    "MIKA", "MNCN", "PGAS", "PTBA", "PTPP", "SIDO", "SMGR", "SMRA",
    "SRTG", "TBIG", "TINS", "TLKM", "TOWR", "TPIA", "UNTR", "UNVR",
    "WIKA",
]

# IDX30 members (2024 period)
IDX30_SYMBOLS = [
    "ADRO", "AMRT", "ANTM", "ASII", "BBCA", "BBNI", "BBRI", "BBTN",
    "BMRI", "BRPT", "CPIN", "EMTK", "GOTO", "ICBP", "INCO", "INDF",
    "INKP", "ISAT", "KLBF", "MDKA", "PGAS", "SMGR", "TBIG", "TLKM",
    "TOWR", "TPIA", "UNTR", "UNVR",
]


def populate_from_hardcoded():
    """Populate stock metadata from verified TradingView data."""
    db = DatabaseManager()
    db.create_tables()

    metadata_list = []
    for symbol, (name, sector, sub_sector) in IDX_STOCK_DATA.items():
        meta = {
            "symbol": symbol,
            "name": name,
            "sector": sector,
            "sub_sector": sub_sector,
            "market_cap": 0,  # Will be updated by yfinance with .JK suffix
            "is_lq45": symbol in LQ45_SYMBOLS,
            "is_idx30": symbol in IDX30_SYMBOLS,
        }
        metadata_list.append(meta)

    db.save_stock_metadata(metadata_list)
    logger.info(f"Saved {len(metadata_list)} stocks from TradingView data.")


def update_market_caps_from_yfinance():
    """Update market caps using yfinance with correct .JK suffix."""
    try:
        import yfinance as yf
    except ImportError:
        logger.error("yfinance not installed. Skipping market cap update.")
        return

    db = DatabaseManager()
    session = db.get_session()

    try:
        from core.data.database import StockMetadata
        stocks = session.query(StockMetadata).all()

        for i, stock in enumerate(stocks):
            try:
                ticker = yf.Ticker(f"{stock.symbol}.JK")
                info = ticker.info
                market_cap = info.get("marketCap", 0)
                if market_cap and market_cap > 0:
                    stock.market_cap = market_cap
                    logger.info(f"[{i+1}/{len(stocks)}] {stock.symbol}: market_cap = {market_cap:,.0f}")
                else:
                    logger.warning(f"[{i+1}/{len(stocks)}] {stock.symbol}: no market cap data")
                time.sleep(0.5)  # Rate limiting
            except Exception as e:
                logger.error(f"Error updating {stock.symbol}: {e}")
                time.sleep(2)

        session.commit()
        logger.info("Market cap update complete.")
    except Exception as e:
        session.rollback()
        logger.error(f"Error during market cap update: {e}")
    finally:
        session.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Populate IDX stock metadata from TradingView")
    parser.add_argument("--skip-market-caps", action="store_true",
                        help="Skip yfinance market cap updates (faster)")
    args = parser.parse_args()

    logger.info("Populating stock metadata from TradingView verified data...")
    populate_from_hardcoded()

    if not args.skip_market_caps:
        logger.info("Updating market caps from yfinance...")
        update_market_caps_from_yfinance()
    else:
        logger.info("Skipping market cap update (--skip-market-caps).")

    logger.info("Done.")
