#!/usr/bin/env python3
"""
Cleanup Sentiment Data

Script to enforce the 30-day retention policy for sentiment data.
Deletes records from sentiment_records, sentiment_daily, and sentiment_sector
that are older than 30 days.
"""

import logging
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.data.database import DatabaseManager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def run_cleanup(days=30):
    """Run sentiment data cleanup."""
    db = DatabaseManager()
    
    logger.info(f"Starting sentiment cleanup for data older than {days} days...")
    try:
        results = db.clean_old_sentiment(days=days)
        logger.info(f"Cleanup complete. Deleted:")
        logger.info(f"  - {results['records_deleted']} raw articles")
        logger.info(f"  - {results['daily_deleted']} daily summaries")
        logger.info(f"  - {results['sector_deleted']} sector summaries")
    except Exception as e:
        logger.error(f"Cleanup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    days = 30
    if len(sys.argv) > 1:
        try:
            days = int(sys.argv[1])
        except ValueError:
            logger.error("Days argument must be an integer.")
            sys.exit(1)
            
    run_cleanup(days)
