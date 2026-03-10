
"""Helper functions for testing stock detail page."""

import requests
from functools import lru_cache

API_URL = "http://localhost:8000"
REQUEST_TIMEOUT = 10


@lru_cache(maxsize=1)
def get_stock_list_cached():
    """Fetch stock list from API with caching."""
    try:
        resp = requests.get(f"{API_URL}/stocks", timeout=REQUEST_TIMEOUT)
        if resp.status_code == 200:
            data = resp.json()
            stocks = data.get("stocks", data) if isinstance(data, dict) else data
            return [s["symbol"] for s in stocks] if isinstance(stocks, list) else ["BBCA"]
    except requests.exceptions.RequestException:
        pass
    return ["BBCA"]


def get_stock_details_safe(symbol: str) -> dict:
    """Fetch stock details from API with error handling."""
    try:
        resp = requests.get(f"{API_URL}/stocks/{symbol}", timeout=REQUEST_TIMEOUT)
        if resp.status_code == 200:
            return resp.json()
    except requests.exceptions.RequestException:
        pass
    return {}
