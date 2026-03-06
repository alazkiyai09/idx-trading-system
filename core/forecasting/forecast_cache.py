"""Forecast caching module for IDX Trading System.

Provides caching mechanisms for price forecasts to:
- Reduce redundant forecast computations
- Speed up repeated forecast access
- Manage forecast lifecycle with TTL

Features:
- In-memory cache with TTL (time-to-live)
- Thread-safe operations
- Symbol-based cache invalidation
"""

import logging
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cache entry for price forecasts.

    Attributes:
        forecast: PriceForecast instance containing forecast data.
        timestamp: Creation timestamp of the cache entry.
        ttl_minutes: Time to live in minutes.
    """

    forecast: "PriceForecast"
    timestamp: datetime
    ttl_minutes: int

    def is_expired(self) -> bool:
        """Check if cache entry has expired.

        Returns:
            True if entry has expired, False otherwise.
        """
        expiry_time = self.timestamp + timedelta(minutes=self.ttl_minutes)
        return datetime.now() > expiry_time


class ForecastCache:
    """Thread-safe cache for price forecasts with TTL support.

    Provides fast access to frequently used forecasts with
    automatic expiration based on configurable TTL.

    Attributes:
        _cache: Internal dictionary storing cached forecasts.
        _ttl_minutes: Default time to live in minutes.
        _lock: Thread lock for thread-safe operations.
    """

    def __init__(self, ttl_minutes: int = 60) -> None:
        """Initialize forecast cache.

        Args:
            ttl_minutes: Default TTL for cache entries in minutes.
        """
        self._cache: Dict[str, CacheEntry] = {}
        self._ttl_minutes = ttl_minutes
        self._lock = threading.Lock()

    def get(self, symbol: str, horizon: int) -> Optional["PriceForecast"]:
        """Get cached forecast if not expired.

        Args:
            symbol: Stock symbol.
            horizon: Forecast horizon.

        Returns:
            Cached PriceForecast if found and not expired, None otherwise.
        """
        key = self._make_key(symbol, horizon)

        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                logger.debug(f"Cache miss for {symbol} (horizon={horizon})")
                return None

            if entry.is_expired():
                del self._cache[key]
                logger.debug(
                    f"Cache expired for {symbol} (horizon={horizon}), removing entry"
                )
                return None

            logger.debug(f"Cache hit for {symbol} (horizon={horizon})")
            return entry.forecast

    def set(
        self, symbol: str, horizon: int, forecast: "PriceForecast", ttl_minutes: Optional[int] = None
    ) -> None:
        """Store forecast in cache.

        Args:
            symbol: Stock symbol.
            horizon: Forecast horizon.
            forecast: PriceForecast to cache.
            ttl_minutes: Optional TTL override in minutes.
        """
        key = self._make_key(symbol, horizon)
        ttl = ttl_minutes if ttl_minutes is not None else self._ttl_minutes

        with self._lock:
            self._cache[key] = CacheEntry(
                forecast=forecast,
                timestamp=datetime.now(),
                ttl_minutes=ttl,
            )
            logger.debug(f"Cached forecast for {symbol} (horizon={horizon}, ttl={ttl}m)")

    def invalidate(self, symbol: str) -> None:
        """Invalidate all cached forecasts for a symbol.

        Args:
            symbol: Stock symbol to invalidate.
        """
        with self._lock:
            keys_to_delete = [k for k in self._cache.keys() if k.startswith(f"{symbol}:")]
            for key in keys_to_delete:
                del self._cache[key]
            logger.debug(f"Invalidated {len(keys_to_delete)} cache entries for {symbol}")

    def clear(self) -> None:
        """Clear all cached forecasts."""
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            logger.info(f"Cleared {count} cached forecasts")

    def cleanup_expired(self) -> int:
        """Remove all expired entries from cache.

        Returns:
            Number of entries removed.
        """
        with self._lock:
            expired_keys = [
                key for key, entry in self._cache.items() if entry.is_expired()
            ]
            for key in expired_keys:
                del self._cache[key]
            if expired_keys:
                logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
            return len(expired_keys)

    def _make_key(self, symbol: str, horizon: int) -> str:
        """Create cache key from symbol and horizon.

        Args:
            symbol: Stock symbol.
            horizon: Forecast horizon.

        Returns:
            Cache key string.
        """
        return f"{symbol}:{horizon}"

    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics.

        Returns:
            Dictionary with cache stats including total entries,
            expired entries, and valid entries.
        """
        with self._lock:
            expired = sum(1 for e in self._cache.values() if e.is_expired())
            return {
                "total_entries": len(self._cache),
                "expired_entries": expired,
                "valid_entries": len(self._cache) - expired,
                "ttl_minutes": self._ttl_minutes,
            }
