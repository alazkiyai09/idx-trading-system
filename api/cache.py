"""Simple in-process cache for API read models."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from threading import Lock
from typing import Any, Callable, Dict


@dataclass
class CacheEntry:
    value: Any
    expires_at: datetime


class ApiCache:
    """Thread-safe in-memory cache for low-churn dashboard read models."""

    def __init__(self) -> None:
        self._entries: Dict[str, CacheEntry] = {}
        self._lock = Lock()

    def get_or_set(self, key: str, ttl_seconds: int, builder: Callable[[], Any]) -> Any:
        now = datetime.now(timezone.utc)
        with self._lock:
            entry = self._entries.get(key)
            if entry and entry.expires_at > now:
                return entry.value

        value = builder()
        expires_at = now + timedelta(seconds=ttl_seconds)
        with self._lock:
            self._entries[key] = CacheEntry(value=value, expires_at=expires_at)
        return value

    def invalidate(self, prefix: str | None = None) -> None:
        with self._lock:
            if prefix is None:
                self._entries.clear()
                return
            doomed = [key for key in self._entries if key.startswith(prefix)]
            for key in doomed:
                self._entries.pop(key, None)


api_cache = ApiCache()
