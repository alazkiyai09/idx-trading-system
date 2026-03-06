"""Tests for cache module."""

import pytest
import time
from datetime import date
from pathlib import Path
import tempfile

from core.data.cache import (
    CacheEntry,
    MemoryCache,
    FileCache,
    DataCacheManager,
    cached,
)


class TestCacheEntry:
    """Tests for CacheEntry class."""

    def test_entry_creation(self):
        """Test creating cache entry."""
        entry = CacheEntry(value="test_value", ttl_seconds=60)

        assert entry.value == "test_value"
        assert not entry.is_expired()

    def test_entry_expiry(self):
        """Test entry expiration."""
        entry = CacheEntry(value="test_value", ttl_seconds=0.01)  # 10ms

        assert not entry.is_expired()
        time.sleep(0.02)
        assert entry.is_expired()


class TestMemoryCache:
    """Tests for MemoryCache class."""

    @pytest.fixture
    def cache(self):
        """Create MemoryCache instance."""
        return MemoryCache(ttl_seconds=60, max_size=10)

    def test_set_and_get(self, cache):
        """Test setting and getting values."""
        cache.set("key1", "value1")
        result = cache.get("key1")

        assert result == "value1"

    def test_get_nonexistent(self, cache):
        """Test getting nonexistent key."""
        result = cache.get("nonexistent")
        assert result is None

    def test_get_expired(self, cache):
        """Test getting expired entry."""
        cache.set("key1", "value1", ttl_seconds=0.01)
        time.sleep(0.02)

        result = cache.get("key1")
        assert result is None

    def test_delete(self, cache):
        """Test deleting entry."""
        cache.set("key1", "value1")
        deleted = cache.delete("key1")

        assert deleted is True
        assert cache.get("key1") is None

    def test_delete_nonexistent(self, cache):
        """Test deleting nonexistent key."""
        deleted = cache.delete("nonexistent")
        assert deleted is False

    def test_clear(self, cache):
        """Test clearing cache."""
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.clear()

        assert cache.get("key1") is None
        assert cache.get("key2") is None

    def test_max_size_eviction(self, cache):
        """Test eviction when max size reached."""
        # Fill cache to max
        for i in range(cache.max_size):
            cache.set(f"key{i}", f"value{i}")

        # Add one more - should evict oldest
        cache.set("new_key", "new_value")

        # First key should be evicted
        assert cache.get("key0") is None
        assert cache.get("new_key") == "new_value"

    def test_cleanup_expired(self, cache):
        """Test cleaning up expired entries."""
        cache.set("key1", "value1", ttl_seconds=0.01)
        cache.set("key2", "value2", ttl_seconds=60)
        time.sleep(0.02)

        removed = cache.cleanup_expired()

        assert removed == 1
        assert cache.get("key1") is None
        assert cache.get("key2") == "value2"

    def test_get_stats(self, cache):
        """Test getting cache statistics."""
        cache.set("key1", "value1", ttl_seconds=0.01)
        cache.set("key2", "value2", ttl_seconds=60)
        time.sleep(0.02)

        stats = cache.get_stats()

        assert stats["total_entries"] == 2
        assert stats["max_size"] == 10


class TestFileCache:
    """Tests for FileCache class."""

    @pytest.fixture
    def file_cache(self, tmp_path):
        """Create FileCache instance with temporary directory."""
        return FileCache(cache_dir=tmp_path / "cache", default_ttl=60)

    def test_set_and_get(self, file_cache):
        """Test setting and getting values."""
        file_cache.set("key1", {"data": "value1"})
        result = file_cache.get("key1")

        assert result == {"data": "value1"}

    def test_get_nonexistent(self, file_cache):
        """Test getting nonexistent key."""
        result = file_cache.get("nonexistent")
        assert result is None

    def test_get_expired(self, file_cache):
        """Test getting expired entry."""
        file_cache.set("key1", "value1", ttl_seconds=0.01)
        time.sleep(0.02)

        result = file_cache.get("key1")
        assert result is None

    def test_delete(self, file_cache):
        """Test deleting entry."""
        file_cache.set("key1", "value1")
        deleted = file_cache.delete("key1")

        assert deleted is True
        assert file_cache.get("key1") is None

    def test_clear(self, file_cache):
        """Test clearing cache."""
        file_cache.set("key1", "value1")
        file_cache.set("key2", "value2")
        count = file_cache.clear()

        assert count == 2
        assert file_cache.get("key1") is None

    def test_cleanup_expired(self, file_cache):
        """Test cleaning up expired entries."""
        file_cache.set("key1", "value1", ttl_seconds=0.01)
        file_cache.set("key2", "value2", ttl_seconds=60)
        time.sleep(0.02)

        removed = file_cache.cleanup_expired()

        assert removed == 1


class TestDataCacheManager:
    """Tests for DataCacheManager class."""

    @pytest.fixture
    def manager(self, tmp_path):
        """Create DataCacheManager instance."""
        manager = DataCacheManager()
        manager.file_cache = FileCache(cache_dir=tmp_path / "cache")
        return manager

    def test_price_data_caching(self, manager):
        """Test caching price data."""
        manager.set_price_data("BBCA", "current", {"price": 9100})
        result = manager.get_price_data("BBCA", "current")

        assert result == {"price": 9100}

    def test_foreign_flow_caching(self, manager):
        """Test caching foreign flow data."""
        manager.set_foreign_flow("BBCA", {"net": 1000000})
        result = manager.get_foreign_flow("BBCA")

        assert result == {"net": 1000000}

    def test_analysis_caching(self, manager):
        """Test caching analysis results."""
        manager.set_analysis("technical", "BBCA_20240115", {"score": 75})
        result = manager.get_analysis("technical", "BBCA_20240115")

        assert result == {"score": 75}

    def test_invalidate_symbol(self, manager):
        """Test invalidating cache for a symbol."""
        manager.set_price_data("BBCA", "current", {"price": 9100})
        manager.set_foreign_flow("BBCA", {"net": 1000000})
        manager.set_price_data("TLKM", "current", {"price": 3500})

        manager.invalidate_symbol("BBCA")

        assert manager.get_price_data("BBCA", "current") is None
        assert manager.get_foreign_flow("BBCA") is None
        assert manager.get_price_data("TLKM", "current") == {"price": 3500}

    def test_clear_all(self, manager):
        """Test clearing all caches."""
        manager.set_price_data("BBCA", "current", {"price": 9100})
        manager.clear_all()

        assert manager.get_price_data("BBCA", "current") is None


class TestCachedDecorator:
    """Tests for @cached decorator."""

    def test_cached_decorator(self):
        """Test cached decorator caches results."""
        call_count = 0

        @cached(ttl_seconds=60, key_prefix="test")
        def expensive_function(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 2

        # First call
        result1 = expensive_function(5)
        assert result1 == 10
        assert call_count == 1

        # Second call - should use cache
        result2 = expensive_function(5)
        assert result2 == 10
        assert call_count == 1  # Not incremented

        # Different argument - should execute
        result3 = expensive_function(10)
        assert result3 == 20
        assert call_count == 2
