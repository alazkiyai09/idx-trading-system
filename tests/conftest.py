"""Pytest configuration and fixtures."""

import asyncio
import pytest
import sys
from pathlib import Path

import fastapi.testclient
import httpx
import starlette.testclient

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class _ASGITestClient:
    """Sync-friendly test client built on httpx ASGITransport.

    Starlette's TestClient currently hangs in this environment, so tests use
    this wrapper instead. Each request runs in a fresh AsyncClient to avoid
    leaking loop state across tests.
    """

    __test__ = False

    def __init__(
        self,
        app,
        base_url: str = "http://testserver",
        headers: dict | None = None,
        follow_redirects: bool = True,
        **_: object,
    ) -> None:
        self.app = app
        self.base_url = base_url
        self.headers = headers or {}
        self.follow_redirects = follow_redirects

    async def _async_request(self, method: str, url: str, **kwargs):
        transport = httpx.ASGITransport(app=self.app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url=self.base_url,
            headers=self.headers,
            follow_redirects=self.follow_redirects,
        ) as client:
            return await client.request(method, url, **kwargs)

    def request(self, method: str, url: str, **kwargs):
        return asyncio.run(self._async_request(method, url, **kwargs))

    def get(self, url: str, **kwargs):
        return self.request("GET", url, **kwargs)

    def post(self, url: str, **kwargs):
        return self.request("POST", url, **kwargs)

    def put(self, url: str, **kwargs):
        return self.request("PUT", url, **kwargs)

    def delete(self, url: str, **kwargs):
        return self.request("DELETE", url, **kwargs)

    def options(self, url: str, **kwargs):
        return self.request("OPTIONS", url, **kwargs)

    def close(self) -> None:
        return None


fastapi.testclient.TestClient = _ASGITestClient
starlette.testclient.TestClient = _ASGITestClient


@pytest.fixture
def sample_ohlcv_data():
    """Sample OHLCV data for testing."""
    from datetime import date
    from core.data.models import OHLCV

    return [
        OHLCV(
            symbol="BBCA",
            date=date(2024, 1, i),
            open=9000.0 + i * 10,
            high=9200.0 + i * 10,
            low=8900.0 + i * 10,
            close=9100.0 + i * 10,
            volume=10000000 + i * 100000,
        )
        for i in range(1, 11)
    ]


@pytest.fixture
def sample_flow_data():
    """Sample foreign flow data for testing."""
    from datetime import date
    from core.data.models import ForeignFlow

    return [
        ForeignFlow(
            symbol="BBCA",
            date=date(2024, 1, i),
            foreign_buy=50000000000.0,
            foreign_sell=30000000000.0 + i * 1000000000,
            foreign_net=20000000000.0 - i * 1000000000,
            total_value=100000000000.0,
            foreign_pct=80.0,
        )
        for i in range(1, 6)
    ]


@pytest.fixture
def temp_db(tmp_path):
    """Create a temporary database for testing."""
    from core.data.database import DatabaseManager

    db_path = tmp_path / "test_trading.db"
    manager = DatabaseManager(f"sqlite:///{db_path}")
    manager.create_tables()
    return manager
