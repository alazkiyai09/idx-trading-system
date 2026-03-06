"""Pytest configuration and fixtures."""

import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


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
