"""Tests for database operations."""

import pytest
from datetime import datetime, date
from unittest.mock import patch, MagicMock

from core.data.database import (
    DatabaseManager,
    PriceHistory,
    ForeignFlowHistory,
    TradeHistory,
    OpenPositions,
    CalibrationSurface,
)


class TestDatabaseManager:
    """Tests for DatabaseManager class."""

    @pytest.fixture
    def db_manager(self, tmp_path):
        """Create a test database manager with temporary database."""
        db_path = tmp_path / "test_trading.db"
        manager = DatabaseManager(f"sqlite:///{db_path}")
        manager.create_tables()
        return manager

    def test_create_tables(self, db_manager):
        """Test that tables are created."""
        # Tables should exist
        from sqlalchemy import inspect

        inspector = inspect(db_manager.engine)
        tables = inspector.get_table_names()

        assert "price_history" in tables
        assert "foreign_flow_history" in tables
        assert "trade_history" in tables
        assert "open_positions" in tables
        assert "calibration_surface" in tables

    def test_save_and_get_prices(self, db_manager):
        """Test saving and retrieving price data."""
        prices = [
            {
                "symbol": "BBCA",
                "date": date(2024, 1, 15),
                "open": 9000.0,
                "high": 9200.0,
                "low": 8900.0,
                "close": 9100.0,
                "volume": 10000000,
            },
            {
                "symbol": "BBCA",
                "date": date(2024, 1, 16),
                "open": 9100.0,
                "high": 9300.0,
                "low": 9000.0,
                "close": 9200.0,
                "volume": 12000000,
            },
        ]

        db_manager.save_prices(prices)

        retrieved = db_manager.get_prices("BBCA", date(2024, 1, 1))
        assert len(retrieved) == 2
        assert retrieved[0].symbol == "BBCA"
        assert retrieved[0].close == 9100.0

    def test_get_prices_with_end_date(self, db_manager):
        """Test retrieving prices with end date filter."""
        prices = [
            {
                "symbol": "TLKM",
                "date": date(2024, 1, 10),
                "open": 3500.0,
                "high": 3550.0,
                "low": 3450.0,
                "close": 3500.0,
                "volume": 50000000,
            },
            {
                "symbol": "TLKM",
                "date": date(2024, 1, 15),
                "open": 3500.0,
                "high": 3600.0,
                "low": 3480.0,
                "close": 3550.0,
                "volume": 55000000,
            },
        ]

        db_manager.save_prices(prices)

        retrieved = db_manager.get_prices(
            "TLKM", date(2024, 1, 1), date(2024, 1, 12)
        )
        assert len(retrieved) == 1
        assert retrieved[0].date == date(2024, 1, 10)

    def test_get_latest_price(self, db_manager):
        """Test getting latest price for a symbol."""
        prices = [
            {
                "symbol": "ASII",
                "date": date(2024, 1, 10),
                "open": 5000.0,
                "high": 5100.0,
                "low": 4950.0,
                "close": 5050.0,
                "volume": 20000000,
            },
            {
                "symbol": "ASII",
                "date": date(2024, 1, 15),
                "open": 5050.0,
                "high": 5150.0,
                "low": 5000.0,
                "close": 5100.0,
                "volume": 22000000,
            },
        ]

        db_manager.save_prices(prices)

        latest = db_manager.get_latest_price("ASII")
        assert latest is not None
        assert latest.date == date(2024, 1, 15)
        assert latest.close == 5100.0

    def test_get_latest_price_not_found(self, db_manager):
        """Test getting latest price for non-existent symbol."""
        latest = db_manager.get_latest_price("NOTEXIST")
        assert latest is None


class TestForeignFlowOperations:
    """Tests for foreign flow operations."""

    @pytest.fixture
    def db_manager(self, tmp_path):
        """Create a test database manager."""
        db_path = tmp_path / "test_trading.db"
        manager = DatabaseManager(f"sqlite:///{db_path}")
        manager.create_tables()
        return manager

    def test_save_and_get_foreign_flows(self, db_manager):
        """Test saving and retrieving foreign flow data."""
        flows = [
            {
                "symbol": "BBCA",
                "date": date(2024, 1, 15),
                "foreign_buy": 50000000000.0,
                "foreign_sell": 30000000000.0,
                "foreign_net": 20000000000.0,
                "total_value": 100000000000.0,
                "foreign_pct": 80.0,
            },
        ]

        db_manager.save_foreign_flows(flows)

        retrieved = db_manager.get_foreign_flows("BBCA", date(2024, 1, 1))
        assert len(retrieved) == 1
        assert retrieved[0].foreign_net == 20000000000.0


class TestTradeOperations:
    """Tests for trade history operations."""

    @pytest.fixture
    def db_manager(self, tmp_path):
        """Create a test database manager."""
        db_path = tmp_path / "test_trading.db"
        manager = DatabaseManager(f"sqlite:///{db_path}")
        manager.create_tables()
        return manager

    def test_save_trade(self, db_manager):
        """Test saving a completed trade."""
        trade = {
            "trade_id": "TRD-20240115-001",
            "symbol": "BBCA",
            "entry_date": date(2024, 1, 15),
            "entry_price": 9100.0,
            "exit_date": date(2024, 1, 17),
            "exit_price": 9400.0,
            "exit_reason": "target_1",
            "quantity": 1100,
            "side": "BUY",
            "gross_pnl": 330000.0,
            "fees": 44000.0,
            "net_pnl": 286000.0,
            "return_pct": 2.86,
            "holding_days": 2,
            "max_favorable": 350000.0,
            "max_adverse": 50000.0,
            "signal_score": 75.0,
            "setup_type": "PULLBACK_TO_MA",
            "rsi_at_entry": 45.0,
            "flow_signal": "buy",
            "flow_consecutive_days": 3,
        }

        db_manager.save_trade(trade)

        trades = db_manager.get_all_trades()
        assert len(trades) == 1
        assert trades[0].trade_id == "TRD-20240115-001"
        assert trades[0].win is True  # return_pct > 0

    def test_get_trades_for_pattern_matching(self, db_manager):
        """Test getting trades for pattern matching."""
        # Save multiple trades
        for i in range(5):
            trade = {
                "trade_id": f"TRD-{i:04d}",
                "symbol": "BBCA",
                "entry_date": date(2024, 1, 15 + i),
                "entry_price": 9100.0,
                "exit_date": date(2024, 1, 17 + i),
                "exit_price": 9400.0,
                "exit_reason": "target_1",
                "quantity": 1100,
                "side": "BUY",
                "return_pct": 2.0 + i * 0.5,
                "holding_days": 2,
                "signal_score": 70.0 + i,
                "setup_type": "PULLBACK_TO_MA",
                "rsi_at_entry": 45.0,
                "flow_signal": "buy",
                "flow_consecutive_days": 3,
            }
            db_manager.save_trade(trade)

        trades = db_manager.get_trades_for_pattern_matching(
            score_range=(70.0, 75.0),
            min_trades=1,
        )
        assert len(trades) >= 1


class TestPositionOperations:
    """Tests for position operations."""

    @pytest.fixture
    def db_manager(self, tmp_path):
        """Create a test database manager."""
        db_path = tmp_path / "test_trading.db"
        manager = DatabaseManager(f"sqlite:///{db_path}")
        manager.create_tables()
        return manager

    def test_save_and_get_positions(self, db_manager):
        """Test saving and retrieving positions."""
        position = {
            "position_id": "POS-20240115-001",
            "symbol": "BBCA",
            "entry_date": date(2024, 1, 15),
            "entry_price": 9100.0,
            "quantity": 1100,
            "stop_loss": 8800.0,
            "target_1": 9400.0,
            "target_2": 9700.0,
            "target_3": 10000.0,
            "setup_type": "PULLBACK_TO_MA",
            "signal_score": 75.0,
            "highest_price": 9100.0,
        }

        db_manager.save_position(position)

        positions = db_manager.get_open_positions()
        assert len(positions) == 1
        assert positions[0].position_id == "POS-20240115-001"

    def test_update_position(self, db_manager):
        """Test updating a position."""
        position = {
            "position_id": "POS-20240115-001",
            "symbol": "BBCA",
            "entry_date": date(2024, 1, 15),
            "entry_price": 9100.0,
            "quantity": 1100,
            "stop_loss": 8800.0,
            "target_1": 9400.0,
            "target_2": 9700.0,
            "target_3": 10000.0,
            "setup_type": "PULLBACK_TO_MA",
            "signal_score": 75.0,
            "highest_price": 9100.0,
        }

        db_manager.save_position(position)
        db_manager.update_position("POS-20240115-001", {"highest_price": 9300.0})

        positions = db_manager.get_open_positions()
        assert positions[0].highest_price == 9300.0

    def test_close_position(self, db_manager):
        """Test closing a position."""
        position = {
            "position_id": "POS-20240115-001",
            "symbol": "BBCA",
            "entry_date": date(2024, 1, 15),
            "entry_price": 9100.0,
            "quantity": 1100,
            "stop_loss": 8800.0,
            "target_1": 9400.0,
            "target_2": 9700.0,
            "target_3": 10000.0,
            "setup_type": "PULLBACK_TO_MA",
            "signal_score": 75.0,
            "highest_price": 9100.0,
        }

        db_manager.save_position(position)
        closed = db_manager.close_position("POS-20240115-001")

        assert closed is not None
        assert closed.position_id == "POS-20240115-001"

        positions = db_manager.get_open_positions()
        assert len(positions) == 0


class TestCalibrationOperations:
    """Tests for calibration surface operations."""

    @pytest.fixture
    def db_manager(self, tmp_path):
        """Create a test database manager."""
        db_path = tmp_path / "test_trading.db"
        manager = DatabaseManager(f"sqlite:///{db_path}")
        manager.create_tables()
        return manager

    def test_save_and_get_calibration_data(self, db_manager):
        """Test saving and retrieving calibration data."""
        data = [
            {
                "strategy": "swing",
                "score_bin": "70-80",
                "days_held": 3,
                "n_trades": 50,
                "win_rate": 65.0,
                "avg_return": 2.5,
            },
            {
                "strategy": "swing",
                "score_bin": "70-80",
                "days_held": 5,
                "n_trades": 40,
                "win_rate": 60.0,
                "avg_return": 2.0,
            },
        ]

        db_manager.save_calibration_data(data)

        surface = db_manager.get_calibration_surface("swing")
        assert len(surface) == 2
        assert surface[0].win_rate == 65.0
