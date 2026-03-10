"""
Integration Tests for Dataclass Validation

Ensures all dataclasses have required fields and proper types.
"""

import pytest
from datetime import datetime, date
from core.data.models import (
    OHLCV, Signal, Position, PortfolioState,
    TechnicalIndicators, Trade
)
from dataclasses import fields


class TestOHLCVValidation:
    """Tests for OHLCV dataclass validation."""

    def test_ohlcv_has_all_required_fields(self):
        """Ensure OHLCV has all required fields"""
        ohlcv = OHLCV(
            timestamp=datetime.now(),
            open=100.0,
            high=105.0,
            low=99.0,
            close=102.0,
            volume=1000000
        )

        # Verify all fields exist
        assert hasattr(ohlcv, 'timestamp')
        assert hasattr(ohlcv, 'open')
        assert hasattr(ohlcv, 'high')
        assert hasattr(ohlcv, 'low')
        assert hasattr(ohlcv, 'close')
        assert hasattr(ohlcv, 'volume')

        # Verify date property for backward compatibility
        assert hasattr(ohlcv, 'date')
        assert ohlcv.date == ohlcv.timestamp.date()

    def test_ohlcv_timestamp_is_datetime(self):
        """Ensure timestamp is datetime type"""
        ohlcv = OHLCV(
            timestamp=datetime.now(),
            open=100.0,
            high=105.0,
            low=99.0,
            close=102.0,
            volume=1000000
        )

        assert isinstance(ohlcv.timestamp, datetime)

    def test_ohlcv_price_types(self):
        """Ensure price fields are floats"""
        ohlcv = OHLCV(
            timestamp=datetime.now(),
            open=100.0,
            high=105.0,
            low=99.0,
            close=102.0,
            volume=1000000
        )

        assert isinstance(ohlcv.open, float)
        assert isinstance(ohlcv.high, float)
        assert isinstance(ohlcv.low, float)
        assert isinstance(ohlcv.close, float)
        assert isinstance(ohlcv.volume, int)


class TestSignalValidation:
    """Tests for Signal dataclass validation."""

    def test_signal_has_all_required_fields(self):
        """Ensure Signal has all required fields"""
        signal = Signal(
            symbol='BBCA',
            timestamp=datetime.now(),
            signal_type='BUY',
            setup_type='Momentum',
            score=75.0,
            entry_price=10000.0,
            stop_loss=9500.0,
            targets=[10500.0, 11000.0, 11500.0]
        )

        # Verify all fields exist
        assert hasattr(signal, 'symbol')
        assert hasattr(signal, 'timestamp')
        assert hasattr(signal, 'signal_type')
        assert hasattr(signal, 'setup_type')
        assert hasattr(signal, 'score')
        assert hasattr(signal, 'entry_price')
        assert hasattr(signal, 'stop_loss')
        assert hasattr(signal, 'targets')
        assert hasattr(signal, 'timeframe')
        assert hasattr(signal, 'risk_reward_ratio')
        assert hasattr(signal, 'key_factors')
        assert hasattr(signal, 'risk_factors')

    def test_signal_score_is_valid(self):
        """Ensure signal score is within valid range"""
        signal = Signal(
            symbol='BBCA',
            timestamp=datetime.now(),
            signal_type='BUY',
            setup_type='Momentum',
            score=75.0,
            entry_price=10000.0,
            stop_loss=9500.0,
            targets=[10500.0, 11000.0, 11500.0]
        )

        assert 0 <= signal.score <= 100


class TestPositionValidation:
    """Tests for Position dataclass validation."""

    def test_position_has_all_required_fields(self):
        """Ensure Position has all required fields"""
        position = Position(
            symbol='BBCA',
            entry_price=10000.0,
            quantity=100,
            entry_date=datetime.now(),
            stop_loss=9500.0,
            targets=[10500.0, 11000.0, 11500.0],
            highest_price=10000.0,
            position_id='POS-001',
            current_price=10200.0,
            unrealized_pnl=20000.0,
            unrealized_pnl_pct=2.0,
            days_held=5,
            setup_type='Momentum',
            signal_score=75.0
        )

        # Verify all fields exist
        assert hasattr(position, 'symbol')
        assert hasattr(position, 'entry_price')
        assert hasattr(position, 'quantity')
        assert hasattr(position, 'entry_date')
        assert hasattr(position, 'stop_loss')
        assert hasattr(position, 'targets')
        assert hasattr(position, 'highest_price')
        assert hasattr(position, 'position_id')
        assert hasattr(position, 'current_price')
        assert hasattr(position, 'unrealized_pnl')
        assert hasattr(position, 'unrealized_pnl_pct')
        assert hasattr(position, 'days_held')
        assert hasattr(position, 'setup_type')
        assert hasattr(position, 'signal_score')
        assert hasattr(position, 'signal')

    def test_position_quantity_is_multiple_of_lot_size(self):
        """Ensure position quantity is valid for IDX"""
        position = Position(
            symbol='BBCA',
            entry_price=10000.0,
            quantity=100,
            entry_date=datetime.now(),
            stop_loss=9500.0,
            targets=[10500.0, 11000.0, 11500.0]
        )

        # IDX lot size is 100 shares
        assert position.quantity % 100 == 0


class TestPortfolioStateValidation:
    """Tests for PortfolioState dataclass validation."""

    def test_portfolio_state_has_all_required_fields(self):
        """Ensure PortfolioState has all required fields"""
        state = PortfolioState(
            timestamp=datetime.now(),
            cash=100_000_000.0,
            total_value=105_000_000.0,
            positions_value=5_000_000.0,
            total_equity=105_000_000.0,
            total_pnl=5_000_000.0,
            total_pnl_pct=5.0,
            daily_pnl=500_000.0,
            daily_pnl_pct=0.5,
            peak_value=106_000_000.0,
            drawdown=1_000_000.0,
            drawdown_pct=0.95,
            open_positions=2,
            positions=[]
        )

        # Verify all fields exist
        assert hasattr(state, 'timestamp')
        assert hasattr(state, 'cash')
        assert hasattr(state, 'total_value')
        assert hasattr(state, 'positions_value')
        assert hasattr(state, 'total_equity')
        assert hasattr(state, 'total_pnl')
        assert hasattr(state, 'total_pnl_pct')
        assert hasattr(state, 'daily_pnl')
        assert hasattr(state, 'daily_pnl_pct')
        assert hasattr(state, 'peak_value')
        assert hasattr(state, 'drawdown')
        assert hasattr(state, 'drawdown_pct')
        assert hasattr(state, 'open_positions')
        assert hasattr(state, 'positions')


class TestContractValidation:
    """Contract tests to ensure dataclasses match their consumers."""

    def test_position_matches_paper_trader_requirements(self):
        """Ensure Position dataclass has all fields PaperTrader uses"""
        # Fields used by PaperTrader
        paper_trader_fields = {
            'position_id', 'symbol', 'entry_price', 'quantity',
            'current_price', 'unrealized_pnl', 'unrealized_pnl_pct',
            'stop_loss', 'target_1', 'target_2', 'target_3',
            'days_held', 'setup_type', 'signal_score', 'signal'
        }

        # Get Position dataclass fields
        from dataclasses import fields
        position_fields = {f.name for f in fields(Position)}

        # Verify all required fields exist
        missing_fields = paper_trader_fields - position_fields
        assert not missing_fields, f"Position missing fields: {missing_fields}"

    def test_portfolio_state_matches_portfolio_manager_requirements(self):
        """Ensure PortfolioState dataclass has all fields PortfolioManager uses"""
        # Fields used by PortfolioManager.get_state()
        portfolio_manager_fields = {
            'timestamp', 'cash', 'total_value', 'positions_value',
            'peak_value', 'drawdown', 'drawdown_pct', 'open_positions',
            'positions', 'total_pnl', 'total_pnl_pct', 'daily_pnl', 'daily_pnl_pct'
        }

        # Get PortfolioState dataclass fields
        from dataclasses import fields
        state_fields = {f.name for f in fields(PortfolioState)}

        # Verify all required fields exist
        missing_fields = portfolio_manager_fields - state_fields
        assert not missing_fields, f"PortfolioState missing fields: {missing_fields}"
