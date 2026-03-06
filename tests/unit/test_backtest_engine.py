"""
Tests for Backtest Engine Module
"""

import pytest
from datetime import date, datetime, timedelta
from unittest.mock import Mock, patch

from backtest.engine import (
    BacktestConfig,
    BacktestResult,
    BacktestEngine,
    run_backtest,
)
from config.trading_modes import TradingMode
from core.data.models import OHLCV, Trade, Signal, SignalType, SetupType, FlowSignal


def create_test_bar(
    symbol: str = "TEST",
    date_val: date = None,
    open_price: float = 9000.0,
    high: float = 9100.0,
    low: float = 8900.0,
    close: float = 9050.0,
    volume: int = 1000000,
) -> OHLCV:
    """Create a test bar."""
    if date_val is None:
        date_val = date.today()

    return OHLCV(
        symbol=symbol,
        date=date_val,
        open=open_price,
        high=high,
        low=low,
        close=close,
        volume=volume,
    )


def create_test_bars(
    symbol: str = "TEST",
    n_days: int = 100,
    start_date: date = None,
    start_price: float = 9000.0,
) -> list:
    """Create a series of test bars."""
    if start_date is None:
        start_date = date.today() - timedelta(days=n_days + 10)

    bars = []
    price = start_price

    for i in range(n_days):
        current_date = start_date + timedelta(days=i)

        # Skip weekends
        if current_date.weekday() >= 5:
            continue

        # Add some price movement
        change = (i % 10 - 5) * 50
        price = start_price + change

        bars.append(create_test_bar(
            symbol=symbol,
            date_val=current_date,
            open_price=price - 20,
            high=price + 50,
            low=price - 50,
            close=price,
        ))

    return bars


class TestBacktestConfig:
    """Tests for BacktestConfig dataclass."""

    def test_config_creation(self):
        """Test creating a config."""
        config = BacktestConfig(
            start_date=date(2023, 1, 1),
            end_date=date(2023, 12, 31),
        )

        assert config.start_date == date(2023, 1, 1)
        assert config.end_date == date(2023, 12, 31)
        assert config.initial_capital == 100_000_000
        assert config.trading_mode == TradingMode.SWING

    def test_config_custom_values(self):
        """Test config with custom values."""
        config = BacktestConfig(
            start_date=date(2023, 1, 1),
            end_date=date(2023, 12, 31),
            initial_capital=50_000_000,
            trading_mode=TradingMode.POSITION,
            position_sizing="empirical_kelly",
            max_positions=3,
        )

        assert config.initial_capital == 50_000_000
        assert config.trading_mode == TradingMode.POSITION
        assert config.position_sizing == "empirical_kelly"
        assert config.max_positions == 3


class TestBacktestResult:
    """Tests for BacktestResult dataclass."""

    def test_empty_result(self):
        """Test empty result."""
        config = BacktestConfig(
            start_date=date(2023, 1, 1),
            end_date=date(2023, 12, 31),
        )
        result = BacktestResult(config=config)

        assert len(result.trades) == 0
        assert len(result.equity_curve) == 0
        assert result.final_capital == 0.0

    def test_to_dict(self):
        """Test dictionary conversion."""
        config = BacktestConfig(
            start_date=date(2023, 1, 1),
            end_date=date(2023, 12, 31),
        )
        result = BacktestResult(
            config=config,
            final_capital=110_000_000,
            total_return_pct=10.0,
        )

        d = result.to_dict()

        assert d["final_capital"] == 110_000_000
        assert d["total_return_pct"] == 10.0
        assert "config" in d


class TestBacktestEngine:
    """Tests for BacktestEngine class."""

    @pytest.fixture
    def config(self):
        """Create test config."""
        return BacktestConfig(
            start_date=date(2023, 1, 1),
            end_date=date(2023, 12, 31),
            initial_capital=100_000_000,
            trading_mode=TradingMode.SWING,
        )

    @pytest.fixture
    def engine(self, config):
        """Create backtest engine."""
        return BacktestEngine(config)

    def test_initialization(self, config):
        """Test engine initialization."""
        engine = BacktestEngine(config)

        assert engine.config == config
        assert len(engine.trades) == 0
        assert len(engine.signals) == 0

    def test_run_empty_data(self, engine):
        """Test running with empty data."""
        result = engine.run({})

        assert result is not None
        assert len(result.trades) == 0

    def test_run_with_data(self, engine):
        """Test running with price data."""
        bars = create_test_bars("BBCA", n_days=100)
        bars_dict = {"BBCA": bars}

        result = engine.run(bars_dict)

        assert result is not None
        assert result.config == engine.config

    def test_calculate_position_size_fixed(self, config):
        """Test fixed position sizing."""
        config.position_sizing = "fixed"
        engine = BacktestEngine(config)

        signal = Signal(
            symbol="BBCA",
            signal_type=SignalType.BUY,
            composite_score=75,
            technical_score=75,
            flow_score=50,
            entry_price=9000.0,
            timestamp=datetime.now(),
            setup_type=SetupType.MOMENTUM,
            flow_signal=FlowSignal.NEUTRAL,
            stop_loss=8550.0,  # 5% below entry
            target_1=9450.0,   # 5% above
            target_2=9900.0,   # 10% above
            target_3=10350.0,  # 15% above
            risk_pct=0.01,
        )

        bar = create_test_bar(close=9000.0)
        size = engine._calculate_position_size(signal, bar)

        # Should be multiple of 100
        assert size % 100 == 0

    def test_record_equity(self, engine):
        """Test equity recording."""
        engine._record_equity(date.today())

        assert len(engine.equity_curve) == 1
        assert engine.equity_curve[0]["equity"] == engine.config.initial_capital

    def test_build_result(self, engine):
        """Test building result."""
        result = engine._build_result()

        assert result is not None
        assert result.config == engine.config


class TestConvenienceFunction:
    """Tests for convenience function."""

    def test_run_backtest(self):
        """Test run_backtest function."""
        bars = create_test_bars("BBCA", n_days=50)
        bars_dict = {"BBCA": bars}

        result = run_backtest(
            bars_by_symbol=bars_dict,
            start_date=date.today() - timedelta(days=60),
            end_date=date.today(),
            initial_capital=50_000_000,
        )

        assert result is not None
        assert result.config.initial_capital == 50_000_000


class TestEdgeCases:
    """Test edge cases."""

    def test_single_symbol(self):
        """Test with single symbol."""
        bars = create_test_bars("BBCA", n_days=50)
        config = BacktestConfig(
            start_date=date.today() - timedelta(days=60),
            end_date=date.today(),
        )
        engine = BacktestEngine(config)
        result = engine.run({"BBCA": bars})

        assert result is not None

    def test_multiple_symbols(self):
        """Test with multiple symbols."""
        bars_bbca = create_test_bars("BBCA", n_days=50)
        bars_bbri = create_test_bars("BBRI", n_days=50, start_price=5000.0)

        config = BacktestConfig(
            start_date=date.today() - timedelta(days=60),
            end_date=date.today(),
        )
        engine = BacktestEngine(config)
        result = engine.run({
            "BBCA": bars_bbca,
            "BBRI": bars_bbri,
        })

        assert result is not None

    def test_insufficient_data(self):
        """Test with insufficient data."""
        bars = create_test_bars("BBCA", n_days=10)  # Too few bars

        config = BacktestConfig(
            start_date=date.today() - timedelta(days=15),
            end_date=date.today(),
        )
        engine = BacktestEngine(config)
        result = engine.run({"BBCA": bars})

        # Should handle gracefully
        assert result is not None

    def test_date_range_filter(self):
        """Test that dates are filtered correctly."""
        bars = create_test_bars("BBCA", n_days=100)

        config = BacktestConfig(
            start_date=date.today() - timedelta(days=30),
            end_date=date.today() - timedelta(days=10),
        )
        engine = BacktestEngine(config)
        result = engine.run({"BBCA": bars})

        assert result is not None
