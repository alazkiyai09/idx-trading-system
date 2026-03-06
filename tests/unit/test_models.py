"""Tests for data models."""

import pytest
from datetime import datetime, date

from core.data.models import (
    OrderSide,
    OrderType,
    OrderStatus,
    SignalType,
    FlowSignal,
    SetupType,
    OHLCV,
    ForeignFlow,
    TechnicalIndicators,
    Signal,
    Position,
    Trade,
    PortfolioState,
    BacktestResult,
)


class TestEnums:
    """Tests for enum types."""

    def test_order_side_values(self):
        """Test OrderSide enum values."""
        assert OrderSide.BUY.value == "BUY"
        assert OrderSide.SELL.value == "SELL"

    def test_signal_type_values(self):
        """Test SignalType enum values."""
        assert SignalType.BUY.value == "BUY"
        assert SignalType.SELL.value == "SELL"
        assert SignalType.HOLD.value == "HOLD"
        assert SignalType.WATCH.value == "WATCH"

    def test_flow_signal_values(self):
        """Test FlowSignal enum values."""
        assert FlowSignal.STRONG_BUY.value == "strong_buy"
        assert FlowSignal.BUY.value == "buy"
        assert FlowSignal.NEUTRAL.value == "neutral"
        assert FlowSignal.SELL.value == "sell"
        assert FlowSignal.STRONG_SELL.value == "strong_sell"

    def test_setup_type_values(self):
        """Test SetupType enum values."""
        assert SetupType.OVERSOLD_BOUNCE.value == "OVERSOLD_BOUNCE"
        assert SetupType.BREAKOUT.value == "BREAKOUT"
        assert SetupType.FOREIGN_ACCUMULATION.value == "FOREIGN_ACCUMULATION"


class TestOHLCV:
    """Tests for OHLCV dataclass."""

    def test_ohlcv_creation(self):
        """Test creating OHLCV instance."""
        ohlcv = OHLCV(
            symbol="BBCA",
            date=date(2024, 1, 15),
            open=9000.0,
            high=9200.0,
            low=8900.0,
            close=9100.0,
            volume=10000000,
        )
        assert ohlcv.symbol == "BBCA"
        assert ohlcv.open == 9000.0
        assert ohlcv.volume == 10000000
        assert ohlcv.value is None

    def test_ohlcv_with_value(self):
        """Test OHLCV with trading value."""
        ohlcv = OHLCV(
            symbol="TLKM",
            date=date(2024, 1, 15),
            open=3500.0,
            high=3550.0,
            low=3450.0,
            close=3500.0,
            volume=50000000,
            value=175000000000.0,
        )
        assert ohlcv.value == 175000000000.0


class TestForeignFlow:
    """Tests for ForeignFlow dataclass."""

    def test_foreign_flow_creation(self):
        """Test creating ForeignFlow instance."""
        flow = ForeignFlow(
            symbol="BBCA",
            date=date(2024, 1, 15),
            foreign_buy=50000000000.0,
            foreign_sell=30000000000.0,
            foreign_net=20000000000.0,
            total_value=100000000000.0,
            foreign_pct=80.0,
        )
        assert flow.symbol == "BBCA"
        assert flow.foreign_net == 20000000000.0
        assert flow.foreign_pct == 80.0


class TestTechnicalIndicators:
    """Tests for TechnicalIndicators dataclass."""

    def test_technical_indicators_creation(self):
        """Test creating TechnicalIndicators instance."""
        indicators = TechnicalIndicators(
            symbol="BBCA",
            date=date(2024, 1, 15),
            ema_20=9000.0,
            ema_50=8800.0,
            sma_200=8500.0,
            rsi=45.0,
            macd=50.0,
            macd_signal=45.0,
            macd_hist=5.0,
            atr=200.0,
            atr_pct=2.2,
            bb_upper=9300.0,
            bb_middle=9000.0,
            bb_lower=8700.0,
            volume_sma_20=10000000.0,
            volume_ratio=1.5,
            trend="uptrend",
        )
        assert indicators.symbol == "BBCA"
        assert indicators.rsi == 45.0
        assert indicators.trend == "uptrend"
        assert indicators.support is None
        assert indicators.resistance is None


class TestSignal:
    """Tests for Signal dataclass."""

    def test_signal_creation(self):
        """Test creating Signal instance."""
        signal = Signal(
            symbol="BBCA",
            timestamp=datetime(2024, 1, 15, 9, 0),
            signal_type=SignalType.BUY,
            composite_score=75.0,
            technical_score=70.0,
            flow_score=80.0,
            fundamental_score=75.0,
            setup_type=SetupType.PULLBACK_TO_MA,
            flow_signal=FlowSignal.BUY,
            entry_price=9100.0,
            stop_loss=8800.0,
            target_1=9400.0,
            target_2=9700.0,
            target_3=10000.0,
            risk_pct=0.01,
            key_factors=["RSI oversold", "Foreign accumulation"],
            risks=["Market volatility"],
        )
        assert signal.symbol == "BBCA"
        assert signal.signal_type == SignalType.BUY
        assert signal.composite_score == 75.0
        assert len(signal.key_factors) == 2
        assert signal.position_size is None

    def test_signal_with_position_details(self):
        """Test Signal with position details."""
        signal = Signal(
            symbol="BBCA",
            timestamp=datetime(2024, 1, 15, 9, 0),
            signal_type=SignalType.BUY,
            composite_score=75.0,
            technical_score=70.0,
            flow_score=80.0,
            setup_type=SetupType.BREAKOUT,
            flow_signal=FlowSignal.STRONG_BUY,
            entry_price=9100.0,
            stop_loss=8800.0,
            target_1=9400.0,
            target_2=9700.0,
            target_3=10000.0,
            risk_pct=0.01,
            position_size=1100,
            position_value=10010000.0,
        )
        assert signal.position_size == 1100
        assert signal.position_value == 10010000.0


class TestPosition:
    """Tests for Position dataclass."""

    def test_position_creation(self):
        """Test creating Position instance."""
        position = Position(
            position_id="POS-20240115-001",
            symbol="BBCA",
            entry_date=date(2024, 1, 15),
            entry_price=9100.0,
            quantity=1100,
            current_price=9200.0,
            unrealized_pnl=110000.0,
            unrealized_pnl_pct=1.1,
            stop_loss=8800.0,
            target_1=9400.0,
            target_2=9700.0,
            target_3=10000.0,
            highest_price=9250.0,
            days_held=2,
            setup_type=SetupType.PULLBACK_TO_MA,
            signal_score=75.0,
        )
        assert position.position_id == "POS-20240115-001"
        assert position.quantity == 1100
        assert position.days_held == 2


class TestTrade:
    """Tests for Trade dataclass."""

    def test_trade_creation(self):
        """Test creating Trade instance."""
        trade = Trade(
            trade_id="TRD-20240115-001",
            symbol="BBCA",
            entry_date=date(2024, 1, 15),
            entry_price=9100.0,
            entry_time=datetime(2024, 1, 15, 9, 5),
            exit_date=date(2024, 1, 17),
            exit_price=9400.0,
            exit_time=datetime(2024, 1, 17, 14, 30),
            exit_reason="target_1",
            quantity=1100,
            side=OrderSide.BUY,
            gross_pnl=330000.0,
            fees=44000.0,
            net_pnl=286000.0,
            return_pct=2.86,
            holding_days=2,
            max_favorable=350000.0,
            max_adverse=50000.0,
            signal_score=75.0,
            setup_type=SetupType.PULLBACK_TO_MA,
            rsi_at_entry=45.0,
            flow_signal=FlowSignal.BUY,
            flow_consecutive_days=3,
        )
        assert trade.trade_id == "TRD-20240115-001"
        assert trade.exit_reason == "target_1"
        assert trade.return_pct == 2.86
        assert trade.holding_days == 2


class TestPortfolioState:
    """Tests for PortfolioState dataclass."""

    def test_portfolio_state_creation(self):
        """Test creating PortfolioState instance."""
        state = PortfolioState(
            timestamp=datetime(2024, 1, 17, 16, 0),
            cash=85000000.0,
            total_value=105000000.0,
            positions_value=20000000.0,
            total_pnl=5000000.0,
            total_pnl_pct=5.0,
            daily_pnl=500000.0,
            daily_pnl_pct=0.5,
            peak_value=106000000.0,
            drawdown=1000000.0,
            drawdown_pct=0.95,
            open_positions=2,
        )
        assert state.cash == 85000000.0
        assert state.total_value == 105000000.0
        assert state.open_positions == 2
        assert len(state.positions) == 0


class TestBacktestResult:
    """Tests for BacktestResult dataclass."""

    def test_backtest_result_creation(self):
        """Test creating BacktestResult instance."""
        result = BacktestResult(
            strategy_name="swing_v1",
            start_date=date(2023, 1, 1),
            end_date=date(2023, 12, 31),
            initial_capital=100000000.0,
            final_capital=120000000.0,
            total_return=20000000.0,
            total_return_pct=20.0,
            annualized_return=20.0,
            max_drawdown=8000000.0,
            sharpe_ratio=1.5,
            sortino_ratio=2.0,
            calmar_ratio=2.5,
            total_trades=50,
            win_rate=60.0,
            profit_factor=1.8,
            avg_win=1000000.0,
            avg_loss=500000.0,
            largest_win=3000000.0,
            largest_loss=1500000.0,
            avg_hold_days=4.5,
        )
        assert result.strategy_name == "swing_v1"
        assert result.total_return_pct == 20.0
        assert result.win_rate == 60.0
        assert result.mc_median_dd is None

    def test_backtest_result_with_monte_carlo(self):
        """Test BacktestResult with Monte Carlo data."""
        result = BacktestResult(
            strategy_name="swing_v1",
            start_date=date(2023, 1, 1),
            end_date=date(2023, 12, 31),
            initial_capital=100000000.0,
            final_capital=120000000.0,
            total_return=20000000.0,
            total_return_pct=20.0,
            annualized_return=20.0,
            max_drawdown=8000000.0,
            sharpe_ratio=1.5,
            sortino_ratio=2.0,
            calmar_ratio=2.5,
            total_trades=50,
            win_rate=60.0,
            profit_factor=1.8,
            avg_win=1000000.0,
            avg_loss=500000.0,
            largest_win=3000000.0,
            largest_loss=1500000.0,
            avg_hold_days=4.5,
            mc_median_dd=5000000.0,
            mc_p95_dd=12000000.0,
            mc_p99_dd=18000000.0,
        )
        assert result.mc_median_dd == 5000000.0
        assert result.mc_p95_dd == 12000000.0
        assert result.mc_p99_dd == 18000000.0
