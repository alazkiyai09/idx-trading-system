"""
Tests for Signal Generator Module

Tests signal generation, composite scoring, and setup detection.
"""

import pytest
from datetime import datetime, timedelta, date
from typing import List

from core.signals.signal_generator import SignalGenerator, CompositeScorer, CompositeScore
from core.data.models import (
    OHLCV,
    Signal,
    SignalType,
    SetupType,
    FlowSignal,
    TechnicalIndicators,
)
from core.data.foreign_flow import FlowAnalysis
from config.trading_modes import TradingMode, get_mode_config


def create_ohlcv_data(
    num_days: int = 50,
    start_price: float = 9000.0,
    trend: str = "up",
    symbol: str = "TEST",
) -> List[OHLCV]:
    """Create test OHLCV data.

    Args:
        num_days: Number of days.
        start_price: Starting price.
        trend: Price trend.
        symbol: Stock symbol.

    Returns:
        List of OHLCV objects.
    """
    data = []
    price = start_price
    base_date = date(2024, 1, 1)

    for i in range(num_days):
        if trend == "up":
            change = price * 0.01
        elif trend == "down":
            change = -price * 0.01
        else:
            change = 0

        open_price = price
        close_price = price + change
        high_price = max(open_price, close_price) + price * 0.01
        low_price = min(open_price, close_price) - price * 0.01

        data.append(
            OHLCV(
                symbol=symbol,
                date=base_date + timedelta(days=i),
                open=open_price,
                high=high_price,
                low=low_price,
                close=close_price,
                volume=10_000_000 + i * 100_000,
            )
        )
        price = close_price

    return data


def create_flow_analysis(
    signal: FlowSignal = FlowSignal.BUY,
    score: float = 70.0,
    consecutive_days: int = 3,
) -> FlowAnalysis:
    """Create test flow analysis.

    Args:
        signal: Flow signal type.
        score: Signal score.
        consecutive_days: Consecutive buy/sell days.

    Returns:
        FlowAnalysis object.
    """
    return FlowAnalysis(
        symbol="BBCA",
        date=datetime.now().date(),
        today_net=3_000_000_000,
        five_day_net=15_000_000_000,
        twenty_day_net=60_000_000_000,
        consecutive_buy_days=consecutive_days if signal in [FlowSignal.BUY, FlowSignal.STRONG_BUY] else 0,
        consecutive_sell_days=consecutive_days if signal in [FlowSignal.SELL, FlowSignal.STRONG_SELL] else 0,
        signal=signal,
        signal_score=score,
        foreign_pct_of_volume=15.0,
        is_unusual_volume=False,
    )


class TestCompositeScore:
    """Tests for CompositeScore dataclass."""

    def test_composite_score_creation(self):
        """Test creating a composite score."""
        score = CompositeScore(
            total=75.0,
            technical=80.0,
            flow=65.0,
            fundamental=None,
            technical_weight=0.6,
            flow_weight=0.4,
            fundamental_weight=0.0,
        )

        assert score.total == 75.0
        assert score.technical == 80.0
        assert score.flow == 65.0
        assert score.fundamental is None


class TestCompositeScorer:
    """Tests for CompositeScorer class."""

    @pytest.fixture
    def config(self):
        """Get swing trading config."""
        return get_mode_config(TradingMode.SWING)

    @pytest.fixture
    def scorer(self, config):
        """Create composite scorer."""
        return CompositeScorer(config)

    def test_initialization(self, config):
        """Test scorer initialization."""
        scorer = CompositeScorer(config)
        assert scorer.config == config

    def test_calculate_basic(self, scorer):
        """Test basic score calculation."""
        result = scorer.calculate(
            technical_score=80.0,
            flow_score=60.0,
        )

        assert isinstance(result, CompositeScore)
        assert 0 <= result.total <= 100

    def test_calculate_with_fundamental(self, scorer):
        """Test score calculation with fundamental score."""
        result = scorer.calculate(
            technical_score=80.0,
            flow_score=60.0,
            fundamental_score=70.0,
        )

        assert result.fundamental == 70.0
        assert result.fundamental_weight > 0

    def test_calculate_weights_adjustment(self, scorer):
        """Test weight adjustment when no fundamental score."""
        result = scorer.calculate(
            technical_score=80.0,
            flow_score=60.0,
            fundamental_score=None,
        )

        # Weights should sum to 1.0 (excluding fundamental)
        assert result.fundamental_weight == 0.0
        assert abs(result.technical_weight + result.flow_weight - 1.0) < 0.01

    def test_calculate_weighted_total(self, scorer):
        """Test weighted total calculation."""
        result = scorer.calculate(
            technical_score=100.0,
            flow_score=0.0,
        )

        # Should be weighted toward technical (higher weight in swing mode)
        assert result.total > 0
        assert result.total < 100  # Not all technical

    def test_calculate_high_scores(self, scorer):
        """Test with high scores in all categories."""
        result = scorer.calculate(
            technical_score=90.0,
            flow_score=85.0,
            fundamental_score=80.0,
        )

        # Should produce high total
        assert result.total >= 80

    def test_calculate_low_scores(self, scorer):
        """Test with low scores."""
        result = scorer.calculate(
            technical_score=30.0,
            flow_score=40.0,
        )

        # Should produce low total
        assert result.total < 50


class TestSignalGenerator:
    """Tests for SignalGenerator class."""

    @pytest.fixture
    def config(self):
        """Get swing trading config."""
        return get_mode_config(TradingMode.SWING)

    @pytest.fixture
    def generator(self, config):
        """Create signal generator."""
        return SignalGenerator(config)

    @pytest.fixture
    def uptrend_data(self):
        """Create uptrend price data."""
        return create_ohlcv_data(num_days=60, trend="up")

    @pytest.fixture
    def buy_flow(self):
        """Create buy flow analysis."""
        return create_flow_analysis(signal=FlowSignal.BUY, score=70.0)

    def test_initialization(self, config):
        """Test generator initialization."""
        gen = SignalGenerator(config)

        assert gen.config == config
        assert gen.technical_analyzer is not None
        assert gen.composite_scorer is not None

    def test_generate_basic(self, generator, uptrend_data):
        """Test basic signal generation."""
        signal = generator.generate(
            symbol="BBCA",
            ohlcv_data=uptrend_data,
        )

        # May or may not generate signal depending on conditions
        if signal:
            assert isinstance(signal, Signal)
            assert signal.symbol == "BBCA"

    def test_generate_with_flow(self, generator, uptrend_data, buy_flow):
        """Test signal generation with flow analysis."""
        signal = generator.generate(
            symbol="BBCA",
            ohlcv_data=uptrend_data,
            flow_analysis=buy_flow,
        )

        if signal:
            assert signal.flow_score == buy_flow.signal_score
            assert signal.flow_signal == FlowSignal.BUY

    def test_generate_insufficient_data(self, generator):
        """Test with insufficient data."""
        data = create_ohlcv_data(num_days=10)  # Too few

        signal = generator.generate(
            symbol="BBCA",
            ohlcv_data=data,
        )

        assert signal is None

    def test_generate_below_min_score(self, generator):
        """Test when score is below minimum threshold."""
        # Create downtrend data (should produce low scores)
        data = create_ohlcv_data(num_days=60, trend="down")

        signal = generator.generate(
            symbol="BBCA",
            ohlcv_data=data,
        )

        # May not generate signal or generate sell signal
        if signal:
            # If generated, could be sell or hold
            assert signal.signal_type in [SignalType.SELL, SignalType.HOLD, SignalType.WATCH]

    def test_generate_buy_signal(self, generator, uptrend_data, buy_flow):
        """Test generating buy signal."""
        signal = generator.generate(
            symbol="BBCA",
            ohlcv_data=uptrend_data,
            flow_analysis=buy_flow,
        )

        if signal and signal.signal_type == SignalType.BUY:
            assert signal.entry_price > 0
            assert signal.stop_loss < signal.entry_price
            assert signal.target_1 > signal.entry_price

    def test_generate_signal_fields(self, generator, uptrend_data, buy_flow):
        """Test signal has all required fields."""
        signal = generator.generate(
            symbol="BBCA",
            ohlcv_data=uptrend_data,
            flow_analysis=buy_flow,
        )

        if signal:
            assert signal.symbol == "BBCA"
            assert signal.timestamp is not None
            assert signal.signal_type is not None
            assert signal.composite_score >= 0
            assert signal.entry_price > 0
            assert signal.stop_loss > 0
            assert signal.target_1 > 0
            assert signal.target_2 > 0
            assert signal.target_3 > 0
            assert signal.risk_pct >= 0
            assert isinstance(signal.key_factors, list)
            assert isinstance(signal.risks, list)

    def test_generate_setup_type(self, generator, uptrend_data, buy_flow):
        """Test setup type detection."""
        signal = generator.generate(
            symbol="BBCA",
            ohlcv_data=uptrend_data,
            flow_analysis=buy_flow,
        )

        if signal:
            assert isinstance(signal.setup_type, SetupType)

    def test_generate_key_factors(self, generator, uptrend_data, buy_flow):
        """Test key factors are populated."""
        signal = generator.generate(
            symbol="BBCA",
            ohlcv_data=uptrend_data,
            flow_analysis=buy_flow,
        )

        if signal and signal.signal_type == SignalType.BUY:
            # Should have some key factors
            assert len(signal.key_factors) > 0

    def test_generate_risks(self, generator, uptrend_data):
        """Test risk factors are populated."""
        signal = generator.generate(
            symbol="BBCA",
            ohlcv_data=uptrend_data,
        )

        if signal:
            # Risks should be a list (may be empty)
            assert isinstance(signal.risks, list)

    def test_generate_stop_loss_calculation(self, generator, uptrend_data):
        """Test stop loss is calculated correctly."""
        signal = generator.generate(
            symbol="BBCA",
            ohlcv_data=uptrend_data,
        )

        if signal:
            # Stop loss should be below entry
            assert signal.stop_loss < signal.entry_price

            # Risk percentage should be reasonable (1-10%)
            risk_pct = (signal.entry_price - signal.stop_loss) / signal.entry_price
            assert 0.01 <= risk_pct <= 0.10

    def test_generate_targets_calculation(self, generator, uptrend_data):
        """Test targets are calculated correctly."""
        signal = generator.generate(
            symbol="BBCA",
            ohlcv_data=uptrend_data,
        )

        if signal:
            # Targets should be above entry and ascending
            assert signal.target_1 > signal.entry_price
            assert signal.target_2 > signal.target_1
            assert signal.target_3 > signal.target_2

    def test_generate_with_strong_buy_flow(self, generator, uptrend_data):
        """Test with strong buy flow signal."""
        strong_flow = create_flow_analysis(
            signal=FlowSignal.STRONG_BUY,
            score=90.0,
            consecutive_days=5,
        )

        signal = generator.generate(
            symbol="BBCA",
            ohlcv_data=uptrend_data,
            flow_analysis=strong_flow,
        )

        if signal:
            # Should have higher flow score
            assert signal.flow_score >= 80

    def test_generate_with_sell_flow(self, generator, uptrend_data):
        """Test with sell flow signal."""
        sell_flow = create_flow_analysis(
            signal=FlowSignal.SELL,
            score=30.0,
        )

        signal = generator.generate(
            symbol="BBCA",
            ohlcv_data=uptrend_data,
            flow_analysis=sell_flow,
        )

        # May generate sell signal or no signal
        if signal:
            # Flow score should reflect sell signal
            assert signal.flow_score < 50

    def test_generate_empty_data(self, generator):
        """Test with empty OHLCV data."""
        signal = generator.generate(
            symbol="BBCA",
            ohlcv_data=[],
        )

        assert signal is None

    def test_different_trading_modes(self, uptrend_data, buy_flow):
        """Test signal generation with different trading modes."""
        modes = [TradingMode.SWING, TradingMode.POSITION]

        for mode in modes:
            config = get_mode_config(mode)
            gen = SignalGenerator(config)

            signal = gen.generate(
                symbol="BBCA",
                ohlcv_data=uptrend_data,
                flow_analysis=buy_flow,
            )

            # Should generate signal or None
            assert signal is None or isinstance(signal, Signal)


class TestSignalTypes:
    """Test different signal type generation."""

    @pytest.fixture
    def config(self):
        """Get swing trading config."""
        return get_mode_config(TradingMode.SWING)

    @pytest.fixture
    def generator(self, config):
        """Create signal generator."""
        return SignalGenerator(config)

    def test_buy_signal_conditions(self, generator):
        """Test conditions that lead to buy signal."""
        # Strong uptrend with buy flow
        data = create_ohlcv_data(num_days=100, trend="up")
        flow = create_flow_analysis(signal=FlowSignal.BUY, score=75.0)

        signal = generator.generate(
            symbol="BBCA",
            ohlcv_data=data,
            flow_analysis=flow,
        )

        if signal and signal.signal_type == SignalType.BUY:
            assert signal.composite_score >= generator.config.min_score

    def test_watch_signal_low_volume(self, generator):
        """Test watch signal for low volume."""
        # Create data with declining volume
        data = create_ohlcv_data(num_days=60, trend="up")
        # Reduce volume
        for i, bar in enumerate(data):
            bar.volume = 100_000  # Very low volume

        signal = generator.generate(
            symbol="BBCA",
            ohlcv_data=data,
        )

        # May generate WATCH signal or no signal
        if signal:
            assert signal.signal_type in [SignalType.BUY, SignalType.WATCH, SignalType.HOLD]


class TestSetupTypes:
    """Test different setup type detection."""

    @pytest.fixture
    def config(self):
        """Get swing trading config."""
        return get_mode_config(TradingMode.SWING)

    @pytest.fixture
    def generator(self, config):
        """Create signal generator."""
        return SignalGenerator(config)

    def test_momentum_setup(self, generator):
        """Test momentum setup detection."""
        data = create_ohlcv_data(num_days=60, trend="up")

        signal = generator.generate(symbol="BBCA", ohlcv_data=data)

        if signal:
            # Default setup is often momentum
            assert isinstance(signal.setup_type, SetupType)

    def test_foreign_accumulation_setup(self, generator):
        """Test foreign accumulation setup detection."""
        data = create_ohlcv_data(num_days=60, trend="up")
        flow = create_flow_analysis(
            signal=FlowSignal.STRONG_BUY,
            score=85.0,
            consecutive_days=5,
        )

        signal = generator.generate(
            symbol="BBCA",
            ohlcv_data=data,
            flow_analysis=flow,
        )

        if signal and signal.setup_type == SetupType.FOREIGN_ACCUMULATION:
            assert signal.flow_signal == FlowSignal.STRONG_BUY


class TestEdgeCases:
    """Test edge cases."""

    @pytest.fixture
    def generator(self):
        """Create signal generator."""
        config = get_mode_config(TradingMode.SWING)
        return SignalGenerator(config)

    def test_extreme_prices(self, generator):
        """Test with extreme price values."""
        data = create_ohlcv_data(num_days=60, start_price=100000.0)  # Very high

        signal = generator.generate(symbol="HIGH", ohlcv_data=data)

        # Should handle without error
        if signal:
            assert signal.entry_price > 0

    def test_low_prices(self, generator):
        """Test with low price values."""
        data = create_ohlcv_data(num_days=60, start_price=100.0)

        signal = generator.generate(symbol="LOW", ohlcv_data=data)

        # Should handle without error
        if signal:
            assert signal.entry_price > 0

    def test_flat_prices(self, generator):
        """Test with flat price movement."""
        data = create_ohlcv_data(num_days=60, trend="flat")

        signal = generator.generate(symbol="FLAT", ohlcv_data=data)

        # May or may not generate signal in flat market
        if signal:
            assert signal.signal_type in [SignalType.HOLD, SignalType.WATCH, SignalType.BUY]

    def test_single_day_data(self, generator):
        """Test with single day of data."""
        data = create_ohlcv_data(num_days=1)

        signal = generator.generate(symbol="SINGLE", ohlcv_data=data)

        # Should return None (insufficient data)
        assert signal is None

    def test_very_volatile_data(self, generator):
        """Test with very volatile data."""
        base_date = date(2024, 1, 1)
        data = []

        for i in range(60):
            price = 9000 + (i % 10 - 5) * 1000  # Oscillating price
            data.append(
                OHLCV(
                    symbol="VOLATILE",
                    date=base_date + timedelta(days=i),
                    open=price,
                    high=price + 500,
                    low=price - 500,
                    close=price + 200,
                    volume=10_000_000,
                )
            )

        signal = generator.generate(symbol="VOLATILE", ohlcv_data=data)

        # Should handle without crashing
        if signal:
            assert signal.entry_price > 0
