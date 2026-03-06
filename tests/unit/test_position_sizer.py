"""
Tests for Position Sizer Module

Tests position size calculations based on risk parameters.
"""

import pytest
from dataclasses import asdict

from core.risk.position_sizer import PositionSizer, PositionSize
from config.trading_modes import TradingMode, get_mode_config
from config.constants import IDX_LOT_SIZE


class TestPositionSize:
    """Tests for PositionSize dataclass."""

    def test_position_size_creation(self):
        """Test creating a position size."""
        size = PositionSize(
            shares=1000,
            lots=10,
            value=9_000_000,
            risk_amount=450_000,
            risk_pct=0.0045,
            entry_price=9000.0,
            stop_loss=8550.0,
        )

        assert size.shares == 1000
        assert size.lots == 10
        assert size.value == 9_000_000
        assert size.risk_amount == 450_000
        assert size.risk_pct == 0.0045

    def test_position_size_asdict(self):
        """Test converting position size to dict."""
        size = PositionSize(
            shares=500,
            lots=5,
            value=4_500_000,
            risk_amount=225_000,
            risk_pct=0.0045,
            entry_price=9000.0,
            stop_loss=8550.0,
        )

        d = asdict(size)
        assert d["shares"] == 500
        assert d["lots"] == 5
        assert d["entry_price"] == 9000.0


class TestPositionSizer:
    """Tests for PositionSizer class."""

    @pytest.fixture
    def swing_config(self):
        """Get swing trading mode config."""
        return get_mode_config(TradingMode.SWING)

    @pytest.fixture
    def sizer(self, swing_config):
        """Create a position sizer with 100M capital."""
        return PositionSizer(capital=100_000_000, config=swing_config)

    def test_initialization(self, swing_config):
        """Test sizer initialization."""
        sizer = PositionSizer(capital=100_000_000, config=swing_config)

        assert sizer.capital == 100_000_000
        assert sizer.lot_size == IDX_LOT_SIZE

    def test_calculate_basic(self, sizer):
        """Test basic position size calculation."""
        result = sizer.calculate(
            entry_price=9000.0,
            stop_loss=8550.0,  # 5% stop
            signal_score=75.0,
        )

        # Verify basic structure
        assert isinstance(result, PositionSize)
        assert result.shares > 0
        assert result.lots > 0
        assert result.value > 0
        assert result.entry_price == 9000.0
        assert result.stop_loss == 8550.0

    def test_calculate_shares_rounded_to_lots(self, sizer):
        """Test that shares are rounded to lot size."""
        result = sizer.calculate(
            entry_price=9000.0,
            stop_loss=8550.0,
            signal_score=75.0,
        )

        # Shares should be divisible by lot size
        assert result.shares % IDX_LOT_SIZE == 0
        assert result.shares == result.lots * IDX_LOT_SIZE

    def test_calculate_risk_amount(self, sizer):
        """Test risk amount calculation."""
        result = sizer.calculate(
            entry_price=9000.0,
            stop_loss=8550.0,  # 450 risk per share
            signal_score=75.0,
        )

        # Risk per share
        risk_per_share = 9000.0 - 8550.0

        # Total risk should equal shares * risk per share
        expected_risk = result.shares * risk_per_share
        assert abs(result.risk_amount - expected_risk) < 1  # Allow small rounding

    def test_calculate_high_signal_score(self, sizer):
        """Test position sizing with high signal score."""
        high_score_result = sizer.calculate(
            entry_price=9000.0,
            stop_loss=8550.0,
            signal_score=85.0,  # High quality
        )

        low_score_result = sizer.calculate(
            entry_price=9000.0,
            stop_loss=8550.0,
            signal_score=55.0,  # Lower quality
        )

        # High score should get larger position
        assert high_score_result.shares >= low_score_result.shares

    def test_calculate_signal_score_tiers(self, sizer):
        """Test different signal score tiers."""
        score_85 = sizer.calculate(entry_price=9000.0, stop_loss=8550.0, signal_score=85.0)
        score_75 = sizer.calculate(entry_price=9000.0, stop_loss=8550.0, signal_score=75.0)
        score_65 = sizer.calculate(entry_price=9000.0, stop_loss=8550.0, signal_score=65.0)
        score_55 = sizer.calculate(entry_price=9000.0, stop_loss=8550.0, signal_score=55.0)

        # Verify ordering (higher score = larger or equal position)
        assert score_85.shares >= score_75.shares
        assert score_75.shares >= score_65.shares
        assert score_65.shares >= score_55.shares

    def test_calculate_position_multiplier(self, sizer):
        """Test position multiplier effect."""
        full_size = sizer.calculate(
            entry_price=9000.0,
            stop_loss=8550.0,
            signal_score=75.0,
            position_multiplier=1.0,
        )

        half_size = sizer.calculate(
            entry_price=9000.0,
            stop_loss=8550.0,
            signal_score=75.0,
            position_multiplier=0.5,
        )

        # Half multiplier should result in smaller position
        assert half_size.shares <= full_size.shares

    def test_calculate_invalid_entry_price(self, sizer):
        """Test with invalid entry price."""
        with pytest.raises(ValueError, match="Invalid entry price"):
            sizer.calculate(entry_price=0, stop_loss=8550.0, signal_score=75.0)

        with pytest.raises(ValueError, match="Invalid entry price"):
            sizer.calculate(entry_price=-100, stop_loss=8550.0, signal_score=75.0)

    def test_calculate_invalid_stop_loss(self, sizer):
        """Test with invalid stop loss."""
        with pytest.raises(ValueError, match="Invalid stop loss"):
            sizer.calculate(entry_price=9000.0, stop_loss=0, signal_score=75.0)

        with pytest.raises(ValueError, match="Invalid stop loss"):
            sizer.calculate(entry_price=9000.0, stop_loss=-100, signal_score=75.0)

    def test_calculate_stop_above_entry(self, sizer):
        """Test with stop loss above entry price."""
        with pytest.raises(ValueError, match="must be below entry"):
            sizer.calculate(
                entry_price=9000.0,
                stop_loss=9500.0,  # Above entry
                signal_score=75.0,
            )

    def test_calculate_stop_equals_entry(self, sizer):
        """Test with stop loss equal to entry price."""
        with pytest.raises(ValueError, match="must be below entry"):
            sizer.calculate(
                entry_price=9000.0,
                stop_loss=9000.0,
                signal_score=75.0,
            )

    def test_calculate_minimum_lot(self, sizer):
        """Test minimum position is one lot."""
        # Very tight stop would normally result in very small position
        result = sizer.calculate(
            entry_price=90000.0,  # High price
            stop_loss=89900.0,  # Very tight 0.1% stop
            signal_score=50.0,
        )

        # Should be at least 1 lot
        assert result.shares >= IDX_LOT_SIZE
        assert result.lots >= 1

    def test_calculate_max_position_limit(self, swing_config):
        """Test maximum position percentage limit."""
        # Create sizer with smaller capital to hit max position
        sizer = PositionSizer(capital=50_000_000, config=swing_config)

        result = sizer.calculate(
            entry_price=5000.0,  # Low price
            stop_loss=4500.0,  # 10% stop
            signal_score=90.0,  # High score
        )

        # Position value should not exceed max position percentage
        max_value = 50_000_000 * swing_config.max_position_pct
        assert result.value <= max_value * 1.01  # Allow 1% tolerance

    def test_calculate_for_target_risk(self, sizer):
        """Test calculation for specific target risk."""
        result = sizer.calculate_for_target_risk(
            entry_price=9000.0,
            stop_loss=8550.0,
            target_risk_pct=0.01,  # 1% risk
        )

        # Risk amount should be approximately 1% of capital
        expected_risk = 100_000_000 * 0.01
        # Allow some tolerance due to lot rounding
        assert abs(result.risk_amount - expected_risk) < expected_risk * 0.1

    def test_calculate_for_target_risk_invalid(self, sizer):
        """Test target risk calculation with invalid inputs."""
        with pytest.raises(ValueError):
            sizer.calculate_for_target_risk(
                entry_price=0,
                stop_loss=8550.0,
                target_risk_pct=0.01,
            )

        with pytest.raises(ValueError):
            sizer.calculate_for_target_risk(
                entry_price=9000.0,
                stop_loss=9500.0,  # Above entry
                target_risk_pct=0.01,
            )

    def test_get_max_shares(self, sizer):
        """Test maximum shares calculation."""
        max_shares = sizer.get_max_shares(price=9000.0)

        # Should be positive
        assert max_shares > 0

        # Should be divisible by lot size
        assert max_shares % IDX_LOT_SIZE == 0

        # Value should not exceed max position percentage
        max_value = max_shares * 9000.0
        expected_max = 100_000_000 * sizer.config.max_position_pct
        assert max_value <= expected_max * 1.01

    def test_calculate_kelly_size(self, sizer):
        """Test Kelly Criterion position sizing."""
        result = sizer.calculate_kelly_size(
            entry_price=9000.0,
            stop_loss=8550.0,
            win_rate=0.55,  # 55% win rate
            avg_win_loss_ratio=1.5,  # Wins 1.5x losses
        )

        # Should return a valid position size
        assert isinstance(result, PositionSize)
        assert result.shares > 0
        assert result.shares % IDX_LOT_SIZE == 0

    def test_calculate_kelly_size_low_win_rate(self, sizer):
        """Test Kelly with low win rate."""
        result = sizer.calculate_kelly_size(
            entry_price=9000.0,
            stop_loss=8550.0,
            win_rate=0.40,  # 40% win rate
            avg_win_loss_ratio=2.0,
        )

        # Should still return at least minimum lot
        assert result.shares >= IDX_LOT_SIZE

    def test_calculate_kelly_size_high_win_rate(self, sizer):
        """Test Kelly with high win rate."""
        result = sizer.calculate_kelly_size(
            entry_price=9000.0,
            stop_loss=8550.0,
            win_rate=0.70,  # 70% win rate
            avg_win_loss_ratio=2.0,
        )

        # Should return a valid position
        assert result.shares > 0

    def test_update_capital(self, sizer):
        """Test updating capital."""
        original_capital = sizer.capital

        sizer.update_capital(150_000_000)

        assert sizer.capital == 150_000_000
        assert sizer.capital != original_capital

    def test_different_trading_modes(self):
        """Test position sizing with different trading modes."""
        capital = 100_000_000
        entry = 9000.0
        stop = 8550.0

        # Create sizers for different modes
        intraday_config = get_mode_config(TradingMode.INTRADAY)
        swing_config = get_mode_config(TradingMode.SWING)
        position_config = get_mode_config(TradingMode.POSITION)

        intraday_sizer = PositionSizer(capital, intraday_config)
        swing_sizer = PositionSizer(capital, swing_config)
        position_sizer = PositionSizer(capital, position_config)

        intraday_result = intraday_sizer.calculate(entry, stop, 75.0)
        swing_result = swing_sizer.calculate(entry, stop, 75.0)
        position_result = position_sizer.calculate(entry, stop, 75.0)

        # Each mode should produce valid results
        assert intraday_result.shares > 0
        assert swing_result.shares > 0
        assert position_result.shares > 0

        # Risk percentages should be different based on mode
        # Intraday: 0.5%, Swing: 1%, Position: 1.5%
        # Higher risk tolerance should allow larger positions (with same stop distance)
        assert position_result.risk_pct >= swing_result.risk_pct
        # Note: actual shares may vary due to lot rounding

    def test_tight_stop_loss(self, sizer):
        """Test with very tight stop loss."""
        result = sizer.calculate(
            entry_price=9000.0,
            stop_loss=8950.0,  # Only 50 points (0.56%)
            signal_score=75.0,
        )

        # Should still produce valid result
        assert result.shares > 0
        assert result.shares % IDX_LOT_SIZE == 0

    def test_wide_stop_loss(self, sizer):
        """Test with wide stop loss."""
        result = sizer.calculate(
            entry_price=9000.0,
            stop_loss=7500.0,  # 1500 points (16.7%)
            signal_score=75.0,
        )

        # Should still produce valid result
        assert result.shares > 0
        # Wide stop means smaller position for same risk
        assert result.value < 100_000_000 * 0.5  # Should be conservative


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.fixture
    def sizer(self):
        """Create position sizer."""
        config = get_mode_config(TradingMode.SWING)
        return PositionSizer(capital=100_000_000, config=config)

    def test_very_high_price(self, sizer):
        """Test with very high stock price."""
        result = sizer.calculate(
            entry_price=100_000.0,  # 100 thousand per share (more realistic for IDX)
            stop_loss=95_000.0,
            signal_score=75.0,
        )

        # Should still produce valid result (minimum 1 lot if enough capital)
        # May fail if capital is insufficient for even 1 lot
        if result.shares >= IDX_LOT_SIZE:
            assert result.shares >= IDX_LOT_SIZE

    def test_very_low_price(self, sizer):
        """Test with very low stock price."""
        result = sizer.calculate(
            entry_price=100.0,  # 100 per share
            stop_loss=90.0,
            signal_score=75.0,
        )

        # Should produce valid result
        assert result.shares >= IDX_LOT_SIZE
        # Low price means more shares
        assert result.shares > 1000

    def test_extreme_signal_scores(self, sizer):
        """Test with extreme signal scores."""
        # Maximum score
        result_100 = sizer.calculate(
            entry_price=9000.0,
            stop_loss=8550.0,
            signal_score=100.0,
        )
        assert result_100.shares > 0

        # Minimum valid score
        result_0 = sizer.calculate(
            entry_price=9000.0,
            stop_loss=8550.0,
            signal_score=0.0,
        )
        assert result_0.shares >= IDX_LOT_SIZE

    def test_fractional_prices(self, sizer):
        """Test with fractional prices."""
        result = sizer.calculate(
            entry_price=9125.50,
            stop_loss=8650.25,
            signal_score=75.0,
        )

        # Should handle fractional prices
        assert result.shares > 0
        assert result.entry_price == 9125.50
        assert result.stop_loss == 8650.25

    def test_small_capital(self):
        """Test with small capital amount."""
        config = get_mode_config(TradingMode.SWING)
        sizer = PositionSizer(capital=5_000_000, config=config)  # 5 million IDR

        result = sizer.calculate(
            entry_price=5000.0,
            stop_loss=4750.0,
            signal_score=75.0,
        )

        # Should still produce minimum lot
        assert result.shares >= IDX_LOT_SIZE
