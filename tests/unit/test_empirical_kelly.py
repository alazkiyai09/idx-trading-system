"""
Tests for Empirical Kelly Module
"""

import pytest
from datetime import date

from core.risk.empirical_kelly import (
    EmpiricalKelly,
    KellyResult,
    calculate_empirical_kelly,
    get_position_size,
)
from core.risk.pattern_matcher import PatternMatchResult, SignalPattern
from core.data.models import Trade, SetupType, FlowSignal, OrderSide


def create_test_trade(
    return_pct: float,
    signal_score: float = 75.0,
    holding_days: int = 3,
) -> Trade:
    """Create a test trade."""
    pnl = return_pct * 1000  # Scale to reasonable P&L
    return Trade(
        trade_id="TEST-001",
        symbol="TEST",
        entry_date=date.today(),
        entry_price=9000.0,
        exit_date=date.today(),
        exit_price=9000.0 * (1 + return_pct / 100),
        exit_reason="test",
        quantity=100,
        side=OrderSide.BUY,
        gross_pnl=pnl,
        fees=100,
        net_pnl=pnl - 100,
        return_pct=return_pct,
        holding_days=holding_days,
        max_favorable=0,
        max_adverse=0,
        signal_score=signal_score,
        setup_type=SetupType.MOMENTUM,
        rsi_at_entry=50.0,
        flow_signal=FlowSignal.NEUTRAL,
        flow_consecutive_days=0,
    )


class TestKellyResult:
    """Tests for KellyResult dataclass."""

    def test_result_creation(self):
        """Test creating a Kelly result."""
        result = KellyResult(
            standard_kelly=0.15,
            empirical_kelly=0.10,
            final_kelly=0.08,
            win_rate=0.60,
            is_valid=True,
        )

        assert result.standard_kelly == 0.15
        assert result.final_kelly == 0.08
        assert result.is_valid is True

    def test_get_position_pct(self):
        """Test position size calculation."""
        result = KellyResult(
            final_kelly=0.10,
            is_valid=True,
        )

        position = result.get_position_pct(100_000_000)
        assert position == 10_000_000

    def test_summary(self):
        """Test summary generation."""
        result = KellyResult(
            standard_kelly=0.15,
            empirical_kelly=0.10,
            final_kelly=0.08,
            win_rate=0.60,
            is_valid=True,
        )

        summary = result.summary()
        assert "KELLY" in summary
        assert "15.0%" in summary  # Standard Kelly


class TestEmpiricalKelly:
    """Tests for EmpiricalKelly class."""

    @pytest.fixture
    def kelly(self):
        """Create Kelly calculator."""
        return EmpiricalKelly()

    def test_initialization(self):
        """Test initialization."""
        kelly = EmpiricalKelly(max_kelly=0.20, min_matches=20)
        assert kelly.max_kelly == 0.20
        assert kelly.min_matches == 20

    def test_calculate_basic(self, kelly):
        """Test basic Kelly calculation."""
        result = kelly.calculate(
            win_rate=0.60,
            avg_win=8.0,
            avg_loss=-4.0,
        )

        assert result.is_valid is True
        assert result.win_rate == 0.60
        assert result.standard_kelly > 0

    def test_calculate_with_cv(self, kelly):
        """Test calculation with CV adjustment."""
        result_no_cv = kelly.calculate(
            win_rate=0.60,
            avg_win=8.0,
            avg_loss=-4.0,
            cv_edge=0.0,  # No uncertainty
        )

        result_high_cv = kelly.calculate(
            win_rate=0.60,
            avg_win=8.0,
            avg_loss=-4.0,
            cv_edge=0.5,  # High uncertainty
        )

        # High CV should reduce Kelly
        assert result_high_cv.final_kelly < result_no_cv.final_kelly

    def test_calculate_with_mc_dd(self, kelly):
        """Test calculation with Monte Carlo DD adjustment."""
        result_ok = kelly.calculate(
            win_rate=0.60,
            avg_win=8.0,
            avg_loss=-4.0,
            mc_p95_dd=0.15,  # Within limits
            max_acceptable_dd=0.20,
        )

        result_high = kelly.calculate(
            win_rate=0.60,
            avg_win=8.0,
            avg_loss=-4.0,
            mc_p95_dd=0.30,  # Exceeds limits
            max_acceptable_dd=0.20,
        )

        # High DD should reduce Kelly
        assert result_high.final_kelly < result_ok.final_kelly

    def test_negative_kelly(self, kelly):
        """Test with negative edge (no advantage)."""
        result = kelly.calculate(
            win_rate=0.40,  # Less than 50%
            avg_win=5.0,
            avg_loss=-5.0,
        )

        assert result.is_valid is False
        assert result.final_kelly == 0

    def test_kelly_cap(self, kelly):
        """Test Kelly capping at maximum."""
        result = kelly.calculate(
            win_rate=0.80,
            avg_win=20.0,
            avg_loss=-5.0,
            cv_edge=0.0,
        )

        # Should be capped at max_kelly
        assert result.final_kelly <= kelly.max_kelly

    def test_high_cv_warning(self, kelly):
        """Test warning for high CV."""
        result = kelly.calculate(
            win_rate=0.60,
            avg_win=8.0,
            avg_loss=-4.0,
            cv_edge=0.9,  # Very high CV
        )

        assert any("High CV" in w for w in result.warnings)

    def test_calculate_from_returns(self, kelly):
        """Test calculation from return list."""
        returns = [5.0, -2.0, 8.0, -3.0, 6.0, -1.0, 4.0, -2.0]

        result = kelly.calculate(
            win_rate=0.625,
            avg_win=5.75,
            avg_loss=-2.0,
            returns=returns,
        )

        assert result.is_valid is True
        assert result.cv_edge > 0

    def test_calculate_from_pattern(self, kelly):
        """Test calculation from pattern match result."""
        trades = [
            create_test_trade(5.0),
            create_test_trade(-2.0),
            create_test_trade(8.0),
            create_test_trade(-3.0),
        ] * 10  # 40 trades

        pattern = SignalPattern(score_range=(70, 80))
        result = PatternMatchResult(pattern=pattern, matches=trades)

        kelly_result = kelly.calculate_from_pattern(result)

        assert kelly_result.is_valid is True

    def test_calculate_from_insufficient_pattern(self, kelly):
        """Test with insufficient pattern matches."""
        trades = [create_test_trade(5.0) for _ in range(10)]  # Only 10

        pattern = SignalPattern(score_range=(70, 80))
        result = PatternMatchResult(pattern=pattern, matches=trades)

        kelly_result = kelly.calculate_from_pattern(result)

        assert kelly_result.is_valid is False

    def test_get_conservative_kelly(self, kelly):
        """Test conservative Kelly value."""
        conservative = kelly.get_conservative_kelly()
        assert conservative == 0.005


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_calculate_empirical_kelly(self):
        """Test convenience function."""
        kelly = calculate_empirical_kelly(
            win_rate=0.60,
            avg_win=8.0,
            avg_loss=-4.0,
            cv_edge=0.3,
        )

        assert 0 < kelly <= 0.25

    def test_get_position_size(self):
        """Test position size calculation."""
        size, result = get_position_size(
            capital=100_000_000,
            win_rate=0.60,
            avg_win=8.0,
            avg_loss=-4.0,
        )

        assert size > 0
        assert result.is_valid is True


class TestEdgeCases:
    """Test edge cases."""

    @pytest.fixture
    def kelly(self):
        """Create Kelly calculator."""
        return EmpiricalKelly()

    def test_zero_avg_loss(self, kelly):
        """Test with zero average loss."""
        result = kelly.calculate(
            win_rate=0.60,
            avg_win=8.0,
            avg_loss=0.0,
        )

        # Should still produce a result
        assert result.is_valid is True

    def test_very_high_win_rate(self, kelly):
        """Test with very high win rate."""
        result = kelly.calculate(
            win_rate=0.95,
            avg_win=5.0,
            avg_loss=-10.0,
        )

        assert result.is_valid is True
        assert result.final_kelly <= kelly.max_kelly

    def test_equal_win_loss(self, kelly):
        """Test with equal win and loss."""
        result = kelly.calculate(
            win_rate=0.55,
            avg_win=5.0,
            avg_loss=-5.0,
        )

        # Should have positive Kelly with >50% win rate
        assert result.is_valid is True

    def test_mc_adjustment_extreme(self, kelly):
        """Test Monte Carlo adjustment with extreme DD."""
        result = kelly.calculate(
            win_rate=0.60,
            avg_win=8.0,
            avg_loss=-4.0,
            mc_p95_dd=0.80,  # 80% DD
            max_acceptable_dd=0.20,
        )

        # Should significantly reduce Kelly
        assert result.mc_multiplier < 0.5
