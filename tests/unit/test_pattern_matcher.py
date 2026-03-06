"""
Tests for Pattern Matcher Module
"""

import pytest
from datetime import date

from core.risk.pattern_matcher import (
    PatternMatcher,
    SignalPattern,
    PatternMatchResult,
)
from core.data.models import Trade, SetupType, FlowSignal, OrderSide


def create_test_trade(
    signal_score: float = 75.0,
    rsi: float = 50.0,
    flow_signal: FlowSignal = FlowSignal.NEUTRAL,
    flow_days: int = 0,
    setup_type: SetupType = SetupType.MOMENTUM,
    return_pct: float = 5.0,
) -> Trade:
    """Create a test trade."""
    pnl = return_pct * 1000
    return Trade(
        trade_id=f"TEST-{signal_score}-{rsi}",
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
        holding_days=3,
        max_favorable=0,
        max_adverse=0,
        signal_score=signal_score,
        setup_type=setup_type,
        rsi_at_entry=rsi,
        flow_signal=flow_signal,
        flow_consecutive_days=flow_days,
    )


class TestSignalPattern:
    """Tests for SignalPattern dataclass."""

    def test_pattern_creation(self):
        """Test creating a pattern."""
        pattern = SignalPattern(
            score_range=(70, 80),
            rsi_range=(30, 50),
            flow_signal="buy",
        )

        assert pattern.score_range == (70, 80)
        assert pattern.rsi_range == (30, 50)

    def test_matches_score(self):
        """Test score matching."""
        pattern = SignalPattern(score_range=(70, 80))
        trade = create_test_trade(signal_score=75.0)

        assert pattern.matches(trade) is True

    def test_matches_score_outside_range(self):
        """Test score outside range."""
        pattern = SignalPattern(score_range=(70, 80))
        trade = create_test_trade(signal_score=65.0)

        assert pattern.matches(trade) is False

    def test_matches_flow_signal(self):
        """Test flow signal matching."""
        pattern = SignalPattern(flow_signal="buy")
        trade = create_test_trade(flow_signal=FlowSignal.BUY)

        assert pattern.matches(trade) is True

    def test_matches_setup_type(self):
        """Test setup type matching."""
        pattern = SignalPattern(setup_type="MOMENTUM")
        trade = create_test_trade(setup_type=SetupType.MOMENTUM)

        assert pattern.matches(trade) is True

    def test_matches_multiple_criteria(self):
        """Test matching multiple criteria."""
        pattern = SignalPattern(
            score_range=(70, 80),
            rsi_range=(30, 50),
            flow_signal="buy",
        )
        trade = create_test_trade(
            signal_score=75.0,
            rsi=40.0,
            flow_signal=FlowSignal.BUY,
        )

        assert pattern.matches(trade) is True


class TestPatternMatchResult:
    """Tests for PatternMatchResult dataclass."""

    def test_empty_result(self):
        """Test empty result."""
        pattern = SignalPattern(score_range=(70, 80))
        result = PatternMatchResult(pattern=pattern)

        assert result.count == 0
        assert result.win_rate == 0.0
        assert result.is_significant is False

    def test_with_matches(self):
        """Test with matches."""
        pattern = SignalPattern(score_range=(0, 100))
        trades = [
            create_test_trade(return_pct=5.0),
            create_test_trade(return_pct=-2.0),
            create_test_trade(return_pct=8.0),
        ]

        result = PatternMatchResult(pattern=pattern, matches=trades)

        assert result.count == 3
        assert result.win_count == 2
        assert result.loss_count == 1
        assert abs(result.win_rate - 2/3) < 0.01

    def test_significance(self):
        """Test significance check."""
        pattern = SignalPattern()

        # 30+ trades is significant
        trades = [create_test_trade() for _ in range(35)]
        result = PatternMatchResult(pattern=pattern, matches=trades)
        assert result.is_significant is True

        # Less than 30 is not
        trades = [create_test_trade() for _ in range(25)]
        result = PatternMatchResult(pattern=pattern, matches=trades)
        assert result.is_significant is False

    def test_return_stats(self):
        """Test return statistics."""
        pattern = SignalPattern()
        trades = [
            create_test_trade(return_pct=10.0),
            create_test_trade(return_pct=5.0),
            create_test_trade(return_pct=-3.0),
        ]

        result = PatternMatchResult(pattern=pattern, matches=trades)

        assert result.avg_return == (10 + 5 - 3) / 3
        assert result.avg_win == 7.5  # (10 + 5) / 2
        assert result.avg_loss == -3.0


class TestPatternMatcher:
    """Tests for PatternMatcher class."""

    @pytest.fixture
    def matcher(self):
        """Create a pattern matcher with sample trades."""
        trades = []

        # Create trades with various characteristics
        for score in [55, 65, 75, 85, 95]:
            for _ in range(10):
                trades.append(create_test_trade(signal_score=score))

        return PatternMatcher(trades=trades)

    def test_initialization(self):
        """Test matcher initialization."""
        matcher = PatternMatcher()
        assert len(matcher.trades) == 0

    def test_initialization_with_trades(self, matcher):
        """Test initialization with trades."""
        assert len(matcher.trades) == 50

    def test_add_trade(self):
        """Test adding a trade."""
        matcher = PatternMatcher()
        trade = create_test_trade()

        matcher.add_trade(trade)
        assert len(matcher.trades) == 1

    def test_match(self, matcher):
        """Test pattern matching."""
        pattern = SignalPattern(score_range=(70, 80))
        result = matcher.match(pattern)

        assert result.count == 10  # Scores 75 only
        assert result.is_significant is False  # Only 10 trades

    def test_match_by_score(self, matcher):
        """Test matching by score."""
        result = matcher.match_by_score(75.0, tolerance=5.0)

        # Should match scores 70-80
        assert result.count == 10

    def test_match_by_flow(self):
        """Test matching by flow signal."""
        trades = [
            create_test_trade(flow_signal=FlowSignal.BUY),
            create_test_trade(flow_signal=FlowSignal.BUY),
            create_test_trade(flow_signal=FlowSignal.SELL),
        ]
        matcher = PatternMatcher(trades=trades)

        result = matcher.match_by_flow("buy")
        assert result.count == 2

    def test_match_by_setup(self):
        """Test matching by setup type."""
        trades = [
            create_test_trade(setup_type=SetupType.MOMENTUM),
            create_test_trade(setup_type=SetupType.BREAKOUT),
            create_test_trade(setup_type=SetupType.MOMENTUM),
        ]
        matcher = PatternMatcher(trades=trades)

        result = matcher.match_by_setup("MOMENTUM")
        assert result.count == 2

    def test_create_pattern_from_signal(self, matcher):
        """Test pattern creation from signal."""
        pattern = matcher.create_pattern_from_signal(
            signal_score=75.0,
            flow_signal="buy",
            setup_type="MOMENTUM",
            rsi=40.0,
        )

        assert pattern.score_range == (70.0, 80.0)
        assert pattern.flow_signal == "buy"
        assert pattern.setup_type == "MOMENTUM"

    def test_get_best_matches(self, matcher):
        """Test finding best matches."""
        result = matcher.get_best_matches(
            signal_score=75.0,
            min_matches=5,
        )

        assert result is not None
        assert result.count >= 5

    def test_get_best_matches_not_enough(self):
        """Test when not enough matches exist."""
        matcher = PatternMatcher(trades=[create_test_trade()])
        result = matcher.get_best_matches(
            signal_score=75.0,
            min_matches=30,
        )

        # Should return best effort
        assert result is not None
        assert result.count < 30

    def test_get_pattern_stats(self, matcher):
        """Test pattern statistics."""
        stats = matcher.get_pattern_stats()

        assert stats["total_trades"] == 50
        assert "score_bins" in stats

    def test_validate_pattern_data(self, matcher):
        """Test pattern data validation."""
        warnings = matcher.validate_pattern_data()

        # Should have warnings for bins with less than 30 trades
        assert isinstance(warnings, list)


class TestEdgeCases:
    """Test edge cases."""

    def test_empty_matcher(self):
        """Test with empty matcher."""
        matcher = PatternMatcher()
        pattern = SignalPattern(score_range=(70, 80))

        result = matcher.match(pattern)
        assert result.count == 0

    def test_no_matching_trades(self):
        """Test pattern with no matches."""
        trades = [create_test_trade(signal_score=75.0) for _ in range(10)]
        matcher = PatternMatcher(trades=trades)
        pattern = SignalPattern(score_range=(0, 10))  # Very low scores
        result = matcher.match(pattern)

        assert result.count == 0

    def test_all_trades_match(self):
        """Test pattern matching all trades."""
        trades = [create_test_trade() for _ in range(10)]
        matcher = PatternMatcher(trades=trades)

        pattern = SignalPattern(score_range=(0, 100))  # All scores
        result = matcher.match(pattern)

        assert result.count == 10

    def test_boundary_scores(self):
        """Test boundary score matching."""
        trades = [
            create_test_trade(signal_score=70.0),
            create_test_trade(signal_score=69.9),
            create_test_trade(signal_score=80.0),
            create_test_trade(signal_score=80.1),
        ]
        matcher = PatternMatcher(trades=trades)

        pattern = SignalPattern(score_range=(70, 80))
        result = matcher.match(pattern)

        # Should match 70.0 and 80.0
        assert result.count == 2

    def test_negative_returns(self):
        """Test with all negative returns."""
        trades = [create_test_trade(return_pct=-5.0) for _ in range(10)]
        matcher = PatternMatcher(trades=trades)

        pattern = SignalPattern(score_range=(0, 100))
        result = matcher.match(pattern)

        assert result.win_count == 0
        assert result.loss_count == 10
        assert result.win_rate == 0.0

    def test_profit_factor_infinite(self):
        """Test infinite profit factor."""
        trades = [create_test_trade(return_pct=5.0) for _ in range(10)]
        matcher = PatternMatcher(trades=trades)

        pattern = SignalPattern(score_range=(0, 100))
        result = matcher.match(pattern)

        # All wins = infinite profit factor
        assert result.profit_factor == float('inf')
