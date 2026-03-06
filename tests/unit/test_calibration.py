"""
Tests for Calibration Surface Module
"""

import pytest
from datetime import date

from research.calibration import (
    CalibrationBuilder,
    CalibrationSurface,
    CalibrationCell,
    build_calibration_surface,
)
from core.data.models import Trade, SetupType, FlowSignal, OrderSide


def create_test_trade(
    signal_score: float,
    holding_days: int,
    return_pct: float,
) -> Trade:
    """Create a test trade."""
    pnl = return_pct * 1000
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


class TestCalibrationCell:
    """Tests for CalibrationCell dataclass."""

    def test_cell_creation(self):
        """Test creating a calibration cell."""
        cell = CalibrationCell(
            score_bin=(70, 80),
            day=3,
            n=50,
            win_rate=0.65,
            avg_return=3.5,
        )

        assert cell.score_bin == (70, 80)
        assert cell.day == 3
        assert cell.n == 50
        assert cell.win_rate == 0.65

    def test_significance(self):
        """Test significance check."""
        significant = CalibrationCell(
            score_bin=(70, 80),
            day=3,
            n=50,
            is_significant=True,
        )

        not_significant = CalibrationCell(
            score_bin=(70, 80),
            day=3,
            n=20,
            is_significant=False,
        )

        assert significant.is_significant is True
        assert not_significant.is_significant is False

    def test_to_dict(self):
        """Test dictionary conversion."""
        cell = CalibrationCell(
            score_bin=(70, 80),
            day=3,
            n=50,
            win_rate=0.65,
            avg_return=3.5,
        )

        d = cell.to_dict()
        assert d["score_range"] == "70-80"
        assert d["day"] == 3
        assert d["win_rate"] == 0.65


class TestCalibrationSurface:
    """Tests for CalibrationSurface class."""

    @pytest.fixture
    def surface(self):
        """Create a calibration surface with sample cells."""
        surface = CalibrationSurface(max_days=7)

        # Add some cells
        for score_low, score_high in [(50, 60), (60, 70), (70, 80), (80, 90), (90, 100)]:
            bin_key = f"{score_low}-{score_high}"
            for day in range(1, 8):
                # Simulate edge decay
                base_win_rate = 0.50 + (score_low - 40) / 200
                decay = (day - 1) * 0.02
                win_rate = max(0.40, base_win_rate - decay)

                cell = CalibrationCell(
                    score_bin=(score_low, score_high),
                    day=day,
                    n=30,
                    win_rate=win_rate,
                    avg_return=win_rate * 5 - (1 - win_rate) * 3,
                    is_significant=True,
                )
                surface.cells[(bin_key, day)] = cell

        return surface

    def test_initialization(self):
        """Test surface initialization."""
        surface = CalibrationSurface()
        assert len(surface.score_bins) == 5
        assert surface.max_days == 7

    def test_get_win_rate(self, surface):
        """Test getting win rate."""
        win_rate = surface.get_win_rate(75, 3)
        assert 0.4 <= win_rate <= 0.8

    def test_get_win_rate_no_data(self):
        """Test getting win rate with no data."""
        surface = CalibrationSurface()
        win_rate = surface.get_win_rate(75, 3)
        assert win_rate == 0.5  # Neutral

    def test_get_optimal_exit_day(self, surface):
        """Test optimal exit day calculation."""
        # Higher scores should have later optimal exits
        exit_90 = surface.get_optimal_exit_day(90)
        exit_60 = surface.get_optimal_exit_day(60)

        # At minimum, these should return valid days
        assert 1 <= exit_90 <= 7
        assert 1 <= exit_60 <= 7

    def test_get_edge_decay(self, surface):
        """Test edge decay calculation."""
        decay = surface.get_edge_decay(75)
        assert len(decay) == 7
        # Generally decreasing (edge decay)
        assert decay[0] >= decay[-1]

    def test_should_exit_by_calibration(self, surface):
        """Test exit decision."""
        # Test with good edge
        should_exit, reason = surface.should_exit_by_calibration(
            score=80,
            days_held=2,
            current_pnl_pct=5.0,
        )

        # Should be a valid response
        assert isinstance(should_exit, bool)
        assert isinstance(reason, str)

    def test_to_matrix(self, surface):
        """Test matrix conversion."""
        matrix, rows, cols = surface.to_matrix()

        assert matrix.shape == (5, 7)
        assert len(rows) == 5
        assert len(cols) == 7

    def test_summary(self, surface):
        """Test summary generation."""
        summary = surface.summary()
        assert "CALIBRATION" in summary
        assert "OPTIMAL EXIT" in summary


class TestCalibrationBuilder:
    """Tests for CalibrationBuilder class."""

    @pytest.fixture
    def builder(self):
        """Create calibration builder."""
        return CalibrationBuilder()

    @pytest.fixture
    def sample_trades(self):
        """Create sample trades for testing."""
        trades = []

        # Create trades across score bins and holding days
        for score in [55, 65, 75, 85, 95]:
            for days in range(1, 8):
                for _ in range(10):  # 10 trades per cell
                    # Higher scores tend to win more
                    win_prob = 0.45 + (score - 50) / 200
                    import random
                    random.seed(42)
                    if random.random() < win_prob:
                        return_pct = random.uniform(2, 10)
                    else:
                        return_pct = random.uniform(-8, -1)

                    trades.append(create_test_trade(score, days, return_pct))

        return trades

    def test_initialization(self):
        """Test builder initialization."""
        builder = CalibrationBuilder(max_days=5)
        assert builder.max_days == 5

    def test_build_empty(self, builder):
        """Test building with no trades."""
        surface = builder.build([])
        assert len(surface.cells) == 0

    def test_build_basic(self, builder, sample_trades):
        """Test basic surface building."""
        surface = builder.build(sample_trades)

        # Should have cells
        assert len(surface.cells) > 0

    def test_build_significance(self, builder):
        """Test significance calculation."""
        # Create trades with varying counts
        trades = []
        for _ in range(35):
            trades.append(create_test_trade(75, 3, 5.0))
        for _ in range(20):
            trades.append(create_test_trade(75, 4, -2.0))

        surface = builder.build(trades)

        # Check significance
        cell_3 = surface.cells.get(("70-80", 3))
        cell_4 = surface.cells.get(("70-80", 4))

        if cell_3:
            assert cell_3.is_significant is True
        if cell_4:
            assert cell_4.is_significant is False

    def test_build_from_returns(self, builder):
        """Test building from return dictionaries."""
        trade_data = [
            {"signal_score": 75, "holding_days": 3, "return_pct": 5.0},
            {"signal_score": 75, "holding_days": 3, "return_pct": -2.0},
        ] * 20

        surface = builder.build_from_returns(trade_data)
        assert len(surface.cells) > 0


class TestConvenienceFunction:
    """Tests for convenience function."""

    def test_build_calibration_surface(self):
        """Test convenience function."""
        trades = []
        for _ in range(40):
            trades.append(create_test_trade(75, 3, 5.0))
            trades.append(create_test_trade(75, 4, -2.0))

        surface = build_calibration_surface(trades)

        assert isinstance(surface, CalibrationSurface)
        assert surface.max_days == 7


class TestEdgeCases:
    """Test edge cases."""

    def test_extreme_scores(self):
        """Test with extreme scores."""
        surface = CalibrationSurface()

        # Score outside bins
        win_rate = surface.get_win_rate(10, 3)
        assert win_rate == 0.5  # No data

        win_rate = surface.get_win_rate(150, 3)
        assert win_rate == 0.5  # No data

    def test_day_outside_range(self):
        """Test with day outside range."""
        surface = CalibrationSurface(max_days=7)
        win_rate = surface.get_win_rate(75, 10)
        assert win_rate == 0.5

    def test_empty_cells(self):
        """Test surface with no cells."""
        surface = CalibrationSurface()

        exit_day = surface.get_optimal_exit_day(75)
        # With no data, win_rate returns 0.5 which is < min_edge (0.52)
        # So optimal exit is 1 (exit immediately)
        assert exit_day == 1

    def test_all_losing_trades(self):
        """Test with all losing trades."""
        builder = CalibrationBuilder()
        trades = [create_test_trade(75, 3, -5.0) for _ in range(40)]

        surface = builder.build(trades)
        cell = surface.cells.get(("70-80", 3))

        if cell:
            assert cell.win_rate == 0.0

    def test_all_winning_trades(self):
        """Test with all winning trades."""
        builder = CalibrationBuilder()
        trades = [create_test_trade(75, 3, 5.0) for _ in range(40)]

        surface = builder.build(trades)
        cell = surface.cells.get(("70-80", 3))

        if cell:
            assert cell.win_rate == 1.0
