"""
Tests for Monte Carlo Simulation Module
"""

import pytest
import numpy as np

from research.monte_carlo import (
    MonteCarloEngine,
    MonteCarloResult,
    EquityPath,
    DrawdownDistribution,
    calculate_equity_curve,
    calculate_max_drawdown,
    run_monte_carlo,
)


class TestEquityPath:
    """Tests for EquityPath dataclass."""

    def test_empty_returns(self):
        """Test with empty returns."""
        path = EquityPath(returns=[])
        assert path.max_drawdown == 0.0
        assert path.final_equity == 1.0
        assert path.equity_curve == []

    def test_positive_returns(self):
        """Test with all positive returns."""
        path = EquityPath(returns=[5.0, 3.0, 4.0])
        assert path.final_equity > 1.0
        assert path.max_drawdown == 0.0

    def test_negative_returns(self):
        """Test with all negative returns."""
        path = EquityPath(returns=[-5.0, -3.0, -4.0])
        assert path.final_equity < 1.0
        assert path.max_drawdown > 0

    def test_mixed_returns(self):
        """Test with mixed returns."""
        path = EquityPath(returns=[5.0, -3.0, 4.0, -2.0])
        assert len(path.equity_curve) == 5  # Initial + 4 returns

    def test_drawdown_calculation(self):
        """Test drawdown calculation."""
        # 10% gain followed by 15% loss
        path = EquityPath(returns=[10.0, -15.0])
        # After 10% gain: 1.10
        # After 15% loss: 1.10 * 0.85 = 0.935
        # DD = (1.10 - 0.935) / 1.10 = 0.15
        assert 0.13 < path.max_drawdown < 0.16


class TestDrawdownDistribution:
    """Tests for DrawdownDistribution dataclass."""

    def test_empty_drawdowns(self):
        """Test with empty drawdowns."""
        dist = DrawdownDistribution(drawdowns=np.array([]))
        assert dist.median == 0.0
        assert dist.p95 == 0.0

    def test_single_drawdown(self):
        """Test with single drawdown."""
        dist = DrawdownDistribution(drawdowns=np.array([0.1]))
        assert dist.median == 0.1

    def test_percentile_calculation(self):
        """Test percentile calculation."""
        drawdowns = np.array([0.05, 0.10, 0.15, 0.20, 0.25])
        dist = DrawdownDistribution(drawdowns=drawdowns)

        assert dist.median == 0.15
        assert dist.p95 > 0.20

    def test_probability_of_exceeding(self):
        """Test probability calculation."""
        drawdowns = np.array([0.05, 0.10, 0.15, 0.20, 0.25])
        dist = DrawdownDistribution(drawdowns=drawdowns)

        # P(DD > 0.15) should be 0.4 (2 out of 5)
        prob = dist.probability_of_dd_exceeding(0.15)
        assert 0.3 < prob < 0.5


class TestMonteCarloEngine:
    """Tests for MonteCarloEngine class."""

    def test_initialization(self):
        """Test engine initialization."""
        engine = MonteCarloEngine(n_simulations=100)
        assert engine.n_simulations == 100

    def test_initialization_with_seed(self):
        """Test engine with seed for reproducibility."""
        engine1 = MonteCarloEngine(n_simulations=100, seed=42)
        engine2 = MonteCarloEngine(n_simulations=100, seed=42)

        returns = [5.0, -2.0, 3.0, -1.0, 4.0]
        result1 = engine1.simulate(returns)
        result2 = engine2.simulate(returns)

        # Results should be identical with same seed
        assert result1.drawdown_distribution.median == pytest.approx(
            result2.drawdown_distribution.median, rel=1e-9
        )

    def test_simulate_empty_returns(self):
        """Test simulation with empty returns."""
        engine = MonteCarloEngine(n_simulations=100)
        result = engine.simulate([])

        assert result.n_simulations == 0

    def test_simulate_basic(self):
        """Test basic simulation."""
        engine = MonteCarloEngine(n_simulations=100)
        returns = [5.0, -2.0, 3.0, -1.0, 4.0, -2.0, 6.0, -3.0, 2.0, -1.0]

        result = engine.simulate(returns)

        assert result.n_simulations == 100
        assert len(result.paths) == 100
        assert result.drawdown_distribution is not None

    def test_simulate_percentiles(self):
        """Test that percentiles are calculated."""
        engine = MonteCarloEngine(n_simulations=1000)
        returns = [5.0, -2.0, 3.0, -1.0, 4.0] * 20

        result = engine.simulate(returns)
        dd = result.drawdown_distribution

        assert dd.median > 0
        assert dd.p95 > dd.median
        assert dd.p99 > dd.p95

    def test_sizing_multiplier(self):
        """Test sizing multiplier calculation."""
        engine = MonteCarloEngine(n_simulations=100, seed=42)
        returns = [5.0, -2.0, 3.0, -1.0, 4.0] * 10

        result = engine.simulate(returns)
        multiplier = result.get_sizing_multiplier(max_acceptable_dd=0.50)

        assert 0 < multiplier <= 1.0

    def test_risk_report(self):
        """Test risk report generation."""
        engine = MonteCarloEngine(n_simulations=100)
        returns = [5.0, -2.0, 3.0, -1.0, 4.0] * 10

        report = engine.get_risk_report(returns)

        assert "MONTE CARLO" in report
        assert "DRAWDOWN" in report


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_calculate_equity_curve(self):
        """Test equity curve calculation."""
        returns = [10.0, -5.0, 10.0]
        curve = calculate_equity_curve(returns)

        assert curve[0] == 1.0
        assert curve[1] == 1.10
        assert abs(curve[2] - 1.045) < 0.01
        assert curve[3] > curve[2]

    def test_calculate_max_drawdown(self):
        """Test max drawdown calculation."""
        # Equity curve: 1.0 -> 1.1 -> 0.95 -> 1.05
        curve = [1.0, 1.1, 0.95, 1.05]
        dd = calculate_max_drawdown(curve)

        # Max DD should be (1.1 - 0.95) / 1.1 = 0.136
        assert 0.13 < dd < 0.15

    def test_calculate_max_drawdown_monotonic_up(self):
        """Test max DD with monotonically increasing equity."""
        curve = [1.0, 1.1, 1.2, 1.3]
        dd = calculate_max_drawdown(curve)
        assert dd == 0.0

    def test_run_monte_carlo(self):
        """Test convenience function."""
        returns = [5.0, -2.0, 3.0, -1.0, 4.0]
        result = run_monte_carlo(returns, n_simulations=50)

        assert result.n_simulations == 50
        assert result.drawdown_distribution is not None


class TestMonteCarloResult:
    """Tests for MonteCarloResult dataclass."""

    def test_backtest_percentile(self):
        """Test backtest percentile calculation."""
        engine = MonteCarloEngine(n_simulations=1000, seed=42)
        returns = [5.0, -2.0, 3.0, -1.0, 4.0] * 10

        result = engine.simulate(returns)

        # Backtest should fall somewhere in distribution
        assert 0 <= result.backtest_percentile <= 100

    def test_safety_margin(self):
        """Test safety margin calculation."""
        engine = MonteCarloEngine(n_simulations=100, seed=42)
        returns = [5.0, -2.0, 3.0, -1.0, 4.0] * 10

        result = engine.simulate(returns)

        # Safety margin should be positive
        assert result.safety_margin > 0


class TestEdgeCases:
    """Test edge cases."""

    def test_very_small_returns(self):
        """Test with very small returns."""
        engine = MonteCarloEngine(n_simulations=50)
        returns = [0.01, -0.01, 0.02, -0.02]

        result = engine.simulate(returns)
        assert result.drawdown_distribution.median < 0.01

    def test_large_returns(self):
        """Test with large returns."""
        engine = MonteCarloEngine(n_simulations=50)
        returns = [50.0, -30.0, 40.0, -25.0]

        result = engine.simulate(returns)
        assert result.drawdown_distribution.p95 > 0.1

    def test_single_return(self):
        """Test with single return."""
        engine = MonteCarloEngine(n_simulations=50)
        result = engine.simulate([5.0])

        assert result.n_simulations == 50
