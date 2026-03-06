"""
Monte Carlo Simulation Module

Generates resampled equity paths to understand drawdown distribution
and calculate risk metrics for institutional-grade position sizing.
"""

import logging
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

# Default number of simulations
DEFAULT_SIMULATIONS = 10_000


@dataclass
class EquityPath:
    """A single equity curve path.

    Attributes:
        returns: List of returns in order.
        equity_curve: Cumulative equity values.
        max_drawdown: Maximum drawdown encountered.
        final_equity: Final equity value.
    """

    returns: List[float]
    equity_curve: List[float] = field(default_factory=list)
    max_drawdown: float = 0.0
    final_equity: float = 1.0

    def __post_init__(self):
        """Calculate equity curve and drawdown."""
        if not self.returns:
            return

        # Calculate equity curve (starting at 1.0)
        equity = 1.0
        peak = equity
        max_dd = 0.0

        self.equity_curve = [equity]

        for r in self.returns:
            equity *= (1 + r / 100)  # r is in percentage
            self.equity_curve.append(equity)

            # Track drawdown
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak
            if dd > max_dd:
                max_dd = dd

        self.max_drawdown = max_dd
        self.final_equity = equity


@dataclass
class DrawdownDistribution:
    """Distribution of drawdowns from Monte Carlo simulation.

    Attributes:
        drawdowns: All max drawdowns from simulations.
        median: 50th percentile drawdown.
        p75: 75th percentile drawdown.
        p90: 90th percentile drawdown.
        p95: 95th percentile drawdown (institutional target).
        p99: 99th percentile drawdown.
        mean: Average drawdown.
        std: Standard deviation of drawdowns.
    """

    drawdowns: np.ndarray
    median: float = 0.0
    p75: float = 0.0
    p90: float = 0.0
    p95: float = 0.0
    p99: float = 0.0
    mean: float = 0.0
    std: float = 0.0

    def __post_init__(self):
        """Calculate percentile statistics."""
        if len(self.drawdowns) == 0:
            return

        self.median = float(np.percentile(self.drawdowns, 50))
        self.p75 = float(np.percentile(self.drawdowns, 75))
        self.p90 = float(np.percentile(self.drawdowns, 90))
        self.p95 = float(np.percentile(self.drawdowns, 95))
        self.p99 = float(np.percentile(self.drawdowns, 99))
        self.mean = float(np.mean(self.drawdowns))
        self.std = float(np.std(self.drawdowns))

    def get_percentile(self, percentile: float) -> float:
        """Get drawdown at a specific percentile.

        Args:
            percentile: Percentile (0-100).

        Returns:
            Drawdown at that percentile.
        """
        return float(np.percentile(self.drawdowns, percentile))

    def probability_of_dd_exceeding(self, threshold: float) -> float:
        """Calculate probability of drawdown exceeding threshold.

        Args:
            threshold: Drawdown threshold (as decimal, e.g., 0.20 for 20%).

        Returns:
            Probability (0-1).
        """
        if len(self.drawdowns) == 0:
            return 0.0
        return float(np.mean(self.drawdowns > threshold))


@dataclass
class MonteCarloResult:
    """Complete Monte Carlo simulation result.

    Attributes:
        n_simulations: Number of simulations run.
        original_returns: Original return sequence.
        paths: All generated equity paths.
        drawdown_distribution: Drawdown distribution analysis.
        backtest_max_dd: Original backtest max drawdown.
        backtest_percentile: Where backtest DD falls in distribution.
        safety_margin: Ratio of p95 DD to backtest DD.
        sizing_multiplier: Recommended position sizing adjustment.
    """

    n_simulations: int
    original_returns: List[float]
    paths: List[EquityPath] = field(default_factory=list)
    drawdown_distribution: Optional[DrawdownDistribution] = None
    backtest_max_dd: float = 0.0
    backtest_percentile: float = 50.0
    safety_margin: float = 1.0
    sizing_multiplier: float = 1.0

    def __post_init__(self):
        """Calculate derived statistics."""
        if not self.paths:
            return

        # Calculate original path drawdown
        original_path = EquityPath(self.original_returns)
        self.backtest_max_dd = original_path.max_drawdown

        # Extract drawdowns
        drawdowns = np.array([p.max_drawdown for p in self.paths])
        self.drawdown_distribution = DrawdownDistribution(drawdowns)

        # Calculate where backtest falls in distribution
        if len(drawdowns) > 0:
            self.backtest_percentile = float(
                np.mean(drawdowns <= self.backtest_max_dd) * 100
            )

        # Calculate safety margin
        if self.backtest_max_dd > 0:
            self.safety_margin = self.drawdown_distribution.p95 / self.backtest_max_dd

    def get_sizing_multiplier(self, max_acceptable_dd: float = 0.20) -> float:
        """Calculate position sizing multiplier.

        If 95th percentile DD exceeds max acceptable, reduce size.

        Args:
            max_acceptable_dd: Maximum acceptable drawdown (default 20%).

        Returns:
            Position sizing multiplier.
        """
        if self.drawdown_distribution is None:
            return 1.0

        if self.drawdown_distribution.p95 <= max_acceptable_dd:
            return 1.0

        return max_acceptable_dd / self.drawdown_distribution.p95


class MonteCarloEngine:
    """Monte Carlo simulation engine for drawdown analysis.

    Generates thousands of resampled equity paths to understand
    the distribution of possible outcomes, not just point estimates.

    Example:
        engine = MonteCarloEngine()
        result = engine.simulate(returns=[5.0, -2.0, 3.0, -1.0, 4.0])
        print(f"95th percentile DD: {result.drawdown_distribution.p95:.1%}")
    """

    def __init__(
        self,
        n_simulations: int = DEFAULT_SIMULATIONS,
        seed: Optional[int] = None,
    ):
        """Initialize Monte Carlo engine.

        Args:
            n_simulations: Number of simulations to run.
            seed: Random seed for reproducibility.
        """
        self.n_simulations = n_simulations
        self.seed = seed

        if seed is not None:
            np.random.seed(seed)

        logger.info(f"Monte Carlo engine initialized: {n_simulations} simulations")

    def simulate(
        self,
        returns: List[float],
        n_simulations: Optional[int] = None,
    ) -> MonteCarloResult:
        """Run Monte Carlo simulation on return series.

        Args:
            returns: List of return percentages.
            n_simulations: Override number of simulations.

        Returns:
            MonteCarloResult with full analysis.
        """
        n_sims = n_simulations or self.n_simulations
        returns_array = np.array(returns)

        if len(returns) == 0:
            logger.warning("Empty returns provided to Monte Carlo simulation")
            return MonteCarloResult(
                n_simulations=0,
                original_returns=[],
            )

        logger.info(f"Running {n_sims} Monte Carlo simulations on {len(returns)} returns")

        # Generate all paths
        paths = []
        for _ in range(n_sims):
            # Resample returns with replacement
            shuffled = np.random.permutation(returns_array)
            path = EquityPath(shuffled.tolist())
            paths.append(path)

        result = MonteCarloResult(
            n_simulations=n_sims,
            original_returns=list(returns),
            paths=paths,
        )

        logger.info(
            f"Monte Carlo complete: "
            f"median DD={result.drawdown_distribution.median:.1%}, "
            f"p95 DD={result.drawdown_distribution.p95:.1%}"
        )

        return result

    def simulate_batch(
        self,
        returns_list: List[List[float]],
        n_simulations: Optional[int] = None,
    ) -> List[MonteCarloResult]:
        """Run Monte Carlo on multiple return series.

        Args:
            returns_list: List of return series.
            n_simulations: Override number of simulations.

        Returns:
            List of MonteCarloResults.
        """
        results = []
        for returns in returns_list:
            result = self.simulate(returns, n_simulations)
            results.append(result)
        return results

    def calculate_dd_probability(
        self,
        returns: List[float],
        threshold: float,
        n_simulations: Optional[int] = None,
    ) -> float:
        """Calculate probability of drawdown exceeding threshold.

        Args:
            returns: Return series.
            threshold: DD threshold (e.g., 0.30 for 30%).
            n_simulations: Override number of simulations.

        Returns:
            Probability (0-1).
        """
        result = self.simulate(returns, n_simulations)
        return result.drawdown_distribution.probability_of_dd_exceeding(threshold)

    def get_risk_report(
        self,
        returns: List[float],
        max_acceptable_dd: float = 0.20,
    ) -> str:
        """Generate a risk report from Monte Carlo simulation.

        Args:
            returns: Return series.
            max_acceptable_dd: Maximum acceptable drawdown.

        Returns:
            Formatted risk report string.
        """
        result = self.simulate(returns)
        dd = result.drawdown_distribution

        lines = [
            "=" * 60,
            "MONTE CARLO RISK ANALYSIS",
            "=" * 60,
            f"Simulations: {result.n_simulations:,}",
            f"Trades analyzed: {len(returns)}",
            "",
            "DRAWDOWN DISTRIBUTION:",
            f"  Median (50th):    {dd.median:.1%}",
            f"  75th percentile:  {dd.p75:.1%}",
            f"  90th percentile:  {dd.p90:.1%}",
            f"  95th percentile:  {dd.p95:.1%}  ← INSTITUTIONAL TARGET",
            f"  99th percentile:  {dd.p99:.1%}",
            "",
            "BACKTEST COMPARISON:",
            f"  Backtest max DD:  {result.backtest_max_dd:.1%}",
            f"  Backtest percentile: {result.backtest_percentile:.0f}th",
            f"  Safety margin:    {result.safety_margin:.1f}x",
            "",
        ]

        # Risk assessment
        sizing_mult = result.get_sizing_multiplier(max_acceptable_dd)
        if dd.p95 > max_acceptable_dd:
            lines.extend([
                "⚠️  RISK WARNING:",
                f"  95th percentile DD ({dd.p95:.1%}) exceeds max ({max_acceptable_dd:.1%})",
                f"  Recommended sizing multiplier: {sizing_mult:.2f}",
                f"  Reduce position sizes by {(1 - sizing_mult) * 100:.0f}%",
            ])
        else:
            lines.extend([
                "✓  RISK ASSESSMENT: ACCEPTABLE",
                f"  95th percentile DD within limits",
            ])

        # Probability of extreme events
        prob_30 = dd.probability_of_dd_exceeding(0.30)
        prob_40 = dd.probability_of_dd_exceeding(0.40)

        lines.extend([
            "",
            "EXTREME EVENT PROBABILITIES:",
            f"  P(DD > 30%): {prob_30:.1%}",
            f"  P(DD > 40%): {prob_40:.1%}",
            "",
            "=" * 60,
        ])

        return "\n".join(lines)


def calculate_equity_curve(returns: List[float]) -> List[float]:
    """Calculate equity curve from returns.

    Args:
        returns: List of return percentages.

    Returns:
        Equity curve starting at 1.0.
    """
    equity = 1.0
    curve = [equity]
    for r in returns:
        equity *= (1 + r / 100)
        curve.append(equity)
    return curve


def calculate_max_drawdown(equity_curve: List[float]) -> float:
    """Calculate maximum drawdown from equity curve.

    Args:
        equity_curve: Equity values over time.

    Returns:
        Maximum drawdown as decimal (e.g., 0.15 for 15%).
    """
    if not equity_curve:
        return 0.0

    peak = equity_curve[0]
    max_dd = 0.0

    for value in equity_curve:
        if value > peak:
            peak = value
        dd = (peak - value) / peak if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd

    return max_dd


def run_monte_carlo(
    returns: List[float],
    n_simulations: int = DEFAULT_SIMULATIONS,
) -> MonteCarloResult:
    """Convenience function to run Monte Carlo simulation.

    Args:
        returns: Return percentages.
        n_simulations: Number of simulations.

    Returns:
        MonteCarloResult.
    """
    engine = MonteCarloEngine(n_simulations=n_simulations)
    return engine.simulate(returns)
