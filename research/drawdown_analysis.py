"""
Drawdown Analysis Module

Analyzes drawdown risk profiles and compares backtest results
to Monte Carlo distributions for institutional-grade risk assessment.
"""

import logging
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple

from research.monte_carlo import MonteCarloResult, DrawdownDistribution

logger = logging.getLogger(__name__)


@dataclass
class DrawdownProfile:
    """Complete drawdown risk profile.

    Attributes:
        backtest_max_dd: Maximum DD from single backtest.
        mc_median_dd: Median DD from Monte Carlo.
        mc_p75_dd: 75th percentile DD.
        mc_p90_dd: 90th percentile DD.
        mc_p95_dd: 95th percentile DD (institutional target).
        mc_p99_dd: 99th percentile DD.
        backtest_percentile: Where backtest falls in MC distribution.
        safety_margin: Ratio of mc_p95 / backtest_dd.
        probability_of_ruin: P(DD > 40%).
        sizing_recommendation: Recommended position sizing multiplier.
        risk_level: Overall risk assessment.
    """

    backtest_max_dd: float = 0.0
    mc_median_dd: float = 0.0
    mc_p75_dd: float = 0.0
    mc_p90_dd: float = 0.0
    mc_p95_dd: float = 0.0
    mc_p99_dd: float = 0.0
    backtest_percentile: float = 50.0
    safety_margin: float = 1.0
    probability_of_ruin: float = 0.0
    sizing_recommendation: float = 1.0
    risk_level: str = "UNKNOWN"

    def is_acceptable(self, max_acceptable_dd: float = 0.20) -> bool:
        """Check if risk profile is acceptable.

        Args:
            max_acceptable_dd: Maximum acceptable DD at 95th percentile.

        Returns:
            True if acceptable.
        """
        return self.mc_p95_dd <= max_acceptable_dd

    def get_risk_level(self) -> str:
        """Determine risk level.

        Returns:
            Risk level string: LOW, MEDIUM, HIGH, EXTREME.
        """
        if self.mc_p95_dd < 0.10:
            return "LOW"
        elif self.mc_p95_dd < 0.20:
            return "MEDIUM"
        elif self.mc_p95_dd < 0.30:
            return "HIGH"
        else:
            return "EXTREME"

    def summary(self) -> str:
        """Get summary string.

        Returns:
            Formatted summary.
        """
        lines = [
            "=" * 60,
            "DRAWDOWN RISK PROFILE",
            "=" * 60,
            "",
            "BACKTEST RESULT:",
            f"  Max Drawdown: {self.backtest_max_dd:.1%}",
            "",
            "MONTE CARLO DISTRIBUTION:",
            f"  Median (50th):  {self.mc_median_dd:.1%}  (typical case)",
            f"  75th percentile: {self.mc_p75_dd:.1%}  (unlucky)",
            f"  90th percentile: {self.mc_p90_dd:.1%}  (bad luck)",
            f"  95th percentile: {self.mc_p95_dd:.1%}  ← INSTITUTIONAL TARGET",
            f"  99th percentile: {self.mc_p99_dd:.1%}  (disaster)",
            "",
            "RISK ASSESSMENT:",
            f"  Backtest percentile: {self.backtest_percentile:.0f}th",
            f"  Safety margin: {self.safety_margin:.1f}x",
            f"  P(DD > 40%): {self.probability_of_ruin:.1%}",
            f"  Risk level: {self.get_risk_level()}",
            "",
        ]

        if self.sizing_recommendation < 1.0:
            lines.append(f"RECOMMENDATION: Reduce position sizes by {(1 - self.sizing_recommendation) * 100:.0f}%")
        else:
            lines.append("RECOMMENDATION: Current sizing acceptable")

        lines.append("=" * 60)

        return "\n".join(lines)


class DrawdownAnalyzer:
    """Analyzes drawdown risk from Monte Carlo simulations.

    Compares single backtest results to distribution of possible
    outcomes to understand true risk profile.

    Example:
        analyzer = DrawdownAnalyzer()
        profile = analyzer.analyze(mc_result, max_acceptable_dd=0.20)
        print(f"Risk level: {profile.get_risk_level()}")
    """

    def __init__(
        self,
        max_acceptable_dd: float = 0.20,
        ruin_threshold: float = 0.40,
    ):
        """Initialize drawdown analyzer.

        Args:
            max_acceptable_dd: Maximum acceptable DD at 95th percentile.
            ruin_threshold: DD threshold for "ruin" scenario.
        """
        self.max_acceptable_dd = max_acceptable_dd
        self.ruin_threshold = ruin_threshold

    def analyze(
        self,
        mc_result: MonteCarloResult,
    ) -> DrawdownProfile:
        """Analyze drawdown from Monte Carlo result.

        Args:
            mc_result: Monte Carlo simulation result.

        Returns:
            DrawdownProfile with full analysis.
        """
        if mc_result.drawdown_distribution is None:
            return DrawdownProfile()

        dd_dist = mc_result.drawdown_distribution

        # Calculate backtest percentile
        backtest_percentile = mc_result.backtest_percentile

        # Calculate safety margin
        if mc_result.backtest_max_dd > 0:
            safety_margin = dd_dist.p95 / mc_result.backtest_max_dd
        else:
            safety_margin = 1.0

        # Calculate probability of ruin
        prob_ruin = dd_dist.probability_of_dd_exceeding(self.ruin_threshold)

        # Calculate sizing recommendation
        if dd_dist.p95 > self.max_acceptable_dd:
            sizing_rec = self.max_acceptable_dd / dd_dist.p95
        else:
            sizing_rec = 1.0

        profile = DrawdownProfile(
            backtest_max_dd=mc_result.backtest_max_dd,
            mc_median_dd=dd_dist.median,
            mc_p75_dd=dd_dist.p75,
            mc_p90_dd=dd_dist.p90,
            mc_p95_dd=dd_dist.p95,
            mc_p99_dd=dd_dist.p99,
            backtest_percentile=backtest_percentile,
            safety_margin=safety_margin,
            probability_of_ruin=prob_ruin,
            sizing_recommendation=sizing_rec,
            risk_level="",  # Will be set below
        )

        profile.risk_level = profile.get_risk_level()

        return profile

    def analyze_returns(
        self,
        returns: List[float],
        n_simulations: int = 10_000,
    ) -> DrawdownProfile:
        """Analyze drawdown from return series.

        Runs Monte Carlo simulation and analyzes results.

        Args:
            returns: Return percentages.
            n_simulations: Number of MC simulations.

        Returns:
            DrawdownProfile.
        """
        from research.monte_carlo import MonteCarloEngine

        engine = MonteCarloEngine(n_simulations=n_simulations)
        mc_result = engine.simulate(returns)

        return self.analyze(mc_result)

    def compare_to_benchmark(
        self,
        returns: List[float],
        benchmark_dd: float,
        n_simulations: int = 10_000,
    ) -> Tuple[DrawdownProfile, str]:
        """Compare drawdown to benchmark.

        Args:
            returns: Return percentages.
            benchmark_dd: Benchmark max drawdown.
            n_simulations: Number of MC simulations.

        Returns:
            Tuple of (profile, comparison_text).
        """
        profile = self.analyze_returns(returns, n_simulations)

        lines = [
            "DRAWDOWN COMPARISON",
            "",
            f"Your backtest DD: {profile.backtest_max_dd:.1%}",
            f"Benchmark DD: {benchmark_dd:.1%}",
            "",
            f"Monte Carlo 95th percentile: {profile.mc_p95_dd:.1%}",
            "",
        ]

        if profile.mc_p95_dd > benchmark_dd:
            lines.extend([
                f"⚠️  Your 95th percentile DD is {profile.mc_p95_dd - benchmark_dd:.1%} higher",
                "   Consider reducing position sizes",
            ])
        else:
            lines.append("✓ Your risk profile is better than benchmark")

        return profile, "\n".join(lines)

    def get_sizing_recommendation(
        self,
        returns: List[float],
        current_risk_pct: float,
        target_dd: float = 0.20,
    ) -> Tuple[float, str]:
        """Get position sizing recommendation.

        Args:
            returns: Return percentages.
            current_risk_pct: Current risk per trade (e.g., 0.01 for 1%).
            target_dd: Target maximum DD.

        Returns:
            Tuple of (recommended_risk_pct, explanation).
        """
        profile = self.analyze_returns(returns)

        if profile.mc_p95_dd <= target_dd:
            return current_risk_pct, "Current sizing acceptable"

        # Calculate adjustment
        multiplier = target_dd / profile.mc_p95_dd
        recommended = current_risk_pct * multiplier

        explanation = (
            f"Reduce risk from {current_risk_pct:.1%} to {recommended:.1%} "
            f"(multiply by {multiplier:.2f}) to target {target_dd:.0%} max DD"
        )

        return recommended, explanation


def analyze_drawdown(
    returns: List[float],
    max_acceptable_dd: float = 0.20,
    n_simulations: int = 10_000,
) -> DrawdownProfile:
    """Convenience function to analyze drawdown.

    Args:
        returns: Return percentages.
        max_acceptable_dd: Maximum acceptable DD.
        n_simulations: Number of MC simulations.

    Returns:
        DrawdownProfile.
    """
    analyzer = DrawdownAnalyzer(max_acceptable_dd=max_acceptable_dd)
    return analyzer.analyze_returns(returns, n_simulations)


def get_sizing_adjustment(
    returns: List[float],
    target_dd: float = 0.20,
) -> float:
    """Get position sizing adjustment factor.

    Args:
        returns: Return percentages.
        target_dd: Target maximum DD.

    Returns:
        Multiplier for position sizing.
    """
    profile = analyze_drawdown(returns, target_dd)
    return profile.sizing_recommendation
