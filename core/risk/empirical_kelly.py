"""
Empirical Kelly Criterion Module

Calculates uncertainty-adjusted position sizing using the Kelly Criterion
with adjustments for edge uncertainty and Monte Carlo drawdown analysis.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

from core.risk.pattern_matcher import PatternMatchResult
from research.monte_carlo import MonteCarloResult
from research.return_distribution import ReturnDistribution

logger = logging.getLogger(__name__)


@dataclass
class KellyResult:
    """Result of Kelly Criterion calculation.

    Attributes:
        standard_kelly: Standard Kelly fraction (before adjustments).
        uncertainty_haircut: Haircut from edge uncertainty (1 - CV).
        empirical_kelly: Kelly after uncertainty adjustment.
        mc_multiplier: Monte Carlo adjustment factor.
        final_kelly: Final recommended Kelly fraction.
        win_rate: Historical win rate.
        avg_win: Average winning return.
        avg_loss: Average losing return.
        cv_edge: Coefficient of variation of edge.
        p95_dd: 95th percentile drawdown from Monte Carlo.
        is_valid: Whether calculation is valid.
        warnings: List of warning messages.
    """

    standard_kelly: float = 0.0
    uncertainty_haircut: float = 1.0
    empirical_kelly: float = 0.0
    mc_multiplier: float = 1.0
    final_kelly: float = 0.0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    cv_edge: float = 0.0
    p95_dd: float = 0.0
    is_valid: bool = False
    warnings: List[str] = None

    def __post_init__(self):
        """Initialize warnings list."""
        if self.warnings is None:
            self.warnings = []

    def get_position_pct(self, capital: float) -> float:
        """Get position size as percentage of capital.

        Args:
            capital: Total capital.

        Returns:
            Position size in currency.
        """
        return self.final_kelly * capital

    def summary(self) -> str:
        """Get summary string.

        Returns:
            Formatted summary.
        """
        lines = [
            "=" * 50,
            "EMPIRICAL KELLY ANALYSIS",
            "=" * 50,
            f"Win Rate: {self.win_rate:.1%}",
            f"Avg Win: {self.avg_win:.2f}%",
            f"Avg Loss: {self.avg_loss:.2f}%",
            "",
            f"Standard Kelly: {self.standard_kelly:.1%}",
            f"Uncertainty Haircut: {self.uncertainty_haircut:.2f} (CV={self.cv_edge:.2f})",
            f"Empirical Kelly: {self.empirical_kelly:.1%}",
            "",
            f"MC Multiplier: {self.mc_multiplier:.2f} (p95 DD={self.p95_dd:.1%})",
            f"Final Kelly: {self.final_kelly:.1%}",
            "",
        ]

        if self.warnings:
            lines.append("WARNINGS:")
            for w in self.warnings:
                lines.append(f"  ⚠️  {w}")

        if self.is_valid:
            lines.append("✓ Calculation valid")
        else:
            lines.append("✗ Calculation invalid - use conservative sizing")

        lines.append("=" * 50)

        return "\n".join(lines)


class EmpiricalKelly:
    """Calculates uncertainty-adjusted Kelly Criterion.

    The Empirical Kelly adjusts the standard Kelly Criterion for:
    1. Edge uncertainty (using CV haircut)
    2. Monte Carlo drawdown risk

    Formula:
        f_empirical = f_kelly × (1 - CV_edge) × mc_multiplier

    Example:
        kelly = EmpiricalKelly()
        result = kelly.calculate(
            win_rate=0.62,
            avg_win=8.5,
            avg_loss=-4.2,
            returns=[5.0, -2.0, 8.0, -3.0, ...],
            mc_p95_dd=0.28,
            max_acceptable_dd=0.20
        )
        print(f"Final Kelly: {result.final_kelly:.1%}")
    """

    # Maximum Kelly fraction (safety cap)
    MAX_KELLY = 0.25  # 25%

    # Minimum matches for valid calculation
    MIN_MATCHES = 30

    # Maximum CV to use Kelly (above this, use minimum)
    MAX_CV = 0.8

    def __init__(
        self,
        max_kelly: float = MAX_KELLY,
        min_matches: int = MIN_MATCHES,
        max_cv: float = MAX_CV,
    ):
        """Initialize Empirical Kelly calculator.

        Args:
            max_kelly: Maximum Kelly fraction allowed.
            min_matches: Minimum matches for valid calculation.
            max_cv: Maximum CV to use Kelly sizing.
        """
        self.max_kelly = max_kelly
        self.min_matches = min_matches
        self.max_cv = max_cv

    def calculate(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        returns: Optional[List[float]] = None,
        cv_edge: Optional[float] = None,
        mc_p95_dd: Optional[float] = None,
        max_acceptable_dd: float = 0.20,
    ) -> KellyResult:
        """Calculate Empirical Kelly.

        Args:
            win_rate: Historical win rate (0-1).
            avg_win: Average winning return (%).
            avg_loss: Average losing return (%).
            returns: List of returns for CV calculation.
            cv_edge: Pre-calculated CV of edge.
            mc_p95_dd: 95th percentile DD from Monte Carlo.
            max_acceptable_dd: Maximum acceptable drawdown.

        Returns:
            KellyResult with full analysis.
        """
        warnings = []

        # Calculate standard Kelly
        # f = (p * b - q) / b
        # where p = win_rate, q = loss_rate, b = |avg_win / avg_loss|
        loss_rate = 1 - win_rate

        if avg_loss == 0:
            # No losses (unlikely but handle)
            standard_kelly = win_rate
        else:
            b = abs(avg_win / avg_loss)  # Odds
            standard_kelly = (win_rate * b - loss_rate) / b

        # Kelly must be positive
        if standard_kelly <= 0:
            return KellyResult(
                standard_kelly=0,
                is_valid=False,
                warnings=["Negative Kelly - no edge detected"],
            )

        # Calculate CV of edge
        if cv_edge is None and returns:
            cv_edge = self._calculate_cv(returns)
        elif cv_edge is None:
            cv_edge = 0.5  # Default assumption

        # Apply uncertainty haircut
        if cv_edge > self.max_cv:
            warnings.append(f"High CV ({cv_edge:.2f}) - using minimum sizing")
            uncertainty_haircut = 0.1
        else:
            uncertainty_haircut = 1 - cv_edge

        empirical_kelly = standard_kelly * uncertainty_haircut

        # Apply Monte Carlo adjustment
        mc_multiplier = 1.0
        if mc_p95_dd is not None and mc_p95_dd > max_acceptable_dd:
            mc_multiplier = max_acceptable_dd / mc_p95_dd
            warnings.append(
                f"MC p95 DD ({mc_p95_dd:.1%}) > max ({max_acceptable_dd:.1%}), "
                f"applying {mc_multiplier:.2f}x multiplier"
            )

        # Calculate final Kelly
        final_kelly = empirical_kelly * mc_multiplier

        # Cap at maximum
        if final_kelly > self.max_kelly:
            warnings.append(f"Kelly ({final_kelly:.1%}) capped at {self.max_kelly:.1%}")
            final_kelly = self.max_kelly

        return KellyResult(
            standard_kelly=standard_kelly,
            uncertainty_haircut=uncertainty_haircut,
            empirical_kelly=empirical_kelly,
            mc_multiplier=mc_multiplier,
            final_kelly=final_kelly,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            cv_edge=cv_edge,
            p95_dd=mc_p95_dd or 0,
            is_valid=True,
            warnings=warnings,
        )

    def calculate_from_pattern(
        self,
        pattern_result: PatternMatchResult,
        mc_p95_dd: Optional[float] = None,
        max_acceptable_dd: float = 0.20,
    ) -> KellyResult:
        """Calculate Kelly from pattern match result.

        Args:
            pattern_result: Pattern matching result.
            mc_p95_dd: 95th percentile DD from Monte Carlo.
            max_acceptable_dd: Maximum acceptable drawdown.

        Returns:
            KellyResult.
        """
        warnings = []

        # Check if we have enough matches
        if pattern_result.count < self.min_matches:
            warnings.append(
                f"Only {pattern_result.count} matches (need {self.min_matches})"
            )
            return KellyResult(
                is_valid=False,
                warnings=warnings,
            )

        # Extract returns
        returns = [t.return_pct for t in pattern_result.matches]

        return self.calculate(
            win_rate=pattern_result.win_rate,
            avg_win=pattern_result.avg_win,
            avg_loss=pattern_result.avg_loss,
            returns=returns,
            mc_p95_dd=mc_p95_dd,
            max_acceptable_dd=max_acceptable_dd,
        )

    def calculate_from_distribution(
        self,
        distribution: ReturnDistribution,
        mc_p95_dd: Optional[float] = None,
        max_acceptable_dd: float = 0.20,
    ) -> KellyResult:
        """Calculate Kelly from return distribution.

        Args:
            distribution: Return distribution analysis.
            mc_p95_dd: 95th percentile DD from Monte Carlo.
            max_acceptable_dd: Maximum acceptable drawdown.

        Returns:
            KellyResult.
        """
        # Estimate win/loss stats from distribution
        returns = distribution.returns
        wins = [r for r in returns if r > 0]
        losses = [r for r in returns if r <= 0]

        win_rate = len(wins) / len(returns) if len(returns) > 0 else 0
        avg_win = sum(wins) / len(wins) if wins else 0
        avg_loss = sum(losses) / len(losses) if losses else 0

        cv_edge = distribution.get_coefficient_of_variation()

        return self.calculate(
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            cv_edge=cv_edge,
            mc_p95_dd=mc_p95_dd,
            max_acceptable_dd=max_acceptable_dd,
        )

    def _calculate_cv(self, returns: List[float]) -> float:
        """Calculate coefficient of variation.

        Args:
            returns: List of returns.

        Returns:
            CV value.
        """
        if not returns:
            return 0.5

        import numpy as np
        returns_array = np.array(returns)

        mean = np.mean(returns_array)
        std = np.std(returns_array)

        if mean == 0:
            return 1.0 if std > 0 else 0.0

        return abs(std / mean)

    def get_conservative_kelly(self) -> float:
        """Get conservative Kelly fraction for uncertain situations.

        Returns:
            Conservative Kelly fraction.
        """
        return 0.005  # 0.5% - very conservative


def calculate_empirical_kelly(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    cv_edge: Optional[float] = None,
    mc_p95_dd: Optional[float] = None,
) -> float:
    """Convenience function to calculate Empirical Kelly.

    Args:
        win_rate: Win rate (0-1).
        avg_win: Average winning return.
        avg_loss: Average losing return.
        cv_edge: CV of edge.
        mc_p95_dd: 95th percentile DD.

    Returns:
        Final Kelly fraction.
    """
    kelly = EmpiricalKelly()
    result = kelly.calculate(
        win_rate=win_rate,
        avg_win=avg_win,
        avg_loss=avg_loss,
        cv_edge=cv_edge,
        mc_p95_dd=mc_p95_dd,
    )
    return result.final_kelly


def get_position_size(
    capital: float,
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    cv_edge: Optional[float] = None,
    mc_p95_dd: Optional[float] = None,
    max_acceptable_dd: float = 0.20,
) -> Tuple[float, KellyResult]:
    """Calculate position size using Empirical Kelly.

    Args:
        capital: Total capital.
        win_rate: Win rate (0-1).
        avg_win: Average winning return.
        avg_loss: Average losing return.
        cv_edge: CV of edge.
        mc_p95_dd: 95th percentile DD.
        max_acceptable_dd: Maximum acceptable DD.

    Returns:
        Tuple of (position_size, kelly_result).
    """
    kelly = EmpiricalKelly()
    result = kelly.calculate(
        win_rate=win_rate,
        avg_win=avg_win,
        avg_loss=avg_loss,
        cv_edge=cv_edge,
        mc_p95_dd=mc_p95_dd,
        max_acceptable_dd=max_acceptable_dd,
    )

    position_size = result.final_kelly * capital
    return position_size, result
