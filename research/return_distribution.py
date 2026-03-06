"""
Return Distribution Analysis Module

Builds empirical return distributions and calculates key statistics
for risk management and position sizing.
"""

import logging
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ReturnDistribution:
    """Empirical return distribution analysis.

    Captures the actual distribution characteristics of trading returns,
    including fat tails, skewness, and risk metrics.

    Attributes:
        returns: Raw return data.
        n: Number of observations.
        mean: Mean return.
        std: Standard deviation.
        median: Median return.
        min: Minimum return.
        max: Maximum return.
        skewness: Distribution skewness.
        kurtosis: Excess kurtosis (vs normal).
        p5: 5th percentile.
        p25: 25th percentile.
        p75: 75th percentile.
        p95: 95th percentile.
        var_95: Value at Risk (95%).
        cvar_95: Conditional VaR (Expected Shortfall).
        is_normal: Whether distribution passes normality test.
        normality_p: P-value from normality test.
    """

    returns: np.ndarray
    n: int = 0
    mean: float = 0.0
    std: float = 0.0
    median: float = 0.0
    min: float = 0.0
    max: float = 0.0
    skewness: float = 0.0
    kurtosis: float = 0.0
    p5: float = 0.0
    p25: float = 0.0
    p75: float = 0.0
    p95: float = 0.0
    var_95: float = 0.0
    cvar_95: float = 0.0
    is_normal: bool = False
    normality_p: float = 0.0

    def __post_init__(self):
        """Calculate all statistics from returns."""
        if len(self.returns) == 0:
            return

        self.n = len(self.returns)

        # Basic statistics
        self.mean = float(np.mean(self.returns))
        self.std = float(np.std(self.returns))
        self.median = float(np.median(self.returns))
        self.min = float(np.min(self.returns))
        self.max = float(np.max(self.returns))

        # Percentiles
        self.p5 = float(np.percentile(self.returns, 5))
        self.p25 = float(np.percentile(self.returns, 25))
        self.p75 = float(np.percentile(self.returns, 75))
        self.p95 = float(np.percentile(self.returns, 95))

        # Higher moments
        self.skewness = self._calculate_skewness()
        self.kurtosis = self._calculate_kurtosis()

        # Risk metrics
        self.var_95 = self._calculate_var(0.95)
        self.cvar_95 = self._calculate_cvar(0.95)

        # Normality test
        self.is_normal, self.normality_p = self._test_normality()

    def _calculate_skewness(self) -> float:
        """Calculate skewness of distribution.

        Positive = right tail, Negative = left tail.
        """
        if self.n < 3 or self.std == 0:
            return 0.0

        n = self.n
        mean = self.mean
        std = self.std

        # Fisher-Pearson coefficient
        skew = (n / ((n - 1) * (n - 2))) * np.sum(
            ((self.returns - mean) / std) ** 3
        )
        return float(skew)

    def _calculate_kurtosis(self) -> float:
        """Calculate excess kurtosis.

        Positive = fat tails, Negative = thin tails.
        Normal distribution has kurtosis = 0.
        """
        if self.n < 4 or self.std == 0:
            return 0.0

        n = self.n
        mean = self.mean
        std = self.std

        # Excess kurtosis
        kurt = (
            (n * (n + 1) / ((n - 1) * (n - 2) * (n - 3))) *
            np.sum(((self.returns - mean) / std) ** 4)
        ) - (3 * (n - 1) ** 2 / ((n - 2) * (n - 3)))

        return float(kurt)

    def _calculate_var(self, confidence: float = 0.95) -> float:
        """Calculate Value at Risk.

        Args:
            confidence: Confidence level (default 95%).

        Returns:
            VaR as negative number (loss).
        """
        if self.n == 0:
            return 0.0

        # VaR is the loss at the (1 - confidence) percentile
        percentile = (1 - confidence) * 100
        return float(np.percentile(self.returns, percentile))

    def _calculate_cvar(self, confidence: float = 0.95) -> float:
        """Calculate Conditional VaR (Expected Shortfall).

        Average of losses beyond VaR threshold.

        Args:
            confidence: Confidence level.

        Returns:
            CVaR as negative number (expected loss).
        """
        if self.n == 0:
            return 0.0

        var = self._calculate_var(confidence)
        # Average of returns below VaR
        tail_returns = self.returns[self.returns <= var]

        if len(tail_returns) == 0:
            return var

        return float(np.mean(tail_returns))

    def _test_normality(self) -> Tuple[bool, float]:
        """Test if distribution is normal using Shapiro-Wilk.

        Returns:
            Tuple of (is_normal, p_value).
        """
        if self.n < 3:
            return False, 0.0

        try:
            from scipy import stats
            stat, p_value = stats.shapiro(self.returns)
            # Consider normal if p > 0.05
            return p_value > 0.05, float(p_value)
        except ImportError:
            # Fallback: check skewness and kurtosis
            is_normal = abs(self.skewness) < 0.5 and abs(self.kurtosis) < 1.0
            return is_normal, 0.0

    def get_coefficient_of_variation(self) -> float:
        """Calculate CV (coefficient of variation).

        CV = std / mean
        Used for uncertainty adjustment in Kelly Criterion.

        Returns:
            CV value. Returns 0 if mean is 0.
        """
        if self.mean == 0:
            return 0.0
        return abs(self.std / self.mean)

    def get_sharpe_ratio(self, risk_free_rate: float = 0.0) -> float:
        """Calculate Sharpe ratio.

        Args:
            risk_free_rate: Risk-free rate (annualized).

        Returns:
            Sharpe ratio.
        """
        if self.std == 0:
            return 0.0
        return (self.mean - risk_free_rate) / self.std

    def get_sortino_ratio(
        self,
        risk_free_rate: float = 0.0,
        target_return: float = 0.0,
    ) -> float:
        """Calculate Sortino ratio.

        Uses downside deviation instead of total std.

        Args:
            risk_free_rate: Risk-free rate.
            target_return: Minimum acceptable return.

        Returns:
            Sortino ratio.
        """
        downside_returns = self.returns[self.returns < target_return]

        if len(downside_returns) == 0:
            return float('inf') if self.mean > risk_free_rate else 0.0

        downside_std = np.std(downside_returns)

        if downside_std == 0:
            return float('inf') if self.mean > risk_free_rate else 0.0

        return (self.mean - risk_free_rate) / downside_std

    def get_tail_ratio(self) -> float:
        """Calculate tail ratio (95th / 5th percentile absolute).

        Higher = fatter right tail (good).
        Lower = fatter left tail (bad).

        Returns:
            Tail ratio.
        """
        if abs(self.p5) < 0.0001:
            return float('inf')
        return abs(self.p95 / self.p5)

    def summary(self) -> str:
        """Get summary string.

        Returns:
            Formatted summary.
        """
        lines = [
            f"Returns: n={self.n}",
            f"  Mean: {self.mean:.2f}%, Std: {self.std:.2f}%",
            f"  Median: {self.median:.2f}%",
            f"  Range: [{self.min:.2f}%, {self.max:.2f}%]",
            f"  Skewness: {self.skewness:.2f}",
            f"  Kurtosis: {self.kurtosis:.2f}",
            f"  VaR(95%): {self.var_95:.2f}%",
            f"  CVaR(95%): {self.cvar_95:.2f}%",
            f"  Normal: {'Yes' if self.is_normal else 'No'} (p={self.normality_p:.3f})",
        ]
        return "\n".join(lines)


class ReturnAnalyzer:
    """Analyzes return distributions for trading strategies.

    Example:
        analyzer = ReturnAnalyzer()
        dist = analyzer.analyze([5.0, -2.0, 3.0, -1.0, 4.0])
        print(f"CV: {dist.get_coefficient_of_variation():.2f}")
    """

    def __init__(self):
        """Initialize return analyzer."""
        pass

    def analyze(self, returns: List[float]) -> ReturnDistribution:
        """Analyze a return series.

        Args:
            returns: List of return percentages.

        Returns:
            ReturnDistribution with full analysis.
        """
        return_array = np.array(returns)
        return ReturnDistribution(returns=return_array)

    def analyze_trades(self, trades: list) -> ReturnDistribution:
        """Analyze returns from trade objects.

        Args:
            trades: List of Trade objects with return_pct attribute.

        Returns:
            ReturnDistribution with full analysis.
        """
        returns = [t.return_pct for t in trades if hasattr(t, 'return_pct')]
        return self.analyze(returns)

    def compare_distributions(
        self,
        returns1: List[float],
        returns2: List[float],
        label1: str = "Distribution 1",
        label2: str = "Distribution 2",
    ) -> str:
        """Compare two return distributions.

        Args:
            returns1: First return series.
            returns2: Second return series.
            label1: Label for first distribution.
            label2: Label for second distribution.

        Returns:
            Formatted comparison string.
        """
        dist1 = self.analyze(returns1)
        dist2 = self.analyze(returns2)

        lines = [
            "=" * 60,
            "RETURN DISTRIBUTION COMPARISON",
            "=" * 60,
            "",
            f"{'Metric':<20} {label1:<20} {label2:<20}",
            "-" * 60,
            f"{'N':<20} {dist1.n:<20} {dist2.n:<20}",
            f"{'Mean':<20} {dist1.mean:<20.2f} {dist2.mean:<20.2f}",
            f"{'Std':<20} {dist1.std:<20.2f} {dist2.std:<20.2f}",
            f"{'Median':<20} {dist1.median:<20.2f} {dist2.median:<20.2f}",
            f"{'Skewness':<20} {dist1.skewness:<20.2f} {dist2.skewness:<20.2f}",
            f"{'Kurtosis':<20} {dist1.kurtosis:<20.2f} {dist2.kurtosis:<20.2f}",
            f"{'VaR(95%)':<20} {dist1.var_95:<20.2f} {dist2.var_95:<20.2f}",
            f"{'CVaR(95%)':<20} {dist1.cvar_95:<20.2f} {dist2.cvar_95:<20.2f}",
            f"{'CV':<20} {dist1.get_coefficient_of_variation():<20.2f} {dist2.get_coefficient_of_variation():<20.2f}",
            "",
            "=" * 60,
        ]

        return "\n".join(lines)

    def get_edge_estimate(
        self,
        returns: List[float],
        confidence: float = 0.95,
    ) -> Tuple[float, float, float]:
        """Estimate edge with confidence interval.

        Args:
            returns: Return series.
            confidence: Confidence level for interval.

        Returns:
            Tuple of (point_estimate, lower_bound, upper_bound).
        """
        dist = self.analyze(returns)

        # Use bootstrap or normal approximation
        if dist.n < 30:
            # Use t-distribution
            from scipy import stats
            t_val = stats.t.ppf((1 + confidence) / 2, dist.n - 1)
            margin = t_val * dist.std / np.sqrt(dist.n)
        else:
            # Use normal approximation
            z_val = 1.96 if confidence == 0.95 else 2.576
            margin = z_val * dist.std / np.sqrt(dist.n)

        return (
            dist.mean,
            dist.mean - margin,
            dist.mean + margin,
        )


def build_return_distribution(returns: List[float]) -> ReturnDistribution:
    """Convenience function to build return distribution.

    Args:
        returns: List of return percentages.

    Returns:
        ReturnDistribution with full analysis.
    """
    analyzer = ReturnAnalyzer()
    return analyzer.analyze(returns)


def calculate_var_cvar(
    returns: List[float],
    confidence: float = 0.95,
) -> Tuple[float, float]:
    """Calculate VaR and CVaR for a return series.

    Args:
        returns: Return series.
        confidence: Confidence level.

    Returns:
        Tuple of (VaR, CVaR).
    """
    dist = build_return_distribution(returns)
    return dist.var_95, dist.cvar_95
