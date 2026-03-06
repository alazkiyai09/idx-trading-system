"""
Pattern Matcher Module

Finds historical trades matching current signal characteristics
for empirical analysis and position sizing.
"""

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import List, Optional, Dict, Any

from core.data.models import Trade, SetupType, FlowSignal

logger = logging.getLogger(__name__)


@dataclass
class SignalPattern:
    """Defines criteria for pattern matching.

    Attributes:
        score_range: (min, max) composite score range.
        rsi_range: (min, max) RSI range at entry.
        trend: Trend direction at entry.
        volume_ratio_min: Minimum volume ratio at entry.
        flow_signal: Foreign flow signal type.
        flow_consecutive_days_min: Minimum consecutive flow days.
        setup_type: Setup type (optional).
    """

    score_range: tuple = (0, 100)
    rsi_range: tuple = (0, 100)
    trend: Optional[str] = None
    volume_ratio_min: float = 0.0
    flow_signal: Optional[str] = None
    flow_consecutive_days_min: int = 0
    setup_type: Optional[str] = None

    def matches(self, trade: Trade) -> bool:
        """Check if a trade matches this pattern.

        Args:
            trade: Trade to check.

        Returns:
            True if trade matches all pattern criteria.
        """
        # Check score range
        if not (self.score_range[0] <= trade.signal_score <= self.score_range[1]):
            return False

        # Check RSI range
        if trade.rsi_at_entry is not None:
            if not (self.rsi_range[0] <= trade.rsi_at_entry <= self.rsi_range[1]):
                return False

        # Check flow signal
        if self.flow_signal and trade.flow_signal:
            if trade.flow_signal.value != self.flow_signal:
                return False

        # Check consecutive days
        if trade.flow_consecutive_days is not None:
            if trade.flow_consecutive_days < self.flow_consecutive_days_min:
                return False

        # Check setup type
        if self.setup_type and trade.setup_type:
            if trade.setup_type.value != self.setup_type:
                return False

        return True


@dataclass
class PatternMatchResult:
    """Result of pattern matching analysis.

    Attributes:
        pattern: The pattern that was matched.
        matches: List of matching trades.
        count: Number of matches.
        win_count: Number of winning trades.
        loss_count: Number of losing trades.
        win_rate: Win rate (0-1).
        avg_return: Average return percentage.
        std_return: Standard deviation of returns.
        avg_win: Average winning return.
        avg_loss: Average losing return.
        total_pnl: Total P&L from matches.
        profit_factor: Gross wins / gross losses.
        is_significant: Whether we have enough matches for statistical validity.
    """

    pattern: SignalPattern
    matches: List[Trade] = field(default_factory=list)
    count: int = 0
    win_count: int = 0
    loss_count: int = 0
    win_rate: float = 0.0
    avg_return: float = 0.0
    std_return: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    total_pnl: float = 0.0
    profit_factor: float = 0.0
    is_significant: bool = False

    def __post_init__(self):
        """Calculate statistics from matches."""
        if not self.matches:
            return

        self.count = len(self.matches)

        # Calculate basic stats
        returns = [t.return_pct for t in self.matches]
        wins = [r for r in returns if r > 0]
        losses = [r for r in returns if r <= 0]

        self.win_count = len(wins)
        self.loss_count = len(losses)
        self.win_rate = self.win_count / self.count if self.count > 0 else 0

        # Calculate return stats
        if returns:
            self.avg_return = sum(returns) / len(returns)
            if len(returns) > 1:
                variance = sum((r - self.avg_return) ** 2 for r in returns) / len(returns)
                self.std_return = variance ** 0.5

        if wins:
            self.avg_win = sum(wins) / len(wins)
        if losses:
            self.avg_loss = sum(losses) / len(losses)

        # Calculate profit factor
        total_wins = sum(wins)
        total_losses = abs(sum(losses))
        self.profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')

        # Calculate total P&L
        self.total_pnl = sum(t.net_pnl for t in self.matches)

        # Check statistical significance (minimum 30 matches)
        self.is_significant = self.count >= 30


class PatternMatcher:
    """Matches current signals to historical trades.

    Finds trades with similar characteristics to understand
    expected outcomes and calculate empirical statistics.

    Example:
        matcher = PatternMatcher(trade_history)
        pattern = SignalPattern(
            score_range=(70, 80),
            flow_signal="buy",
            setup_type="MOMENTUM"
        )
        result = matcher.match(pattern)
        if result.is_significant:
            print(f"Win rate: {result.win_rate:.1%}")
    """

    # Minimum matches for statistical significance
    MIN_MATCHES = 30

    def __init__(self, trades: Optional[List[Trade]] = None):
        """Initialize pattern matcher.

        Args:
            trades: Historical trade database.
        """
        self.trades: List[Trade] = trades or []
        logger.info(f"Pattern matcher initialized with {len(self.trades)} trades")

    def add_trade(self, trade: Trade) -> None:
        """Add a trade to the history.

        Args:
            trade: Trade to add.
        """
        self.trades.append(trade)

    def add_trades(self, trades: List[Trade]) -> None:
        """Add multiple trades to history.

        Args:
            trades: Trades to add.
        """
        self.trades.extend(trades)

    def match(self, pattern: SignalPattern) -> PatternMatchResult:
        """Find all trades matching a pattern.

        Args:
            pattern: Pattern to match.

        Returns:
            PatternMatchResult with matching trades and statistics.
        """
        matches = [t for t in self.trades if pattern.matches(t)]

        result = PatternMatchResult(
            pattern=pattern,
            matches=matches,
        )

        logger.info(
            f"Pattern match: {result.count} matches, "
            f"win_rate={result.win_rate:.1%}, "
            f"significant={result.is_significant}"
        )

        return result

    def match_by_score(
        self,
        score: float,
        tolerance: float = 5.0,
    ) -> PatternMatchResult:
        """Find trades matching a score within tolerance.

        Args:
            score: Target composite score.
            tolerance: Score range (+/-) to include.

        Returns:
            PatternMatchResult with matching trades.
        """
        pattern = SignalPattern(
            score_range=(score - tolerance, score + tolerance)
        )
        return self.match(pattern)

    def match_by_flow(
        self,
        flow_signal: str,
        consecutive_days_min: int = 0,
    ) -> PatternMatchResult:
        """Find trades matching foreign flow criteria.

        Args:
            flow_signal: Flow signal type.
            consecutive_days_min: Minimum consecutive days.

        Returns:
            PatternMatchResult with matching trades.
        """
        pattern = SignalPattern(
            flow_signal=flow_signal,
            flow_consecutive_days_min=consecutive_days_min,
        )
        return self.match(pattern)

    def match_by_setup(
        self,
        setup_type: str,
        score_min: float = 60.0,
    ) -> PatternMatchResult:
        """Find trades matching a setup type.

        Args:
            setup_type: Setup type to match.
            score_min: Minimum signal score.

        Returns:
            PatternMatchResult with matching trades.
        """
        pattern = SignalPattern(
            setup_type=setup_type,
            score_range=(score_min, 100),
        )
        return self.match(pattern)

    def create_pattern_from_signal(
        self,
        signal_score: float,
        flow_signal: Optional[str] = None,
        setup_type: Optional[str] = None,
        rsi: Optional[float] = None,
        score_tolerance: float = 5.0,
        rsi_tolerance: float = 10.0,
    ) -> SignalPattern:
        """Create a pattern from signal characteristics.

        Args:
            signal_score: Composite signal score.
            flow_signal: Foreign flow signal.
            setup_type: Setup type.
            rsi: RSI at entry.
            score_tolerance: Score range tolerance.
            rsi_tolerance: RSI range tolerance.

        Returns:
            SignalPattern matching the criteria.
        """
        rsi_min = max(0, (rsi or 50) - rsi_tolerance)
        rsi_max = min(100, (rsi or 50) + rsi_tolerance)

        return SignalPattern(
            score_range=(signal_score - score_tolerance, signal_score + score_tolerance),
            rsi_range=(rsi_min, rsi_max),
            flow_signal=flow_signal,
            setup_type=setup_type,
        )

    def get_best_matches(
        self,
        signal_score: float,
        flow_signal: Optional[str] = None,
        setup_type: Optional[str] = None,
        rsi: Optional[float] = None,
        min_matches: int = 30,
    ) -> Optional[PatternMatchResult]:
        """Get best matching pattern with sufficient data.

        Starts with strict criteria and progressively relaxes
        until minimum matches are found.

        Args:
            signal_score: Composite signal score.
            flow_signal: Foreign flow signal.
            setup_type: Setup type.
            rsi: RSI at entry.
            min_matches: Minimum matches required.

        Returns:
            PatternMatchResult if enough matches found, None otherwise.
        """
        # Try strict criteria first
        pattern = self.create_pattern_from_signal(
            signal_score=signal_score,
            flow_signal=flow_signal,
            setup_type=setup_type,
            rsi=rsi,
            score_tolerance=5.0,
            rsi_tolerance=10.0,
        )
        result = self.match(pattern)

        if result.count >= min_matches:
            return result

        # Relax score tolerance
        pattern.score_range = (signal_score - 10, signal_score + 10)
        result = self.match(pattern)

        if result.count >= min_matches:
            return result

        # Relax RSI tolerance
        pattern.rsi_range = (0, 100)
        result = self.match(pattern)

        if result.count >= min_matches:
            return result

        # Drop setup type requirement
        pattern.setup_type = None
        result = self.match(pattern)

        if result.count >= min_matches:
            return result

        # Drop flow signal requirement
        pattern.flow_signal = None
        result = self.match(pattern)

        if result.count >= min_matches:
            return result

        # Still not enough matches
        logger.warning(
            f"Could not find {min_matches} matches for pattern. "
            f"Best result: {result.count} matches"
        )
        return result if result.count > 0 else None

    def get_pattern_stats(self) -> Dict[str, Any]:
        """Get statistics about available patterns.

        Returns:
            Dictionary with pattern statistics.
        """
        if not self.trades:
            return {
                "total_trades": 0,
                "patterns_available": 0,
            }

        # Group by score bins
        score_bins = {
            "50-60": 0,
            "60-70": 0,
            "70-80": 0,
            "80-90": 0,
            "90-100": 0,
        }

        for trade in self.trades:
            score = trade.signal_score
            if 50 <= score < 60:
                score_bins["50-60"] += 1
            elif 60 <= score < 70:
                score_bins["60-70"] += 1
            elif 70 <= score < 80:
                score_bins["70-80"] += 1
            elif 80 <= score < 90:
                score_bins["80-90"] += 1
            elif 90 <= score <= 100:
                score_bins["90-100"] += 1

        # Count patterns with sufficient data
        significant_patterns = sum(1 for count in score_bins.values() if count >= self.MIN_MATCHES)

        # Group by flow signal
        flow_counts: Dict[str, int] = {}
        for trade in self.trades:
            if trade.flow_signal:
                signal = trade.flow_signal.value
                flow_counts[signal] = flow_counts.get(signal, 0) + 1

        return {
            "total_trades": len(self.trades),
            "score_bins": score_bins,
            "significant_patterns": significant_patterns,
            "flow_counts": flow_counts,
            "min_matches_required": self.MIN_MATCHES,
        }

    def validate_pattern_data(self) -> List[str]:
        """Validate that we have enough data for pattern matching.

        Returns:
            List of validation warnings.
        """
        warnings = []
        stats = self.get_pattern_stats()

        if stats["total_trades"] < self.MIN_MATCHES:
            warnings.append(
                f"Only {stats['total_trades']} trades available. "
                f"Need at least {self.MIN_MATCHES} for pattern matching."
            )

        for bin_name, count in stats.get("score_bins", {}).items():
            if 0 < count < self.MIN_MATCHES:
                warnings.append(
                    f"Score bin {bin_name} has only {count} trades. "
                    f"Need {self.MIN_MATCHES} for reliable statistics."
                )

        return warnings
