"""
Calibration Surface Module

Builds 2D calibration surfaces to understand how edge decays
over time based on signal score and holding period.
"""

import logging
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, Any

from core.data.models import Trade

logger = logging.getLogger(__name__)


@dataclass
class CalibrationCell:
    """A single cell in the calibration surface.

    Attributes:
        score_bin: Score range (min, max).
        day: Days held.
        n: Number of trades in cell.
        win_rate: Win rate (0-1).
        avg_return: Average return percentage.
        std_return: Standard deviation of returns.
        total_pnl: Total P&L.
        is_significant: Whether cell has enough data.
    """

    score_bin: Tuple[int, int]
    day: int
    n: int = 0
    win_rate: float = 0.0
    avg_return: float = 0.0
    std_return: float = 0.0
    total_pnl: float = 0.0
    is_significant: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "score_range": f"{self.score_bin[0]}-{self.score_bin[1]}",
            "day": self.day,
            "n": self.n,
            "win_rate": self.win_rate,
            "avg_return": self.avg_return,
            "std_return": self.std_return,
            "is_significant": self.is_significant,
        }


@dataclass
class CalibrationSurface:
    """2D calibration surface C(score, days_held).

    Maps signal score and holding period to empirical win rate
    and expected return.

    Attributes:
        cells: Dictionary mapping (score_bin, day) to CalibrationCell.
        score_bins: List of score bin boundaries.
        max_days: Maximum days held.
        min_edge: Minimum win rate to consider having edge.
    """

    cells: Dict[Tuple[str, int], CalibrationCell] = field(default_factory=dict)
    score_bins: List[Tuple[int, int]] = field(default_factory=list)
    max_days: int = 7
    min_edge: float = 0.52

    def __post_init__(self):
        """Initialize default score bins."""
        if not self.score_bins:
            self.score_bins = [
                (50, 60),
                (60, 70),
                (70, 80),
                (80, 90),
                (90, 100),
            ]

    def get_win_rate(self, score: float, day: int) -> float:
        """Get win rate for a score and day.

        Args:
            score: Signal score.
            day: Days held.

        Returns:
            Win rate (0-1), or 0.5 if no data.
        """
        bin_key = self._get_score_bin(score)
        cell = self.cells.get((bin_key, day))

        if cell is None or not cell.is_significant:
            return 0.5  # Neutral if no data

        return cell.win_rate

    def get_avg_return(self, score: float, day: int) -> float:
        """Get average return for a score and day.

        Args:
            score: Signal score.
            day: Days held.

        Returns:
            Average return percentage, or 0 if no data.
        """
        bin_key = self._get_score_bin(score)
        cell = self.cells.get((bin_key, day))

        if cell is None:
            return 0.0

        return cell.avg_return

    def get_optimal_exit_day(self, score: float) -> int:
        """Find optimal exit day based on edge decay.

        The day when edge decays below minimum threshold.

        Args:
            score: Signal score.

        Returns:
            Optimal exit day.
        """
        for day in range(1, self.max_days + 1):
            win_rate = self.get_win_rate(score, day)
            if win_rate < self.min_edge:
                return max(1, day - 1)  # Exit day before edge disappears

        return self.max_days  # Hold max if edge persists

    def get_edge_decay(self, score: float) -> List[float]:
        """Get edge decay over time for a score.

        Args:
            score: Signal score.

        Returns:
            List of win rates by day.
        """
        decay = []
        for day in range(1, self.max_days + 1):
            decay.append(self.get_win_rate(score, day))
        return decay

    def get_daily_decay_rate(self, score: float) -> float:
        """Calculate average daily edge decay rate.

        Args:
            score: Signal score.

        Returns:
            Average daily decay (negative = edge decreasing).
        """
        decay = self.get_edge_decay(score)
        if len(decay) < 2:
            return 0.0

        total_decay = 0.0
        count = 0

        for i in range(1, len(decay)):
            diff = decay[i] - decay[i - 1]
            total_decay += diff
            count += 1

        return total_decay / count if count > 0 else 0.0

    def should_exit_by_calibration(
        self,
        score: float,
        days_held: int,
        current_pnl_pct: float,
    ) -> Tuple[bool, str]:
        """Determine if position should exit based on calibration.

        Args:
            score: Entry signal score.
            days_held: Days position has been held.
            current_pnl_pct: Current P&L percentage.

        Returns:
            Tuple of (should_exit, reason).
        """
        current_win_rate = self.get_win_rate(score, days_held)
        optimal_exit = self.get_optimal_exit_day(score)

        # Edge decayed below threshold
        if current_win_rate < self.min_edge:
            return True, f"Edge decayed: {current_win_rate:.1%} win rate"

        # Past optimal exit and profitable
        if days_held >= optimal_exit and current_pnl_pct > 0:
            return True, f"Past optimal exit day ({optimal_exit}), taking profit"

        # Check for rapid decay
        if days_held > 1:
            prev_win_rate = self.get_win_rate(score, days_held - 1)
            if prev_win_rate - current_win_rate > 0.05:
                return True, "Rapid edge decay detected (>5%)"

        return False, "Hold - edge still present"

    def _get_score_bin(self, score: float) -> str:
        """Get score bin key for a score.

        Args:
            score: Signal score.

        Returns:
            Score bin string like "70-80".
        """
        for low, high in self.score_bins:
            if low <= score < high:
                return f"{low}-{high}"

        # Default to last bin for scores >= 100
        if score >= 90:
            return "90-100"
        return "50-60"

    def to_matrix(self) -> Tuple[np.ndarray, List[str], List[int]]:
        """Convert surface to matrix form.

        Returns:
            Tuple of (win_rate_matrix, row_labels, col_labels).
        """
        row_labels = [f"{low}-{high}" for low, high in self.score_bins]
        col_labels = list(range(1, self.max_days + 1))

        matrix = np.zeros((len(self.score_bins), self.max_days))

        for i, (low, high) in enumerate(self.score_bins):
            bin_key = f"{low}-{high}"
            for j, day in enumerate(col_labels):
                cell = self.cells.get((bin_key, day))
                if cell:
                    matrix[i, j] = cell.win_rate
                else:
                    matrix[i, j] = 0.5  # Neutral

        return matrix, row_labels, col_labels

    def summary(self) -> str:
        """Get summary string of calibration surface.

        Returns:
            Formatted summary.
        """
        lines = [
            "=" * 70,
            "CALIBRATION SURFACE ANALYSIS",
            "=" * 70,
            "",
            "Win Rate by Signal Score and Days Held:",
            "",
        ]

        # Header
        header = f"{'Score':<10}"
        for day in range(1, self.max_days + 1):
            header += f"  Day{day:<5}"
        lines.append(header)
        lines.append("-" * 70)

        # Rows
        for low, high in self.score_bins:
            bin_key = f"{low}-{high}"
            row = f"{bin_key:<10}"
            for day in range(1, self.max_days + 1):
                cell = self.cells.get((bin_key, day))
                if cell and cell.is_significant:
                    row += f"  {cell.win_rate:.0%}    "
                elif cell:
                    row += f"  {cell.win_rate:.0%}*   "
                else:
                    row += "   --    "
            lines.append(row)

        lines.extend([
            "",
            "* = Insufficient data (< 30 trades)",
            "",
            "OPTIMAL EXIT TIMING:",
        ])

        for low, high in self.score_bins:
            bin_key = f"{low}-{high}"
            optimal = self.get_optimal_exit_day((low + high) / 2)
            lines.append(f"  Score {bin_key}: Exit by day {optimal}")

        avg_decay = self.get_daily_decay_rate(75)
        lines.append(f"\nAverage edge decay: {avg_decay:.1%} per day")

        lines.append("")
        lines.append("=" * 70)

        return "\n".join(lines)


class CalibrationBuilder:
    """Builds calibration surfaces from trade history.

    Example:
        builder = CalibrationBuilder()
        surface = builder.build(trades)
        print(surface.get_optimal_exit_day(75))
    """

    # Minimum trades per cell for significance
    MIN_TRADES_PER_CELL = 30

    def __init__(
        self,
        score_bins: Optional[List[Tuple[int, int]]] = None,
        max_days: int = 7,
        min_edge: float = 0.52,
    ):
        """Initialize calibration builder.

        Args:
            score_bins: Score bin boundaries.
            max_days: Maximum days to analyze.
            min_edge: Minimum win rate for having edge.
        """
        self.score_bins = score_bins or [
            (50, 60),
            (60, 70),
            (70, 80),
            (80, 90),
            (90, 100),
        ]
        self.max_days = max_days
        self.min_edge = min_edge

    def build(self, trades: List[Trade]) -> CalibrationSurface:
        """Build calibration surface from trades.

        Args:
            trades: List of historical trades.

        Returns:
            CalibrationSurface with populated cells.
        """
        surface = CalibrationSurface(
            score_bins=self.score_bins,
            max_days=self.max_days,
            min_edge=self.min_edge,
        )

        # Group trades by (score_bin, holding_days)
        grouped: Dict[Tuple[str, int], List[Trade]] = {}

        for trade in trades:
            score = trade.signal_score
            days = trade.holding_days

            if days < 1 or days > self.max_days:
                continue

            # Find score bin
            bin_key = None
            for low, high in self.score_bins:
                if low <= score < high:
                    bin_key = f"{low}-{high}"
                    break

            if bin_key is None:
                continue

            key = (bin_key, days)
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(trade)

        # Calculate stats for each cell
        for (bin_key, day), cell_trades in grouped.items():
            if not cell_trades:
                continue

            n = len(cell_trades)
            wins = sum(1 for t in cell_trades if t.return_pct > 0)
            win_rate = wins / n if n > 0 else 0
            returns = [t.return_pct for t in cell_trades]
            avg_return = sum(returns) / n if n > 0 else 0
            std_return = (sum((r - avg_return) ** 2 for r in returns) / n) ** 0.5 if n > 1 else 0
            total_pnl = sum(t.net_pnl for t in cell_trades)

            # Parse bin back to tuple
            low, high = map(int, bin_key.split("-"))

            cell = CalibrationCell(
                score_bin=(low, high),
                day=day,
                n=n,
                win_rate=win_rate,
                avg_return=avg_return,
                std_return=std_return,
                total_pnl=total_pnl,
                is_significant=n >= self.MIN_TRADES_PER_CELL,
            )

            surface.cells[(bin_key, day)] = cell

        logger.info(
            f"Built calibration surface with {len(surface.cells)} cells, "
            f"{sum(1 for c in surface.cells.values() if c.is_significant)} significant"
        )

        return surface

    def build_from_returns(
        self,
        trade_data: List[Dict[str, Any]],
    ) -> CalibrationSurface:
        """Build calibration from trade data dictionaries.

        Args:
            trade_data: List of dicts with 'signal_score', 'holding_days', 'return_pct'.

        Returns:
            CalibrationSurface.
        """
        trades = []
        for i, data in enumerate(trade_data):
            trade = Trade(
                trade_id=f"TRADE-{i}",
                symbol=data.get("symbol", "UNKNOWN"),
                entry_date=data.get("entry_date"),
                entry_price=data.get("entry_price", 0),
                exit_date=data.get("exit_date"),
                exit_price=data.get("exit_price", 0),
                exit_reason=data.get("exit_reason", "unknown"),
                quantity=data.get("quantity", 0),
                side=data.get("side", "BUY"),
                gross_pnl=data.get("gross_pnl", 0),
                fees=data.get("fees", 0),
                net_pnl=data.get("net_pnl", 0),
                return_pct=data.get("return_pct", 0),
                holding_days=data.get("holding_days", 0),
                max_favorable=0,
                max_adverse=0,
                signal_score=data.get("signal_score", 50),
                setup_type=data.get("setup_type"),
                rsi_at_entry=data.get("rsi_at_entry", 50),
                flow_signal=data.get("flow_signal"),
                flow_consecutive_days=data.get("flow_consecutive_days", 0),
            )
            trades.append(trade)

        return self.build(trades)


def build_calibration_surface(
    trades: List[Trade],
    max_days: int = 7,
) -> CalibrationSurface:
    """Convenience function to build calibration surface.

    Args:
        trades: List of historical trades.
        max_days: Maximum days to analyze.

    Returns:
        CalibrationSurface.
    """
    builder = CalibrationBuilder(max_days=max_days)
    return builder.build(trades)
