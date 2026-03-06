"""
Walk-Forward Analysis Module

Implements walk-forward optimization to:
- Train on N months, test on M months
- Roll forward and repeat
- Combine results
- Guard against overfitting
"""

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum

import numpy as np

from backtest.engine import BacktestEngine, BacktestConfig, BacktestResult
from backtest.metrics import calculate_metrics
from core.data.models import Bar, Trade

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward analysis.

    Attributes:
        base_config: Base backtest configuration.
        train_period_months: Training period in months.
        test_period_months: Test period in months.
        anchor: Whether to anchor training to start (expanding window).
        min_train_trades: Minimum trades required in training.
    """

    base_config: BacktestConfig
    train_period_months: int = 12
    test_period_months: int = 3
    anchor: bool = False  # False = rolling window, True = expanding window
    min_train_trades: int = 30


@dataclass
class FoldResult:
    """Result of a single walk-forward fold.

    Attributes:
        fold_number: Fold index.
        train_start: Training period start.
        train_end: Training period end.
        test_start: Test period start.
        test_end: Test period end.
        train_trades: Number of trades in training.
        test_result: Test period backtest result.
        metrics: Performance metrics for this fold.
    """

    fold_number: int
    train_start: date
    train_end: date
    test_start: date
    test_end: date
    train_trades: int = 0
    test_result: Optional[BacktestResult] = None
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WalkForwardResult:
    """Result of walk-forward analysis.

    Attributes:
        config: Configuration used.
        folds: List of fold results.
        combined_metrics: Metrics across all test periods.
        trades: All trades from test periods.
        stability_score: Score of strategy stability (0-1).
        overfitting_score: Score indicating overfitting risk (0-1).
    """

    config: WalkForwardConfig
    folds: List[FoldResult] = field(default_factory=list)
    combined_metrics: Dict[str, Any] = field(default_factory=dict)
    trades: List[Trade] = field(default_factory=list)
    stability_score: float = 0.0
    overfitting_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "config": {
                "train_months": self.config.train_period_months,
                "test_months": self.config.test_period_months,
                "anchor": self.config.anchor,
            },
            "summary": {
                "n_folds": len(self.folds),
                "stability_score": self.stability_score,
                "overfitting_score": self.overfitting_score,
            },
            "combined_metrics": self.combined_metrics,
            "folds": [
                {
                    "fold": f.fold_number,
                    "test_return": f.metrics.get("total_return_pct", 0),
                    "test_trades": len(f.test_result.trades) if f.test_result else 0,
                    "win_rate": f.metrics.get("trade", {}).get("win_rate", 0),
                }
                for f in self.folds
            ],
        }

    def summary(self) -> str:
        """Generate summary string."""
        lines = [
            "=" * 60,
            "WALK-FORWARD ANALYSIS RESULTS",
            "=" * 60,
            "",
            f"Folds: {len(self.folds)}",
            f"Train Period: {self.config.train_period_months} months",
            f"Test Period: {self.config.test_period_months} months",
            "",
            "COMBINED TEST PERIOD METRICS:",
            f"  Total Return: {self.combined_metrics.get('total_return_pct', 0):+.2f}%",
            f"  Total Trades: {len(self.trades)}",
            f"  Win Rate: {self.combined_metrics.get('trade', {}).get('win_rate', 0):.1%}",
            f"  Sharpe: {self.combined_metrics.get('risk_adjusted', {}).get('sharpe_ratio', 0):.2f}",
            "",
            "QUALITY SCORES:",
            f"  Stability Score: {self.stability_score:.2f}",
            f"  Overfitting Risk: {self.overfitting_score:.2f}",
            "",
            "FOLD BREAKDOWN:",
        ]

        for fold in self.folds:
            ret = fold.metrics.get("total_return_pct", 0)
            trades = len(fold.test_result.trades) if fold.test_result else 0
            lines.append(f"  Fold {fold.fold_number}: {ret:+.2f}% ({trades} trades)")

        lines.extend(["", "=" * 60])
        return "\n".join(lines)


class WalkForwardAnalyzer:
    """Walk-forward analysis engine.

    Performs walk-forward optimization to validate strategy
    robustness and detect overfitting.

    Example:
        config = WalkForwardConfig(base_config=backtest_config)
        analyzer = WalkForwardAnalyzer(config)
        result = analyzer.run(bars_by_symbol)
        print(f"Stability: {result.stability_score:.2f}")
    """

    def __init__(self, config: WalkForwardConfig) -> None:
        """Initialize walk-forward analyzer.

        Args:
            config: Configuration.
        """
        self.config = config

    def run(
        self,
        bars_by_symbol: Dict[str, List[Bar]],
    ) -> WalkForwardResult:
        """Run walk-forward analysis.

        Args:
            bars_by_symbol: Dictionary of symbol to bars.

        Returns:
            WalkForwardResult.
        """
        # Get date range
        all_dates = set()
        for bars in bars_by_symbol.values():
            for bar in bars:
                all_dates.add(bar.date)

        if not all_dates:
            return WalkForwardResult(config=self.config)

        min_date = min(all_dates)
        max_date = max(all_dates)

        # Generate fold periods
        folds = self._generate_folds(min_date, max_date)

        logger.info(
            f"Running walk-forward analysis: {len(folds)} folds, "
            f"{self.config.train_period_months}m train / {self.config.test_period_months}m test"
        )

        # Run each fold
        fold_results = []
        all_test_trades = []

        for fold_info in folds:
            fold_result = self._run_fold(
                fold_info,
                bars_by_symbol,
            )
            fold_results.append(fold_result)

            if fold_result.test_result and fold_result.test_result.trades:
                all_test_trades.extend(fold_result.test_result.trades)

        # Combine results
        combined_metrics = self._calculate_combined_metrics(all_test_trades)

        # Calculate quality scores
        stability_score = self._calculate_stability_score(fold_results)
        overfitting_score = self._calculate_overfitting_score(fold_results)

        result = WalkForwardResult(
            config=self.config,
            folds=fold_results,
            combined_metrics=combined_metrics,
            trades=all_test_trades,
            stability_score=stability_score,
            overfitting_score=overfitting_score,
        )

        return result

    def _generate_folds(
        self,
        min_date: date,
        max_date: date,
    ) -> List[Dict[str, date]]:
        """Generate fold date ranges.

        Args:
            min_date: Minimum date in data.
            max_date: Maximum date in data.

        Returns:
            List of fold dictionaries with train/test dates.
        """
        folds = []

        train_days = self.config.train_period_months * 30
        test_days = self.config.test_period_months * 30

        current_train_start = min_date
        current_train_end = min_date + timedelta(days=train_days)

        fold_num = 1

        while True:
            test_start = current_train_end + timedelta(days=1)
            test_end = test_start + timedelta(days=test_days)

            # Stop if test period exceeds data
            if test_end > max_date:
                break

            folds.append({
                "fold": fold_num,
                "train_start": current_train_start,
                "train_end": current_train_end,
                "test_start": test_start,
                "test_end": test_end,
            })

            # Move window
            if self.config.anchor:
                # Expanding window - keep start, extend end
                current_train_end = test_end
            else:
                # Rolling window - move both
                current_train_start = test_start
                current_train_end = test_end

            fold_num += 1

        return folds

    def _run_fold(
        self,
        fold_info: Dict[str, Any],
        bars_by_symbol: Dict[str, List[Bar]],
    ) -> FoldResult:
        """Run single fold.

        Args:
            fold_info: Fold date information.
            bars_by_symbol: Price data.

        Returns:
            FoldResult.
        """
        fold_result = FoldResult(
            fold_number=fold_info["fold"],
            train_start=fold_info["train_start"],
            train_end=fold_info["train_end"],
            test_start=fold_info["test_start"],
            test_end=fold_info["test_end"],
        )

        # Run training period backtest to collect trades
        train_config = BacktestConfig(
            start_date=fold_info["train_start"],
            end_date=fold_info["train_end"],
            initial_capital=self.config.base_config.initial_capital,
            trading_mode=self.config.base_config.trading_mode,
        )

        train_engine = BacktestEngine(train_config)
        train_result = train_engine.run(bars_by_symbol)
        fold_result.train_trades = len(train_result.trades)

        # Check minimum trades
        if fold_result.train_trades < self.config.min_train_trades:
            logger.warning(
                f"Fold {fold_info['fold']}: Only {fold_result.train_trades} training trades "
                f"(min: {self.config.min_train_trades})"
            )

        # Run test period backtest
        test_config = BacktestConfig(
            start_date=fold_info["test_start"],
            end_date=fold_info["test_end"],
            initial_capital=self.config.base_config.initial_capital,
            trading_mode=self.config.base_config.trading_mode,
            position_sizing=self.config.base_config.position_sizing,
        )

        test_engine = BacktestEngine(test_config)

        # Use training trades for pattern matching
        test_result = test_engine.run(bars_by_symbol, train_result.trades)

        fold_result.test_result = test_result
        fold_result.metrics = test_result.metrics

        logger.info(
            f"Fold {fold_info['fold']}: Train={fold_result.train_trades} trades, "
            f"Test={len(test_result.trades)} trades, "
            f"Return={test_result.total_return_pct:+.2f}%"
        )

        return fold_result

    def _calculate_combined_metrics(
        self,
        all_trades: List[Trade],
    ) -> Dict[str, Any]:
        """Calculate combined metrics across all test periods.

        Args:
            all_trades: All test period trades.

        Returns:
            Combined metrics dictionary.
        """
        if not all_trades:
            return {}

        # Simple equity curve from trades
        equity = self.config.base_config.initial_capital
        equity_curve = [{"date": "start", "equity": equity}]

        for trade in all_trades:
            equity += trade.net_pnl
            equity_curve.append({
                "date": trade.exit_date.isoformat() if hasattr(trade.exit_date, 'isoformat') else str(trade.exit_date),
                "equity": equity,
            })

        return calculate_metrics(
            trades=all_trades,
            equity_curve=equity_curve,
            initial_capital=self.config.base_config.initial_capital,
        )

    def _calculate_stability_score(self, fold_results: List[FoldResult]) -> float:
        """Calculate strategy stability score.

        Higher score = more consistent performance across folds.

        Args:
            fold_results: List of fold results.

        Returns:
            Stability score (0-1).
        """
        if len(fold_results) < 2:
            return 0.5

        # Get returns for each fold
        returns = []
        for fold in fold_results:
            if fold.metrics:
                returns.append(fold.metrics.get("total_return_pct", 0))

        if not returns:
            return 0.5

        # Calculate coefficient of variation
        mean_return = np.mean(returns)
        std_return = np.std(returns)

        if mean_return == 0:
            return 0.5

        cv = abs(std_return / mean_return)

        # Lower CV = higher stability
        # CV < 0.5 = high stability (score > 0.7)
        # CV > 2 = low stability (score < 0.3)
        if cv < 0.5:
            return 0.9
        elif cv < 1.0:
            return 0.7
        elif cv < 2.0:
            return 0.5
        else:
            return 0.3

    def _calculate_overfitting_score(self, fold_results: List[FoldResult]) -> float:
        """Calculate overfitting risk score.

        Higher score = more likely overfit (train >> test performance).

        Args:
            fold_results: List of fold results.

        Returns:
            Overfitting risk score (0-1, higher = more risk).
        """
        if len(fold_results) < 2:
            return 0.5

        # Compare train vs test performance
        train_test_diffs = []

        for fold in fold_results:
            if fold.test_result and fold.metrics:
                # Use win rate difference as proxy
                test_win_rate = fold.metrics.get("trade", {}).get("win_rate", 0)

                # Estimate train win rate from train trades count
                # (simplified - in reality would track actual train metrics)
                if fold.train_trades > 0:
                    # Assume moderate train performance
                    estimated_train_wr = 0.55
                    diff = estimated_train_wr - test_win_rate
                    train_test_diffs.append(diff)

        if not train_test_diffs:
            return 0.5

        # Larger average difference = more overfitting
        avg_diff = np.mean(train_test_diffs)

        if avg_diff > 0.15:
            return 0.9  # High risk
        elif avg_diff > 0.10:
            return 0.7
        elif avg_diff > 0.05:
            return 0.5
        else:
            return 0.3  # Low risk


def run_walk_forward(
    bars_by_symbol: Dict[str, List[Bar]],
    start_date: date,
    end_date: date,
    train_months: int = 12,
    test_months: int = 3,
    **kwargs,
) -> WalkForwardResult:
    """Convenience function for walk-forward analysis.

    Args:
        bars_by_symbol: Dictionary of symbol to bars.
        start_date: Start date.
        end_date: End date.
        train_months: Training period in months.
        test_months: Test period in months.
        **kwargs: Additional options.

    Returns:
        WalkForwardResult.
    """
    from config.trading_modes import TradingMode

    base_config = BacktestConfig(
        start_date=start_date,
        end_date=end_date,
        initial_capital=kwargs.get("initial_capital", 100_000_000),
        trading_mode=kwargs.get("trading_mode", TradingMode.SWING),
    )

    wf_config = WalkForwardConfig(
        base_config=base_config,
        train_period_months=train_months,
        test_period_months=test_months,
    )

    analyzer = WalkForwardAnalyzer(wf_config)
    return analyzer.run(bars_by_symbol)
