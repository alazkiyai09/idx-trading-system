"""
Monte Carlo Backtest Module

Wraps standard backtest with Monte Carlo simulation to:
- Run 10,000 resampled paths
- Calculate drawdown distribution
- Compare sizing strategies
- Generate enhanced reports
"""

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum

import numpy as np

from backtest.engine import BacktestEngine, BacktestConfig, BacktestResult
from research.monte_carlo import MonteCarloEngine, MonteCarloResult
from research.drawdown_analysis import DrawdownAnalyzer, DrawdownProfile
from core.risk.empirical_kelly import EmpiricalKelly
from core.data.models import Trade

logger = logging.getLogger(__name__)


@dataclass
class MonteCarloBacktestConfig:
    """Configuration for Monte Carlo backtest.

    Attributes:
        base_config: Base backtest configuration.
        n_simulations: Number of MC simulations.
        max_acceptable_dd: Maximum acceptable drawdown.
        compare_sizing: Whether to compare sizing strategies.
    """

    base_config: BacktestConfig
    n_simulations: int = 10_000
    max_acceptable_dd: float = 0.20
    compare_sizing: bool = True
    seed: Optional[int] = 42


@dataclass
class SizingComparison:
    """Result of sizing strategy comparison.

    Attributes:
        strategy_name: Name of sizing strategy.
        final_capital: Final capital after simulation.
        total_return: Total return percentage.
        max_drawdown: Maximum drawdown.
        sharpe_ratio: Sharpe ratio.
        win_rate: Win rate.
        profit_factor: Profit factor.
    """

    strategy_name: str
    final_capital: float = 0.0
    total_return: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0


@dataclass
class MonteCarloBacktestResult:
    """Result of Monte Carlo enhanced backtest.

    Attributes:
        base_result: Standard backtest result.
        mc_result: Monte Carlo simulation result.
        drawdown_profile: Drawdown analysis profile.
        sizing_comparison: Comparison of sizing strategies.
        recommendations: List of recommendations.
    """

    base_result: BacktestResult
    mc_result: Optional[MonteCarloResult] = None
    drawdown_profile: Optional[DrawdownProfile] = None
    sizing_comparison: List[SizingComparison] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "base_result": self.base_result.to_dict(),
            "mc_analysis": {
                "n_simulations": self.mc_result.n_simulations if self.mc_result else 0,
                "backtest_max_dd": self.mc_result.backtest_max_dd if self.mc_result else 0,
                "drawdown_distribution": {
                    "median": self.mc_result.drawdown_distribution.median if self.mc_result and self.mc_result.drawdown_distribution else 0,
                    "p95": self.mc_result.drawdown_distribution.p95 if self.mc_result and self.mc_result.drawdown_distribution else 0,
                    "p99": self.mc_result.drawdown_distribution.p99 if self.mc_result and self.mc_result.drawdown_distribution else 0,
                } if self.mc_result and self.mc_result.drawdown_distribution else {},
                "backtest_percentile": self.mc_result.backtest_percentile if self.mc_result else 50,
                "sizing_multiplier": self.mc_result.get_sizing_multiplier() if self.mc_result else 1.0,
            },
            "drawdown_profile": {
                "risk_level": self.drawdown_profile.risk_level if self.drawdown_profile else "UNKNOWN",
                "safety_margin": self.drawdown_profile.safety_margin if self.drawdown_profile else 1.0,
                "sizing_recommendation": self.drawdown_profile.sizing_recommendation if self.drawdown_profile else 1.0,
            } if self.drawdown_profile else {},
            "sizing_comparison": [
                {
                    "strategy": s.strategy_name,
                    "return": s.total_return,
                    "max_dd": s.max_drawdown,
                    "sharpe": s.sharpe_ratio,
                }
                for s in self.sizing_comparison
            ],
            "recommendations": self.recommendations,
        }


class MonteCarloBacktest:
    """Monte Carlo enhanced backtest engine.

    Runs standard backtest followed by Monte Carlo simulation
    to understand distribution of possible outcomes.

    Example:
        config = MonteCarloBacktestConfig(base_config=backtest_config)
        mc_backtest = MonteCarloBacktest(config)
        result = mc_backtest.run(bars_by_symbol)
        print(f"95th percentile DD: {result.mc_result.drawdown_distribution.p95:.2%}")
    """

    def __init__(self, config: MonteCarloBacktestConfig) -> None:
        """Initialize Monte Carlo backtest.

        Args:
            config: Configuration.
        """
        self.config = config
        self.mc_engine = MonteCarloEngine(
            n_simulations=config.n_simulations,
            seed=config.seed,
        )
        self.dd_analyzer = DrawdownAnalyzer(
            max_acceptable_dd=config.max_acceptable_dd,
        )

    def run(
        self,
        bars_by_symbol: Dict[str, List],
        historical_trades: Optional[List[Trade]] = None,
    ) -> MonteCarloBacktestResult:
        """Run Monte Carlo enhanced backtest.

        Args:
            bars_by_symbol: Dictionary of symbol to bars.
            historical_trades: Historical trades for pattern matching.

        Returns:
            MonteCarloBacktestResult with enhanced analysis.
        """
        # Run standard backtest
        logger.info("Running standard backtest...")
        engine = BacktestEngine(self.config.base_config)
        base_result = engine.run(bars_by_symbol, historical_trades)

        # Run Monte Carlo simulation
        logger.info(f"Running Monte Carlo simulation ({self.config.n_simulations:,} paths)...")
        mc_result = self._run_monte_carlo(base_result)

        # Analyze drawdown
        logger.info("Analyzing drawdown profile...")
        drawdown_profile = self._analyze_drawdown(mc_result)

        # Compare sizing strategies
        sizing_comparison = []
        if self.config.compare_sizing:
            logger.info("Comparing sizing strategies...")
            sizing_comparison = self._compare_sizing_strategies(
                base_result,
                bars_by_symbol,
                historical_trades,
            )

        # Generate recommendations
        recommendations = self._generate_recommendations(
            base_result,
            mc_result,
            drawdown_profile,
        )

        return MonteCarloBacktestResult(
            base_result=base_result,
            mc_result=mc_result,
            drawdown_profile=drawdown_profile,
            sizing_comparison=sizing_comparison,
            recommendations=recommendations,
        )

    def _run_monte_carlo(self, base_result: BacktestResult) -> MonteCarloResult:
        """Run Monte Carlo simulation on backtest returns.

        Args:
            base_result: Standard backtest result.

        Returns:
            MonteCarloResult.
        """
        # Extract returns from trades
        if not base_result.trades:
            return MonteCarloResult(
                n_simulations=0,
                paths=[],
                drawdown_distribution=None,
                backtest_max_dd=0,
            )

        returns = [trade.return_pct for trade in base_result.trades]

        # Run simulation
        mc_result = self.mc_engine.simulate(returns)

        # Calculate backtest max drawdown
        if base_result.equity_curve:
            equity_values = [e["equity"] for e in base_result.equity_curve]
            running_max = np.maximum.accumulate(equity_values)
            drawdowns = (running_max - equity_values) / running_max
            mc_result.backtest_max_dd = float(np.max(drawdowns))

        return mc_result

    def _analyze_drawdown(self, mc_result: MonteCarloResult) -> DrawdownProfile:
        """Analyze drawdown from Monte Carlo result.

        Args:
            mc_result: Monte Carlo simulation result.

        Returns:
            DrawdownProfile.
        """
        return self.dd_analyzer.analyze(mc_result)

    def _compare_sizing_strategies(
        self,
        base_result: BacktestResult,
        bars_by_symbol: Dict[str, List],
        historical_trades: Optional[List[Trade]],
    ) -> List[SizingComparison]:
        """Compare different position sizing strategies.

        Args:
            base_result: Base backtest result.
            bars_by_symbol: Price data.
            historical_trades: Historical trades.

        Returns:
            List of SizingComparison results.
        """
        comparisons = []
        strategies = ["fixed", "kelly", "empirical_kelly"]

        for strategy in strategies:
            try:
                # Create config with different sizing
                config = BacktestConfig(
                    start_date=self.config.base_config.start_date,
                    end_date=self.config.base_config.end_date,
                    initial_capital=self.config.base_config.initial_capital,
                    trading_mode=self.config.base_config.trading_mode,
                    position_sizing=strategy,
                )

                # Run backtest
                engine = BacktestEngine(config)
                result = engine.run(bars_by_symbol, historical_trades)

                # Extract metrics
                comparison = SizingComparison(
                    strategy_name=strategy,
                    final_capital=result.final_capital,
                    total_return=result.total_return_pct,
                    max_drawdown=result.metrics.get("drawdown", {}).get("max_drawdown_pct", 0),
                    sharpe_ratio=result.metrics.get("risk_adjusted", {}).get("sharpe_ratio", 0),
                    win_rate=result.metrics.get("trade", {}).get("win_rate", 0),
                    profit_factor=result.metrics.get("trade", {}).get("profit_factor", 0),
                )
                comparisons.append(comparison)

            except Exception as e:
                logger.warning(f"Failed to run backtest with {strategy} sizing: {e}")

        return comparisons

    def _generate_recommendations(
        self,
        base_result: BacktestResult,
        mc_result: Optional[MonteCarloResult],
        drawdown_profile: Optional[DrawdownProfile],
    ) -> List[str]:
        """Generate recommendations based on analysis.

        Args:
            base_result: Base backtest result.
            mc_result: Monte Carlo result.
            drawdown_profile: Drawdown profile.

        Returns:
            List of recommendations.
        """
        recommendations = []

        # Trade count
        if len(base_result.trades) < 30:
            recommendations.append(
                "Low trade count - extend backtest period or use more symbols for statistical significance"
            )

        # Drawdown warnings
        if mc_result and mc_result.drawdown_distribution:
            p95_dd = mc_result.drawdown_distribution.p95

            if p95_dd > 0.30:
                recommendations.append(
                    f"CRITICAL: 95th percentile DD ({p95_dd:.1%}) exceeds 30% - "
                    "significantly reduce position sizes"
                )
            elif p95_dd > 0.20:
                recommendations.append(
                    f"WARNING: 95th percentile DD ({p95_dd:.1%}) exceeds 20% - "
                    "consider reducing position sizes"
                )

        # Sizing recommendation
        if drawdown_profile and drawdown_profile.sizing_recommendation < 1.0:
            reduction = (1 - drawdown_profile.sizing_recommendation) * 100
            recommendations.append(
                f"Reduce position sizes by {reduction:.0f}% to target 20% max DD"
            )

        # Strategy quality
        win_rate = base_result.metrics.get("trade", {}).get("win_rate", 0)
        profit_factor = base_result.metrics.get("trade", {}).get("profit_factor", 0)

        if win_rate < 0.4:
            recommendations.append(
                "Low win rate - review entry criteria and signal quality"
            )

        if profit_factor < 1.5:
            recommendations.append(
                "Low profit factor - improve risk/reward ratio or cut losers faster"
            )

        if not recommendations:
            recommendations.append(
                "Strategy metrics look healthy - continue monitoring in live trading"
            )

        return recommendations


def run_monte_carlo_backtest(
    bars_by_symbol: Dict[str, List],
    start_date: date,
    end_date: date,
    initial_capital: float = 100_000_000,
    n_simulations: int = 10_000,
    **kwargs,
) -> MonteCarloBacktestResult:
    """Convenience function for Monte Carlo backtest.

    Args:
        bars_by_symbol: Dictionary of symbol to bars.
        start_date: Start date.
        end_date: End date.
        initial_capital: Initial capital.
        n_simulations: Number of MC simulations.
        **kwargs: Additional options.

    Returns:
        MonteCarloBacktestResult.
    """
    from config.trading_modes import TradingMode

    base_config = BacktestConfig(
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
        trading_mode=kwargs.get("trading_mode", TradingMode.SWING),
    )

    mc_config = MonteCarloBacktestConfig(
        base_config=base_config,
        n_simulations=n_simulations,
        max_acceptable_dd=kwargs.get("max_acceptable_dd", 0.20),
    )

    mc_backtest = MonteCarloBacktest(mc_config)
    return mc_backtest.run(bars_by_symbol)
