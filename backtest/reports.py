"""
Backtest Reports Module

Generates comprehensive backtest reports including:
- Standard metrics section
- Monte Carlo analysis section
- Calibration surface section
- Execution quality section
- Position sizing comparison section
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Optional, Any

from backtest.metrics import PerformanceMetrics

logger = logging.getLogger(__name__)


@dataclass
class ReportConfig:
    """Configuration for report generation.

    Attributes:
        output_dir: Directory for output files.
        include_trades: Include trade list in report.
        include_equity_curve: Include equity curve data.
        include_daily_returns: Include daily returns.
        decimal_places: Number of decimal places for numbers.
    """

    output_dir: str = "reports"
    include_trades: bool = True
    include_equity_curve: bool = False
    include_daily_returns: bool = False
    decimal_places: int = 2


class BacktestReport:
    """Generates comprehensive backtest reports.

    Example:
        report = BacktestReport(config)
        report.generate(backtest_result, "my_backtest")
    """

    def __init__(self, config: Optional[ReportConfig] = None) -> None:
        """Initialize report generator.

        Args:
            config: Report configuration.
        """
        self.config = config or ReportConfig()
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        result: Dict[str, Any],
        name: str,
        include_mc: bool = False,
        mc_data: Optional[Dict] = None,
        include_calibration: bool = False,
        calibration_data: Optional[Dict] = None,
    ) -> str:
        """Generate comprehensive backtest report.

        Args:
            result: Backtest result dictionary.
            name: Report name.
            include_mc: Include Monte Carlo section.
            mc_data: Monte Carlo analysis data.
            include_calibration: Include calibration section.
            calibration_data: Calibration surface data.

        Returns:
            Report content as markdown string.
        """
        sections = []

        # Header
        sections.append(self._generate_header(name, result))

        # Summary
        sections.append(self._generate_summary_section(result))

        # Performance Metrics
        sections.append(self._generate_metrics_section(result))

        # Trade Analysis
        if self.config.include_trades and "trades" in result:
            sections.append(self._generate_trades_section(result))

        # Monte Carlo Section
        if include_mc and mc_data:
            sections.append(self._generate_mc_section(mc_data))

        # Calibration Section
        if include_calibration and calibration_data:
            sections.append(self._generate_calibration_section(calibration_data))

        # Risk Analysis
        sections.append(self._generate_risk_section(result))

        # Recommendations
        sections.append(self._generate_recommendations(result, mc_data))

        # Footer
        sections.append(self._generate_footer())

        report_content = "\n\n".join(sections)

        # Save to files
        self._save_report(report_content, name, result)

        return report_content

    def _generate_header(self, name: str, result: Dict) -> str:
        """Generate report header."""
        config = result.get("config", {})
        return f"""# Backtest Report: {name}

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Configuration

| Parameter | Value |
|-----------|-------|
| Start Date | {config.get('start_date', 'N/A')} |
| End Date | {config.get('end_date', 'N/A')} |
| Initial Capital | {config.get('initial_capital', 0):,.0f} IDR |
| Trading Mode | {config.get('trading_mode', 'N/A')} |
| Position Sizing | {config.get('position_sizing', 'N/A')} |
"""

    def _generate_summary_section(self, result: Dict) -> str:
        """Generate summary section."""
        metrics = result.get("metrics", {})
        trade = metrics.get("trade", {})
        drawdown = metrics.get("drawdown", {})
        risk = metrics.get("risk_adjusted", {})

        return f"""## Summary

### Returns
| Metric | Value |
|--------|-------|
| Total Return | {metrics.get('total_return_pct', 0):+.2f}% |
| CAGR | {metrics.get('cagr', 0):+.2f}% |
| Final Capital | {metrics.get('final_capital', 0):,.0f} IDR |

### Trade Statistics
| Metric | Value |
|--------|-------|
| Total Trades | {trade.get('total_trades', 0)} |
| Win Rate | {trade.get('win_rate', 0):.1%} |
| Profit Factor | {trade.get('profit_factor', 0):.2f} |
| Avg Trade | {trade.get('avg_trade', 0):+.2f}% |

### Risk Metrics
| Metric | Value |
|--------|-------|
| Max Drawdown | {drawdown.get('max_drawdown_pct', 0):.2f}% |
| Sharpe Ratio | {risk.get('sharpe_ratio', 0):.2f} |
| Sortino Ratio | {risk.get('sortino_ratio', 0):.2f} |
| Calmar Ratio | {risk.get('calmar_ratio', 0):.2f} |
"""

    def _generate_metrics_section(self, result: Dict) -> str:
        """Generate detailed metrics section."""
        metrics = result.get("metrics", {})
        trade = metrics.get("trade", {})
        drawdown = metrics.get("drawdown", {})
        risk = metrics.get("risk_adjusted", {})

        return f"""## Detailed Metrics

### Trade Analysis
| Metric | Value |
|--------|-------|
| Total Trades | {trade.get('total_trades', 0)} |
| Winning Trades | {result.get('winning_trades', 0)} |
| Losing Trades | {result.get('losing_trades', 0)} |
| Win Rate | {trade.get('win_rate', 0):.1%} |
| Avg Win | {trade.get('avg_win', 0):+.2f}% |
| Avg Loss | {trade.get('avg_loss', 0):+.2f}% |
| Largest Win | {trade.get('largest_win', 0):+.2f}% |
| Largest Loss | {trade.get('largest_loss', 0):+.2f}% |
| Profit Factor | {trade.get('profit_factor', 0):.2f} |
| Expectancy | {trade.get('expectancy', 0):+.2f}% |
| Avg Holding Days | {trade.get('avg_holding_days', 0):.1f} |

### Drawdown Analysis
| Metric | Value |
|--------|-------|
| Max Drawdown | {drawdown.get('max_drawdown_pct', 0):.2f}% |
| Avg Drawdown | {drawdown.get('avg_drawdown', 0):.2f}% |
| Max Duration | {drawdown.get('max_drawdown_duration', 0)} days |
| Drawdown Periods | {drawdown.get('drawdown_periods', 0)} |
| Recovery Factor | {drawdown.get('recovery_factor', 0):.2f} |
| Ulcer Index | {drawdown.get('ulcer_index', 0):.2f} |

### Risk-Adjusted Returns
| Metric | Value |
|--------|-------|
| Annual Return | {risk.get('annual_return', 0):+.2f}% |
| Annual Volatility | {risk.get('annual_volatility', 0):.2f}% |
| Sharpe Ratio | {risk.get('sharpe_ratio', 0):.2f} |
| Sortino Ratio | {risk.get('sortino_ratio', 0):.2f} |
| Calmar Ratio | {risk.get('calmar_ratio', 0):.2f} |
"""

    def _generate_trades_section(self, result: Dict) -> str:
        """Generate trades section."""
        trades = result.get("trades", [])

        if not trades:
            return "## Trade Log\n\nNo trades executed."

        lines = ["## Trade Log\n"]
        lines.append("| # | Symbol | Entry | Exit | Days | Return | P&L | Reason |")
        lines.append("|---|--------|-------|------|------|--------|-----|--------|")

        for i, trade in enumerate(trades[:50], 1):  # Limit to 50 trades
            lines.append(
                f"| {i} | {trade.get('symbol', 'N/A')} | "
                f"{trade.get('entry_price', 0):,.0f} | "
                f"{trade.get('exit_price', 0):,.0f} | "
                f"{trade.get('holding_days', 0)} | "
                f"{trade.get('return_pct', 0):+.2f}% | "
                f"{trade.get('net_pnl', 0):+,.0f} | "
                f"{trade.get('exit_reason', 'N/A')} |"
            )

        if len(trades) > 50:
            lines.append(f"\n*... and {len(trades) - 50} more trades*")

        return "\n".join(lines)

    def _generate_mc_section(self, mc_data: Dict) -> str:
        """Generate Monte Carlo analysis section."""
        dd_dist = mc_data.get("drawdown_distribution", {})

        return f"""## Monte Carlo Analysis

### Simulation Parameters
| Parameter | Value |
|-----------|-------|
| Simulations | {mc_data.get('n_simulations', 0):,} |
| Backtest DD | {mc_data.get('backtest_max_dd', 0):.2%} |

### Drawdown Distribution
| Percentile | Drawdown |
|------------|----------|
| 50th (Median) | {dd_dist.get('median', 0):.2%} |
| 75th | {dd_dist.get('p75', 0):.2%} |
| 90th | {dd_dist.get('p90', 0):.2f}% |
| 95th | {dd_dist.get('p95', 0):.2%} ← Institutional Target |
| 99th | {dd_dist.get('p99', 0):.2%} |

### Risk Assessment
| Metric | Value |
|--------|-------|
| Backtest Percentile | {mc_data.get('backtest_percentile', 0):.0f}th |
| Safety Margin | {mc_data.get('safety_margin', 0):.2f}x |
| Sizing Multiplier | {mc_data.get('sizing_multiplier', 1.0):.2f} |
"""

    def _generate_calibration_section(self, cal_data: Dict) -> str:
        """Generate calibration surface section."""
        return f"""## Calibration Surface

### Edge Decay by Score
{self._format_calibration_table(cal_data)}

### Optimal Exit Days
| Score Range | Optimal Exit |
|-------------|--------------|
| 50-60 | Day {cal_data.get('optimal_exit_50', 3)} |
| 60-70 | Day {cal_data.get('optimal_exit_60', 4)} |
| 70-80 | Day {cal_data.get('optimal_exit_70', 5)} |
| 80-90 | Day {cal_data.get('optimal_exit_80', 5)} |
| 90-100 | Day {cal_data.get('optimal_exit_90', 7)} |
"""

    def _format_calibration_table(self, cal_data: Dict) -> str:
        """Format calibration data as table."""
        cells = cal_data.get("cells", [])

        if not cells:
            return "*No calibration data available*"

        lines = ["| Score \\ Day | D1 | D2 | D3 | D4 | D5 | D6 | D7 |"]
        lines.append("|-------------|----|----|----|----|----|----|----|")

        # Group by score bin
        for score_bin in ["50-60", "60-70", "70-80", "80-90", "90-100"]:
            row = [score_bin]
            for day in range(1, 8):
                # Find cell for this bin/day
                win_rate = 0.50  # Default
                for cell in cells:
                    if cell.get("score_range") == score_bin and cell.get("day") == day:
                        win_rate = cell.get("win_rate", 0.50)
                        break
                row.append(f"{win_rate:.0%}")
            lines.append("| " + " | ".join(row) + " |")

        return "\n".join(lines)

    def _generate_risk_section(self, result: Dict) -> str:
        """Generate risk analysis section."""
        metrics = result.get("metrics", {})
        drawdown = metrics.get("drawdown", {})

        risk_level = "LOW"
        max_dd = drawdown.get("max_drawdown_pct", 0)
        if max_dd >= 30:
            risk_level = "EXTREME"
        elif max_dd >= 20:
            risk_level = "HIGH"
        elif max_dd >= 10:
            risk_level = "MEDIUM"

        return f"""## Risk Analysis

### Risk Level: {risk_level}

| Assessment | Status |
|------------|--------|
| Max DD < 10% | {"✅" if max_dd < 10 else "❌"} |
| Max DD < 20% | {"✅" if max_dd < 20 else "❌"} |
| Max DD < 30% | {"✅" if max_dd < 30 else "❌"} |
| Recovery Factor > 2 | {"✅" if drawdown.get('recovery_factor', 0) > 2 else "❌"} |

### Warnings
{self._generate_warnings(result)}
"""

    def _generate_warnings(self, result: Dict) -> str:
        """Generate warnings based on metrics."""
        warnings = []
        metrics = result.get("metrics", {})
        trade = metrics.get("trade", {})
        drawdown = metrics.get("drawdown", {})

        if trade.get("total_trades", 0) < 30:
            warnings.append("- ⚠️ **Low trade count** - Results may not be statistically significant")

        if trade.get("win_rate", 0) < 0.4:
            warnings.append("- ⚠️ **Low win rate** - Strategy may need refinement")

        if drawdown.get("max_drawdown_pct", 0) > 25:
            warnings.append("- ⚠️ **High drawdown** - Consider reducing position sizes")

        if trade.get("profit_factor", 0) < 1.0:
            warnings.append("- ⚠️ **Negative expectancy** - Strategy loses money on average")

        if not warnings:
            warnings.append("- No significant warnings")

        return "\n".join(warnings)

    def _generate_recommendations(
        self,
        result: Dict,
        mc_data: Optional[Dict] = None,
    ) -> str:
        """Generate recommendations section."""
        recommendations = []

        metrics = result.get("metrics", {})
        drawdown = metrics.get("drawdown", {})

        # Position sizing
        if mc_data:
            multiplier = mc_data.get("sizing_multiplier", 1.0)
            if multiplier < 1.0:
                recommendations.append(
                    f"1. **Reduce position sizes by {(1 - multiplier) * 100:.0f}%** "
                    f"based on Monte Carlo analysis"
                )
            else:
                recommendations.append(
                    "1. Current position sizing appears appropriate"
                )
        else:
            recommendations.append(
                "1. Run Monte Carlo analysis for position sizing recommendations"
            )

        # Drawdown
        max_dd = drawdown.get("max_drawdown_pct", 0)
        if max_dd > 20:
            recommendations.append(
                "2. **High drawdown detected** - Consider tighter risk management"
            )
        elif max_dd > 10:
            recommendations.append(
                "2. Moderate drawdown - Monitor risk closely"
            )

        # Trade frequency
        trade = metrics.get("trade", {})
        if trade.get("total_trades", 0) < 50:
            recommendations.append(
                "3. Limited trade sample - Extend backtest period if possible"
            )

        return f"""## Recommendations

{chr(10).join(recommendations)}
"""

    def _generate_footer(self) -> str:
        """Generate report footer."""
        return f"""---

*Report generated by IDX Trading System Backtest Engine*
*Timestamp: {datetime.now().isoformat()}*
"""

    def _save_report(
        self,
        content: str,
        name: str,
        result: Dict,
    ) -> None:
        """Save report to files.

        Args:
            content: Markdown content.
            name: Report name.
            result: Full result for JSON export.
        """
        # Save markdown
        md_path = self.output_dir / f"{name}.md"
        with open(md_path, "w") as f:
            f.write(content)

        # Save JSON
        json_path = self.output_dir / f"{name}.json"
        with open(json_path, "w") as f:
            json.dump(result, f, indent=2, default=str)

        logger.info(f"Reports saved to {self.output_dir}")


def generate_report(
    result: Dict[str, Any],
    name: str = "backtest",
    output_dir: str = "reports",
    **kwargs,
) -> str:
    """Convenience function to generate a report.

    Args:
        result: Backtest result dictionary.
        name: Report name.
        output_dir: Output directory.
        **kwargs: Additional options.

    Returns:
        Report content as markdown string.
    """
    config = ReportConfig(output_dir=output_dir)
    report = BacktestReport(config)
    return report.generate(result, name, **kwargs)
