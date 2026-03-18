"""Run an IMSS backtest simulation.

Usage:
    python3 scripts/imss_run_backtest.py
    python3 scripts/imss_run_backtest.py --start 2024-07-01 --end 2024-09-30
"""

from __future__ import annotations

import argparse
import asyncio
import logging

from rich.console import Console
from rich.table import Table

from imss.config import SimulationConfig
from imss.simulation.engine import SimulationEngine

console = Console()
logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")


def on_step_display(step, sim_date, ohlcv, actions, agents):
    """Rich callback for per-step display."""
    buys = sum(1 for a in actions if a.action == "BUY")
    sells = sum(1 for a in actions if a.action == "SELL")
    holds = sum(1 for a in actions if a.action == "HOLD")
    console.print(
        f"  Step {step:3d} | {sim_date} | Close: {ohlcv['close']:,.0f} | "
        f"Vol: {ohlcv['volume']:>12,} | B:{buys} S:{sells} H:{holds}"
    )


async def main():
    parser = argparse.ArgumentParser(description="IMSS Backtest Runner")
    parser.add_argument("--stock", default="BBRI")
    parser.add_argument("--start", default="2024-07-01")
    parser.add_argument("--end", default="2024-09-30")
    parser.add_argument("--tier2-count", type=int, default=4)
    parser.add_argument("--tier3-count", type=int, default=50)
    parser.add_argument("--runs", type=int, default=1, help="Number of simulation runs")
    args = parser.parse_args()

    config = SimulationConfig(
        target_stocks=[args.stock],
        backtest_start=args.start,
        backtest_end=args.end,
        tier2_per_archetype=args.tier2_count,
        tier3_total=args.tier3_count,
        num_parallel_runs=args.runs,
    )

    console.print(f"[bold blue]IMSS Backtest: {args.stock} from {args.start} to {args.end}[/]")
    console.print(f"Agents: {len(config.tier1_personas)} T1 + {args.tier2_count * len(config.tier2_archetypes)} T2 + {args.tier3_count} T3")
    console.print()

    engine = SimulationEngine()

    if args.runs > 1:
        from imss.simulation.aggregator import MultiRunResult
        console.print(f"Running {args.runs} simulation runs...")
        multi_result = await engine.run_multi(config, on_step=on_step_display)

        console.print()
        console.print(f"[bold green]Multi-Run Complete ({multi_result.num_runs} runs)[/]")
        console.print(f"Total batch cost: ${multi_result.total_batch_cost_usd:.4f}")
        console.print(f"Mean LLM calls/run: {multi_result.mean_total_llm_calls:.0f}")

        table = Table(title="Agent Statistics (Across Runs)")
        table.add_column("Agent Type", style="cyan")
        table.add_column("Samples")
        table.add_column("Mean Cash", justify="right")
        table.add_column("Std Cash", justify="right")
        table.add_column("Mean P&L%", justify="right")
        table.add_column("Std P&L%", justify="right")
        for persona_type, stats in sorted(multi_result.agent_stats.items()):
            table.add_row(
                persona_type, str(stats.num_samples),
                f"IDR {stats.mean_final_cash:,.0f}",
                f"IDR {stats.std_final_cash:,.0f}",
                f"{stats.mean_pnl_pct:+.2f}%",
                f"{stats.std_pnl_pct:.2f}%",
            )
        console.print(table)
    else:
        result = await engine.run_single(config, on_step=on_step_display)

        console.print()
        console.print(f"[bold green]Simulation {result.status}[/]")
        console.print(f"Steps: {result.step_count} | LLM calls: {result.total_llm_calls} | Tokens: {result.total_tokens_used:,}")
        console.print(f"Cost: ${result.estimated_cost_usd:.4f} | JSON parse rate: {result.json_parse_success_rate:.1%}")

        table = Table(title="Agent Results")
        table.add_column("Agent", style="cyan")
        table.add_column("Tier")
        table.add_column("Final Cash", justify="right")
        table.add_column("P&L%", justify="right")
        table.add_column("Holdings")
        for a in sorted(result.agents_final, key=lambda x: x.tier):
            holdings_str = ", ".join(f"{s}:{q}" for s, q in a.holdings.items()) if a.holdings else "-"
            table.add_row(a.id, str(a.tier), f"IDR {a.final_cash:,.0f}", f"{a.pnl_pct:+.2f}%", holdings_str)
        console.print(table)


if __name__ == "__main__":
    asyncio.run(main())
