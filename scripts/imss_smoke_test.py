"""IMSS smoke test — minimal end-to-end validation.

Runs 5 trading days with 3 Tier 1 + 0 Tier 2 + 10 Tier 3 agents.
Target: <2 minutes. Exit code 0 on pass, 1 on fail.

Usage:
    python3 scripts/imss_smoke_test.py
"""

from __future__ import annotations

import asyncio
import sys

from rich.console import Console

from imss.config import SimulationConfig
from imss.simulation.engine import SimulationEngine

console = Console()


async def smoke_test() -> bool:
    config = SimulationConfig(
        target_stocks=["BBRI"],
        mode="BACKTEST",
        backtest_start="2024-07-01",
        backtest_end="2024-07-07",
        tier1_personas=["pak_budi", "sarah", "andi"],
        tier2_per_archetype=0,
        tier2_archetypes=[],
        tier3_total=10,
        num_parallel_runs=1,
    )

    console.print("[bold blue]IMSS Smoke Test[/]")
    console.print(f"Config: {len(config.tier1_personas)} T1 + 0 T2 + {config.tier3_total} T3, 5 days")

    engine = SimulationEngine()

    try:
        result = await engine.run_single(config)
    except Exception as e:
        console.print(f"[red]FAIL: Simulation crashed: {e}[/]")
        return False

    # Validate
    passed = True

    if result.status != "COMPLETED":
        console.print(f"[red]FAIL: Status is {result.status}, expected COMPLETED[/]")
        passed = False

    if result.step_count == 0:
        console.print("[red]FAIL: No step logs created[/]")
        passed = False

    for agent in result.agents_final:
        if agent.final_cash < 0:
            console.print(f"[red]FAIL: Agent {agent.id} has negative cash: {agent.final_cash}[/]")
            passed = False

    if passed:
        console.print(f"[bold green]PASS[/] — {result.step_count} steps, {result.total_llm_calls} LLM calls, ${result.estimated_cost_usd:.4f}")
    return passed


async def smoke_test_multi_run() -> bool:
    """Minimal 2-run multi-run test."""
    config = SimulationConfig(
        target_stocks=["BBRI"],
        mode="BACKTEST",
        backtest_start="2024-07-01",
        backtest_end="2024-07-05",
        tier1_personas=["pak_budi"],
        tier2_per_archetype=0,
        tier2_archetypes=[],
        tier3_total=5,
        num_parallel_runs=2,
    )

    console.print("\n[bold blue]IMSS Multi-Run Smoke Test (2 runs)[/]")

    engine = SimulationEngine()
    try:
        result = await engine.run_multi(config)
    except Exception as e:
        console.print(f"[red]FAIL: Multi-run crashed: {e}[/]")
        return False

    if result.num_runs != 2:
        console.print(f"[red]FAIL: Expected 2 runs, got {result.num_runs}[/]")
        return False

    console.print(f"[bold green]PASS[/] — {result.num_runs} runs, batch cost ${result.total_batch_cost_usd:.4f}")
    return True


if __name__ == "__main__":
    ok = asyncio.run(smoke_test())
    if ok:
        ok = asyncio.run(smoke_test_multi_run())
    sys.exit(0 if ok else 1)
