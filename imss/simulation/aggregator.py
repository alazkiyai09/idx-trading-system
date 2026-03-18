"""Multi-run aggregation for IMSS simulations."""

from __future__ import annotations

import uuid
from collections import defaultdict

import numpy as np
from pydantic import BaseModel

from imss.simulation.engine import SimulationResult


class AgentRunStats(BaseModel):
    """Aggregated statistics for one agent type across multiple runs."""

    persona_type: str
    num_samples: int
    mean_final_cash: float
    std_final_cash: float
    mean_pnl_pct: float
    std_pnl_pct: float
    buy_rate: float = 0.0
    sell_rate: float = 0.0
    hold_rate: float = 0.0


class MultiRunResult(BaseModel):
    """Aggregated result from N simulation runs."""

    simulation_id: str
    num_runs: int
    individual_results: list[SimulationResult]
    mean_total_llm_calls: float
    mean_estimated_cost_usd: float
    total_batch_cost_usd: float
    agent_stats: dict[str, AgentRunStats]


def aggregate_runs(results: list[SimulationResult]) -> MultiRunResult:
    """Aggregate multiple SimulationResult objects into a MultiRunResult.

    Designed to be consumed by a future Observer agent.
    """
    if not results:
        raise ValueError("Cannot aggregate empty results list")

    # Group agents by persona_type across runs
    agents_by_type: dict[str, list[dict]] = defaultdict(list)
    for res in results:
        for agent in res.agents_final:
            agents_by_type[agent.persona_type].append({
                "final_cash": agent.final_cash,
                "pnl_pct": agent.pnl_pct,
            })

    # Count actions per persona_type from step logs
    action_counts: dict[str, dict[str, int]] = defaultdict(lambda: {"BUY": 0, "SELL": 0, "HOLD": 0})
    for res in results:
        # Build agent_id → persona_type map for this run
        id_to_type = {a.id: a.persona_type for a in res.agents_final}
        for log in res.step_logs:
            for act in log.get("agent_actions", []):
                persona = id_to_type.get(act.get("agent_id", ""))
                action = act.get("action", "HOLD")
                if persona and action in ("BUY", "SELL", "HOLD"):
                    action_counts[persona][action] += 1

    # Compute per-type stats
    agent_stats: dict[str, AgentRunStats] = {}
    for persona_type, agent_data in agents_by_type.items():
        cash_values = [d["final_cash"] for d in agent_data]
        pnl_values = [d["pnl_pct"] for d in agent_data]
        n = len(agent_data)

        counts = action_counts.get(persona_type, {"BUY": 0, "SELL": 0, "HOLD": 0})
        total_actions = counts["BUY"] + counts["SELL"] + counts["HOLD"]
        buy_rate = counts["BUY"] / total_actions if total_actions > 0 else 0.0
        sell_rate = counts["SELL"] / total_actions if total_actions > 0 else 0.0
        hold_rate = counts["HOLD"] / total_actions if total_actions > 0 else 0.0

        agent_stats[persona_type] = AgentRunStats(
            persona_type=persona_type,
            num_samples=n,
            mean_final_cash=float(np.mean(cash_values)),
            std_final_cash=float(np.std(cash_values)) if n > 1 else 0.0,
            mean_pnl_pct=float(np.mean(pnl_values)),
            std_pnl_pct=float(np.std(pnl_values)) if n > 1 else 0.0,
            buy_rate=round(buy_rate, 4),
            sell_rate=round(sell_rate, 4),
            hold_rate=round(hold_rate, 4),
        )

    # Aggregate costs
    llm_calls = [r.total_llm_calls for r in results]
    costs = [r.estimated_cost_usd for r in results]

    return MultiRunResult(
        simulation_id=str(uuid.uuid4()),
        num_runs=len(results),
        individual_results=results,
        mean_total_llm_calls=float(np.mean(llm_calls)),
        mean_estimated_cost_usd=float(np.mean(costs)),
        total_batch_cost_usd=float(np.sum(costs)),
        agent_stats=agent_stats,
    )
