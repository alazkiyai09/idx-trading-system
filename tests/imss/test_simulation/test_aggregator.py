"""Tests for multi-run aggregation."""

import pytest
from imss.simulation.aggregator import aggregate_runs, MultiRunResult, AgentRunStats
from imss.simulation.engine import SimulationResult, AgentSummary


def _make_result(run_number: int, pnl_offset: float = 0.0) -> SimulationResult:
    """Create a mock SimulationResult for testing."""
    # Build step logs with agent actions for action rate testing
    step_logs = []
    for step in range(5):
        step_logs.append({
            "step_number": step,
            "simulated_date": f"2024-07-0{step + 1}",
            "market_state": {"symbol": "BBRI", "close": 5100, "volume": 80000000},
            "agent_actions": [
                {"agent_id": "pak_budi", "action": "BUY", "stock": "BBRI", "quantity": 100},
                {"agent_id": "andi", "action": "HOLD", "stock": "BBRI", "quantity": 0},
            ],
            "events_active": [],
            "aggregate_sentiment": 0.0,
            "aggregate_order_imbalance": 0.0,
        })
    # Last step: pak_budi sells instead
    step_logs[-1]["agent_actions"][0] = {
        "agent_id": "pak_budi", "action": "SELL", "stock": "BBRI", "quantity": 100,
    }

    return SimulationResult(
        simulation_id=f"sim-{run_number}",
        status="COMPLETED",
        total_steps=5,
        agents_final=[
            AgentSummary(id="pak_budi", tier=1, persona_type="tier1_pak_budi",
                         final_cash=10e9 + run_number * 1e8, holdings={"BBRI": 500_000},
                         pnl_pct=1.5 + pnl_offset + run_number * 0.5),
            AgentSummary(id="andi", tier=1, persona_type="tier1_andi",
                         final_cash=90e6 + run_number * 5e6, holdings={},
                         pnl_pct=-2.0 + pnl_offset + run_number * 0.3),
        ],
        total_llm_calls=10 + run_number,
        total_tokens_used=5000 + run_number * 100,
        estimated_cost_usd=0.01 + run_number * 0.001,
        json_parse_success_rate=0.95,
        step_count=5,
        run_number=run_number,
        step_logs=step_logs,
    )


class TestAggregateRuns:
    def test_aggregate_three_runs(self):
        results = [_make_result(i) for i in range(3)]
        multi = aggregate_runs(results)
        assert multi.num_runs == 3
        assert len(multi.individual_results) == 3
        assert "tier1_pak_budi" in multi.agent_stats
        assert "tier1_andi" in multi.agent_stats

    def test_mean_std_computed(self):
        results = [_make_result(i) for i in range(3)]
        multi = aggregate_runs(results)
        stats = multi.agent_stats["tier1_pak_budi"]
        assert stats.num_samples == 3
        assert stats.mean_final_cash > 0
        assert stats.std_final_cash >= 0
        assert stats.mean_pnl_pct > 0

    def test_total_batch_cost_sums(self):
        results = [_make_result(i) for i in range(3)]
        multi = aggregate_runs(results)
        expected_cost = sum(r.estimated_cost_usd for r in results)
        assert abs(multi.total_batch_cost_usd - expected_cost) < 1e-6

    def test_single_run_zero_std(self):
        results = [_make_result(0)]
        multi = aggregate_runs(results)
        stats = multi.agent_stats["tier1_pak_budi"]
        assert stats.std_final_cash == 0.0
        assert stats.std_pnl_pct == 0.0

    def test_action_rates_computed(self):
        """Action rates are computed from step log agent_actions."""
        results = [_make_result(i) for i in range(3)]
        multi = aggregate_runs(results)

        # pak_budi: 4 BUY + 1 SELL per run × 3 runs = 12 BUY + 3 SELL
        pak = multi.agent_stats["tier1_pak_budi"]
        assert pak.buy_rate == pytest.approx(12 / 15, abs=0.01)
        assert pak.sell_rate == pytest.approx(3 / 15, abs=0.01)
        assert pak.hold_rate == pytest.approx(0.0, abs=0.01)

        # andi: 5 HOLD per run × 3 runs = 15 HOLD
        andi = multi.agent_stats["tier1_andi"]
        assert andi.buy_rate == pytest.approx(0.0, abs=0.01)
        assert andi.sell_rate == pytest.approx(0.0, abs=0.01)
        assert andi.hold_rate == pytest.approx(1.0, abs=0.01)

        # Rates sum to ~1.0
        assert pak.buy_rate + pak.sell_rate + pak.hold_rate == pytest.approx(1.0, abs=0.01)
        assert andi.buy_rate + andi.sell_rate + andi.hold_rate == pytest.approx(1.0, abs=0.01)

    def test_empty_results_raises(self):
        with pytest.raises(ValueError):
            aggregate_runs([])
