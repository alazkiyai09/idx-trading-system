# IMSS Phase 2B — New Agents & Multi-Run Execution

**Date:** 2026-03-18
**Status:** Design approved
**Prerequisite:** Phase 1 complete (29/29 tests, `feat/imss-phase1` branch)

---

## 1. Overview

This sub-project adds 4 new agent personas/archetypes and multi-run parallel execution to the IMSS simulation engine.

**New Tier 1 agents:**
- **Dr. Lim** — Contrarian value analyst with access to fundamental data (P/E, P/B, DY, ROE)
- **MarketBot** — Hybrid market maker with rule-based direction + LLM-driven reasoning

**New Tier 2 archetypes:**
- **dividend_holder** — Yield-focused, patient, slow reactor (latency=2)
- **sector_rotator** — Macro-aware timing, neutral sentiment (latency=1)

**Multi-run execution:**
- Execute N simulation runs with different random seeds
- Statistical aggregation (mean, std, confidence intervals) across runs
- No Observer agent in this phase (follow-up)

**Out of scope:** Episodic/social/causal memory, Observer agent, OpenViking, Phase 1 debt cleanup, multi-stock expansion.

---

## 2. Dr. Lim — Contrarian Value Analyst (Tier 1)

### 2.1 Persona Configuration

Added to `PERSONAS` dict in `imss/agents/tier1/personas.py`:

| Field | Value |
|-------|-------|
| key | `dr_lim` |
| name | Dr. Lim |
| description | Independent equity analyst with PhD in finance. Deep fundamental analysis. Goes against crowd when valuations extreme. |
| risk_tolerance | 0.4 |
| max_allocation | 35% |
| stop_loss_pct | 20 |
| holding_period | Months |
| initial_cash | IDR 5,000,000,000 (5B) |
| initial_holdings | `{"BBRI": 300_000}` |

**Behavioral rules:**
- Contrarian approach — buys when fear high, valuations low
- Evaluates P/E, P/B, dividend yield, ROE fundamentals
- Patient — holds losing positions months if thesis intact
- Ignores short-term noise and social media hype
- Increases position sizes when fear high and valuations low
- Statistical thinker — believes in mean reversion

### 2.2 Fundamentals Data Layer

**New DB model:** `StockFundamentals` in `imss/db/models.py`

| Column | Type | Description |
|--------|------|-------------|
| id | Integer PK | Auto-increment |
| symbol | String(10), indexed | Stock ticker |
| period | String(10) | e.g. "2024-Q2" |
| pe_ratio | Float | Price-to-earnings |
| pb_ratio | Float | Price-to-book |
| dividend_yield_pct | Float | Annual dividend yield % |
| roe_pct | Float | Return on equity % |
| market_cap_trillion_idr | Float | Market cap in trillion IDR |

**Seed data** (static BBRI snapshot in `data/seed_events/bbri_fundamentals.json`):

```json
{
  "symbol": "BBRI",
  "period": "2024-Q2",
  "pe_ratio": 12.5,
  "pb_ratio": 2.1,
  "dividend_yield_pct": 5.2,
  "roe_pct": 18.3,
  "market_cap_trillion_idr": 620
}
```

### 2.3 Prompt Enhancement

`build_tier1_user_prompt()` in `imss/llm/prompts/tier1_decision.py` gains an optional `fundamentals: dict | None = None` parameter. When provided, appends:

```
== Fundamental Data ==
P/E Ratio: 12.5
P/B Ratio: 2.1
Dividend Yield: 5.2%
ROE: 18.3%
Market Cap: IDR 620T
```

Only Dr. Lim receives this section. Other Tier 1 agents pass `fundamentals=None`.

### 2.4 Engine Integration

`SimulationEngine.run_single()` loads `StockFundamentals` for the target stock at simulation start. The fundamentals dict is injected into `market_state["fundamentals"]` as `dict | None`. This avoids changing the `BaseAgent.decide(market_state, events, step)` signature. Dr. Lim's `decide()` reads `market_state["fundamentals"]` and passes it to the prompt builder; all other Tier 1 agents ignore the field.

---

## 3. MarketBot — Hybrid Market Maker (Tier 1)

### 3.1 Persona Configuration

Added to `PERSONAS` dict:

| Field | Value |
|-------|-------|
| key | `marketbot` |
| name | MarketBot |
| description | Automated market making algorithm. Provides liquidity, profits from bid-ask spread. No directional bias. |
| risk_tolerance | 0.2 |
| max_allocation | 20% |
| stop_loss_pct | 1 |
| holding_period | Intraday |
| initial_cash | IDR 20,000,000,000 (20B) |
| initial_holdings | `{"BBRI": 200_000}` |

**Behavioral rules:**
- Provides liquidity — buys when others selling, sells when others buying
- No directional opinion — goal is capture spread, not predict direction
- Reduces activity (widens spread) during high volatility
- Increases activity during calm, range-bound markets
- Never holds large net positions — rebalances toward neutral
- Executes based on aggregate order flow, not events

### 3.2 Hybrid decide() Architecture

**New file:** `imss/agents/tier1/marketbot.py`

`MarketBotAgent` subclass of `Tier1Agent` with overridden `decide()`:

1. **Rule-based direction:** Read `market_state["prev_aggregate_order_imbalance"]` (see Section 3.4 for loop changes).
   - If imbalance > 0.1 (net buying pressure) → pre-determined direction = SELL
   - If imbalance < -0.1 (net selling pressure) → pre-determined direction = BUY
   - If abs(imbalance) <= 0.1 → pre-determined direction = HOLD
2. **Volatility gate:** If `market_state["pct_change_5d"]` exceeds ±5%, reduce position sizing by 50% (simulates "widening spread").
3. **LLM reasoning:** Call LLM with the pre-determined direction, asking for quantity, confidence, and reasoning justification.
4. **Return:** `AgentAction` with the rule-based direction but LLM-reasoned quantity/confidence.

### 3.4 Loop Changes for MarketBot

**File:** `imss/simulation/loop.py`

`run_simulation_loop()` must maintain a `prev_aggregate_order_imbalance` variable, initialized to `0.0`. After each step's order resolution and aggregate computation, store the computed `aggregate_order_imbalance` value. At the start of the next step, inject it into `market_state["prev_aggregate_order_imbalance"]`.

This ensures MarketBot reads the **previous** step's imbalance, not the current step's (which hasn't been computed yet).

Note: `create_tier1_agent()` must return `MarketBotAgent` when `persona_key == "marketbot"`. Import added to `imss/agents/tier1/__init__.py`.

### 3.3 Dedicated Prompt

**New file:** `imss/llm/prompts/marketbot_decision.py`

- `build_marketbot_system_prompt()` — Explains market-maker role, emphasizes liquidity provision and delta-neutral goal
- `build_marketbot_user_prompt(direction, imbalance, volatility, cash, holdings, close_price)` — Presents the pre-computed direction, current portfolio state, asks for quantity and confidence justification

---

## 4. New Tier 2 Archetypes

### 4.1 dividend_holder

Added to `ARCHETYPES` dict in `imss/agents/tier2/archetypes.py`:

| Field | Value |
|-------|-------|
| key | `dividend_holder` |
| name | Dividend Holder |
| one_liner | "You buy high-dividend stocks and hold long term. You rarely sell unless dividends are cut." |
| risk_tolerance_range | (0.3, 0.5) |
| sentiment_bias_range | (0.1, 0.3) — mildly bullish |
| decision_latency | 2 (slow reactor) |
| initial_cash_range | (100_000_000, 500_000_000) |

### 4.2 sector_rotator

| Field | Value |
|-------|-------|
| key | `sector_rotator` |
| name | Sector Rotator |
| one_liner | "You shift allocation based on economic cycle and macro data. Rate cuts mean buy banking, inflation means sell growth." |
| risk_tolerance_range | (0.4, 0.7) |
| sentiment_bias_range | (-0.1, 0.1) — neutral |
| decision_latency | 1 |
| initial_cash_range | (150_000_000, 600_000_000) |

### 4.3 Config Update

`SimulationConfig.tier2_archetypes` default changes to:
```python
["momentum_chaser", "panic_seller", "news_reactive", "dividend_holder", "sector_rotator"]
```

With `tier2_per_archetype=4`, total Tier 2 agents increases from 12 to 20.

### 4.4 Decision Latency Implementation

No changes to `propagation.py`. The existing `PROPAGATION_DELAYS` matrix applies uniformly to all Tier 2 agents via MEDIUM access.

The `decision_latency` field is a **separate** concept from propagation delay — it controls how many steps the agent waits before acting on received events. **This is NOT implemented in Phase 1** (all existing archetypes have `decision_latency=0`).

**Required change in `Tier2Agent.decide()`** (`imss/agents/tier2/archetypes.py`): Before processing events, filter them:
```python
if self.decision_latency > 0:
    events = [e for e in events if step - e.get("_step", 0) >= self.decision_latency]
```

This requires the simulation loop to tag each event with `_step` (the step number when the event was first distributed to this tier). Add `_step` tagging in `distribute_events()` or in the loop when events are passed to agents.

---

## 5. Multi-Run Parallel Execution

### 5.1 Seed Propagation

Each run gets seed `base_seed + run_number` (base_seed = 42 by default).

**Seed computation in `run_single()`:**
```python
seed = 42 + run_number
```

**Modified factory calls in `run_single()`:**
```python
# Currently: create_tier2_agents(archetype, config.tier2_per_archetype)
# Changed to: create_tier2_agents(archetype, config.tier2_per_archetype, seed=seed)
# Currently: create_tier3_agents(config.tier3_total, config.tier3_distribution)
# Changed to: create_tier3_agents(config.tier3_total, config.tier3_distribution, seed=seed)
```

**Affected call sites:**
- `create_tier2_agents(archetype_key, count, seed)` — randomizes cash, risk tolerance
- `create_tier3_agents(total, distribution, seed)` — randomizes cash, agent behavior (RandomWalkAgent)
- `run_simulation_loop(..., seed)` — passed to `distribute_events()` rng for delay jitter

### 5.2 run_multi() Method

**File:** `imss/simulation/engine.py`

```python
async def run_multi(self, config: SimulationConfig, on_step=None) -> MultiRunResult:
    results = []
    for i in range(config.num_parallel_runs):
        result = await self.run_single(config, run_number=i, on_step=on_step)
        results.append(result)
    return aggregate_runs(results)
```

Runs execute **sequentially** to avoid SQLite write contention. The `runs_batch_size` config field is reserved for future async batching with PostgreSQL.

**CostTracker reset:** The `LLMRouter` has a single shared `CostTracker`. Before each `run_single()` call within `run_multi()`, snapshot the tracker state (total_calls, total_input_tokens, etc.). After the run completes, compute deltas to populate the per-run `SimulationResult` cost fields. This ensures individual results have accurate per-run costs, not cumulative totals.

Implementation: Add `CostTracker.snapshot() -> dict` and compute deltas in `run_single()`:
```python
pre = self._router.cost_tracker.snapshot()
# ... run simulation ...
post = self._router.cost_tracker.snapshot()
llm_calls = post["total_calls"] - pre["total_calls"]
```

### 5.3 Aggregation

**New file:** `imss/simulation/aggregator.py`

**Models:**

```python
class AgentRunStats(BaseModel):
    persona_type: str
    num_runs: int
    mean_final_cash: float
    std_final_cash: float
    mean_pnl_pct: float
    std_pnl_pct: float
    buy_rate: float    # fraction of total actions that were BUY
    sell_rate: float
    hold_rate: float

class MultiRunResult(BaseModel):
    simulation_id: str
    num_runs: int
    individual_results: list[SimulationResult]
    mean_total_llm_calls: float
    mean_estimated_cost_usd: float
    total_batch_cost_usd: float
    agent_stats: dict[str, AgentRunStats]  # keyed by persona_type

# Also add to SimulationResult:
#   run_number: int = 0  (for traceability in multi-run aggregation)
```

**Function:** `aggregate_runs(results: list[SimulationResult]) -> MultiRunResult`

Logic:
1. Group `AgentSummary` objects across all runs by `persona_type`
2. Compute mean/std for `final_cash` and `pnl_pct` per group
3. Action rates computed from **post-resolution** step logs: count BUY/SELL/HOLD per agent type across all runs, divide by total actions
4. Sum costs across runs for `total_batch_cost_usd`
5. Generate shared `simulation_id` UUID for the batch

**Note:** The aggregator is designed to be consumed by a future Observer agent. Its output model (`MultiRunResult`) should be treated as the Observer's input contract, not final user-facing output.

### 5.4 Config Changes

In `SimulationConfig`:
- `num_parallel_runs`: default stays at 1 (backward compatible). CLI `--runs` flag overrides.
- `runs_batch_size`: unchanged (reserved for Phase 3 PostgreSQL)

### 5.5 Step Log Enhancement

Each `SimulationStepLog` already has a `run_number` column. The aggregator reads step logs from `individual_results` (in-memory), not from the database.

### 5.6 P&L Calculation Fix

Phase 1 left `pnl_pct` hardcoded to `0.0` (pending item P1). This Phase 2B fixes it to make aggregated stats meaningful.

**In `SimulationEngine.run_single()`:** Record each agent's initial portfolio value (cash + holdings * close prices on day 1) at simulation start. At completion, compute:
```python
pnl_pct = ((final_value - initial_value) / initial_value) * 100
```
Store in `AgentSummary.pnl_pct`.

### 5.7 Cost Estimates

With the expanded agent population (5 T1 + 20 T2 + 50 T3 = 75 agents):

| Metric | Phase 1 (65 agents) | Phase 2B (75 agents) | Change |
|--------|--------------------|--------------------|--------|
| LLM calls/step | 15 (3 T1 + 12 T2) | 25 (5 T1 + 20 T2) | +67% |
| Est. cost/run | ~$0.03 | ~$0.05 | +67% |
| Est. cost/5-run batch | N/A | ~$0.25 | — |

The `$5 cost alert threshold` remains appropriate. Smoke test target (2 min) should still hold — the extra agents add ~10 LLM calls per step but the mocked-LLM smoke test is not affected.

---

## 6. File Changes Summary

### New Files

| Path | Responsibility |
|------|---------------|
| `imss/agents/tier1/marketbot.py` | `MarketBotAgent` subclass with hybrid decide() |
| `imss/llm/prompts/marketbot_decision.py` | MarketBot-specific prompt builders |
| `imss/simulation/aggregator.py` | `aggregate_runs()`, `MultiRunResult`, `AgentRunStats` |
| `data/seed_events/bbri_fundamentals.json` | Static BBRI fundamentals snapshot |
| `tests/imss/test_agents/test_tier1_new.py` | Dr. Lim + MarketBot tests |
| `tests/imss/test_agents/test_tier2_new.py` | dividend_holder + sector_rotator tests |
| `tests/imss/test_simulation/test_aggregator.py` | Aggregation unit tests |
| `tests/imss/test_simulation/test_multi_run.py` | Multi-run integration test |

### Modified Files

| Path | Changes |
|------|---------|
| `imss/agents/tier1/personas.py` | Add `dr_lim` to PERSONAS |
| `imss/agents/tier1/__init__.py` | Export `MarketBotAgent` |
| `imss/agents/tier2/archetypes.py` | Add `dividend_holder`, `sector_rotator` to ARCHETYPES; implement `decision_latency` filtering in `Tier2Agent.decide()` |
| `imss/db/models.py` | Add `StockFundamentals` table |
| `imss/config.py` | Update `tier2_archetypes` default list |
| `imss/simulation/engine.py` | Add `run_multi()`, seed propagation, fundamentals loading, P&L calculation, CostTracker snapshots |
| `imss/simulation/loop.py` | Accept `seed` param, carry forward `prev_aggregate_order_imbalance` into `market_state`, tag events with `_step` |
| `imss/llm/prompts/tier1_decision.py` | Add optional `fundamentals` param to user prompt |
| `scripts/imss_run_backtest.py` | Add `--runs` CLI flag, multi-run output |
| `scripts/imss_smoke_test.py` | Add multi-run smoke test case |
| `scripts/imss_seed_data.py` | Load BBRI fundamentals into DB |

---

## 7. Testing Strategy

| Test File | Cases | Type |
|-----------|-------|------|
| `test_tier1_new.py` | Dr. Lim persona config exists and has correct fields; Dr. Lim prompt includes fundamentals section when provided; Dr. Lim prompt omits fundamentals when None; MarketBot counters positive imbalance with SELL; MarketBot counters negative with BUY; MarketBot holds on near-zero imbalance; MarketBot reduces sizing in high volatility | Unit (mocked LLM) |
| `test_tier2_new.py` | dividend_holder config exists with latency=2; sector_rotator config exists with latency=1; create_tier2_agents produces correct count for each new archetype; all 5 archetypes registered in ARCHETYPES dict; generated agents have randomized params within ranges | Unit |
| `test_aggregator.py` | Aggregates 3 mock SimulationResults correctly; mean/std computed per agent type; action rates sum to ~1.0; handles single run (no std); empty results raises error | Unit |
| `test_multi_run.py` | engine.run_multi with mocked LLM, 2 runs; validates MultiRunResult.num_runs == 2; different seeds produce different Tier 3 actions; total_batch_cost_usd sums individual costs | Integration (mocked LLM, tmp_path DB) |

**Target:** All existing 29 tests continue passing + ~20 new tests.

---

## 8. Dependency Order

1. `StockFundamentals` model + seed data (no dependencies)
2. Dr. Lim persona config + prompt changes (depends on 1)
3. MarketBot agent + prompt (independent of 1-2)
4. dividend_holder + sector_rotator archetypes (independent)
5. Aggregator module (independent)
6. `run_multi()` + seed propagation in engine/loop (depends on 5)
7. CLI + smoke test updates (depends on 6)
8. All tests (parallel with each component)
