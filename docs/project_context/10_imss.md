# IMSS (IDX Market Swarm Simulator) Context

## Overview

IMSS is a multi-agent LLM-powered market simulation engine that models IDX trading behavior through three tiers of agents with different sophistication levels. It lives entirely under `imss/` as a self-contained package, separate from the main trading system in `core/`.

**Current state:** Phase 1 + Phase 2B complete (58 tests passing). Phase 2A (episodic/social memory) and Phase 2C (debt cleanup) not yet started.

## Architecture

### Three-Tier Agent Model

| Tier | Count | Decision Method | LLM | Latency |
|------|-------|----------------|-----|---------|
| 1 | 5 named personas | Full LLM reasoning with persona-specific prompts | Yes | Real-time |
| 2 | 20 typed archetypes (4 per type) | LLM with shorter prompts, randomized params | Yes | 0-2 steps |
| 3 | 50 rule-based | Heuristic (momentum, mean-reversion, random walk, volume) | No | Instant |

### Tier 1 Personas

| Key | Name | Style | Special |
|-----|------|-------|---------|
| `pak_budi` | Pak Budi | Conservative, long-term holder | — |
| `sarah` | Sarah | Growth-focused, momentum trader | — |
| `andi` | Andi | Speculative day trader | — |
| `dr_lim` | Dr. Lim | Contrarian value analyst | Receives fundamentals data (P/E, P/B, DY, ROE) |
| `marketbot` | MarketBot | Hybrid market maker | Rule-based direction from order imbalance + LLM sizing |

### Tier 2 Archetypes

| Key | Latency | Behavior |
|-----|---------|----------|
| `momentum_chaser` | 0 | Follows price trends |
| `panic_seller` | 0 | Sells on bad news |
| `news_reactive` | 0 | Acts on event sentiment |
| `dividend_holder` | 2 | Patient, yield-focused, rarely trades |
| `sector_rotator` | 1 | Macro-aware, rotates by cycle |

### Tier 3 Distribution

- `momentum_follower` (30%), `mean_reversion` (25%), `random_walk` (30%), `volume_follower` (15%)

## Data Flow

```
Historical prices (Yahoo Finance) + Seed events (JSON)
    ↓
SQLite DB (async SQLAlchemy 2.0) — data/imss.db
    ↓
SimulationEngine.run_single() / run_multi()
    ↓
Loop: for each trading day
    → Distribute events (propagation delays by tier)
    → Tier 3 decide (heuristic)
    → Tier 2 decide (LLM, with decision latency filtering)
    → Tier 1 decide (LLM, full context)
    → Order resolution (lot-aligned, cash/holdings constraints)
    → Aggregate sentiment + order imbalance
    ↓
SimulationResult / MultiRunResult (with agent stats + action rates)
```

## Key Module Responsibilities

### `imss/config.py`
- `IMSSSettings`: env-driven settings (GLM API key, base URL, model, DB URL)
- `SimulationConfig`: per-run config (stocks, dates, agent population, multi-run params)

### `imss/agents/`
- `base.py`: `BaseAgent`, `AgentAction`, `WorkingMemory` base classes
- `tier1/personas.py`: `PERSONAS` dict, `PersonaConfig`, `Tier1Agent`, `create_tier1_agent()`
- `tier1/marketbot.py`: `MarketBotAgent` — hybrid decide() with imbalance thresholds (±0.1) and volatility gate (5%)
- `tier2/archetypes.py`: `ARCHETYPES` dict, `Tier2Agent`, `create_tier2_agents()`, decision latency filtering
- `tier3/heuristic.py`: `HeuristicAgent` subclasses, `create_tier3_agents()`

### `imss/llm/`
- `router.py`: `LLMRouter` (Anthropic SDK → GLM-5 at api.z.ai/api/anthropic), `CostTracker`, JSON parsing
- `batcher.py`: `LLMBatcher` for concurrent calls (initialized but not yet wired for T2)
- `prompts/`: `tier1_decision.py` (with optional fundamentals), `tier2_decision.py`, `marketbot_decision.py`

### `imss/simulation/`
- `engine.py`: `SimulationEngine` with `run_single()` and `run_multi()`, P&L calculation, cost tracking
- `loop.py`: `run_simulation_loop()` — turn-based loop, event distribution, order imbalance carry-forward
- `order_book.py`: `resolve_backtest_orders()` — lot-aligned fills with cash/holdings constraints
- `propagation.py`: Event delay matrix by tier (FAST/MEDIUM/SLOW access)
- `aggregator.py`: `aggregate_runs()` → `MultiRunResult` with `AgentRunStats` (mean/std/action rates)

### `imss/db/`
- `models.py`: 8 tables — `StockOHLCV`, `Event`, `EventEntity`, `CausalLink`, `SimulationRun`, `AgentConfig`, `SimulationStepLog`, `StockFundamentals`

### `imss/data/`
- `price_feed.py`: Yahoo Finance price fetcher + DB storage
- `embedder.py`: Zhipu embedding client (not currently used — embedding API unavailable)

### `imss/memory/`
- `working.py`: `WorkingMemory` (cash, holdings, portfolio value calculation)
- Episodic/social memory not yet implemented (Phase 2A)

## LLM Configuration

- **Provider:** GLM-5 via Anthropic-compatible API at `https://api.z.ai/api/anthropic`
- **SDK:** `anthropic.AsyncAnthropic` (switched from OpenAI SDK in Phase 2B)
- **Env var:** `GLM_API_KEY` (required), `IMSS_GLM_BASE_URL` (optional override)
- **Cost:** ~$0.03 per single run (75 agents, 5 days), ~$0.06 per 2-run batch

## Multi-Run Execution

- `run_multi()` executes N sequential runs with seeds `42 + run_number`
- Each run creates independent Tier 2/3 agents with seed-driven randomization
- `CostTracker.snapshot()` enables per-run cost delta computation
- `aggregate_runs()` produces mean/std for cash and P&L, plus action rates (BUY/SELL/HOLD) per persona type

## Database

- **File:** `data/imss.db` (SQLite via async SQLAlchemy 2.0 + aiosqlite)
- **Separate from main system:** Does not share tables with `data/trading.db`
- **Seed script:** `scripts/imss_seed_data.py` — loads BBRI prices, events, and fundamentals
- **ChromaDB:** `data/chroma/` — reserved for episodic memory (Phase 2A), not currently populated

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/imss_seed_data.py` | Bootstrap DB with BBRI prices + events + fundamentals |
| `scripts/imss_run_backtest.py` | CLI for single/multi-run backtests (`--runs N`) |
| `scripts/imss_smoke_test.py` | E2E validation: single run + multi-run with live LLM |

## Test Suite

58 tests in `tests/imss/`, all passing. Run with: `python3 -m pytest tests/imss/ -v`

| Test File | Count | Focus |
|-----------|-------|-------|
| `test_agents/test_base.py` | 4 | Base agent creation, working memory |
| `test_agents/test_tier1_new.py` | 11 | Dr. Lim config/prompts, MarketBot imbalance/volatility |
| `test_agents/test_tier2_new.py` | 9 | Archetype configs, creation, latency filtering |
| `test_agents/test_tier3.py` | 5 | Heuristic agent creation and behavior |
| `test_data/test_db.py` | 3 | Table creation (8 tables), fundamentals |
| `test_data/test_price_feed.py` | 3 | Price data validation |
| `test_llm/test_router.py` | 7 | JSON parsing, fence stripping |
| `test_simulation/test_aggregator.py` | 6 | Multi-run aggregation, action rates |
| `test_simulation/test_engine.py` | 1 | Full engine integration (mocked LLM) |
| `test_simulation/test_multi_run.py` | 2 | Multi-run integration, seed divergence |
| `test_simulation/test_order_resolution.py` | 6 | Order fills, lot enforcement |

## Known Pending Items

### Phase 2A (Memory & Intelligence) — Not Started
- Episodic memory (ChromaDB-based for T1, sliding-window for T2)
- Social memory and trust dynamics
- Causal knowledge graph

### Phase 1 Debt (Phase 2C) — Not Started
- P2: Wire LLMBatcher for Tier 2 concurrent calls
- P4: Alembic migrations
- P5: Mock test for embedder
- P8: Cost alert warning during simulation
- P9: Switch to structlog
- P10: ARA/ARB price limits

### Future Phases
- Phase 3: FastAPI endpoints, prediction mode, multi-stock, PostgreSQL
- Phase 4: Dashboard UI, WebSocket streaming, agent visualization

## Edit Guidance

- IMSS is self-contained under `imss/` — it does not import from `core/`, `api/`, or `dashboard/`
- The main trading system does not import from `imss/`
- If adding new agent types, follow the pattern in `personas.py` (T1) or `archetypes.py` (T2)
- All LLM calls go through `LLMRouter.call()` — never call the Anthropic SDK directly from agents
- Tests mock at the `router.call()` level, not the underlying SDK — SDK changes don't break tests
- The simulation loop carries forward `prev_aggregate_order_imbalance` for MarketBot and tags events with `_step` for latency filtering
- `SimulationResult.step_logs` contains full action history — used by aggregator for action rates
