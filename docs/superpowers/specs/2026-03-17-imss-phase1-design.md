# IMSS Phase 1 Design — Foundation

**Date:** 2026-03-17
**Status:** Approved
**Scope:** Phase 1 only — single-stock, single-run backtest with basic agents, no memory

---

## 1. Overview

The IDX Market Swarm Simulator (IMSS) is a multi-agent market simulation engine for the Indonesian Stock Exchange. Phase 1 delivers a working single-stock backtest with differentiated agent behavior across three tiers of agents.

IMSS lives as a **self-contained `imss/` package** inside the `idx-trading-system` repo. It has its own async patterns, database, and LLM client — minimal coupling with the existing synchronous codebase.

**Rationale for separation:**
- IMSS requires async (AsyncOpenAI, aiosqlite); existing codebase is synchronous
- IMSS uses OpenAI-compatible SDK for GLM-5; existing `glm_client.py` uses Anthropic-compatible wrapper
- Separate SQLite database (`data/imss.db`) avoids table conflicts
- Self-contained package is easier to test and eventually extract

---

## 2. Project Structure

```
idx-trading-system/
├── imss/                          # Self-contained IMSS package
│   ├── __init__.py
│   ├── config.py                  # Pydantic Settings, SimulationConfig
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── base.py                # BaseAgent, WorkingMemory, AgentAction
│   │   ├── tier1/
│   │   │   ├── __init__.py
│   │   │   └── personas.py        # Pak Budi, Sarah, Andi
│   │   ├── tier2/
│   │   │   ├── __init__.py
│   │   │   └── archetypes.py      # momentum_chaser, panic_seller, news_reactive
│   │   └── tier3/
│   │       ├── __init__.py
│   │       └── heuristic.py       # 4 rule-based strategies
│   ├── memory/
│   │   ├── __init__.py
│   │   └── working.py             # WorkingMemory (Phase 1 only)
│   ├── simulation/
│   │   ├── __init__.py
│   │   ├── engine.py              # SimulationEngine orchestrator
│   │   ├── loop.py                # Turn-based simulation loop
│   │   └── order_book.py          # Order resolution (backtest mode)
│   ├── data/
│   │   ├── __init__.py
│   │   ├── price_feed.py          # yfinance IDX data ingestion
│   │   └── embedder.py            # Zhipu embedding-3 via api.z.ai
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── router.py              # LLMRouter (async OpenAI-compatible)
│   │   ├── batcher.py             # Semaphore-based concurrent calls
│   │   └── prompts/
│   │       ├── __init__.py
│   │       ├── tier1_decision.py   # Full persona decision prompt
│   │       └── tier2_decision.py   # Simplified archetype prompt
│   └── db/
│       ├── __init__.py
│       └── models.py              # SQLAlchemy async models
├── data/
│   ├── imss.db                    # IMSS SQLite database
│   ├── seed_events/
│   │   └── bbri_events_2024.json  # 50 curated events
│   └── chroma/                    # ChromaDB persistence
├── scripts/
│   ├── imss_seed_data.py          # Price download + event loading
│   ├── imss_run_backtest.py       # CLI backtest runner
│   └── imss_smoke_test.py         # Quick validation
├── tests/
│   └── imss/
│       ├── conftest.py
│       ├── fixtures/
│       │   ├── mock_prices.json
│       │   └── mock_events.json
│       ├── test_agents/
│       │   ├── test_base.py
│       │   └── test_tier3.py
│       ├── test_simulation/
│       │   ├── test_order_resolution.py
│       │   └── test_engine.py
│       └── test_data/
│           └── test_price_feed.py
└── docs/
    ├── IDX_Market_Swarm_Simulator_Spec.md
    ├── IDX_Market_Swarm_Simulator_Implementation.md
    ├── IMSS_Update_LLM_Configuration.md
    └── IMSS_Update_OpenViking_Integration.md
```

---

## 3. Dependencies

New dependencies added to `requirements.txt` (not already present):

```
# IMSS-specific
chromadb>=0.4.22
aiosqlite>=0.20
structlog>=24.1
rich>=13.7
tiktoken>=0.6
tqdm>=4.66
```

Already present in the repo: `openai`, `pydantic`, `pydantic-settings`, `python-dotenv`, `yfinance`, `pandas`, `numpy`, `sqlalchemy`, `httpx`, `aiohttp`.

---

## 4. Environment Configuration

New variables added to `.env.example`:

```bash
# === IMSS Configuration ===
IMSS_DATABASE_URL=sqlite+aiosqlite:///./data/imss.db
IMSS_CHROMA_PERSIST_DIR=./data/chroma

# IMSS LLM (reuses existing GLM_API_KEY)
IMSS_GLM_BASE_URL=https://api.z.ai/api/paas/v4/
IMSS_GLM_MODEL=glm-5
IMSS_EMBEDDING_MODEL=embedding-3
IMSS_EMBEDDING_DIMENSION=1024

# IMSS Simulation Defaults
IMSS_MAX_CONCURRENT_LLM_CALLS=5
IMSS_LLM_REQUEST_TIMEOUT=30
IMSS_COST_ALERT_THRESHOLD_USD=5.00
```

`imss/config.py` uses Pydantic `BaseSettings` with `env_prefix="IMSS_"` where appropriate, plus reads `GLM_API_KEY` directly from the shared `.env`.

`SimulationConfig` is a separate Pydantic `BaseModel` (not settings) with Phase 1 defaults:
- target_stocks: `["BBRI"]`
- mode: `"BACKTEST"`
- backtest_start/end: `"2024-07-01"` / `"2024-09-30"`
- tier1_personas: `["pak_budi", "sarah", "andi"]`
- tier2_per_archetype: 4 (3 archetypes = 12 agents)
- tier3_total: 50
- num_parallel_runs: 1
- enable_episodic_memory / social_memory / causal_retrieval: all `False`

---

## 5. Database Schema

Async SQLite at `data/imss.db`. Own `DeclarativeBase`. Tables created via `create_all()` (no Alembic in Phase 1 — introduce Alembic in Phase 2 before schema stabilizes for PostgreSQL migration).

**Write serialization**: All database writes are batched into a single transaction at the end of each simulation step. Agent actions are collected in memory during the step, then written in one `async with session.begin()` block. This avoids SQLite concurrent write issues (`database is locked`) that would occur if multiple async coroutines wrote simultaneously.

### 5.1 stocks_ohlcv

| Column | Type | Notes |
|--------|------|-------|
| id | Integer PK | Auto-increment |
| symbol | String(10) | Indexed |
| timestamp | DateTime | Indexed |
| open | Float | |
| high | Float | |
| low | Float | |
| close | Float | |
| volume | BigInteger | |
| adjusted_close | Float | |

Unique constraint: `(symbol, timestamp)`

### 5.2 events

| Column | Type | Notes |
|--------|------|-------|
| id | UUID PK | |
| timestamp | DateTime | Indexed |
| category | String(20) | REGULATORY, EARNINGS, MACRO, NEWS, RUMOR, POLITICAL |
| source | String(100) | |
| title | Text | |
| summary | Text | |
| raw_content | Text | Nullable |
| sentiment_score | Float | -1.0 to 1.0 |
| magnitude_score | Float | 0.0 to 1.0 |
| embedding_id | String(100) | ChromaDB document ID |
| created_at | DateTime | |

### 5.3 event_entities

| Column | Type | Notes |
|--------|------|-------|
| id | Integer PK | Auto-increment |
| event_id | UUID FK→events | |
| entity_type | String(10) | STOCK, SECTOR |
| entity_symbol | String(20) | e.g., BBRI, BANKING |

### 5.4 causal_links (schema only, populated in Phase 2)

| Column | Type | Notes |
|--------|------|-------|
| id | UUID PK | |
| event_id | UUID FK→events | Nullable |
| event_category | String(20) | |
| stock_symbol | String(10) | |
| lag_days | Integer | |
| direction | String(10) | POSITIVE, NEGATIVE, NEUTRAL |
| correlation_strength | Float | 0.0 to 1.0 |
| confidence | Float | 0.0 to 1.0 |
| sample_count | Integer | |
| last_updated | DateTime | |

### 5.5 simulation_runs

| Column | Type | Notes |
|--------|------|-------|
| id | UUID PK | |
| config_json | Text | JSON-serialized SimulationConfig |
| mode | String(10) | BACKTEST, PREDICT |
| status | String(20) | PENDING, RUNNING, COMPLETED, FAILED |
| started_at | DateTime | |
| completed_at | DateTime | Nullable |
| results_summary_json | Text | Nullable |
| total_llm_calls | Integer | Default 0 |
| total_tokens_used | Integer | Default 0 |
| estimated_cost_usd | Float | Default 0.0 |

### 5.6 agent_configs

| Column | Type | Notes |
|--------|------|-------|
| id | UUID PK | |
| simulation_id | UUID FK→simulation_runs | |
| agent_id | String(50) | e.g., "pak_budi", "momentum_chaser_003" |
| tier | Integer | 1, 2, or 3 |
| persona_type | String(50) | |
| parameters_json | Text | JSON: cash, risk_tolerance, holdings, archetype params |

Stores per-agent randomized parameters for post-hoc analysis.

### 5.7 simulation_step_logs

| Column | Type | Notes |
|--------|------|-------|
| id | Integer PK | Auto-increment |
| simulation_id | UUID FK→simulation_runs | |
| run_number | Integer | |
| step_number | Integer | |
| simulated_date | Date | |
| market_state_json | Text | Prices, volumes |
| agent_actions_json | Text | All agent decisions |
| events_active_json | Text | Events this step |
| aggregate_sentiment | Float | |
| aggregate_order_imbalance | Float | |

---

## 6. LLM Client

### 6.1 LLMRouter

Async OpenAI-compatible client. Single provider (GLM-5) in Phase 1.

```python
# Initialization
client = AsyncOpenAI(
    api_key=settings.glm_api_key,
    base_url="https://api.z.ai/api/paas/v4/"
)

# Routing
Tier 1 → model="glm-5", temperature=0.7, max_tokens=1024
Tier 2 → model="glm-5", temperature=0.5, max_tokens=512
```

**JSON parsing**: Strip markdown fences (`\`\`\`json` / `\`\`\``), `json.loads()`, validate required keys. On parse failure → default HOLD action.

**Cost tracking**: Log `response.usage.prompt_tokens` and `completion_tokens` per call. Aggregate by tier and agent_id. Use tiktoken `cl100k_base` as token count approximation for prompt budgeting.

**Error handling**: Retry 3x with exponential backoff on 429/500/timeout. After 3 failures → return default HOLD.

### 6.2 LLMBatcher

```python
class LLMBatcher:
    semaphore = asyncio.Semaphore(5)

    async def execute_batch(requests) -> list[AgentAction]:
        results = await asyncio.gather(*[
            self._limited_call(req) for req in requests
        ], return_exceptions=True)
        # Replace exceptions with default HOLD
        return [r if not isinstance(r, Exception) else default_hold(req)
                for r, req in zip(results, requests)]
```

### 6.3 Embedder

Zhipu `embedding-3` via the same OpenAI client at `api.z.ai`. Dimension: 1024. Pre-computes embeddings and passes them to ChromaDB (bypasses ChromaDB's built-in embedding function).

**Important: ChromaDB query pattern with external embeddings:**
When querying, embed the query text via Zhipu first, then use `collection.query(query_embeddings=[embedding_vector], n_results=k)`. Never use `query_texts` — ChromaDB's default embedding function won't match the Zhipu embedding space.

**Note on tiktoken approximation:** `cl100k_base` is used for prompt budget estimation only. GLM-5 uses a different tokenizer, so counts may be off by ~20-30%. This is acceptable for cost alerts but should not be relied upon for precise token limits.

---

## 7. Agent Architecture

### 7.1 Base Classes

```python
class AgentAction(BaseModel):
    agent_id: str
    step: int                              # simulation step when action was taken
    action: Literal["BUY", "SELL", "HOLD"]
    stock: str
    quantity: int                          # multiple of 100
    price: float = 0.0                     # fill price, populated after order resolution
    confidence: float                      # 0.0-1.0
    reasoning: str
    sentiment_update: float                # -1.0 to 1.0

class WorkingMemory(BaseModel):
    current_step: int = 0
    cash: float
    holdings: dict[str, int] = {}
    portfolio_value_history: list[tuple[int, float]] = []
    recent_actions: list[AgentAction] = []  # last 10
    recent_observations: list[dict] = []  # last 20
    current_sentiment: float = 0.0

# Note: BaseModel + ABC works in Pydantic v2. Subclasses implement decide() as async def.
# The abstract method is excluded from Pydantic serialization automatically.
class BaseAgent(BaseModel, ABC):
    id: str
    tier: int
    name: str
    persona_type: str
    working_memory: WorkingMemory

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @abstractmethod
    async def decide(self, market_state, events, step) -> AgentAction: ...

    def execute(self, action: AgentAction, price: float) -> None:
        # Update cash/holdings based on action
        ...
```

**`sentiment_update` usage:** After each agent decides, `working_memory.current_sentiment` is updated to `action.sentiment_update`. At step end, `aggregate_sentiment` in the step log is computed as the mean of all agents' `sentiment_update` values.

### 7.2 Tier 1 — Named Personas (3 in Phase 1)

| Persona | Risk Tolerance | Style | Initial Cash (IDR) |
|---------|---------------|-------|-------------------|
| Pak Budi | 0.3 | Conservative, regulation-aware, HOLD-biased | 10B |
| Sarah | 0.6 | Foreign institutional, USD-focused, quick loss-cutting | 15B |
| Andi | 0.8 | Aggressive retail, momentum-chasing, overtrades | 100M |

**Phase 1 prompt template** (stripped of memory sections to save tokens):
- System prompt: persona description, behavioral rules, risk profile, JSON response format
- User prompt: portfolio state, market data (OHLCV + 5d/20d changes), events for today
- **Omitted in Phase 1** (re-introduced in Phase 2): historical parallels, episodic memory, social signals sections. This saves ~200-300 tokens per call × 3 agents × 65 steps = ~40K-60K tokens per run.

### 7.3 Tier 2 — Typed Archetypes (12 in Phase 1)

| Archetype | Count | Behavior |
|-----------|-------|----------|
| momentum_chaser | 4 | Follows price trends, buys rising stocks |
| panic_seller | 4 | Overreacts to bad news, sells first |
| news_reactive | 4 | Trades headlines, first to react |

Each agent generated with random parameters sampled from archetype-defined ranges. Simplified prompt from Section 5.2.

### 7.4 Tier 3 — Statistical Agents (50 in Phase 1)

| Heuristic | Count | Rule |
|-----------|-------|------|
| momentum_follower | 15 | Buy if 5d change > +3%, sell if < -3% |
| mean_reversion | 13 | Trade toward 20-day MA when deviation > 1.5 std |
| random_walk | 15 | 10% chance of action each step, 50/50 buy/sell |
| volume_follower | 7 | Buy on volume spike > 2x average |

Pure Python, no LLM. Position sizes: 5% of cash, enforced to IDX lot size (100 shares).

Counts derived from percentage distribution (30%/25%/30%/15%) rounded to nearest integer, remainder assigned to momentum_follower.

---

## 8. Simulation Engine

### 8.1 Execution Flow

```
SimulationEngine.run_single(config, run_number=0)

  INIT:
    Load prices from stocks_ohlcv for [backtest_start, backtest_end]
    Load events, index by date
    Initialize 65 agents (3 T1 + 12 T2 + 50 T3)
    Create SimulationRun record (RUNNING)

  FOR EACH trading day:
    1. ENVIRONMENT UPDATE
       - Get OHLCV for date
       - Get events for date
       - Compute 5d/20d pct changes, volume ratio

    2. INFORMATION DISTRIBUTION (category-dependent delays)
       Tier 1 (HIGH access):
         - REGULATORY, EARNINGS, major NEWS: step T+0
         - RUMOR: step T+1
       Tier 2 (MEDIUM access):
         - REGULATORY: step T+1
         - Major NEWS: step T+0 or T+1 (random)
         - EARNINGS: step T+1
         - RUMOR: step T+0 to T+2 (random)
       Tier 3 (LOW access):
         - No events — react only to price changes

    3. AGENT EXECUTION
       - Tier 3: synchronous rule-based (instant)
       - Tier 2: batched async LLM (12 calls, semaphore=5)
       - Tier 1: sequential async LLM (3 calls)

    4. ORDER RESOLUTION
       - Fill at historical close price
       - Validate cash/holdings sufficiency
       - Enforce IDX lot sizes and tick sizes
       - Reject invalid orders (log reason)

    5. POST-STEP
       - Update working memory
       - Log to simulation_step_logs
       - Print step summary (rich)

  FINALIZE:
    Compute per-agent final P&L
    Update SimulationRun (COMPLETED, cost summary)
    Return SimulationResult
```

### 8.3 SimulationResult Model

```python
class SimulationResult(BaseModel):
    simulation_id: str
    status: str                        # COMPLETED or FAILED
    total_steps: int
    agents_final: list[AgentSummary]   # per-agent: id, tier, final_cash, holdings, pnl_pct
    total_llm_calls: int
    total_tokens_used: int
    estimated_cost_usd: float
    json_parse_success_rate: float     # fraction of valid JSON responses
    step_count: int                    # number of logged steps
```

### 8.4 Cost Estimate (Phase 1)

With 3 Tier 1 + 12 Tier 2 agents over ~65 trading days:
- Tier 1: 3 × 65 = 195 calls × ~2K tokens = ~390K tokens
- Tier 2: 12 × 65 = 780 calls × ~800 tokens = ~624K tokens
- Total: ~1M tokens per run
- At GLM-5 pricing (~$0.001/1K input, ~$0.002/1K output): **estimated < $0.50 per run**
- The $5 alert threshold is very conservative for Phase 1

### 8.2 Order Resolution Rules

**IDX tick sizes:**

| Price Range | Tick |
|------------|------|
| < 200 | IDR 1 |
| 200-500 | IDR 2 |
| 500-2000 | IDR 5 |
| 2000-5000 | IDR 10 |
| > 5000 | IDR 25 |

**Lot size:** 100 shares. All quantities rounded down to nearest 100.

**Auto-reject (ARA/ARB):** ±20% for regular board stocks, ±35% for acceleration board. Not enforced in backtest mode (uses historical prices), but the validation function is built now for Phase 3 prediction mode. Note: IDX updated ARA/ARB limits in 2023 from the old ±7% regime — the project's CLAUDE.md reference to ±7% is outdated.

---

## 9. CLI Scripts

### 9.1 imss_seed_data.py

Downloads BBRI.JK price data via yfinance (2024-01-01 to 2025-12-31). Loads 50 curated events from `data/seed_events/bbri_events_2024.json`. Embeds event summaries via Zhipu embedding-3 into ChromaDB. Creates all DB tables. Rich console output with progress bars.

### 9.2 imss_run_backtest.py

CLI arguments: `--stock`, `--start`, `--end`, `--tier1-personas`, `--tier2-count`, `--tier3-count`. Runs `SimulationEngine.run_single()`. Rich live display: per-step date, prices, agent action table, portfolio summary. Final report: per-agent P&L ranking, action frequency counts, total LLM cost.

### 9.3 imss_smoke_test.py

Minimal config: 5 trading days (2024-07-01 to 2024-07-07, within the main backtest window to ensure event coverage), 3 T1 + 0 T2 + 10 T3. Validates: completion, non-negative cash, step logs exist. Target: <2 minutes. Exit code 0/1.

---

## 10. Testing Strategy

### 10.1 Unit Tests (mocked LLM)

- `test_base.py`: Agent init, working memory deque behavior, portfolio value calculation
- `test_tier3.py`: Each heuristic produces correct action for given price history, lot size compliance
- `test_order_resolution.py`: Fill logic, cash validation, holding validation, tick size rounding, lot size enforcement
- `test_price_feed.py`: yfinance download schema validation, data quality checks

### 10.2 Test Fixtures

- `mock_prices.json`: 5 days BBRI OHLCV (realistic values around IDR 5,100-5,200)
- `mock_events.json`: 2 sample events (1 EARNINGS positive, 1 REGULATORY negative)

### 10.3 LLM Mocking Strategy

Unit tests patch `LLMRouter.call()` to return canned JSON:
```python
{"action": "BUY", "stock": "BBRI", "quantity": 1000,
 "confidence": 0.7, "reasoning": "test", "sentiment_update": 0.3}
```

Only `imss_smoke_test.py` hits the real GLM API.

---

## 11. Validation Criteria

| Criterion | Target |
|-----------|--------|
| Differentiated behavior | Andi trades > 3x more frequently than Pak Budi |
| Portfolio conservation | No agent has negative cash or phantom shares |
| LLM JSON parse rate | > 90% valid JSON responses |
| Simulation completes | 65 trading days without crash |
| Cost per run | Documented, alert if > $5 |
| Smoke test | Passes in < 2 minutes |

---

## 12. What's Deferred to Phase 2+

- Episodic memory (vector-retrieved for T1, sliding window for T2)
- Social memory and trust score dynamics
- Causal knowledge graph retrieval
- Multi-run parallel execution
- Observer/Analyst agent
- Prediction mode with price impact model
- Dr. Lim and MarketBot personas
- dividend_holder and sector_rotator archetypes
- FastAPI routes
- Frontend dashboard
- News scraping automation
- PostgreSQL migration
- OpenViking integration
