# IMSS — Phase 1 Pending Items & Phase 2 Roadmap

**Date:** 2026-03-18
**Status:** Phase 1 complete (29/29 tests passing), ready for Phase 2

---

## Phase 1 — Pending / Known Issues

These items were identified during Phase 1 implementation but intentionally deferred to keep the initial delivery focused.

### Code-Level Items

| # | Item | Location | Priority | Notes |
|---|------|----------|----------|-------|
| P1 | **P&L calculation is placeholder** | `imss/simulation/engine.py:227` | Medium | `pnl_pct` is always `0.0`. Needs initial portfolio value tracking per agent to compute actual realized + unrealized P&L. |
| P2 | **Tier 2 agents not batched** | `imss/simulation/loop.py:132-135` | Medium | Tier 2 LLM calls execute sequentially. The `LLMBatcher` is initialized but not wired into the loop. Should use `batcher.execute_batch()` for T2 concurrency. |
| P3 | **`datetime.utcnow()` deprecation** | `imss/db/models.py`, `imss/simulation/engine.py:237` | Low | Replace with `datetime.now(datetime.UTC)` per Python 3.12 deprecation. |
| P4 | **No Alembic migrations** | `imss/db/` | Low | Tables created via `create_all()`. Introduce Alembic before PostgreSQL migration in Phase 3. |
| P5 | **Embedder has no unit test** | `imss/data/embedder.py` | Low | Requires live GLM API key. Could add a mock test. |
| P6 | **Seed data script not validated end-to-end** | `scripts/imss_seed_data.py` | Medium | Requires GLM_API_KEY for embeddings. Need to run once with real credentials and verify DB + ChromaDB state. |
| P7 | **`causal_links` table empty** | `imss/db/models.py` | N/A | Schema only — populated in Phase 2 with causal knowledge graph. |
| P8 | **No cost alerting** | `imss/config.py:IMSS_COST_ALERT_THRESHOLD_USD` | Low | Config exists but no runtime check against threshold. Add warning when cumulative cost exceeds threshold during simulation. |
| P9 | **structlog not wired** | `imss/` | Low | `structlog` is in requirements but all modules use stdlib `logging`. Switch to structlog for structured JSON logs in Phase 2. |
| P10 | **ARA/ARB price limits not enforced** | `imss/simulation/order_book.py` | Medium | IDX ±20%/±35% daily price limits exist in spec but not enforced in backtest mode (prices come from historical data). Relevant for prediction mode in Phase 3. |

### Missing Personas / Archetypes (Deferred from Spec)

| Agent | Tier | Spec Reference | Phase |
|-------|------|----------------|-------|
| Dr. Lim (Quantitative Analyst) | 1 | Implementation Guide §6.1 | 2 |
| MarketBot (Algorithmic Trader) | 1 | Implementation Guide §6.1 | 2 |
| dividend_holder | 2 | Implementation Guide §6.2 | 2 |
| sector_rotator | 2 | Implementation Guide §6.2 | 2 |

### Test Coverage Gaps

- No test for `imss/simulation/propagation.py` (event delay filtering)
- No test for `imss/simulation/loop.py` (turn-based loop logic, tested only via engine integration)
- No test for `imss/agents/tier1/personas.py` or `imss/agents/tier2/archetypes.py` (LLM-dependent, tested only via engine mock)
- No negative/edge-case tests for engine (empty events, single trading day, missing stock data)

---

## Phase 2 — Memory & Intelligence

**Goal:** Add episodic memory, social memory, causal knowledge graph, multi-run support, and 2 new personas.

**Prerequisite:** Run `scripts/imss_seed_data.py` with live GLM_API_KEY and validate smoke test passes end-to-end.

### 2.1 Episodic Memory

- [ ] **Tier 1**: ChromaDB-based episodic memory — store each decision + outcome as an embedding, retrieve top-K similar past situations during decision prompt
- [ ] **Tier 2**: Sliding-window episodic memory — last N decisions stored in working memory, no vector retrieval
- [ ] Add episodic memory sections back to Tier 1 prompt templates (stripped in Phase 1 to save ~60K tokens/run)
- [ ] Config flags: `enable_episodic_memory` already exists in `SimulationConfig`

### 2.2 Social Memory & Trust Dynamics

- [ ] Track inter-agent trust scores (who was right, who was wrong)
- [ ] Tier 1 agents can reference "what other agents did" in decisions
- [ ] Social influence propagation: Tier 1 decisions influence Tier 2 sentiment
- [ ] Config flag: `enable_social_memory` already exists in `SimulationConfig`

### 2.3 Causal Knowledge Graph

- [ ] Populate `causal_links` table from historical event-price correlations
- [ ] Build retrieval pipeline: given current events, find historical parallels via causal graph
- [ ] Integrate into Tier 1 decision prompt as "Historical Parallels" section
- [ ] Config flag: `enable_causal_retrieval` already exists in `SimulationConfig`

### 2.4 New Agents

- [ ] **Dr. Lim** (Tier 1): Quantitative analyst persona — data-driven, statistical, skeptical of narrative
- [ ] **MarketBot** (Tier 1): Algorithmic trader — follows systematic rules, no emotion
- [ ] **dividend_holder** (Tier 2): Yield-focused, holds for dividends, rarely trades
- [ ] **sector_rotator** (Tier 2): Rotates between sectors based on macro signals

### 2.5 Multi-Run Parallel Execution

- [ ] Execute N simulation runs with different random seeds
- [ ] Aggregate results across runs (mean, std, confidence intervals)
- [ ] Basic Observer agent that analyzes cross-run patterns
- [ ] Wire `num_parallel_runs` and `runs_batch_size` config (already in `SimulationConfig`)

### 2.6 OpenViking Integration (Optional)

Per `docs/IMSS_Update_OpenViking_Integration.md`:
- [ ] Evaluate replacing ChromaDB event store with OpenViking `viking://resources/events/` hierarchy
- [ ] Evaluate replacing manual episodic memory with OpenViking session management
- [ ] Decision gate: only proceed if OpenViking provides clear benefit over current ChromaDB approach

### 2.7 Phase 1 Debt Cleanup

- [ ] Fix P1: Implement proper P&L tracking per agent
- [ ] Fix P2: Wire LLMBatcher for Tier 2 concurrent calls
- [ ] Fix P3: Replace `datetime.utcnow()` with `datetime.now(datetime.UTC)`
- [ ] Fix P5: Add mock test for embedder
- [ ] Fix P6: Validate seed data script end-to-end
- [ ] Fix P8: Add cost alert warning during simulation
- [ ] Fix P9: Switch to structlog
- [ ] Add missing unit tests for propagation, loop, Tier 1/2 agents

---

## Phase 3 — Integration & Scale (Future)

- FastAPI backend with simulation CRUD endpoints
- Prediction mode with price impact model
- Screener output → simulation input pipeline
- Simulation output → virtual trading module
- Multi-stock support (BBRI, BMRI, BBCA, TLKM, ASII)
- Automated news scraper for Indonesian financial sources
- LLM-based entity tagging and sentiment scoring pipeline
- Event injection API for "what-if" scenarios
- PostgreSQL migration (introduce Alembic)
- ARA/ARB price limit enforcement for prediction mode

## Phase 4 — UI & Polish (Future)

- Simulation dashboard (run status, progress, cost tracking)
- Scenario report viewer (Observer agent output)
- Agent network visualization (D3.js social graph)
- Post-simulation agent chat interface
- WebSocket support for live simulation streaming
- Performance optimization (batching, caching, prompt compression)
