# IDX Market Swarm Simulator (IMSS)

## A Multi-Agent Market Intelligence Engine for Indonesian Stock Exchange Prediction

**Version:** 0.1.0-draft
**Author:** Ahmad Whafa Azka Al Azkiyai
**Date:** March 2026
**Status:** Specification & Architecture Document

---

## 1. Executive Summary

The IDX Market Swarm Simulator (IMSS) is a domain-specific multi-agent simulation engine designed for the Indonesian Stock Exchange. Unlike general-purpose swarm prediction systems (e.g., MiroFish), IMSS is hardened for financial market dynamics — integrating real IDX price data, Indonesian financial news, and OJK/BI regulatory signals into a simulation environment where LLM-powered agents with persistent memory, social awareness, and learned historical correlations produce scenario-based market forecasts.

IMSS operates in two modes: **Backtesting Mode** (replaying historical data to calibrate agent behavior and build causal memory) and **Prediction Mode** (forward simulation producing probability-weighted scenario distributions). The system integrates directly into an existing IDX automated trading pipeline as a scenario analysis module.

### Key Differentiators

- Domain-specialized for IDX rather than general-purpose prediction
- Three-tier agent architecture balancing simulation fidelity with compute cost
- Information-price causal memory that learns correlations between news events and price movements through backtesting
- Social memory with trust dynamics producing realistic herding, panic cascades, and information asymmetry
- Multi-LLM cost-optimized routing (cheap models for swarm, premium models for analysis)
- Direct integration with real IDX data feeds and an existing trading system

---

## 2. System Goals

### 2.1 Primary Goals

- Produce actionable scenario-based forecasts for IDX stocks that outperform naive sentiment analysis
- Serve as a scenario analysis module within an existing IDX trading system (v3.1)
- Demonstrate emergent market phenomena (herding, panic cascades, information asymmetry, mean reversion) through agent interaction
- Build and maintain a causal knowledge base linking Indonesian financial events to historical price movements

### 2.2 Portfolio & Career Goals

- Showcase multi-agent orchestration, LLM integration at scale, real financial data pipelines, and cost-aware model routing
- Target audience: AI Engineer hiring managers at Indonesian fintechs, digital banks, and AI startups
- Demonstrate production-grade engineering (not a toy demo)

### 2.3 Trading Integration Goals

- Accept stock screening output and sentiment signals as simulation seeds
- Return probability-weighted scenario distributions consumable by a virtual trading module
- Support "what-if" event injection for manual scenario exploration

---

## 3. Architecture Overview

### 3.1 High-Level System Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                        IMSS - System Overview                       │
│                                                                     │
│  ┌──────────────┐    ┌──────────────┐    ┌───────────────────────┐  │
│  │  Data Ingest  │───▶│  Knowledge   │───▶│  Simulation Engine   │  │
│  │    Layer      │    │    Graph     │    │                       │  │
│  │              │    │   + Causal   │    │  ┌─────────────────┐  │  │
│  │ • IDX Prices │    │    Memory    │    │  │  Tier 1: Named  │  │  │
│  │ • News/Events│    │              │    │  │  (5-10 agents)  │  │  │
│  │ • OJK/BI     │    │ • Entities   │    │  ├─────────────────┤  │  │
│  │ • Macro Data │    │ • Events     │    │  │  Tier 2: Typed  │  │  │
│  └──────────────┘    │ • Causal     │    │  │  (30-50 agents) │  │  │
│                      │   Links      │    │  ├─────────────────┤  │  │
│                      │ • Embeddings │    │  │  Tier 3: Stat   │  │  │
│                      └──────────────┘    │  │  (100s agents)  │  │  │
│                                          │  └─────────────────┘  │  │
│                                          └──────────┬────────────┘  │
│                                                     │               │
│                      ┌──────────────────────────────▼────────────┐  │
│                      │          Observer / Analyst Agent          │  │
│                      │  • Multi-run aggregation                  │  │
│                      │  • Scenario probability distribution      │  │
│                      │  • Report generation                      │  │
│                      └──────────────────────────────┬────────────┘  │
│                                                     │               │
│  ┌──────────────────────────────────────────────────▼────────────┐  │
│  │                    Output / Integration Layer                  │  │
│  │  • Scenario reports   • Trading system API   • Interactive UI │  │
│  └───────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 Component Breakdown

| Component | Responsibility | Key Technologies |
|---|---|---|
| Data Ingest Layer | Collect, normalize, and timestamp all market data, news, and regulatory signals | Python, REST APIs, web scrapers, scheduled jobs |
| Knowledge Graph + Causal Memory | Store entities, events, price actions, and learned causal links; provide semantic retrieval | PostgreSQL + TimescaleDB, ChromaDB/Qdrant (vector), NetworkX (graph) |
| Simulation Engine | Orchestrate agent lifecycle, manage turn-based execution, handle concurrent LLM calls | Python async, agent framework, batched LLM calls |
| Agent Layer (Tiers 1-3) | Individual decision-making entities with varying levels of intelligence and memory | Multi-LLM harness (GLM, DeepSeek, Claude) |
| Observer / Analyst Agent | Aggregate simulation runs, detect emergent patterns, generate scenario reports | Claude API (premium model) |
| Output / Integration Layer | Serve results to trading system, render UI, expose API | FastAPI, React/Vue dashboard |

---

## 4. Agent Architecture

### 4.1 Three-Tier Agent Design

The agent population is stratified into three tiers to balance simulation fidelity with computational cost.

#### Tier 1 — Named Agents (5-10 per simulation)

- **Role:** Major institutional players, key market makers, influential analysts
- **LLM:** DeepSeek-V3 or Claude Haiku (higher quality, moderate cost)
- **Memory:** Full episodic memory (vector-retrieved) + full social memory (relationship map with trust scores) + access to causal knowledge graph
- **Decision process:** Multi-step reasoning with historical parallel retrieval
- **State:** Detailed portfolio, risk parameters, investment thesis, behavioral profile
- **Example personas:**
  - Conservative state-owned bank fund manager (risk-averse, regulatory-sensitive)
  - Aggressive foreign institutional investor (momentum-driven, USD/IDR sensitive)
  - Contrarian value fund (mean-reversion strategy, long time horizon)
  - Market maker (liquidity provider, spread-sensitive)

#### Tier 2 — Typed Agents (30-50 per simulation)

- **Role:** Mid-market participants — fund managers, active retail traders, sector specialists
- **LLM:** GLM-4 or equivalent (cost-optimized)
- **Memory:** Lightweight social memory (tracks Tier 1 agents only, JSON structure, no vector retrieval) + simplified episodic memory (last N events only, sliding window)
- **Decision process:** Single-step LLM call with structured context
- **State:** Simplified portfolio, behavioral type tag, sentiment score
- **Example archetypes:**
  - Retail momentum chaser
  - Retail panic seller
  - Sector rotation trader
  - Dividend-focused retail investor
  - News-reactive day trader

#### Tier 3 — Statistical Agents (100-500 per simulation)

- **Role:** Market liquidity and volume realism
- **LLM:** None — purely rule-based
- **Memory:** None
- **Decision process:** Parameterized heuristics (momentum following, mean reversion, random walk, volume-weighted)
- **State:** Position size, behavioral parameters (derived from real IDX order flow statistics)
- **Purpose:** Provide realistic market microstructure without LLM cost

### 4.2 Agent Base Properties

Every agent (regardless of tier) has the following base properties:

```
Agent:
  id: unique identifier
  tier: 1 | 2 | 3
  persona_type: string (e.g., "institutional_conservative", "retail_momentum")
  
  # Portfolio State
  holdings: map<stock_symbol, quantity>
  cash: float (IDR)
  portfolio_value_history: list<timestamped_value>
  
  # Behavioral Parameters
  risk_tolerance: float [0.0 - 1.0]
  information_access_level: enum [HIGH, MEDIUM, LOW]
  decision_latency: int (simulation steps delay before reacting to events)
  sentiment_bias: float [-1.0 to 1.0] (bearish to bullish)
  
  # Social Graph (Tier 1-2 only)
  observation_targets: list<agent_id>
  trust_scores: map<agent_id, float>
  social_memory: list<social_observation_record>
  
  # Episodic Memory (Tier 1-2 only)
  episodic_memory_store: vector_store_reference | sliding_window
  
  # Action Interface
  observe(environment_state) -> observation
  decide(observation, memory_context) -> action
  execute(action) -> updated_state
  reflect(outcome) -> memory_update
```

### 4.3 Agent Decision Flow

```
For each simulation step:
  
  1. OBSERVE
     ├── Current price data for observed stocks
     ├── New events injected this step (news, regulatory, macro)
     ├── Observable actions from other agents (based on social graph)
     └── Market aggregate signals (volume, breadth, sector performance)
  
  2. RETRIEVE (Tier 1-2 only)
     ├── Query causal knowledge graph: "events similar to current situation"
     │   └── Returns: historical parallels with price outcomes
     ├── Query episodic memory: "my past experience in similar contexts"
     │   └── Returns: personal history of decisions and outcomes
     └── Query social memory: "what are trusted agents doing?"
         └── Returns: recent actions of observed agents + trust scores
  
  3. DECIDE
     ├── Tier 1: Full LLM reasoning with all retrieved context
     ├── Tier 2: LLM call with structured context (social + simplified history)
     └── Tier 3: Rule-based heuristic evaluation
     
     Output: Action = { type: BUY|SELL|HOLD, stock, quantity, reasoning }
  
  4. EXECUTE
     ├── Update portfolio state
     ├── Register action in simulation order book
     └── Broadcast action to observing agents (with propagation delay)
  
  5. REFLECT (Tier 1-2 only)
     ├── Evaluate outcome of past decisions against actual price movement
     ├── Update episodic memory with new experience
     ├── Update trust scores for observed agents based on their prediction accuracy
     └── Strengthen/weaken causal links in knowledge graph
```

---

## 5. Memory System

### 5.1 Working Memory (All Tiers)

In-memory structured state maintained during a simulation run. Reset between runs unless explicitly carried over for multi-session backtesting.

```
WorkingMemory:
  current_step: int
  portfolio_snapshot: { holdings, cash, unrealized_pnl }
  recent_actions: deque(maxlen=10)  # own actions
  recent_observations: deque(maxlen=20)  # observed events and prices
  current_sentiment: float
  active_theses: list<string>  # Tier 1 only — current investment reasoning
```

### 5.2 Episodic Memory (Tier 1 Full, Tier 2 Simplified)

**Tier 1 — Vector-Retrieved Episodic Memory:**
- Each decision and its outcome is summarized into a text record and embedded
- Stored in a per-agent namespace within a vector store (ChromaDB or Qdrant)
- At decision time, the current situation is embedded and top-K similar past experiences are retrieved
- Maximum store size: 500 records per agent (oldest pruned)

**Tier 2 — Sliding Window Episodic Memory:**
- Last N decision-outcome pairs stored as structured JSON (no embeddings)
- N = 20 by default
- Simple keyword matching rather than semantic retrieval
- Sufficient for pattern recognition without vector store overhead

### 5.3 Social Memory (Tier 1-2 Only)

```
SocialMemory:
  relationships: map<agent_id, SocialRecord>
  
SocialRecord:
  agent_id: string
  observed_actions: deque(maxlen=20)  # what this agent did recently
  prediction_accuracy: float  # how often following this agent was profitable
  trust_score: float [0.0 - 1.0]  # Bayesian updated based on outcomes
  last_interaction_step: int
  notes: list<string>  # Tier 1 only — LLM-generated observations about this agent
```

**Trust Score Update Rule:**
After each simulation step where Agent A observed Agent B's action and can evaluate the outcome:
- If B's action (e.g., buying) aligned with a profitable outcome → increase trust
- If B's action aligned with a loss → decrease trust
- Decay toward 0.5 over time (uncertainty grows without new observations)

**Observation Network Rules:**
- Tier 1 agents can observe all Tier 1 agents and a random subset of Tier 2
- Tier 2 agents can observe all Tier 1 agents only
- Tier 3 agents observe nothing (no social memory)
- Observation has a configurable delay (1-3 steps) to simulate information propagation latency

### 5.4 Causal Knowledge Graph (System-Level)

This is not per-agent memory — it's a shared knowledge base that agents query.

```
Event Node:
  id: unique
  timestamp: datetime
  category: enum [REGULATORY, EARNINGS, MACRO, NEWS, RUMOR, POLITICAL]
  source: string
  title: string
  summary: string
  embedding: vector
  affected_entities: list<stock_symbol | sector>
  sentiment_score: float [-1.0 to 1.0]
  magnitude_score: float [0.0 to 1.0]  # how significant

PriceAction Node:
  stock_symbol: string
  timestamp: datetime
  open, high, low, close, volume: float
  derived_signals: { volatility, volume_spike, sector_relative_strength }

CausalLink Edge:
  event_id: reference
  price_action_id: reference
  lag_days: int  # how many days between event and price reaction
  correlation_strength: float [0.0 to 1.0]
  direction: enum [POSITIVE, NEGATIVE, NEUTRAL]
  confidence: float  # increases with more backtest confirmations
  sample_count: int  # how many backtest runs confirmed this link
```

---

## 6. Simulation Engine

### 6.1 Simulation Configuration

```
SimulationConfig:
  # Scope
  target_stocks: list<string>  # e.g., ["BBRI", "BMRI", "BBCA"]
  simulation_period: { start_date, end_date }  # for backtesting
  prediction_horizon: int  # days forward for prediction mode
  
  # Agent Population
  tier1_agents: list<AgentConfig>  # manually defined personas
  tier2_count: int  # auto-generated from archetype templates
  tier2_archetype_distribution: map<archetype, float>  # e.g., {"momentum": 0.3, "panic_seller": 0.2, ...}
  tier3_count: int
  tier3_behavioral_params: StatisticalAgentConfig
  
  # Simulation Parameters
  steps_per_day: int  # granularity (1 = daily, default)
  total_steps: int  # auto-calculated or manually set
  num_parallel_runs: int  # for prediction mode Monte Carlo (default: 100)
  
  # Event Injection
  seed_events: list<Event>  # for prediction mode — hypothetical events to test
  use_historical_events: bool  # for backtesting mode
  
  # LLM Configuration
  tier1_model: string  # e.g., "deepseek-v3"
  tier2_model: string  # e.g., "glm-4"
  analyst_model: string  # e.g., "claude-sonnet-4-20250514"
  max_concurrent_llm_calls: int
  
  # Memory Configuration
  enable_episodic_memory: bool
  enable_social_memory: bool
  enable_causal_retrieval: bool
  episodic_top_k: int  # how many past experiences to retrieve (default: 5)
  causal_top_k: int  # how many historical parallels to retrieve (default: 3)
```

### 6.2 Simulation Loop (Turn-Based)

```
For each simulation step t:
  
  1. ENVIRONMENT UPDATE
     ├── Inject price data for step t (historical in backtest, projected in prediction)
     ├── Inject any events scheduled for step t
     └── Update market aggregate signals
  
  2. INFORMATION PROPAGATION
     ├── Distribute events to agents based on information_access_level
     ├── Apply decision_latency delays per agent
     └── Propagate Tier 1 actions to Tier 2 observers (with delay)
  
  3. AGENT EXECUTION (parallelized by tier)
     ├── Tier 3 first (instant, rule-based, updates order book)
     ├── Tier 2 next (batched LLM calls)
     └── Tier 1 last (individual LLM calls with full context)
  
  4. ORDER RESOLUTION
     ├── Aggregate all buy/sell orders
     ├── Calculate price impact based on order imbalance
     │   (in prediction mode — backtesting uses real prices)
     └── Execute fills, update agent portfolios
  
  5. POST-STEP
     ├── Agents reflect and update memory
     ├── Update causal link strengths (backtesting mode)
     ├── Log simulation state for observer agent
     └── Check termination conditions
```

### 6.3 Price Impact Model (Prediction Mode Only)

In backtesting mode, real prices are used. In prediction mode, the simulation needs to generate plausible price movements based on agent behavior:

```
PriceImpactModel:
  # Net order imbalance drives price
  order_imbalance = (total_buy_volume - total_sell_volume) / total_volume
  
  # Price change = f(imbalance, volatility, liquidity)
  price_change_pct = order_imbalance * volatility_multiplier * liquidity_factor
  
  # Add noise calibrated from historical IDX intraday volatility
  noise = normal(0, historical_volatility_for_stock)
  
  # Final price
  new_price = prev_price * (1 + price_change_pct + noise)
```

This model should be calibrated per-stock using historical order flow data.

---

## 7. Operating Modes

### 7.1 Backtesting Mode

**Purpose:** Calibrate agent behavior, build causal memory, validate simulation against historical reality.

**Input:**
- Target stock(s) and date range
- Historical price data (OHLCV)
- Historical events (news, regulatory, earnings) with timestamps

**Process:**
1. Initialize agent population with default parameters
2. Run simulation over historical period using real prices
3. At each step, agents make decisions; track their P&L against the market
4. After the run, compare emergent statistics (aggregate agent sentiment vs actual price direction, volume patterns, reaction timing) against historical ground truth
5. Update causal links based on observed event → price relationships
6. Output calibration metrics

**Output:**
- Calibration report: agent population accuracy, parameter recommendations
- Updated causal knowledge graph with strengthened/weakened links
- Agent memory stores populated with historical experience

**Validation Metrics:**
- Direction accuracy: % of steps where aggregate agent sentiment correctly predicted next-step price direction
- Magnitude calibration: how closely simulated volatility matches historical volatility
- Event reaction timing: whether agents reacted to events with appropriate lag
- Emergent pattern detection: did herding/cascades occur at historically appropriate moments

### 7.2 Prediction Mode

**Purpose:** Generate forward-looking scenario distributions for trading decisions.

**Input:**
- Target stock(s)
- Current market state (latest prices, recent events)
- Seed events: either breaking news or hypothetical "what-if" scenarios
- Prediction horizon (e.g., 5 days, 20 days)

**Process:**
1. Load calibrated agent population (with memory from backtesting)
2. Run N parallel simulations (default: 100) with controlled randomness
   - Agent decisions vary due to LLM temperature
   - Tier 3 statistical agents add stochastic volume noise
   - Information propagation timing varies slightly
3. For each run, record price trajectory and key agent decisions
4. Observer/Analyst agent aggregates across all runs

**Output:**
- Probability distribution of price outcomes at prediction horizon
- Key scenario clusters with narrative explanations
- Confidence intervals (e.g., "70% probability BBRI stays between 4,800-5,200 over 5 days")
- Identified risk events and their estimated impact

---

## 8. Data Requirements

### 8.1 Price Data

| Data | Source | Granularity | History Needed |
|---|---|---|---|
| IDX OHLCV | Existing trading system / IDX API / Yahoo Finance | Daily | 2+ years |
| Volume profiles | IDX order flow / broker data | Daily | 1+ year (for Tier 3 calibration) |
| Sector indices | IDX sectoral data | Daily | 2+ years |

### 8.2 Event / News Data

| Data | Source | Format | History Needed |
|---|---|---|---|
| Indonesian financial news | Kontan, Bisnis Indonesia, CNBC Indonesia | Scraped articles → timestamped + entity-tagged + sentiment-scored | 2+ years |
| OJK announcements | ojk.go.id official releases | Structured regulatory text | 2+ years |
| BI rate decisions & statements | bi.go.id | Structured macro data | 2+ years |
| Company earnings releases | IDX filings, RTI Business | Structured financial data | 2+ years |
| USD/IDR exchange rate | BI, forex APIs | Daily | 2+ years |
| Commodity prices (coal, palm oil, nickel) | Investing.com, Bloomberg | Daily | 2+ years (for IDX resource sector) |

### 8.3 Data Processing Pipeline

```
Raw Data → Ingestion → Processing → Storage

1. INGESTION
   ├── Scheduled scrapers for news sources (daily)
   ├── API pulls for price data (daily close)
   ├── Manual upload for historical regulatory documents
   └── Macro data feeds (BI rate, commodities, FX)

2. PROCESSING
   ├── Entity extraction: which stocks/sectors does this event affect?
   │   (LLM-based entity tagging in batch)
   ├── Sentiment scoring: LLM-based or fine-tuned classifier
   ├── Magnitude scoring: how significant is this event? (LLM-based)
   ├── Embedding generation: for semantic similarity retrieval
   └── Timestamp normalization: all events mapped to IDX trading days

3. STORAGE
   ├── TimescaleDB: price time-series data
   ├── PostgreSQL: event records, agent configurations, simulation results
   ├── ChromaDB/Qdrant: event embeddings for semantic retrieval
   └── File storage: raw scraped content, simulation logs
```

---

## 9. Information Propagation & Delay Model

Events do not reach all agents simultaneously. This models real-world information asymmetry.

```
Information Propagation Rules:

  HIGH access (Tier 1 institutional agents):
    - Receive regulatory events at step T+0
    - Receive major news at step T+0
    - Receive earnings at step T+0
    - Receive rumors at step T+1
    
  MEDIUM access (Tier 2 active retail):
    - Receive regulatory events at step T+1
    - Receive major news at step T+0 to T+1 (random)
    - Receive earnings at step T+1
    - Receive rumors at step T+0 to T+2 (random)
    
  LOW access (Tier 3 statistical):
    - Receive no events directly
    - React only to price changes (momentum/mean-reversion heuristics)

Social Signal Propagation:
  - When a Tier 1 agent acts, observing Tier 2 agents see it at step T+1 to T+3
  - Propagation creates a cascade: Tier 1 acts → Tier 2 follows → Tier 3 reacts to price
  - This naturally produces realistic herding dynamics
```

---

## 10. Observer / Analyst Agent

### 10.1 Role

A non-trading meta-agent that watches all simulation runs and synthesizes findings. Runs on a premium model (Claude) because output quality matters most here.

### 10.2 Capabilities

- Access to all simulation run logs (agent decisions, price trajectories, event timelines)
- Statistical aggregation across N parallel runs
- Pattern recognition: identifying scenario clusters, bifurcation points, consensus/divergence moments
- Natural language report generation
- Interactive Q&A about simulation results

### 10.3 Report Structure

```
Scenario Analysis Report:
  
  1. Executive Summary
     - Target stock(s) and prediction horizon
     - Overall directional bias with confidence
     - Key risk factors identified
  
  2. Scenario Clusters
     - Cluster A (probability X%): description, price range, key drivers
     - Cluster B (probability Y%): description, price range, key drivers
     - ...
  
  3. Critical Events & Sensitivities
     - Which injected events had the largest impact?
     - How sensitive are outcomes to specific agent behaviors?
     - Identified bifurcation points (where scenarios diverge)
  
  4. Agent Behavior Summary
     - Institutional consensus/divergence
     - Retail sentiment trajectory
     - Herding events detected
  
  5. Historical Parallel Analysis
     - Most relevant historical precedents from causal memory
     - How current situation differs from historical parallels
  
  6. Actionable Recommendations
     - Suggested position sizing given scenario distribution
     - Key signals to watch that would confirm/invalidate scenarios
     - Recommended re-simulation triggers
```

---

## 11. Trading System Integration

### 11.1 Integration Points with IDX Trading System v3.1

```
Trading System v3.1                    IMSS
                                        
  Stock Screener ─────────────────────▶ target_stocks input
  Sentiment Dashboard ────────────────▶ seed_events / market mood
  
                                        ┌─────────────────────┐
                                        │ Simulation Engine    │
                                        │ runs N scenarios     │
                                        └──────────┬──────────┘
                                                   │
  Virtual Trading Module ◀────────────── scenario_distribution output
  Risk Management ◀───────────────────── risk_factors output
  Alert System ◀──────────────────────── critical_signals output
```

### 11.2 API Contract

```
POST /api/simulate
  Request:
    target_stocks: list<string>
    mode: "backtest" | "predict"
    prediction_horizon_days: int (prediction mode)
    backtest_range: { start, end } (backtest mode)
    seed_events: list<Event> (optional, prediction mode)
    num_runs: int (default: 100)
    
  Response:
    simulation_id: string
    status: "running" | "completed" | "failed"

GET /api/simulation/{id}/results
  Response:
    scenario_clusters: list<ScenarioCluster>
    price_distribution: { p10, p25, p50, p75, p90 }
    directional_bias: { direction, confidence }
    risk_factors: list<RiskFactor>
    report_markdown: string
    
POST /api/simulate/{id}/inject-event
  Purpose: Mid-simulation event injection for interactive exploration
  Request:
    event: Event
    
GET /api/simulation/{id}/agent/{agent_id}/chat
  Purpose: Interactive conversation with any agent post-simulation
  Request:
    message: string
  Response:
    agent_response: string
```

---

## 12. Technology Stack

### 12.1 Core Technologies

| Layer | Technology | Rationale |
|---|---|---|
| Language | Python 3.11+ | Ecosystem for ML/AI, async support |
| Web Framework | FastAPI | Async, high performance, auto-docs |
| Task Queue | Celery + Redis | Long-running simulation orchestration |
| Database (relational) | PostgreSQL + TimescaleDB | Time-series price data, simulation records |
| Vector Store | ChromaDB (dev) / Qdrant (prod) | Event embeddings, episodic memory retrieval |
| Graph (in-memory) | NetworkX | Agent social graph, entity relationships |
| Cache / State | Redis | Simulation state, agent working memory |
| Frontend | Vue.js or React | Dashboard and interactive UI |
| Containerization | Docker + Docker Compose | Reproducible deployment |

### 12.2 LLM Configuration

| Role | Model | Provider | Rationale |
|---|---|---|---|
| Tier 1 agents | DeepSeek-V3 | DeepSeek API | High quality, cost-effective for complex reasoning |
| Tier 2 agents | GLM-4 | Zhipu API | Cheapest viable model for structured decisions |
| Tier 3 agents | None | N/A | Rule-based, no LLM needed |
| Observer/Analyst | Claude Sonnet | Anthropic API | Best reasoning for synthesis and report generation |
| Data processing (entity extraction, sentiment) | GLM-4 or DeepSeek | Batch processing | High volume, moderate quality sufficient |

### 12.3 LLM Cost Estimation (per simulation run)

```
Assumptions: 50 steps, 100 parallel runs

Tier 1 (10 agents × 50 steps × 1 run):       500 calls × ~2K tokens = 1M tokens
Tier 2 (40 agents × 50 steps × 1 run):       2,000 calls × ~800 tokens = 1.6M tokens  
Observer (1 agent × 1 call per run):          100 calls × ~4K tokens = 400K tokens

Per-run total: ~3M tokens
100 parallel runs: ~300M tokens

At typical API pricing:
  - DeepSeek-V3 (Tier 1): ~$0.30/M input → ~$0.15
  - GLM-4 (Tier 2): ~$0.10/M input → ~$0.16
  - Claude Sonnet (Observer): ~$3/M input → ~$1.20
  
Estimated total per 100-run prediction: ~$1.50 - $3.00

Note: Actual costs depend on current pricing. Budget ceiling: $5 per prediction run.
```

---

## 13. Data Schema

### 13.1 PostgreSQL Tables

```
-- Price data (TimescaleDB hypertable)
stocks_ohlcv:
  symbol          VARCHAR(10)
  timestamp       TIMESTAMPTZ
  open            DECIMAL
  high            DECIMAL
  low             DECIMAL
  close           DECIMAL
  volume          BIGINT
  adjusted_close  DECIMAL

-- Events
events:
  id              UUID PRIMARY KEY
  timestamp       TIMESTAMPTZ
  category        VARCHAR(20)  -- REGULATORY, EARNINGS, MACRO, NEWS, RUMOR, POLITICAL
  source          VARCHAR(100)
  title           TEXT
  summary         TEXT
  raw_content     TEXT
  sentiment_score DECIMAL
  magnitude_score DECIMAL
  embedding_id    VARCHAR(100)  -- reference to vector store

-- Event-entity associations
event_entities:
  event_id        UUID REFERENCES events
  entity_type     VARCHAR(10)  -- STOCK, SECTOR
  entity_symbol   VARCHAR(20)

-- Causal links (learned through backtesting)
causal_links:
  id              UUID PRIMARY KEY
  event_id        UUID REFERENCES events
  stock_symbol    VARCHAR(10)
  lag_days        INT
  direction       VARCHAR(10)  -- POSITIVE, NEGATIVE, NEUTRAL
  correlation     DECIMAL
  confidence      DECIMAL
  sample_count    INT
  last_updated    TIMESTAMPTZ

-- Simulation runs
simulation_runs:
  id              UUID PRIMARY KEY
  config          JSONB
  mode            VARCHAR(10)  -- BACKTEST, PREDICT
  status          VARCHAR(20)
  started_at      TIMESTAMPTZ
  completed_at    TIMESTAMPTZ
  results_summary JSONB

-- Agent configurations
agent_configs:
  id              UUID PRIMARY KEY
  simulation_id   UUID REFERENCES simulation_runs
  tier            INT
  persona_type    VARCHAR(50)
  parameters      JSONB

-- Simulation step logs
simulation_steps:
  simulation_id   UUID REFERENCES simulation_runs
  run_number      INT
  step_number     INT
  timestamp       TIMESTAMPTZ  -- simulated time
  market_state    JSONB
  agent_actions   JSONB
  events_active   JSONB
```

### 13.2 Vector Store Collections

```
Collection: event_embeddings
  id: event UUID
  vector: embedding of event summary
  metadata: { category, timestamp, sentiment, magnitude, entities }

Collection: agent_episodic_{agent_id}
  id: experience UUID
  vector: embedding of experience summary
  metadata: { step, action, outcome, pnl_impact, timestamp }
```

---

## 14. Project Structure

```
idx-market-swarm/
├── README.md
├── docker-compose.yml
├── .env.example
├── pyproject.toml
│
├── imss/                          # Core Python package
│   ├── __init__.py
│   ├── config.py                  # Global configuration & env vars
│   │
│   ├── agents/                    # Agent definitions
│   │   ├── __init__.py
│   │   ├── base.py                # Base agent class
│   │   ├── tier1/                 # Named agent implementations
│   │   │   ├── institutional.py
│   │   │   ├── market_maker.py
│   │   │   └── personas.py        # Persona templates
│   │   ├── tier2/                 # Typed agent implementations
│   │   │   ├── retail.py
│   │   │   └── archetypes.py      # Archetype templates
│   │   └── tier3/                 # Statistical agents
│   │       ├── heuristic.py
│   │       └── calibration.py
│   │
│   ├── memory/                    # Memory systems
│   │   ├── __init__.py
│   │   ├── working.py             # In-memory working state
│   │   ├── episodic.py            # Vector-based episodic memory
│   │   ├── social.py              # Social memory & trust scores
│   │   └── causal.py              # Causal knowledge graph interface
│   │
│   ├── simulation/                # Simulation engine
│   │   ├── __init__.py
│   │   ├── engine.py              # Main simulation orchestrator
│   │   ├── loop.py                # Turn-based simulation loop
│   │   ├── order_book.py          # Order aggregation & price impact
│   │   ├── propagation.py         # Information propagation model
│   │   └── runner.py              # Multi-run parallel executor
│   │
│   ├── observer/                  # Observer / Analyst agent
│   │   ├── __init__.py
│   │   ├── aggregator.py          # Cross-run statistical aggregation
│   │   ├── pattern_detector.py    # Scenario clustering
│   │   └── report_generator.py    # Report synthesis via Claude
│   │
│   ├── data/                      # Data ingestion & processing
│   │   ├── __init__.py
│   │   ├── price_feed.py          # IDX price data ingestion
│   │   ├── news_scraper.py        # News collection
│   │   ├── regulatory_feed.py     # OJK / BI announcements
│   │   ├── macro_feed.py          # Macro data (FX, commodities)
│   │   ├── entity_tagger.py       # LLM-based entity extraction
│   │   ├── sentiment_scorer.py    # LLM-based sentiment analysis
│   │   └── embedder.py            # Embedding generation
│   │
│   ├── llm/                       # LLM interface layer
│   │   ├── __init__.py
│   │   ├── router.py              # Model routing (tier → model)
│   │   ├── batcher.py             # Batched async LLM calls
│   │   ├── prompts/               # Prompt templates
│   │   │   ├── tier1_decision.py
│   │   │   ├── tier2_decision.py
│   │   │   ├── observer_analysis.py
│   │   │   └── data_processing.py
│   │   └── providers/             # Provider-specific clients
│   │       ├── deepseek.py
│   │       ├── glm.py
│   │       └── claude.py
│   │
│   ├── db/                        # Database layer
│   │   ├── __init__.py
│   │   ├── models.py              # SQLAlchemy models
│   │   ├── vector_store.py        # ChromaDB / Qdrant interface
│   │   └── migrations/            # Alembic migrations
│   │
│   └── api/                       # FastAPI application
│       ├── __init__.py
│       ├── main.py                # FastAPI app entry point
│       ├── routes/
│       │   ├── simulation.py      # Simulation CRUD & control
│       │   ├── results.py         # Results retrieval
│       │   ├── agents.py          # Agent interaction (chat)
│       │   └── data.py            # Data management endpoints
│       └── websocket.py           # Live simulation streaming
│
├── frontend/                      # Dashboard UI
│   ├── package.json
│   └── src/
│       ├── views/
│       │   ├── SimulationDashboard.vue
│       │   ├── ScenarioReport.vue
│       │   ├── AgentNetwork.vue    # Social graph visualization
│       │   └── AgentChat.vue       # Post-simulation agent interaction
│       └── components/
│
├── scripts/                       # Utility scripts
│   ├── seed_historical_data.py    # Initial data population
│   ├── run_backtest.py            # CLI backtest runner
│   └── calibrate_agents.py        # Agent parameter tuning
│
└── tests/
    ├── test_agents/
    ├── test_memory/
    ├── test_simulation/
    └── test_integration/
```

---

## 15. Development Phases

### Phase 1: Foundation (Weeks 1-2)

**Goal:** Single-stock, single-run simulation with basic agents, no memory.

- [ ] Set up project structure, Docker, database schema
- [ ] Implement data ingestion for BBRI (price + 50 manually curated historical events)
- [ ] Implement base agent class with working memory only
- [ ] Create 3 Tier 1 personas and 5 Tier 2 archetypes
- [ ] Implement Tier 3 statistical agents with IDX-calibrated heuristics
- [ ] Build turn-based simulation loop (daily granularity)
- [ ] CLI-only output: simulation step log with agent decisions and portfolio state
- [ ] Run first backtest on 3 months of BBRI data
- [ ] Validate: do agents produce differentiated behavior? Does the simulation complete without errors?

**Deliverable:** Working single-stock simulation with differentiated agent behavior.

### Phase 2: Memory & Intelligence (Weeks 3-4)

**Goal:** Add episodic memory, social memory, and causal knowledge graph. Multi-run support.

- [ ] Implement ChromaDB-based episodic memory for Tier 1 agents
- [ ] Implement sliding-window episodic memory for Tier 2 agents
- [ ] Implement social memory with trust score dynamics
- [ ] Build causal knowledge graph schema and population pipeline
- [ ] Implement historical parallel retrieval in agent decision flow
- [ ] Add information propagation delay model
- [ ] Implement multi-run parallel execution
- [ ] Build basic Observer agent with cross-run aggregation
- [ ] Run 6-month backtest with memory enabled, validate calibration improvement
- [ ] Cost tracking: measure actual LLM token usage per run

**Deliverable:** Memory-enhanced simulation with social dynamics and multi-run scenario analysis.

### Phase 3: Integration & Scale (Weeks 5-6)

**Goal:** Connect to trading system, add prediction mode, expand stock coverage.

- [ ] Build FastAPI backend with simulation CRUD endpoints
- [ ] Implement prediction mode with price impact model
- [ ] Connect screener output → simulation input pipeline
- [ ] Connect simulation output → virtual trading module
- [ ] Expand to 5 stocks (BBRI, BMRI, BBCA, TLKM, ASII)
- [ ] Build automated news scraper for at least 2 Indonesian financial news sources
- [ ] Implement LLM-based entity tagging and sentiment scoring pipeline
- [ ] Add event injection API for "what-if" scenarios
- [ ] Load test: validate system handles 100 parallel runs without timeout

**Deliverable:** Production-ready API integrated with trading system, multi-stock support.

### Phase 4: UI & Polish (Weeks 7-8)

**Goal:** Interactive dashboard, agent chat, portfolio-ready presentation.

- [ ] Build simulation dashboard (run status, progress, cost tracking)
- [ ] Build scenario report viewer (rendered from Observer agent output)
- [ ] Build agent network visualization (D3.js / vis.js graph of social connections and information flow)
- [ ] Build post-simulation agent chat interface
- [ ] Add WebSocket support for live simulation streaming
- [ ] Write comprehensive README with architecture diagrams
- [ ] Record demo video showing full workflow: seed → simulate → predict → trade
- [ ] Performance optimization pass (batching, caching, prompt compression)

**Deliverable:** Complete system with interactive UI, ready for portfolio and demos.

---

## 16. Risk & Mitigation

| Risk | Impact | Mitigation |
|---|---|---|
| LLM costs exceed budget | Cannot run enough simulations | Tier system limits expensive calls; Tier 3 is free; monitor costs per run |
| LLM latency makes simulation slow | Bad user experience, limits run count | Batch calls, use fast models for Tier 2, async execution |
| Agent behavior not realistic | Predictions not useful | Extensive backtesting with calibration feedback loop |
| IDX data hard to obtain | Cannot backtest properly | Start with Yahoo Finance (free), upgrade to paid sources later |
| Indonesian news scraping unreliable | Causal memory gaps | Start with manually curated events, automate incrementally |
| Emergent behavior too chaotic | No signal in predictions | Calibrate agent population ratios; increase Tier 3 stabilizing agents |
| Overfitting to historical patterns | Poor forward prediction | Use walk-forward validation (train on period A, predict period B) |

---

## 17. Success Criteria

### Technical Success

- Backtest direction accuracy > 55% (above random) for daily predictions on BBRI
- Simulation produces observable herding and cascade events that correlate with historical volatility spikes
- System completes 100-run prediction in under 30 minutes
- LLM cost per prediction run stays under $5

### Portfolio Success

- Clean, well-documented GitHub repository with architecture diagrams
- Working demo that can be shown in a 5-minute screen recording
- System demonstrates multi-agent orchestration, cost-optimized LLM routing, real data integration, and actionable output
- Narrative: "Built a multi-agent market simulation engine with 3-tier LLM-powered agents, causal memory, and social dynamics, producing scenario-based IDX stock forecasts integrated into an automated trading pipeline"

---

## 18. Future Extensions (Post-MVP)

- **Fine-tuned agent models:** Train small models on historical trader behavior data to replace generic LLM calls for Tier 2
- **Real-time mode:** Ingest live IDX data feed and run continuous simulation
- **Multi-market correlation:** Add cross-market signals (US markets, regional Asian markets)
- **Reinforcement learning calibration:** Use RL to optimize agent population parameters
- **Regulatory sandbox integration:** Partner with OJK innovation hub for validation
- **Explainability layer:** Trace any prediction back to specific agent decisions and causal links
- **Community agent marketplace:** Allow users to define and share custom agent personas
