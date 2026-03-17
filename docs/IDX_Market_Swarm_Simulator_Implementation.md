# IDX Market Swarm Simulator — Implementation Guide

## Companion to: IDX_Market_Swarm_Simulator_Spec.md

**Purpose:** This document provides the exact implementation details needed to build IMSS using Claude Code. It includes prompt templates, configuration values, dependency specifications, database setup, and step-by-step build instructions for each phase.

**Target runtime:** Python 3.11+, SQLite for Phase 1-2 → PostgreSQL for Phase 3+
**Primary LLM:** GLM-5 (Zhipu AI) for all agent tiers
**Secondary LLMs (optional):** Kimi (Moonshot AI), Qwen (Alibaba)
**Data sources:** Yahoo Finance (yfinance), manual event curation, web scraping

---

## Table of Contents

1. [Environment & Dependencies](#1-environment--dependencies)
2. [LLM Provider Configuration](#2-llm-provider-configuration)
3. [Database Setup](#3-database-setup)
4. [Data Ingestion Details](#4-data-ingestion-details)
5. [Agent Prompt Templates](#5-agent-prompt-templates)
6. [Agent Persona Definitions](#6-agent-persona-definitions)
7. [Memory Implementation Details](#7-memory-implementation-details)
8. [Simulation Engine Details](#8-simulation-engine-details)
9. [Observer Agent Details](#9-observer-agent-details)
10. [Configuration Defaults](#10-configuration-defaults)
11. [Testing Strategy](#11-testing-strategy)
12. [Phase-by-Phase Build Instructions](#12-phase-by-phase-build-instructions)
13. [Claude Code Usage Notes](#13-claude-code-usage-notes)

---

## 1. Environment & Dependencies

### 1.1 Python Dependencies

```toml
# pyproject.toml
[project]
name = "imss"
version = "0.1.0"
requires-python = ">=3.11"

dependencies = [
    # Core
    "pydantic>=2.5",
    "pydantic-settings>=2.1",
    "python-dotenv>=1.0",

    # LLM Clients
    "openai>=1.12",            # OpenAI-compatible client (works with GLM, Kimi, Qwen)
    "httpx>=0.27",             # Async HTTP for custom LLM calls
    "tiktoken>=0.6",           # Token counting

    # Data
    "yfinance>=0.2.36",        # IDX price data via Yahoo Finance
    "pandas>=2.2",
    "numpy>=1.26",

    # Database
    "sqlalchemy>=2.0",
    "aiosqlite>=0.20",         # Async SQLite for Phase 1-2
    "alembic>=1.13",           # Migrations
    # "asyncpg>=0.29",         # Uncomment for Phase 3 PostgreSQL
    # "psycopg2-binary>=2.9",  # Uncomment for Phase 3 PostgreSQL

    # Vector Store
    "chromadb>=0.4.22",        # Local vector store, no server needed
    "sentence-transformers>=2.5",  # Local embedding model

    # Async & Concurrency
    "asyncio>=3.4",
    "aiohttp>=3.9",

    # API (Phase 3+)
    "fastapi>=0.109",
    "uvicorn>=0.27",
    "celery[redis]>=5.3",      # Phase 3+ task queue

    # Utilities
    "rich>=13.7",              # CLI output formatting
    "structlog>=24.1",         # Structured logging
    "tqdm>=4.66",              # Progress bars

    # Scraping (Phase 3+)
    "beautifulsoup4>=4.12",
    "httpx>=0.27",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
    "pytest-cov>=4.1",
    "ruff>=0.2",
]
```

### 1.2 Environment Variables

```bash
# .env

# === LLM Configuration ===

# GLM-5 (Primary — used for all tiers)
GLM_API_KEY=your_zhipu_api_key
GLM_BASE_URL=https://open.bigmodel.cn/api/paas/v4
GLM_MODEL=glm-5

# Kimi / Moonshot (Optional — fallback or comparison)
KIMI_API_KEY=your_moonshot_api_key
KIMI_BASE_URL=https://api.moonshot.cn/v1
KIMI_MODEL=moonshot-v1-8k

# Qwen (Optional — fallback or comparison)
QWEN_API_KEY=your_dashscope_api_key
QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
QWEN_MODEL=qwen-plus

# === Database ===
DATABASE_URL=sqlite+aiosqlite:///./data/imss.db
# For Phase 3: DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/imss

# === Vector Store ===
CHROMA_PERSIST_DIR=./data/chroma

# === Embedding Model ===
# Local model — no API needed
EMBEDDING_MODEL=all-MiniLM-L6-v2

# === Simulation Defaults ===
DEFAULT_NUM_RUNS=20
DEFAULT_STEPS_PER_RUN=50
MAX_CONCURRENT_LLM_CALLS=5
LLM_REQUEST_TIMEOUT=30
LLM_TEMPERATURE_TIER1=0.7
LLM_TEMPERATURE_TIER2=0.5
LLM_TEMPERATURE_OBSERVER=0.3

# === Cost Tracking ===
COST_ALERT_THRESHOLD_USD=5.00
```

### 1.3 Project Initialization Commands

```bash
# Claude Code should run these in order:

# 1. Create project structure
mkdir -p idx-market-swarm/{imss/{agents/{tier1,tier2,tier3},memory,simulation,observer,data,llm/{prompts,providers},db,api/routes},frontend,scripts,tests/{test_agents,test_memory,test_simulation,test_integration},data}

# 2. Initialize Python project
cd idx-market-swarm
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# 3. Initialize ChromaDB directory
mkdir -p data/chroma

# 4. Initialize SQLite database (Phase 1)
# Alembic will handle this, but for quick start:
python -c "from imss.db.models import create_tables; create_tables()"
```

---

## 2. LLM Provider Configuration

### 2.1 Unified OpenAI-Compatible Client

All three providers (GLM, Kimi, Qwen) support the OpenAI SDK format. Use a single abstraction layer:

```
LLMRouter:
  Purpose: Route agent LLM calls to the appropriate provider and model
  
  Routing Strategy (with GLM-5 as primary):
    Tier 1 agents → GLM-5 (temperature=0.7, max_tokens=1024)
    Tier 2 agents → GLM-5 (temperature=0.5, max_tokens=512)
    Observer agent → GLM-5 (temperature=0.3, max_tokens=4096)
    Data processing → GLM-5 (temperature=0.1, max_tokens=512)
    
  Fallback Chain: GLM-5 → Kimi → Qwen (if primary fails)
  
  Rate Limiting:
    Max concurrent calls: 5 (configurable via env)
    Retry on 429: exponential backoff, max 3 retries
    Timeout per call: 30 seconds
    
  Cost Tracking:
    Log input_tokens and output_tokens per call
    Aggregate by tier, agent_id, simulation_run
    Alert if cumulative cost exceeds threshold
```

### 2.2 GLM-5 API Specifics

```
Endpoint: https://open.bigmodel.cn/api/paas/v4/chat/completions
Auth: Authorization: Bearer {GLM_API_KEY}
Format: OpenAI-compatible (use openai Python SDK with custom base_url)

Client initialization:
  openai.AsyncOpenAI(
      api_key=GLM_API_KEY,
      base_url="https://open.bigmodel.cn/api/paas/v4"
  )

Model string: "glm-5"

Supported parameters:
  - temperature: 0.0-1.0
  - max_tokens: up to 4096
  - top_p: supported
  - stream: supported
  - tools/function_calling: supported (useful for structured agent output)

GLM-5 Strengths:
  - Strong Chinese + English bilingual (good for Indonesian financial news that may reference Chinese sources)
  - Good structured output / JSON mode
  - Competitive reasoning capability
  
GLM-5 Considerations:
  - Check current rate limits on your plan
  - Token counting: use tiktoken with cl100k_base as approximation
  - Response format: request JSON output via system prompt instruction, not via response_format parameter (verify if GLM-5 supports response_format natively)
```

### 2.3 Structured Output Strategy

All agent decision calls should request JSON output for reliable parsing:

```
System prompt suffix for all agent calls:

"You MUST respond with valid JSON only. No markdown, no explanation outside the JSON.
Response format:
{
  "action": "BUY" | "SELL" | "HOLD",
  "stock": "<symbol>",
  "quantity": <integer>,
  "confidence": <float 0-1>,
  "reasoning": "<one paragraph explaining decision>"
}"
```

If GLM-5 supports function calling / tool_use, prefer that over JSON-in-prompt for more reliable structured output. Implementation should try function calling first, fall back to JSON prompt if unsupported.

---

## 3. Database Setup

### 3.1 Phase 1-2: SQLite

```
File: data/imss.db

SQLAlchemy engine:
  create_async_engine("sqlite+aiosqlite:///./data/imss.db")

Limitations to be aware of:
  - No concurrent writes (use write queue or serialize writes)
  - No TimescaleDB time-series optimizations
  - Sufficient for single-stock, <100 runs

Migration path:
  - Use Alembic from day 1 so schema migrations carry over to PostgreSQL
  - Keep all SQL standard-compliant (no SQLite-specific syntax)
```

### 3.2 SQLAlchemy Model Definitions

```
Models to implement (in imss/db/models.py):

class StockOHLCV:
    """Daily price data"""
    id: int (primary key, auto)
    symbol: str (indexed)
    timestamp: datetime (indexed)
    open: float
    high: float
    low: float
    close: float
    volume: int
    adjusted_close: float
    
    Composite index: (symbol, timestamp) unique

class Event:
    """News, regulatory, earnings, macro events"""
    id: uuid (primary key)
    timestamp: datetime (indexed)
    category: str  # REGULATORY, EARNINGS, MACRO, NEWS, RUMOR, POLITICAL
    source: str
    title: str
    summary: str  # LLM-generated summary
    raw_content: str  # nullable, original text
    sentiment_score: float  # -1.0 to 1.0
    magnitude_score: float  # 0.0 to 1.0
    embedding_id: str  # reference to ChromaDB document ID
    created_at: datetime

class EventEntity:
    """Many-to-many: which stocks/sectors an event affects"""
    id: int (primary key, auto)
    event_id: uuid (foreign key → Event)
    entity_type: str  # STOCK, SECTOR
    entity_symbol: str  # e.g., BBRI, BANKING, MINING

class CausalLink:
    """Learned event→price correlations"""
    id: uuid (primary key)
    event_id: uuid (foreign key → Event)  # nullable for pattern-based links
    event_category: str  # for pattern matching without specific event
    stock_symbol: str
    lag_days: int
    direction: str  # POSITIVE, NEGATIVE, NEUTRAL
    correlation_strength: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    sample_count: int  # how many backtest confirmations
    last_updated: datetime

class SimulationRun:
    """Metadata for each simulation execution"""
    id: uuid (primary key)
    config_json: str  # JSON-serialized SimulationConfig
    mode: str  # BACKTEST, PREDICT
    status: str  # PENDING, RUNNING, COMPLETED, FAILED
    started_at: datetime
    completed_at: datetime (nullable)
    results_summary_json: str (nullable)  # JSON-serialized results
    total_llm_calls: int (default 0)
    total_tokens_used: int (default 0)
    estimated_cost_usd: float (default 0.0)

class SimulationStepLog:
    """Per-step log for analysis and replay"""
    id: int (primary key, auto)
    simulation_id: uuid (foreign key → SimulationRun)
    run_number: int
    step_number: int
    simulated_date: date
    market_state_json: str  # prices, volumes at this step
    agent_actions_json: str  # all agent decisions this step
    events_active_json: str  # events injected/propagated this step
    aggregate_sentiment: float
    aggregate_order_imbalance: float
```

### 3.3 ChromaDB Setup

```
Collection: "event_embeddings"
  Purpose: Semantic search over historical events for causal retrieval
  Embedding model: all-MiniLM-L6-v2 (local, via sentence-transformers)
  Document: event summary text
  Metadata: {
    event_id: str,
    category: str,
    timestamp: str (ISO format),
    sentiment: float,
    magnitude: float,
    entities: str (comma-separated symbols)
  }
  
Collection: "agent_episodic_{agent_id}" (created per Tier 1 agent)
  Purpose: Per-agent episodic memory retrieval
  Embedding model: same as above
  Document: experience summary text
  Metadata: {
    step: int,
    simulated_date: str,
    action: str,
    stock: str,
    outcome_pnl: float,
    was_profitable: bool
  }

Initialization:
  import chromadb
  client = chromadb.PersistentClient(path="./data/chroma")
  event_collection = client.get_or_create_collection(
      name="event_embeddings",
      metadata={"hnsw:space": "cosine"}
  )
```

---

## 4. Data Ingestion Details

### 4.1 IDX Price Data via Yahoo Finance

```
Yahoo Finance ticker mapping for IDX:
  BBRI → "BBRI.JK"
  BMRI → "BMRI.JK"
  BBCA → "BBCA.JK"
  TLKM → "TLKM.JK"
  ASII → "ASII.JK"
  IHSG (composite index) → "^JKSE"

Implementation notes:
  - Use yfinance.download(tickers, start, end, interval="1d")
  - Yahoo Finance IDX data may have gaps on Indonesian holidays
  - Store adjusted close for dividend-adjusted analysis
  - Fetch at least 2 years: start="2024-01-01" to present
  - Run data validation: check for NaN, zero volume days, outlier prices
  - Schedule daily refresh via cron or manual trigger

Data quality checks:
  - Remove rows where volume == 0 (non-trading days that weren't filtered)
  - Forward-fill missing adjusted_close if gaps ≤ 2 days
  - Flag and log any price changes > 25% in a single day (likely corporate action)
```

### 4.2 Historical Event Curation (Phase 1)

For Phase 1, manually curate ~50 significant events for BBRI spanning 2024-2025. This is faster than building a scraper and gives you control over data quality.

```
Event curation format (CSV or JSON):

{
  "timestamp": "2024-03-15T09:00:00+07:00",
  "category": "REGULATORY",
  "source": "OJK",
  "title": "OJK tightens NPL provisioning for digital lending",
  "summary": "OJK issued new regulation requiring banks to increase provisioning reserves for digital lending products, effective Q3 2024. Expected to impact net interest margins for banks with significant digital lending exposure.",
  "affected_entities": ["BBRI", "BMRI", "BBCA", "BANKING"],
  "sentiment_score": -0.3,
  "magnitude_score": 0.6
}

Categories to cover (aim for ~10 events per category):
  REGULATORY: OJK regulations, BI policy changes, government financial policy
  EARNINGS: BBRI quarterly earnings releases and guidance
  MACRO: BI rate decisions, USD/IDR major movements, inflation data
  NEWS: Major banking sector news, BRI-specific news
  POLITICAL: Election impacts, cabinet changes affecting finance ministry

Where to find historical events:
  - ojk.go.id press releases
  - bi.go.id monetary policy announcements
  - kontan.co.id financial news archive
  - Google search: "BBRI news 2024" filtered by date
  - BRI investor relations page for earnings dates

Store as: data/seed_events/bbri_events_2024.json
```

### 4.3 Event Processing Pipeline

```
For each raw event (whether manually curated or scraped):

1. ENTITY TAGGING (if not manually provided)
   Prompt to GLM-5:
   """
   Given this financial news headline and summary, identify all affected Indonesian stock tickers and sectors.
   
   Headline: {title}
   Summary: {summary}
   
   Respond with JSON only:
   {
     "stocks": ["BBRI", "BMRI"],
     "sectors": ["BANKING", "FINANCE"]
   }
   
   Valid sectors: BANKING, FINANCE, MINING, ENERGY, CONSUMER, TELECOM, INFRASTRUCTURE, PROPERTY, MANUFACTURING, TECHNOLOGY
   Valid stocks: Any IDX ticker (4 letters + optional .JK suffix)
   """

2. SENTIMENT SCORING (if not manually provided)
   Prompt to GLM-5:
   """
   Rate the sentiment of this financial event for the Indonesian stock market.
   
   Event: {title} — {summary}
   
   Respond with JSON only:
   {
     "sentiment": <float from -1.0 (very bearish) to 1.0 (very bullish)>,
     "magnitude": <float from 0.0 (insignificant) to 1.0 (market-moving)>,
     "reasoning": "<one sentence>"
   }
   """

3. EMBEDDING
   Use sentence-transformers locally:
   model = SentenceTransformer('all-MiniLM-L6-v2')
   embedding = model.encode(f"{title}. {summary}")
   
   Store in ChromaDB event_embeddings collection

4. DATABASE INSERT
   Insert Event record + EventEntity associations + ChromaDB document
```

### 4.4 Automated News Scraping (Phase 3)

```
Target sources (when ready to automate):

Kontan.co.id:
  URL pattern: https://www.kontan.co.id/search/?search={query}
  Scraping: BeautifulSoup on article list, follow links for full text
  Rate limit: 1 request per 3 seconds
  
Bisnis.com:
  URL pattern: https://www.bisnis.com/topic/{topic}
  Similar scraping approach
  
CNBC Indonesia:
  URL: https://www.cnbcindonesia.com/market
  RSS feed may be available

For each scraped article:
  1. Extract title, date, full text
  2. Run through entity tagging + sentiment scoring pipeline
  3. Deduplicate against existing events (cosine similarity > 0.9 = duplicate)
  4. Store with source attribution
```

---

## 5. Agent Prompt Templates

### 5.1 Tier 1 — Full Decision Prompt

```
SYSTEM PROMPT:
"""
You are {agent_name}, a {persona_description}.

PERSONALITY & BEHAVIOR:
{persona_behavioral_rules}

RISK PROFILE:
- Risk tolerance: {risk_tolerance}/1.0
- Maximum single-stock allocation: {max_allocation}%
- Stop-loss threshold: {stop_loss_pct}%
- Preferred holding period: {holding_period}

You make investment decisions for Indonesian Stock Exchange (IDX) stocks.
You MUST respond with valid JSON only. No other text.

Response format:
{
  "action": "BUY" | "SELL" | "HOLD",
  "stock": "<IDX ticker>",
  "quantity": <integer shares, 0 if HOLD>,
  "confidence": <float 0.0-1.0>,
  "reasoning": "<2-3 sentences explaining your decision>",
  "sentiment_update": <float -1.0 to 1.0, your current market outlook>
}
"""

USER PROMPT (constructed each simulation step):
"""
=== SIMULATION STEP {step} — Date: {simulated_date} ===

YOUR CURRENT PORTFOLIO:
- Cash: IDR {cash:,.0f}
- Holdings: {holdings_formatted}
- Portfolio value: IDR {portfolio_value:,.0f}
- Unrealized P&L: {unrealized_pnl:+.2f}%

MARKET DATA TODAY:
{stock_symbol}: Open {open} | High {high} | Low {low} | Close {close} | Volume {volume:,}
5-day change: {pct_change_5d:+.2f}% | 20-day change: {pct_change_20d:+.2f}%
Sector ({sector}): {sector_change:+.2f}% today

NEW EVENTS TODAY:
{events_formatted}
(If no events: "No significant events today.")

HISTORICAL PARALLELS (from knowledge base):
{causal_parallels_formatted}
(Format each as: "- [{date}] {event_title} → {stock} moved {direction} {magnitude}% over {lag} days (confidence: {conf})")
(If none found: "No strong historical parallels found.")

YOUR PAST EXPERIENCE IN SIMILAR SITUATIONS:
{episodic_memories_formatted}
(Format each as: "- Step {step}: You {action} {stock}. Outcome: {outcome}. P&L: {pnl}")
(If no episodic memory: "No relevant past experience.")

SIGNALS FROM AGENTS YOU FOLLOW:
{social_signals_formatted}
(Format each as: "- {agent_name} (trust: {trust_score:.2f}): {action} {stock} at step {step}")
(If no social signals: "No observable agent activity.")

Make your investment decision.
"""
```

### 5.2 Tier 2 — Simplified Decision Prompt

```
SYSTEM PROMPT:
"""
You are a {archetype_name} trader on the Indonesian Stock Exchange.
Behavioral tendency: {archetype_one_liner}

Respond with JSON only:
{
  "action": "BUY" | "SELL" | "HOLD",
  "stock": "<ticker>",
  "quantity": <integer>,
  "confidence": <float 0-1>,
  "reasoning": "<one sentence>"
}
"""

USER PROMPT:
"""
Portfolio: Cash IDR {cash:,.0f} | {holdings_summary}
{stock}: {close} ({pct_change_1d:+.2f}% today, {pct_change_5d:+.2f}% 5d)

Events: {events_brief}
(Max 2 most relevant events, one line each)

Institutional signals: {tier1_signals_brief}
(Format: "{agent_name} is {action}ing {stock}")

Recent memory: {recent_decisions_brief}
(Last 3 own decisions, one line each)

Decision:
"""
```

### 5.3 Observer / Analyst Prompt

```
SYSTEM PROMPT:
"""
You are a senior quantitative analyst synthesizing results from {num_runs} parallel market simulation runs for the Indonesian Stock Exchange.

Your task is to identify scenario clusters, assess probability distributions, and produce actionable investment analysis.

Be specific with numbers. Cite which simulation runs support each scenario.
Acknowledge uncertainty. Distinguish high-confidence findings from speculative ones.

Respond in structured markdown format.
"""

USER PROMPT:
"""
=== SIMULATION ANALYSIS REQUEST ===

Target: {stock_symbol}
Prediction horizon: {horizon_days} days
Number of simulation runs: {num_runs}
Simulation period: {start_date} to {end_date}

AGGREGATE RESULTS:
- Mean final price: {mean_price} ({mean_return:+.2f}%)
- Median final price: {median_price} ({median_return:+.2f}%)
- Std deviation of returns: {std_return:.2f}%
- Min final price: {min_price} (Run #{min_run})
- Max final price: {max_price} (Run #{max_run})

PRICE DISTRIBUTION:
- P10: {p10_price} ({p10_return:+.2f}%)
- P25: {p25_price}
- P50: {p50_price}
- P75: {p75_price}
- P90: {p90_price} ({p90_return:+.2f}%)

DIRECTIONAL BIAS:
- Runs ending higher: {pct_up:.0f}%
- Runs ending lower: {pct_down:.0f}%
- Runs ending flat (±1%): {pct_flat:.0f}%

KEY EVENTS INJECTED:
{seed_events_formatted}

SCENARIO CLUSTERS (pre-computed via k-means on price trajectories):
{clusters_formatted}
(For each cluster: run IDs, mean trajectory, common agent behavior pattern)

NOTABLE AGENT BEHAVIORS:
- Herding events detected in runs: {herding_runs}
- Panic cascade events: {cascade_runs}
- Institutional consensus moments: {consensus_description}

Produce a scenario analysis report with:
1. Executive summary (3-4 sentences)
2. Top 2-3 scenario clusters with probability, price target range, and narrative
3. Key risk factors and sensitivity analysis
4. Actionable trading recommendations with position sizing guidance
5. Signals to watch that would confirm or invalidate each scenario
"""
```

### 5.4 Data Processing Prompts

```
ENTITY EXTRACTION PROMPT:
"""
Extract IDX stock tickers and market sectors affected by this financial event.

Event: {title}
Details: {summary}

Respond JSON only:
{"stocks": ["BBRI"], "sectors": ["BANKING"]}

Valid sectors: BANKING, FINANCE, MINING, ENERGY, CONSUMER, TELECOM, INFRASTRUCTURE, PROPERTY, MANUFACTURING, TECHNOLOGY, AUTOMOTIVE, PLANTATION
"""

SENTIMENT SCORING PROMPT:
"""
Score this event's impact on Indonesian stock market sentiment.

Event: {title} — {summary}

JSON only:
{"sentiment": <-1.0 to 1.0>, "magnitude": <0.0 to 1.0>}

Guidance:
  sentiment: -1.0 = very bearish, 0.0 = neutral, 1.0 = very bullish
  magnitude: 0.0 = no market impact, 0.5 = moderate, 1.0 = major market mover
"""

EVENT SIMILARITY SUMMARY PROMPT (for causal link narrative):
"""
Briefly describe how this historical event relates to the current situation.

Historical event: {historical_event_summary} (Date: {date}, Impact: {stock} {direction} {pct}%)
Current situation: {current_event_summary}

One sentence only:
"""
```

---

## 6. Agent Persona Definitions

### 6.1 Tier 1 Named Personas

```
PERSONA 1: "Pak Budi" — State Bank Fund Manager
  agent_name: "Pak Budi"
  persona_description: "Senior fund manager at a major Indonesian state-owned pension fund. 25 years of experience in IDX. Conservative, regulation-aware, long-term oriented."
  persona_behavioral_rules: |
    - You prioritize capital preservation over aggressive returns
    - You pay close attention to OJK and BI regulatory signals — these often affect your portfolio directly
    - You rarely panic sell; you believe in fundamental value and Indonesian economic growth
    - You are skeptical of momentum-driven moves and prefer to buy on dips
    - You hold positions for weeks to months, not days
    - You are uncomfortable with more than 30% of portfolio in a single stock
    - When uncertain, you HOLD rather than act
  risk_tolerance: 0.3
  max_allocation: 30
  stop_loss_pct: 15
  holding_period: "weeks to months"
  initial_cash: 10_000_000_000  # IDR 10 billion
  initial_holdings: {"BBRI": 500000, "BMRI": 300000}

PERSONA 2: "Sarah" — Foreign Institutional Investor
  agent_name: "Sarah"
  persona_description: "Portfolio manager at a Singapore-based emerging markets fund. Analytical, USD-return focused, sensitive to currency risk and foreign flow data."
  persona_behavioral_rules: |
    - You evaluate IDX stocks in USD terms — a stock rising 5% in IDR but with 3% IDR depreciation is only +2% for you
    - You track foreign fund flow data closely; net foreign outflows make you nervous
    - You are more aggressive than local institutions — willing to take concentrated positions
    - You follow global macro trends and correlate IDX with regional markets (SET, KLCI, STI)
    - You are quick to cut losses if your thesis is invalidated
    - Political instability or regulatory uncertainty triggers position reduction
    - You like liquid large-caps; you won't touch stocks with low daily volume
  risk_tolerance: 0.6
  max_allocation: 40
  stop_loss_pct: 8
  holding_period: "days to weeks"
  initial_cash: 15_000_000_000  # IDR 15 billion
  initial_holdings: {"BBRI": 200000, "BBCA": 300000}

PERSONA 3: "Andi" — Aggressive Retail Trader
  agent_name: "Andi"
  persona_description: "Full-time retail day trader. Active on Stockbit community. Momentum-driven, influenced by social media sentiment and influencer calls."
  persona_behavioral_rules: |
    - You chase momentum — if a stock is running, you want in
    - You follow what popular Stockbit influencers recommend
    - You panic sell quickly when things go against you
    - You overtrade — you prefer action to sitting still
    - You have strong recency bias — recent events weigh heavily in your decisions
    - You use simple technical signals: "breaking resistance" or "support broken"
    - You tend to buy at highs and sell at lows due to emotional decision-making
    - Small position sizes relative to institutional agents
  risk_tolerance: 0.8
  max_allocation: 50
  stop_loss_pct: 5
  holding_period: "hours to days"
  initial_cash: 100_000_000  # IDR 100 million
  initial_holdings: {}

PERSONA 4: "Dr. Lim" — Contrarian Value Analyst
  agent_name: "Dr. Lim"
  persona_description: "Independent equity analyst with a PhD in finance. Deep fundamental analysis. Goes against the crowd when valuations are extreme."
  persona_behavioral_rules: |
    - You are a contrarian — when everyone is selling, you look for buying opportunities
    - You evaluate stocks on P/E, P/B, dividend yield, and ROE fundamentals
    - You are patient; you can hold a losing position for months if your thesis is intact
    - You ignore short-term noise and social media hype
    - You increase position sizes when fear is high and valuations are low
    - You take profits when euphoria pushes valuations above historical ranges
    - You document your investment thesis and only change it based on fundamental changes
  risk_tolerance: 0.5
  max_allocation: 35
  stop_loss_pct: 20
  holding_period: "months"
  initial_cash: 5_000_000_000  # IDR 5 billion
  initial_holdings: {"BBRI": 300000}

PERSONA 5: "MarketBot" — Algorithmic Market Maker
  agent_name: "MarketBot"
  persona_description: "Automated market making algorithm. Provides liquidity, profits from bid-ask spread. No directional bias."
  persona_behavioral_rules: |
    - You provide liquidity — you buy when others are selling and sell when others are buying
    - You have no directional opinion; your goal is to capture spread, not predict direction
    - You reduce activity (widen spread) during high volatility
    - You increase activity during calm, range-bound markets
    - You never hold large net positions — you rebalance toward neutral
    - News events cause you to temporarily widen spread (reduce activity) until volatility subsides
    - You are purely mechanical; sentiment does not affect you
  risk_tolerance: 0.2
  max_allocation: 20
  stop_loss_pct: 3
  holding_period: "intraday to 1 day"
  initial_cash: 20_000_000_000  # IDR 20 billion
  initial_holdings: {"BBRI": 100000}
```

### 6.2 Tier 2 Archetype Templates

```
Archetypes are parameterized templates — multiple agents are generated from each.

ARCHETYPE: "momentum_chaser"
  description: "Buys stocks that are going up, sells stocks that are going down"
  one_liner: "You follow price momentum. Rising stocks attract you, falling stocks repel you."
  risk_tolerance_range: [0.6, 0.9]
  sentiment_bias_range: [0.0, 0.3]  # slightly bullish
  decision_latency: 0  # reacts immediately
  initial_cash_range: [50_000_000, 200_000_000]

ARCHETYPE: "panic_seller"
  description: "Overreacts to negative news, sells first asks questions later"
  one_liner: "You are risk-averse and react strongly to bad news. Preservation of capital is everything."
  risk_tolerance_range: [0.1, 0.3]
  sentiment_bias_range: [-0.3, 0.0]  # slightly bearish
  decision_latency: 0
  initial_cash_range: [50_000_000, 150_000_000]

ARCHETYPE: "dividend_holder"
  description: "Buy-and-hold investor focused on dividend income"
  one_liner: "You buy high-dividend stocks and hold long term. You rarely sell unless dividends are cut."
  risk_tolerance_range: [0.3, 0.5]
  sentiment_bias_range: [0.1, 0.3]  # mildly bullish
  decision_latency: 2  # slow to react
  initial_cash_range: [100_000_000, 500_000_000]

ARCHETYPE: "sector_rotator"
  description: "Moves money between sectors based on macro signals"
  one_liner: "You shift allocation between sectors based on economic cycle and macro data."
  risk_tolerance_range: [0.4, 0.7]
  sentiment_bias_range: [-0.1, 0.1]  # neutral
  decision_latency: 1
  initial_cash_range: [200_000_000, 1_000_000_000]

ARCHETYPE: "news_reactive"
  description: "Trades primarily based on breaking news and headlines"
  one_liner: "You trade the news. Headlines drive your decisions. First to react, sometimes wrong."
  risk_tolerance_range: [0.5, 0.8]
  sentiment_bias_range: [-0.1, 0.1]
  decision_latency: 0
  initial_cash_range: [30_000_000, 100_000_000]

Agent Generation:
  For each archetype, generate N agents (default: 8 per archetype = 40 total)
  For each generated agent:
    - Randomly sample parameters from defined ranges
    - Assign unique agent_id: "{archetype}_{index:03d}"
    - Randomly assign initial stock holdings (0-2 stocks, small positions)
```

### 6.3 Tier 3 Behavioral Heuristics

```
Tier 3 agents use NO LLM. They follow rule-based strategies:

HEURISTIC: "momentum_follower"
  Rule: If stock price change over last N days > threshold, BUY. If < -threshold, SELL.
  Parameters:
    lookback_days: 5
    buy_threshold: 0.03  # +3%
    sell_threshold: -0.03  # -3%
    position_size: fixed percentage of cash (5%)

HEURISTIC: "mean_reversion"
  Rule: If stock deviates more than Z standard deviations from N-day moving average, trade toward mean.
  Parameters:
    lookback_days: 20
    z_threshold: 1.5
    position_size: proportional to deviation magnitude

HEURISTIC: "random_walk"
  Rule: Randomly buy or sell with low probability each step. Adds noise/liquidity.
  Parameters:
    action_probability: 0.1  # 10% chance of action each step
    buy_bias: 0.5  # equal buy/sell probability
    position_size: small random amount

HEURISTIC: "volume_follower"
  Rule: Buy when volume spikes above average, sell when volume drops.
  Parameters:
    volume_lookback: 10
    spike_threshold: 2.0  # 2x average volume triggers buy signal
    position_size: proportional to volume spike magnitude

Distribution per simulation:
  momentum_follower: 30% of Tier 3
  mean_reversion: 25%
  random_walk: 30%
  volume_follower: 15%
  
  Default total Tier 3: 200 agents
```

---

## 7. Memory Implementation Details

### 7.1 Working Memory (In-Memory Python Object)

```
Implement as a Pydantic model:

class WorkingMemory(BaseModel):
    current_step: int = 0
    cash: float
    holdings: dict[str, int] = {}  # symbol → quantity
    portfolio_value_history: list[tuple[int, float]] = []  # (step, value)
    recent_actions: deque[AgentAction] = deque(maxlen=10)
    recent_observations: deque[Observation] = deque(maxlen=20)
    current_sentiment: float = 0.0
    
class AgentAction(BaseModel):
    step: int
    action: str  # BUY, SELL, HOLD
    stock: str
    quantity: int
    price: float
    reasoning: str
    
class Observation(BaseModel):
    step: int
    type: str  # PRICE, EVENT, SOCIAL_SIGNAL
    content: str
    source: str  # agent_id or "market" or event_id

Reset between simulation runs.
Serializable to JSON for step logging.
```

### 7.2 Episodic Memory (ChromaDB-backed)

```
Tier 1 Implementation:

class EpisodicMemory:
    collection_name: "agent_episodic_{agent_id}"
    max_records: 500
    
    store_experience(step, simulated_date, action, stock, outcome_pnl, context_summary):
        # Generate text summary
        summary = f"On {simulated_date}, I decided to {action} {stock}. " \
                  f"Context: {context_summary}. " \
                  f"Outcome: {'profit' if outcome_pnl > 0 else 'loss'} of {outcome_pnl:.2f}%."
        
        # Embed and store
        collection.add(
            documents=[summary],
            metadatas=[{
                "step": step,
                "simulated_date": str(simulated_date),
                "action": action,
                "stock": stock,
                "outcome_pnl": outcome_pnl,
                "was_profitable": outcome_pnl > 0
            }],
            ids=[f"{agent_id}_exp_{step}"]
        )
        
        # Prune if over max
        if collection.count() > max_records:
            # Delete oldest entries
            ...
    
    retrieve_relevant(current_context, top_k=5):
        results = collection.query(
            query_texts=[current_context],
            n_results=top_k
        )
        return results["documents"], results["metadatas"]

Tier 2 Implementation (simpler, no ChromaDB):

class SimplifiedEpisodicMemory:
    window_size: 20
    experiences: deque[dict] = deque(maxlen=20)
    
    store_experience(step, action, stock, outcome_pnl):
        experiences.append({
            "step": step,
            "action": action,
            "stock": stock,
            "outcome_pnl": outcome_pnl
        })
    
    retrieve_relevant(stock=None, action=None, limit=5):
        # Simple filtering — no semantic search
        filtered = [e for e in experiences 
                    if (stock is None or e["stock"] == stock)
                    and (action is None or e["action"] == action)]
        return filtered[-limit:]
```

### 7.3 Social Memory

```
class SocialMemory:
    relationships: dict[str, SocialRecord] = {}
    
class SocialRecord(BaseModel):
    agent_id: str
    agent_name: str
    observed_actions: deque[ObservedAction] = deque(maxlen=20)
    trust_score: float = 0.5  # start neutral
    prediction_accuracy: float = 0.5
    total_observations: int = 0
    correct_predictions: int = 0
    
class ObservedAction(BaseModel):
    step: int
    action: str
    stock: str
    price_at_action: float
    price_after_5_steps: float | None = None  # filled in later for accuracy calc

Trust Score Update Algorithm:
    
    def update_trust(record, was_correct: bool):
        """Bayesian-like trust update"""
        record.total_observations += 1
        if was_correct:
            record.correct_predictions += 1
        
        # Running accuracy
        record.prediction_accuracy = record.correct_predictions / record.total_observations
        
        # Trust = weighted blend of accuracy and prior
        # Prior pulls toward 0.5 (uncertainty), weakens as observations grow
        prior_weight = 5.0  # equivalent to 5 "virtual" observations at 0.5
        record.trust_score = (
            (record.correct_predictions + prior_weight * 0.5) /
            (record.total_observations + prior_weight)
        )
    
    "was_correct" definition:
        If observed agent BOUGHT and price went up within 5 steps → correct
        If observed agent SOLD and price went down within 5 steps → correct
        If observed agent HELD and price stayed within ±2% → correct
        Otherwise → incorrect

Observation Network Initialization:
    
    # Tier 1 agents observe all other Tier 1 agents
    for agent in tier1_agents:
        agent.social_memory.relationships = {
            other.id: SocialRecord(agent_id=other.id, agent_name=other.name)
            for other in tier1_agents if other.id != agent.id
        }
    
    # Tier 2 agents observe all Tier 1 agents (only)
    for agent in tier2_agents:
        agent.social_memory.relationships = {
            t1.id: SocialRecord(agent_id=t1.id, agent_name=t1.name)
            for t1 in tier1_agents
        }

Information Propagation Delay:
    When Tier 1 agent acts at step T:
        Other Tier 1 agents see it at step T+1
        Tier 2 agents see it at step T + random(1, 3)
        Tier 3 agents never see it directly (only react to price)
```

### 7.4 Causal Knowledge Graph Retrieval

```
class CausalMemory:
    """Interface to the causal knowledge graph for historical parallel retrieval"""
    
    find_parallels(current_event_summary, stock_symbol, top_k=3):
        """Find historically similar events and their price outcomes"""
        
        # Step 1: Semantic search for similar events
        similar_events = event_collection.query(
            query_texts=[current_event_summary],
            n_results=top_k * 2,  # fetch extra, filter by stock
            where={"entities": {"$contains": stock_symbol}}  # or filter post-query
        )
        
        # Step 2: For each similar event, look up causal links
        parallels = []
        for event_id in similar_events["ids"]:
            links = db.query(CausalLink).filter(
                CausalLink.event_id == event_id,
                CausalLink.stock_symbol == stock_symbol,
                CausalLink.confidence > 0.3  # minimum confidence threshold
            ).all()
            
            for link in links:
                parallels.append({
                    "event_date": event.timestamp,
                    "event_title": event.title,
                    "stock": stock_symbol,
                    "direction": link.direction,
                    "magnitude_pct": link.correlation_strength * 100,
                    "lag_days": link.lag_days,
                    "confidence": link.confidence,
                    "sample_count": link.sample_count
                })
        
        # Step 3: Sort by confidence, return top_k
        parallels.sort(key=lambda x: x["confidence"], reverse=True)
        return parallels[:top_k]
    
    update_link(event_id, stock_symbol, actual_direction, actual_magnitude, actual_lag):
        """Called during backtesting to strengthen/weaken causal links"""
        
        existing = db.query(CausalLink).filter(
            CausalLink.event_id == event_id,
            CausalLink.stock_symbol == stock_symbol
        ).first()
        
        if existing:
            # Update running average
            n = existing.sample_count
            existing.correlation_strength = (
                (existing.correlation_strength * n + actual_magnitude) / (n + 1)
            )
            existing.sample_count += 1
            existing.confidence = min(1.0, existing.sample_count / 10)  # confidence grows with samples
            existing.last_updated = now()
        else:
            # Create new link
            new_link = CausalLink(
                event_id=event_id,
                stock_symbol=stock_symbol,
                lag_days=actual_lag,
                direction=actual_direction,
                correlation_strength=actual_magnitude,
                confidence=0.1,  # low initial confidence
                sample_count=1
            )
            db.add(new_link)
```

---

## 8. Simulation Engine Details

### 8.1 Simulation Orchestrator

```
class SimulationEngine:
    """Main orchestrator for running simulations"""
    
    Responsibilities:
    1. Initialize agent population from config
    2. Load historical data for the simulation period
    3. Run the turn-based simulation loop
    4. Manage LLM call batching and concurrency
    5. Log all steps for observer analysis
    6. Handle multi-run parallel execution
    
    Key methods:
    
    async run_single(config: SimulationConfig, run_number: int) -> SimulationResult:
        """Execute one complete simulation run"""
        agents = initialize_agents(config)
        market_data = load_market_data(config.target_stocks, config.simulation_period)
        event_timeline = load_event_timeline(config.simulation_period)
        
        for step in range(config.total_steps):
            simulated_date = get_date_for_step(step, config)
            
            # 1. Environment update
            prices = market_data.get_prices(simulated_date)
            events = event_timeline.get_events(simulated_date)
            
            # 2. Information propagation
            distribute_information(agents, events, step)
            propagate_social_signals(agents, step)
            
            # 3. Agent execution (batched by tier)
            tier3_actions = execute_tier3(agents.tier3, prices)
            tier2_actions = await execute_tier2_batched(agents.tier2, prices, step)
            tier1_actions = await execute_tier1(agents.tier1, prices, step)
            
            all_actions = tier3_actions + tier2_actions + tier1_actions
            
            # 4. Order resolution
            if config.mode == "BACKTEST":
                resolve_orders_backtest(all_actions, prices)  # use real prices
            else:
                new_prices = resolve_orders_predict(all_actions, prices)  # simulate price impact
            
            # 5. Post-step
            update_agent_memories(agents, all_actions, prices, step)
            log_step(simulation_id, run_number, step, market_state, all_actions)
        
        return compile_results(agents, market_data)
    
    async run_multi(config: SimulationConfig) -> MultiRunResult:
        """Execute N parallel simulation runs"""
        results = []
        
        # Run in batches to control concurrency
        batch_size = 5  # 5 runs at a time
        for batch_start in range(0, config.num_parallel_runs, batch_size):
            batch = range(batch_start, min(batch_start + batch_size, config.num_parallel_runs))
            batch_results = await asyncio.gather(*[
                run_single(config, run_number=i) for i in batch
            ])
            results.extend(batch_results)
        
        return aggregate_results(results)
```

### 8.2 LLM Call Batching

```
class LLMBatcher:
    """Batch multiple agent LLM calls for efficient execution"""
    
    max_concurrent: int = 5  # from env MAX_CONCURRENT_LLM_CALLS
    
    async execute_batch(requests: list[LLMRequest]) -> list[LLMResponse]:
        """Execute a batch of LLM requests with concurrency control"""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def limited_call(request):
            async with semaphore:
                return await llm_router.call(request)
        
        results = await asyncio.gather(*[
            limited_call(req) for req in requests
        ], return_exceptions=True)
        
        # Handle failures
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # Log error, return HOLD action as fallback
                results[i] = default_hold_response(requests[i].agent_id)
        
        return results

    Tier 2 execution per step:
        All 40 Tier 2 agents → 40 LLM calls
        With max_concurrent=5 → 8 sequential batches
        At ~2s per call → ~16 seconds per step for Tier 2
        
    Tier 1 execution per step:
        5-10 individual calls (sequential, each needs full context)
        At ~3s per call → ~15-30 seconds per step for Tier 1
        
    Total per step: ~30-45 seconds
    Total per run (50 steps): ~25-38 minutes
    
    Optimization: Cache Tier 2 prompts that haven't changed (no new events, same price → same decision → skip call)
```

### 8.3 Order Resolution

```
Backtest Mode:
    # Simple — use historical prices as execution prices
    for action in all_actions:
        if action.type == "BUY":
            cost = action.quantity * historical_close_price
            if agent.cash >= cost:
                agent.cash -= cost
                agent.holdings[stock] += action.quantity
        elif action.type == "SELL":
            if agent.holdings.get(stock, 0) >= action.quantity:
                agent.cash += action.quantity * historical_close_price
                agent.holdings[stock] -= action.quantity

Prediction Mode:
    # Order imbalance drives price
    total_buy_volume = sum(a.quantity for a in actions if a.type == "BUY")
    total_sell_volume = sum(a.quantity for a in actions if a.type == "SELL")
    total_volume = total_buy_volume + total_sell_volume
    
    if total_volume == 0:
        price_change_pct = random.gauss(0, historical_daily_vol)
    else:
        order_imbalance = (total_buy_volume - total_sell_volume) / total_volume
        
        # Calibration parameters (tune during backtesting)
        impact_coefficient = 0.02  # 1% imbalance → 0.02% price change
        
        price_change_pct = (
            order_imbalance * impact_coefficient +
            random.gauss(0, historical_daily_vol * 0.5)  # reduced noise
        )
    
    new_price = prev_price * (1 + price_change_pct)
    
    # Execute fills at new_price
    ...

IDX-Specific Rules:
    - Price tick size: IDR 1 for prices < 200, IDR 2 for 200-500, IDR 5 for 500-2000, IDR 10 for 2000-5000, IDR 25 for > 5000
    - Auto-reject limit: ±25% for regular stocks, ±35% for acceleration board
    - Lot size: 100 shares per lot
    - Implement: round prices to valid tick, enforce lot size, reject orders exceeding auto-reject limits
```

### 8.4 Scenario Clustering (Post-Simulation)

```
After all N runs complete:

1. Collect final price trajectory from each run
   trajectories = np.array of shape (num_runs, num_steps)

2. Normalize trajectories to returns
   returns = (trajectories / trajectories[:, 0:1]) - 1

3. Cluster using K-Means (k=3 as default)
   from sklearn.cluster import KMeans
   kmeans = KMeans(n_clusters=3, random_state=42)
   labels = kmeans.fit_predict(returns)

4. For each cluster:
   - Mean trajectory
   - Probability = count / total_runs
   - Price range (min, max within cluster)
   - Representative runs (closest to centroid)
   - Common agent behavior: analyze tier1 actions in representative runs

5. Pass to Observer agent for narrative generation

Dependency: scikit-learn (add to pyproject.toml)
```

---

## 9. Observer Agent Details

### 9.1 Aggregation Pipeline

```
class ObserverAgent:
    
    async analyze(multi_run_result: MultiRunResult) -> ScenarioReport:
        
        # Step 1: Compute statistics
        stats = compute_statistics(multi_run_result)
        # mean, median, std, percentiles, directional bias
        
        # Step 2: Cluster scenarios
        clusters = cluster_scenarios(multi_run_result.trajectories, n_clusters=3)
        
        # Step 3: Detect notable patterns
        patterns = detect_patterns(multi_run_result)
        # herding events, cascade events, consensus moments
        
        # Step 4: Generate report via LLM
        prompt = build_observer_prompt(stats, clusters, patterns)
        report_markdown = await llm_router.call(
            model=observer_model,  # GLM-5 with temperature=0.3
            prompt=prompt,
            max_tokens=4096
        )
        
        # Step 5: Parse and structure
        return ScenarioReport(
            stats=stats,
            clusters=clusters,
            patterns=patterns,
            narrative=report_markdown,
            generated_at=now()
        )
```

### 9.2 Pattern Detection Heuristics

```
Herding Detection:
    For each step, calculate:
        action_consensus = |count_buy - count_sell| / total_acting_agents
    If action_consensus > 0.7 and total_acting_agents > 10:
        Mark as herding event
    Record which agents led (acted first) and which followed

Panic Cascade Detection:
    If in consecutive steps (t, t+1, t+2):
        sell_ratio increases each step AND
        cumulative price drop > 3%
    Mark as panic cascade
    Record trigger event and initiating agent

Institutional Consensus:
    If all Tier 1 agents agree on direction (all buying or all selling):
        Record as consensus moment
    If Tier 1 agents diverge while Tier 2 herds:
        Record as smart money / dumb money divergence
```

---

## 10. Configuration Defaults

### 10.1 Master Configuration Object

```
class SimulationConfig(BaseModel):
    """All configurable parameters with sensible defaults"""
    
    # === Target ===
    target_stocks: list[str] = ["BBRI"]
    
    # === Mode ===
    mode: str = "BACKTEST"  # BACKTEST or PREDICT
    
    # === Time ===
    backtest_start: str = "2024-01-01"  # ISO date
    backtest_end: str = "2024-12-31"
    prediction_horizon_days: int = 20
    steps_per_day: int = 1  # daily granularity
    
    # === Agent Population ===
    tier1_personas: list[str] = [
        "pak_budi", "sarah", "andi", "dr_lim", "marketbot"
    ]
    tier2_per_archetype: int = 8
    tier2_archetypes: list[str] = [
        "momentum_chaser", "panic_seller", "dividend_holder",
        "sector_rotator", "news_reactive"
    ]
    tier3_total: int = 200
    tier3_distribution: dict = {
        "momentum_follower": 0.30,
        "mean_reversion": 0.25,
        "random_walk": 0.30,
        "volume_follower": 0.15
    }
    
    # === Multi-Run ===
    num_parallel_runs: int = 20  # start conservative, increase if budget allows
    runs_batch_size: int = 5  # concurrent runs
    
    # === LLM ===
    tier1_model: str = "glm-5"
    tier1_temperature: float = 0.7
    tier1_max_tokens: int = 1024
    tier2_model: str = "glm-5"
    tier2_temperature: float = 0.5
    tier2_max_tokens: int = 512
    observer_model: str = "glm-5"
    observer_temperature: float = 0.3
    observer_max_tokens: int = 4096
    max_concurrent_llm_calls: int = 5
    llm_timeout_seconds: int = 30
    
    # === Memory ===
    enable_episodic_memory: bool = True  # set False for Phase 1
    enable_social_memory: bool = True  # set False for Phase 1
    enable_causal_retrieval: bool = True  # set False for Phase 1
    episodic_top_k: int = 5
    causal_top_k: int = 3
    social_propagation_delay_min: int = 1
    social_propagation_delay_max: int = 3
    
    # === Trust Dynamics ===
    trust_prior_weight: float = 5.0
    trust_initial: float = 0.5
    
    # === Price Impact (Prediction Mode) ===
    impact_coefficient: float = 0.02
    noise_multiplier: float = 0.5
    
    # === Scenario Clustering ===
    num_scenario_clusters: int = 3
    
    # === Cost ===
    cost_alert_threshold_usd: float = 5.0
```

### 10.2 Phase-Specific Configuration Overrides

```
PHASE 1 CONFIG (minimal):
    target_stocks: ["BBRI"]
    mode: "BACKTEST"
    backtest_start: "2024-07-01"
    backtest_end: "2024-09-30"  # 3 months only
    tier1_personas: ["pak_budi", "sarah", "andi"]  # 3 only
    tier2_per_archetype: 4  # 20 total
    tier3_total: 50
    num_parallel_runs: 1  # single run
    enable_episodic_memory: false
    enable_social_memory: false
    enable_causal_retrieval: false

PHASE 2 CONFIG (memory enabled):
    backtest_start: "2024-01-01"
    backtest_end: "2024-12-31"  # full year
    tier1_personas: all 5
    tier2_per_archetype: 8  # 40 total
    tier3_total: 200
    num_parallel_runs: 10
    enable_episodic_memory: true
    enable_social_memory: true
    enable_causal_retrieval: true

PHASE 3 CONFIG (full):
    target_stocks: ["BBRI", "BMRI", "BBCA", "TLKM", "ASII"]
    num_parallel_runs: 50-100
    All memory features enabled
```

---

## 11. Testing Strategy

### 11.1 Unit Tests

```
test_agents/test_base.py:
    - test_agent_initialization: verify all persona fields populated
    - test_working_memory_update: add actions, verify deque behavior
    - test_portfolio_value_calculation: correct with holdings + cash
    - test_action_validation: reject invalid quantities, negative cash

test_agents/test_tier3.py:
    - test_momentum_heuristic: given rising prices → outputs BUY
    - test_mean_reversion_heuristic: given deviation from MA → correct direction
    - test_random_walk_distribution: over 1000 steps, roughly 50/50 buy/sell
    - test_lot_size_compliance: all quantities are multiples of 100

test_memory/test_episodic.py:
    - test_store_and_retrieve: store 10 experiences, query retrieves relevant ones
    - test_max_records_pruning: store 600 records, verify oldest pruned to 500
    - test_empty_retrieval: query with no stored records returns empty gracefully

test_memory/test_social.py:
    - test_trust_score_update: correct prediction increases trust
    - test_trust_decay_toward_neutral: without observations, trust moves to 0.5
    - test_observation_network: tier1 sees tier1, tier2 sees tier1 only

test_simulation/test_order_resolution.py:
    - test_backtest_fill: buy order fills at historical close price
    - test_insufficient_cash: buy order rejected if cash < cost
    - test_insufficient_holdings: sell order rejected if holdings < quantity
    - test_price_impact: large buy order moves price up in prediction mode
    - test_idx_tick_size: prices round to valid IDX tick sizes
    - test_lot_size_enforcement: quantities round to IDX lot sizes
```

### 11.2 Integration Tests

```
test_integration/test_single_run.py:
    - test_full_backtest_run: run 10 steps on BBRI with 3 agents, verify completion
    - test_agent_portfolios_consistent: after run, sum of all holdings + cash is conserved
    - test_step_logging: all steps logged to database
    - test_llm_fallback: simulate LLM timeout, verify HOLD fallback

test_integration/test_data_pipeline.py:
    - test_yfinance_download: fetch 1 month BBRI data, verify schema
    - test_event_processing: process sample event through tagging + sentiment + embedding
    - test_causal_retrieval: store 10 events, query similar, verify relevance
```

### 11.3 Smoke Test / Quick Validation

```
scripts/smoke_test.py:
    """
    Minimal end-to-end test that can run in <2 minutes.
    Uses only 3 Tier 1 agents, 0 Tier 2, 10 Tier 3.
    5 simulation steps. 1 run. Single stock.
    Verifies the system doesn't crash and produces output.
    """
    
    config = SimulationConfig(
        target_stocks=["BBRI"],
        mode="BACKTEST",
        backtest_start="2024-10-01",
        backtest_end="2024-10-07",  # 5 trading days
        tier1_personas=["pak_budi", "sarah", "andi"],
        tier2_per_archetype=0,
        tier3_total=10,
        num_parallel_runs=1,
        enable_episodic_memory=False,
        enable_social_memory=False,
        enable_causal_retrieval=False,
    )
    
    result = await engine.run_single(config, run_number=0)
    
    assert result.status == "COMPLETED"
    assert len(result.step_logs) == 5
    assert all(agent.cash >= 0 for agent in result.agents)
    print("Smoke test passed!")
```

### 11.4 Mock Data for Testing

```
Create: tests/fixtures/

mock_prices.json:
    5 days of BBRI OHLCV data (real or realistic values)
    {
        "2024-10-01": {"open": 5125, "high": 5175, "low": 5100, "close": 5150, "volume": 85000000},
        "2024-10-02": {"open": 5150, "high": 5200, "low": 5125, "close": 5175, "volume": 92000000},
        ...
    }

mock_events.json:
    2 sample events within the test period
    [
        {
            "timestamp": "2024-10-01T09:00:00+07:00",
            "category": "EARNINGS",
            "title": "BRI reports Q3 2024 net profit up 12% YoY",
            "summary": "Bank Rakyat Indonesia reported third quarter net profit of IDR 15.2 trillion, up 12% year-over-year, driven by strong micro-lending growth.",
            "affected_entities": ["BBRI", "BANKING"],
            "sentiment_score": 0.6,
            "magnitude_score": 0.7
        },
        ...
    ]
```

---

## 12. Phase-by-Phase Build Instructions

### Phase 1: Foundation (Weeks 1-2)

```
STEP-BY-STEP INSTRUCTIONS FOR CLAUDE CODE:

Step 1.1: Project Setup
    - Create directory structure as defined in spec Section 14
    - Create pyproject.toml with dependencies (Section 1.1)
    - Create .env.example (Section 1.2)
    - Create .gitignore (standard Python + data/ + .env)
    - Initialize git repo

Step 1.2: Database Models
    - Implement imss/db/models.py with SQLAlchemy models (Section 3.2)
    - Create imss/db/__init__.py with database initialization function
    - Use async SQLite for now
    - Create initial Alembic migration
    - Test: database creates successfully, tables exist

Step 1.3: Data Ingestion
    - Implement imss/data/price_feed.py
      - Function: fetch_idx_prices(symbols, start_date, end_date) → DataFrame
      - Uses yfinance with .JK suffix mapping
      - Stores results in StockOHLCV table
      - Includes data validation (no NaN, no zero volume)
    - Create scripts/seed_historical_data.py
      - Downloads BBRI.JK data for 2024
      - Stores in database
    - Test: run seed script, verify data in SQLite

Step 1.4: Event Seeding
    - Create data/seed_events/bbri_events_2024.json with 50 manually curated events
      (Instruct Claude Code to generate realistic events based on actual 2024 BBRI/IDX events it knows about — it should note these are illustrative and should be verified)
    - Implement imss/data/embedder.py
      - Initialize sentence-transformers model
      - Function: embed_text(text) → vector
    - Implement event loading script that:
      - Reads JSON events
      - Scores sentiment (use GLM if available, otherwise use manually provided scores)
      - Embeds summaries into ChromaDB
      - Stores in Event + EventEntity tables
    - Test: events loaded, queryable in ChromaDB

Step 1.5: LLM Client
    - Implement imss/llm/router.py
      - LLMRouter class with OpenAI-compatible client
      - Configure for GLM-5 as primary
      - Structured output parsing (JSON)
      - Error handling with retry logic
      - Token counting and cost tracking
    - Implement imss/llm/batcher.py
      - Async batched execution with semaphore
    - Test: make a single GLM-5 call, verify JSON response

Step 1.6: Agent Base Classes
    - Implement imss/agents/base.py
      - BaseAgent abstract class with observe/decide/execute/reflect interface
      - WorkingMemory Pydantic model
      - AgentAction model
    - Implement imss/agents/tier1/ persona files
      - Create 3 personas: pak_budi, sarah, andi (save dr_lim and marketbot for Phase 2)
      - Each has full persona prompt from Section 6.1
      - Decision method: builds full prompt, calls LLM, parses JSON response
    - Implement imss/agents/tier2/ archetype files
      - Create 3 archetypes: momentum_chaser, panic_seller, news_reactive
      - Simplified prompt from Section 5.2
    - Implement imss/agents/tier3/ heuristic agents
      - All 4 heuristic types (momentum_follower, mean_reversion, random_walk, volume_follower)
      - Pure Python, no LLM
    - Test: instantiate each agent type, verify prompt generation

Step 1.7: Simulation Loop
    - Implement imss/simulation/loop.py
      - Turn-based simulation loop as described in Section 8.1
      - For Phase 1: NO memory retrieval steps (skip episodic, social, causal)
      - Agent execution order: Tier 3 → Tier 2 (batched) → Tier 1
    - Implement imss/simulation/order_book.py
      - Backtest mode only: fill orders at historical close price
      - Validate cash sufficiency, holding sufficiency
      - Enforce IDX lot sizes (100 shares)
    - Implement imss/simulation/engine.py
      - SimulationEngine.run_single() method
      - Step logging to database
    - Test: run smoke test (5 steps, 3 agents)

Step 1.8: CLI Runner
    - Implement scripts/run_backtest.py
      - CLI script that runs a backtest with configurable parameters
      - Uses rich library for formatted output
      - Prints per-step summary: date, prices, agent actions, portfolio values
    - Test: full 3-month backtest on BBRI with Phase 1 config

Step 1.9: Validation
    - Do agents produce differentiated behavior? (Andi should trade more than Pak Budi)
    - Do portfolios remain consistent? (no negative cash, no phantom shares)
    - Are LLM responses valid JSON? (track parse error rate)
    - What's the actual cost of a single run?
    
    Document findings in: docs/phase1_results.md
```

### Phase 2: Memory & Intelligence (Weeks 3-4)

```
Step 2.1: Episodic Memory
    - Implement imss/memory/episodic.py (both full and simplified versions)
    - Integrate into Tier 1 agent decision flow (retrieve before deciding)
    - Integrate simplified version into Tier 2 agents
    - After each step, agents store their experience
    - Test: run 20-step simulation, verify memory grows and retrieval works

Step 2.2: Social Memory
    - Implement imss/memory/social.py
    - Initialize observation network (Section 7.3)
    - Implement trust score updates
    - Implement information propagation delays
    - Integrate social signals into agent prompts
    - Test: verify Tier 2 agents observe Tier 1, trust scores update correctly

Step 2.3: Causal Knowledge Graph
    - Implement imss/memory/causal.py
    - Build initial causal links from seeded events + historical prices
      (For each event, measure actual price change in subsequent 1, 3, 5 days)
    - Implement retrieval: given current event, find similar historical events
    - Integrate into Tier 1 agent prompts
    - Test: query causal memory with a sample event, verify relevant parallels returned

Step 2.4: Add Remaining Agents
    - Add Tier 1: dr_lim, marketbot personas
    - Add Tier 2: dividend_holder, sector_rotator archetypes
    - Scale to full agent count (5 Tier 1, 40 Tier 2, 200 Tier 3)

Step 2.5: Multi-Run Execution
    - Implement imss/simulation/runner.py
      - Multi-run parallel executor
      - Batched execution (5 runs at a time)
    - Implement scenario clustering (k-means on trajectories)
    - Test: run 10 parallel simulations, verify aggregation

Step 2.6: Observer Agent
    - Implement imss/observer/aggregator.py (statistics)
    - Implement imss/observer/pattern_detector.py (herding, cascades)
    - Implement imss/observer/report_generator.py (GLM-5 synthesis)
    - Test: generate scenario report from 10-run results

Step 2.7: 6-Month Backtest Validation
    - Run full backtest: BBRI, Jan-Jun 2024, all agents, memory enabled
    - Compare direction accuracy against baseline (random = 50%)
    - Verify herding/cascade events correlate with historical volatility spikes
    - Measure cost per run
    - Document in: docs/phase2_results.md
```

### Phase 3: Integration & Scale (Weeks 5-6)

```
Step 3.1: FastAPI Backend
    - Implement imss/api/main.py (FastAPI app)
    - Implement routes: simulation CRUD, results retrieval, agent chat
    - Implement WebSocket for live simulation progress
    - Test: API endpoints work via curl/httpie

Step 3.2: Prediction Mode
    - Implement price impact model in order_book.py
    - Implement prediction mode in simulation engine
    - Calibrate impact_coefficient using backtest data
    - Test: run forward prediction, verify plausible price trajectories

Step 3.3: Trading System Integration
    - Define input/output interfaces matching trading system v3.1
    - Implement scenario distribution output format
    - Test: end-to-end flow from screener signal → simulation → scenario output

Step 3.4: Multi-Stock Expansion
    - Add 4 more stocks: BMRI, BBCA, TLKM, ASII
    - Seed events for each stock (at least 20 events each)
    - Verify cross-stock simulation works

Step 3.5: News Scraping (Basic)
    - Implement basic Kontan.co.id scraper
    - Implement entity tagging + sentiment pipeline
    - Schedule daily data refresh
    - Test: scrape last 7 days, verify event quality

Step 3.6: PostgreSQL Migration (Optional)
    - If SQLite becomes a bottleneck:
      - Update DATABASE_URL to PostgreSQL
      - Run Alembic migrations
      - Verify everything works on new database
```

### Phase 4: UI & Polish (Weeks 7-8)

```
Step 4.1: Frontend Setup
    - Initialize Vue.js or React project in frontend/
    - Set up API client connecting to FastAPI backend

Step 4.2: Simulation Dashboard
    - Run configuration form
    - Real-time progress via WebSocket
    - Cost tracking display
    - Run history table

Step 4.3: Scenario Report View
    - Render observer agent's markdown report
    - Interactive price distribution chart (Recharts / ECharts)
    - Scenario cluster visualization

Step 4.4: Agent Network Visualization
    - D3.js or vis.js force-directed graph
    - Nodes = agents (sized by portfolio value, colored by tier)
    - Edges = observation relationships (thickness by trust score)
    - Animate information propagation and herding events

Step 4.5: Agent Chat Interface
    - Post-simulation chat with any agent
    - Agent responds in character using its persona + simulation memory
    - Shows agent's portfolio history and decision timeline

Step 4.6: Demo & Documentation
    - Record 5-minute demo video
    - Write comprehensive README with architecture diagram
    - Publish to GitHub with clean commit history
```

---

## 13. Claude Code Usage Notes

### 13.1 How to Feed This to Claude Code

```
Recommended approach:

1. Start a new Claude Code session
2. Provide both documents:
   - IDX_Market_Swarm_Simulator_Spec.md (architecture & requirements)
   - IDX_Market_Swarm_Simulator_Implementation.md (this file)
3. Say: "Implement Phase 1 following the step-by-step instructions in the implementation guide, starting from Step 1.1."
4. Let Claude Code work through each step sequentially
5. After each step, verify the output before moving to the next
6. If Claude Code makes assumptions, redirect it to the specific section of the implementation guide

Important: Do NOT ask Claude Code to implement everything at once.
Go step by step within each phase. Verify each step works before proceeding.
```

### 13.2 Common Pitfalls to Watch For

```
1. GLM-5 JSON output reliability
   - GLM-5 may sometimes return JSON with markdown backticks
   - Implementation MUST strip ```json and ``` before parsing
   - Implementation MUST handle parse failures gracefully (default to HOLD)

2. Async SQLite concurrency
   - SQLite doesn't support concurrent writes well
   - Use a write queue or serialize all DB writes through a single connection
   - This is a Phase 1-2 issue; PostgreSQL resolves it in Phase 3

3. ChromaDB embedding model download
   - First run of sentence-transformers will download ~90MB model
   - This needs internet access; cache it in the project data/ directory
   - Alternative: use ChromaDB's built-in default embedding function

4. Yahoo Finance IDX data gaps
   - Indonesian holidays → missing data
   - Some tickers may have sparse historical data
   - Always validate data completeness before running simulation

5. LLM rate limits
   - GLM-5 may have request-per-minute limits
   - The batcher's semaphore (max_concurrent=5) should help
   - If hitting limits, reduce to max_concurrent=3 and add delays

6. Agent portfolio conservation
   - After every step, verify: sum(all_agent_cash) + sum(all_holdings * price) ≈ constant
   - Small differences due to bid-ask simulation are OK
   - Large discrepancies indicate a bug in order resolution

7. Prompt length management
   - Tier 1 prompts with full memory can get very long
   - Track prompt token count; if > 3000 tokens, reduce episodic_top_k or causal_top_k
   - GLM-5 context window should be sufficient but monitor

8. Event timestamp alignment
   - All events must map to actual IDX trading days
   - Events on weekends/holidays should be mapped to next trading day
   - Use pandas.tseries.offsets.CustomBusinessDay with IDX holiday calendar
```

### 13.3 File Execution Order for Claude Code

```
If Claude Code needs to know the order to create files:

Round 1 (Infrastructure):
  1. pyproject.toml
  2. .env.example
  3. .gitignore
  4. imss/__init__.py
  5. imss/config.py (load .env, define SimulationConfig)

Round 2 (Database):
  6. imss/db/__init__.py
  7. imss/db/models.py
  8. scripts/init_db.py

Round 3 (Data):
  9. imss/data/price_feed.py
  10. imss/data/embedder.py
  11. data/seed_events/bbri_events_2024.json
  12. scripts/seed_historical_data.py

Round 4 (LLM):
  13. imss/llm/providers/__init__.py
  14. imss/llm/providers/glm.py (or unified openai-compatible client)
  15. imss/llm/router.py
  16. imss/llm/batcher.py
  17. imss/llm/prompts/tier1_decision.py
  18. imss/llm/prompts/tier2_decision.py

Round 5 (Agents):
  19. imss/agents/base.py
  20. imss/agents/tier1/personas.py
  21. imss/agents/tier2/archetypes.py
  22. imss/agents/tier3/heuristic.py

Round 6 (Simulation):
  23. imss/simulation/order_book.py
  24. imss/simulation/loop.py
  25. imss/simulation/engine.py

Round 7 (Runner & Tests):
  26. scripts/run_backtest.py
  27. scripts/smoke_test.py
  28. tests/fixtures/mock_prices.json
  29. tests/fixtures/mock_events.json
  30. tests/test_agents/test_base.py
  31. tests/test_simulation/test_order_resolution.py

This order ensures dependencies are available when needed.
```

---

## Appendix A: IDX Market Reference Data

```
IDX Trading Hours:
  Pre-opening: 08:45 - 09:00 WIB (UTC+7)
  Session 1: 09:00 - 11:30 WIB
  Session 2: 13:30 - 14:50 WIB (Mon-Thu), 14:00 - 14:50 WIB (Fri)
  Pre-closing: 14:50 - 15:00 WIB
  Post-closing: 15:00 - 15:15 WIB

IDX Lot Size: 100 shares

IDX Price Tick Sizes:
  Price < IDR 200: tick = IDR 1
  IDR 200 - 500: tick = IDR 2
  IDR 500 - 2,000: tick = IDR 5
  IDR 2,000 - 5,000: tick = IDR 10
  Price > IDR 5,000: tick = IDR 25

Auto-Reject Limits (daily price change limits):
  Regular: ±25%
  Acceleration board: ±35%

Key IDX Sectors:
  BANKING: BBRI, BMRI, BBCA, BBNI, BRIS
  TELECOM: TLKM, EXCL, ISAT
  AUTOMOTIVE: ASII
  MINING: ADRO, ANTM, INCO, PTBA
  CONSUMER: UNVR, ICBP, INDF
  PROPERTY: BSDE, CTRA, SMRA

Major IDX Indices:
  IHSG (^JKSE): Composite index
  LQ45: Top 45 liquid stocks
  IDX30: Top 30 stocks
```

## Appendix B: GLM-5 API Quick Reference

```
Base URL: https://open.bigmodel.cn/api/paas/v4
Auth: Authorization: Bearer {API_KEY}

Chat Completion:
  POST /chat/completions
  {
    "model": "glm-5",
    "messages": [
      {"role": "system", "content": "..."},
      {"role": "user", "content": "..."}
    ],
    "temperature": 0.7,
    "max_tokens": 1024,
    "top_p": 0.9
  }

Response:
  {
    "choices": [
      {
        "message": {
          "role": "assistant",
          "content": "..."
        },
        "finish_reason": "stop"
      }
    ],
    "usage": {
      "prompt_tokens": 100,
      "completion_tokens": 50,
      "total_tokens": 150
    }
  }

Using OpenAI SDK:
  from openai import AsyncOpenAI
  client = AsyncOpenAI(
      api_key="your_key",
      base_url="https://open.bigmodel.cn/api/paas/v4"
  )
  response = await client.chat.completions.create(
      model="glm-5",
      messages=[...],
      temperature=0.7
  )
  content = response.choices[0].message.content
  tokens = response.usage.total_tokens
```
