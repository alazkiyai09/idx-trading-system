# IMSS Phase 2B — New Agents & Multi-Run Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add 4 new agent personas/archetypes (Dr. Lim, MarketBot, dividend_holder, sector_rotator) and multi-run parallel execution with statistical aggregation to the IMSS simulation engine.

**Architecture:** Extends the existing three-tier agent system with 2 new Tier 1 personas and 2 new Tier 2 archetypes. MarketBot uses a hybrid decide() pattern (rule-based direction + LLM reasoning). Multi-run execution calls run_single() sequentially with different seeds, then aggregates results via a pure-function aggregator.

**Tech Stack:** Python 3.12, async SQLAlchemy 2.0, Pydantic v2, AsyncOpenAI (GLM-5), pytest + pytest-asyncio

**Spec:** `docs/superpowers/specs/2026-03-18-imss-phase2b-agents-multirun-design.md`

---

## Chunk 1: StockFundamentals + Dr. Lim Persona

### Task 1: StockFundamentals DB Model

**Files:**
- Modify: `imss/db/models.py:149` (add new model before `get_engine()`)
- Modify: `imss/db/__init__.py` (add export)
- Test: `tests/imss/test_data/test_db.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/imss/test_data/test_db.py`:

```python
@pytest.mark.asyncio
async def test_stock_fundamentals_table_exists(tmp_path):
    """StockFundamentals table is created alongside other tables."""
    url = f"sqlite+aiosqlite:///{tmp_path}/test.db"
    from sqlalchemy.ext.asyncio import create_async_engine
    from imss.db.models import Base

    engine = create_async_engine(url)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with engine.connect() as conn:
        result = await conn.run_sync(
            lambda sync_conn: sync_conn.execute(
                __import__("sqlalchemy").text(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='stock_fundamentals'"
                )
            ).fetchone()
        )
    assert result is not None, "stock_fundamentals table should exist"
    await engine.dispose()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/imss/test_data/test_db.py::test_stock_fundamentals_table_exists -v`
Expected: FAIL — table does not exist yet

- [ ] **Step 3: Add StockFundamentals model**

In `imss/db/models.py`, add before the `# --- Database initialization ---` comment (line 151):

```python
class StockFundamentals(Base):
    __tablename__ = "stock_fundamentals"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(10), index=True)
    period: Mapped[str] = mapped_column(String(10))  # e.g. "2024-Q2"
    pe_ratio: Mapped[float] = mapped_column(Float)
    pb_ratio: Mapped[float] = mapped_column(Float)
    dividend_yield_pct: Mapped[float] = mapped_column(Float)
    roe_pct: Mapped[float] = mapped_column(Float)
    market_cap_trillion_idr: Mapped[float] = mapped_column(Float)
```

- [ ] **Step 4: Update `imss/db/__init__.py`**

Add `StockFundamentals` to both the import and `__all__`:

```python
from imss.db.models import (
    AgentConfig,
    Base,
    CausalLink,
    Event,
    EventEntity,
    SimulationRun,
    SimulationStepLog,
    StockFundamentals,
    StockOHLCV,
    create_tables,
    get_engine,
    get_session_factory,
)

__all__ = [
    "AgentConfig",
    "Base",
    "CausalLink",
    "Event",
    "EventEntity",
    "SimulationRun",
    "SimulationStepLog",
    "StockFundamentals",
    "StockOHLCV",
    "create_tables",
    "get_engine",
    "get_session_factory",
]
```

- [ ] **Step 5: Update existing DB test to include new table**

In `tests/imss/test_data/test_db.py`, update the existing test's expected set (line 12 comment and lines 24-32):

Change the docstring from `"""All 7 IMSS tables are created in SQLite."""` to `"""All 8 IMSS tables are created in SQLite."""`

Add `"stock_fundamentals"` to the expected set:

```python
    expected = {
        "stocks_ohlcv",
        "events",
        "event_entities",
        "causal_links",
        "simulation_runs",
        "agent_configs",
        "simulation_step_logs",
        "stock_fundamentals",
    }
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `python3 -m pytest tests/imss/test_data/test_db.py -v`
Expected: ALL PASS (existing updated + new)

- [ ] **Step 7: Create seed data file**

Create `data/seed_events/bbri_fundamentals.json`:

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

- [ ] **Step 8: Update seed data script**

In `scripts/imss_seed_data.py`, add after the events processing block (after line 116):

```python
    # 6. Load fundamentals
    fundamentals_path = Path("data/seed_events/bbri_fundamentals.json")
    if fundamentals_path.exists():
        console.print("[bold blue]Loading fundamentals data...[/]")
        fund_data = json.loads(fundamentals_path.read_text())
        from imss.db.models import StockFundamentals
        async with session_factory() as session:
            async with session.begin():
                session.add(StockFundamentals(
                    symbol=fund_data["symbol"],
                    period=fund_data["period"],
                    pe_ratio=fund_data["pe_ratio"],
                    pb_ratio=fund_data["pb_ratio"],
                    dividend_yield_pct=fund_data["dividend_yield_pct"],
                    roe_pct=fund_data["roe_pct"],
                    market_cap_trillion_idr=fund_data["market_cap_trillion_idr"],
                ))
        console.print("[green]Loaded fundamentals for BBRI[/]")
```

- [ ] **Step 9: Commit**

```bash
git add -f imss/db/models.py imss/db/__init__.py data/seed_events/bbri_fundamentals.json scripts/imss_seed_data.py tests/imss/test_data/test_db.py
git commit -m "feat(imss): add StockFundamentals model and BBRI seed data"
```

---

### Task 2: Dr. Lim Persona + Prompt Enhancement

**Files:**
- Modify: `imss/agents/tier1/personas.py:94` (add to PERSONAS dict)
- Modify: `imss/llm/prompts/tier1_decision.py:40-68` (add fundamentals param)
- Test: `tests/imss/test_agents/test_tier1_new.py` (create)

- [ ] **Step 1: Write the failing tests**

Create `tests/imss/test_agents/test_tier1_new.py`:

```python
"""Tests for Phase 2B Tier 1 agents: Dr. Lim and MarketBot."""

import pytest
from imss.agents.tier1.personas import PERSONAS, create_tier1_agent


class TestDrLimPersona:
    def test_dr_lim_exists_in_personas(self):
        assert "dr_lim" in PERSONAS

    def test_dr_lim_config_fields(self):
        pc = PERSONAS["dr_lim"]
        assert pc.name == "Dr. Lim"
        assert pc.risk_tolerance == 0.4
        assert pc.stop_loss_pct == 20
        assert pc.initial_cash == 5_000_000_000
        assert pc.initial_holdings == {"BBRI": 300_000}

    def test_dr_lim_agent_creation(self):
        agent = create_tier1_agent("dr_lim")
        assert agent.tier == 1
        assert agent.persona_type == "tier1_dr_lim"
        assert agent.working_memory.cash == 5_000_000_000
        assert agent.working_memory.holdings["BBRI"] == 300_000


class TestTier1PromptFundamentals:
    def test_prompt_includes_fundamentals_when_provided(self):
        from imss.llm.prompts.tier1_decision import build_tier1_user_prompt

        fundamentals = {
            "pe_ratio": 12.5,
            "pb_ratio": 2.1,
            "dividend_yield_pct": 5.2,
            "roe_pct": 18.3,
            "market_cap_trillion_idr": 620,
        }
        prompt = build_tier1_user_prompt(
            step=1, simulated_date="2024-07-01", cash=5e9,
            holdings_formatted="BBRI: 300,000", portfolio_value=6.5e9,
            unrealized_pnl=2.5, stock_symbol="BBRI",
            ohlcv={"open": 5100, "high": 5200, "low": 5050, "close": 5150, "volume": 80000000},
            pct_change_5d=1.5, pct_change_20d=3.2,
            events_formatted="No events",
            fundamentals=fundamentals,
        )
        assert "P/E Ratio: 12.5" in prompt
        assert "Dividend Yield: 5.2%" in prompt
        assert "ROE: 18.3%" in prompt

    def test_prompt_omits_fundamentals_when_none(self):
        from imss.llm.prompts.tier1_decision import build_tier1_user_prompt

        prompt = build_tier1_user_prompt(
            step=1, simulated_date="2024-07-01", cash=5e9,
            holdings_formatted="None", portfolio_value=5e9,
            unrealized_pnl=0.0, stock_symbol="BBRI",
            ohlcv={"open": 5100, "high": 5200, "low": 5050, "close": 5150, "volume": 80000000},
            pct_change_5d=0.0, pct_change_20d=0.0,
            events_formatted="",
            fundamentals=None,
        )
        assert "Fundamental Data" not in prompt
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/imss/test_agents/test_tier1_new.py -v`
Expected: FAIL — `dr_lim` not in PERSONAS, `fundamentals` not accepted

- [ ] **Step 3: Add Dr. Lim to PERSONAS dict**

In `imss/agents/tier1/personas.py`, add after the `"andi"` entry (after line 93, before the closing `}`):

```python
    "dr_lim": PersonaConfig(
        key="dr_lim",
        name="Dr. Lim",
        description="Independent equity analyst with PhD in finance. Deep fundamental analysis. Goes against the crowd when valuations are extreme.",
        behavioral_rules=(
            "- You are a contrarian — you buy when fear is high and valuations are low\n"
            "- You evaluate stocks on P/E, P/B, dividend yield, and ROE fundamentals\n"
            "- You are patient — you hold losing positions for months if your thesis is intact\n"
            "- You ignore short-term noise and social media hype\n"
            "- You increase position sizes when fear is high and valuations are low\n"
            "- You are a statistical thinker who believes in mean reversion\n"
            "- When the market is euphoric and prices are high, you reduce positions"
        ),
        risk_tolerance=0.4,
        max_allocation=35,
        stop_loss_pct=20,
        holding_period="months",
        initial_cash=5_000_000_000,
        initial_holdings={"BBRI": 300_000},
    ),
```

- [ ] **Step 4: Add fundamentals parameter to prompt builder**

In `imss/llm/prompts/tier1_decision.py`, modify `build_tier1_user_prompt()` (line 40) to accept an optional `fundamentals` param:

Change the function signature from:
```python
def build_tier1_user_prompt(
    step: int,
    simulated_date: str,
    cash: float,
    holdings_formatted: str,
    portfolio_value: float,
    unrealized_pnl: float,
    stock_symbol: str,
    ohlcv: dict,
    pct_change_5d: float,
    pct_change_20d: float,
    events_formatted: str,
) -> str:
```

To:
```python
def build_tier1_user_prompt(
    step: int,
    simulated_date: str,
    cash: float,
    holdings_formatted: str,
    portfolio_value: float,
    unrealized_pnl: float,
    stock_symbol: str,
    ohlcv: dict,
    pct_change_5d: float,
    pct_change_20d: float,
    events_formatted: str,
    fundamentals: dict | None = None,
) -> str:
```

And append the fundamentals section at the end of the f-string, before the final `Make your investment decision.` line. Replace the return statement with:

```python
    base = f"""=== SIMULATION STEP {step} — Date: {simulated_date} ===

YOUR CURRENT PORTFOLIO:
- Cash: IDR {cash:,.0f}
- Holdings: {holdings_formatted}
- Portfolio value: IDR {portfolio_value:,.0f}
- Unrealized P&L: {unrealized_pnl:+.2f}%

MARKET DATA TODAY:
{stock_symbol}: Open {ohlcv['open']} | High {ohlcv['high']} | Low {ohlcv['low']} | Close {ohlcv['close']} | Volume {ohlcv['volume']:,}
5-day change: {pct_change_5d:+.2f}% | 20-day change: {pct_change_20d:+.2f}%

NEW EVENTS TODAY:
{events_formatted if events_formatted else "No significant events today."}"""

    if fundamentals:
        base += f"""

== Fundamental Data ==
P/E Ratio: {fundamentals['pe_ratio']}
P/B Ratio: {fundamentals['pb_ratio']}
Dividend Yield: {fundamentals['dividend_yield_pct']}%
ROE: {fundamentals['roe_pct']}%
Market Cap: IDR {fundamentals['market_cap_trillion_idr']}T"""

    base += "\n\nMake your investment decision."
    return base
```

- [ ] **Step 5: Wire fundamentals through Tier1Agent.decide()**

In `imss/agents/tier1/personas.py`, modify the `build_tier1_user_prompt()` call inside `Tier1Agent.decide()` (around line 155-167). Add the `fundamentals` kwarg — only Dr. Lim gets fundamentals:

```python
        # Pass fundamentals only to Dr. Lim
        agent_fundamentals = market_state.get("fundamentals") if self.persona_config.key == "dr_lim" else None

        user_prompt = build_tier1_user_prompt(
            step=step,
            simulated_date=market_state.get("date", ""),
            cash=self.working_memory.cash,
            holdings_formatted=holdings_formatted,
            portfolio_value=portfolio_value,
            unrealized_pnl=unrealized_pnl,
            stock_symbol=stock,
            ohlcv=ohlcv,
            pct_change_5d=market_state.get("pct_change_5d", 0),
            pct_change_20d=market_state.get("pct_change_20d", 0),
            events_formatted=events_formatted,
            fundamentals=agent_fundamentals,
        )
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `python3 -m pytest tests/imss/test_agents/test_tier1_new.py::TestDrLimPersona -v && python3 -m pytest tests/imss/test_agents/test_tier1_new.py::TestTier1PromptFundamentals -v`
Expected: ALL PASS

- [ ] **Step 7: Run all existing tests to verify no regressions**

Run: `python3 -m pytest tests/imss/ -v`
Expected: ALL 29 existing tests + 5 new tests PASS

- [ ] **Step 8: Commit**

```bash
git add -f imss/agents/tier1/personas.py imss/llm/prompts/tier1_decision.py tests/imss/test_agents/test_tier1_new.py
git commit -m "feat(imss): add Dr. Lim persona with fundamentals prompt support"
```

---

## Chunk 2: MarketBot Hybrid Agent

### Task 3: MarketBot Prompt Templates

**Files:**
- Create: `imss/llm/prompts/marketbot_decision.py`

- [ ] **Step 1: Create the MarketBot prompt module**

Create `imss/llm/prompts/marketbot_decision.py`:

```python
"""MarketBot market-maker prompt templates."""

from __future__ import annotations


def build_marketbot_system_prompt() -> str:
    return """You are MarketBot, an automated market-making algorithm on the Indonesia Stock Exchange (IDX).

YOUR ROLE:
- You provide LIQUIDITY to the market
- You COUNTER the aggregate order flow: when others buy, you sell; when others sell, you buy
- You have NO directional opinion — your goal is to capture spread, not predict direction
- You reduce activity (smaller quantities) during high volatility
- You increase activity during calm markets
- You never hold large net positions; you rebalance toward neutral

The system has already determined your DIRECTION for this step based on aggregate order flow.
Your job is to determine the QUANTITY and provide CONFIDENCE + REASONING.

You MUST respond with valid JSON only. No other text.

Response format:
{
  "action": "<DIRECTION WILL BE PROVIDED — use it exactly>",
  "stock": "<IDX ticker>",
  "quantity": <integer shares, multiple of 100>,
  "confidence": <float 0.0-1.0>,
  "reasoning": "<2-3 sentences explaining your sizing decision>",
  "sentiment_update": <float -1.0 to 1.0, always near 0.0 for market makers>
}"""


def build_marketbot_user_prompt(
    direction: str,
    imbalance: float,
    pct_change_5d: float,
    cash: float,
    holdings_formatted: str,
    stock_symbol: str,
    close_price: float,
) -> str:
    volatility_note = ""
    if abs(pct_change_5d) > 5:
        volatility_note = "HIGH VOLATILITY DETECTED — reduce position sizing by 50%."
    else:
        volatility_note = "Normal volatility — standard position sizing."

    return f"""=== MARKET MAKER DECISION ===

PRE-DETERMINED DIRECTION: {direction}
(Based on aggregate order imbalance: {imbalance:+.3f})

YOUR PORTFOLIO:
- Cash: IDR {cash:,.0f}
- Holdings: {holdings_formatted}

MARKET STATE:
- {stock_symbol} close: {close_price:,.0f}
- 5-day price change: {pct_change_5d:+.2f}%
- {volatility_note}

Determine the quantity (multiple of 100 shares) for your {direction} order.
Set action to "{direction}" in your response."""
```

- [ ] **Step 2: Commit**

```bash
git add -f imss/llm/prompts/marketbot_decision.py
git commit -m "feat(imss): add MarketBot prompt templates"
```

---

### Task 4: MarketBotAgent Class

**Files:**
- Create: `imss/agents/tier1/marketbot.py`
- Modify: `imss/agents/tier1/personas.py` (add marketbot PersonaConfig)
- Modify: `imss/agents/tier1/__init__.py` (add exports)
- Test: `tests/imss/test_agents/test_tier1_new.py` (add MarketBot tests)

- [ ] **Step 1: Write the failing tests**

Add to `tests/imss/test_agents/test_tier1_new.py`:

```python
from unittest.mock import AsyncMock, patch
from imss.llm.router import LLMResponse


class TestMarketBotAgent:
    def test_marketbot_exists_in_personas(self):
        assert "marketbot" in PERSONAS

    def test_marketbot_config_fields(self):
        pc = PERSONAS["marketbot"]
        assert pc.name == "MarketBot"
        assert pc.risk_tolerance == 0.2
        assert pc.stop_loss_pct == 1
        assert pc.initial_cash == 20_000_000_000
        assert pc.initial_holdings == {"BBRI": 200_000}

    @pytest.mark.asyncio
    async def test_marketbot_sells_on_positive_imbalance(self):
        from imss.agents.tier1.marketbot import MarketBotAgent
        agent = create_tier1_agent("marketbot")
        assert isinstance(agent, MarketBotAgent)

        mock_response = LLMResponse(
            content='{"action": "SELL", "stock": "BBRI", "quantity": 1000, "confidence": 0.7, "reasoning": "providing liquidity", "sentiment_update": 0.0}',
            parsed_json={"action": "SELL", "stock": "BBRI", "quantity": 1000, "confidence": 0.7, "reasoning": "providing liquidity", "sentiment_update": 0.0},
            input_tokens=100, output_tokens=50, model="glm-5", latency_ms=100,
        )
        mock_router = AsyncMock()
        mock_router.call = AsyncMock(return_value=mock_response)
        agent.set_router(mock_router)

        market_state = {
            "symbol": "BBRI", "date": "2024-07-01",
            "ohlcv": {"open": 5100, "high": 5200, "low": 5050, "close": 5150, "volume": 80000000},
            "prices": {"BBRI": 5150}, "price_history": [5100, 5150],
            "volume_history": [80000000], "pct_change_1d": 0.5,
            "pct_change_5d": 1.0, "pct_change_20d": 2.0,
            "prev_aggregate_order_imbalance": 0.5,  # positive = net buying
        }
        action = await agent.decide(market_state, [], 1)
        assert action.action == "SELL"

    @pytest.mark.asyncio
    async def test_marketbot_buys_on_negative_imbalance(self):
        from imss.agents.tier1.marketbot import MarketBotAgent
        agent = create_tier1_agent("marketbot")

        mock_response = LLMResponse(
            content='{"action": "BUY", "stock": "BBRI", "quantity": 500, "confidence": 0.6, "reasoning": "absorbing sell pressure", "sentiment_update": 0.0}',
            parsed_json={"action": "BUY", "stock": "BBRI", "quantity": 500, "confidence": 0.6, "reasoning": "absorbing sell pressure", "sentiment_update": 0.0},
            input_tokens=100, output_tokens=50, model="glm-5", latency_ms=100,
        )
        mock_router = AsyncMock()
        mock_router.call = AsyncMock(return_value=mock_response)
        agent.set_router(mock_router)

        market_state = {
            "symbol": "BBRI", "date": "2024-07-01",
            "ohlcv": {"open": 5100, "high": 5200, "low": 5050, "close": 5150, "volume": 80000000},
            "prices": {"BBRI": 5150}, "price_history": [5100, 5150],
            "volume_history": [80000000], "pct_change_1d": -0.5,
            "pct_change_5d": -1.0, "pct_change_20d": -2.0,
            "prev_aggregate_order_imbalance": -0.5,  # negative = net selling
        }
        action = await agent.decide(market_state, [], 1)
        assert action.action == "BUY"

    @pytest.mark.asyncio
    async def test_marketbot_holds_on_near_zero_imbalance(self):
        from imss.agents.tier1.marketbot import MarketBotAgent
        agent = create_tier1_agent("marketbot")
        # No router needed — HOLD doesn't call LLM

        market_state = {
            "symbol": "BBRI", "date": "2024-07-01",
            "ohlcv": {"open": 5100, "high": 5200, "low": 5050, "close": 5150, "volume": 80000000},
            "prices": {"BBRI": 5150}, "price_history": [5100, 5150],
            "volume_history": [80000000], "pct_change_1d": 0.0,
            "pct_change_5d": 0.0, "pct_change_20d": 0.0,
            "prev_aggregate_order_imbalance": 0.05,  # near zero
        }
        action = await agent.decide(market_state, [], 1)
        assert action.action == "HOLD"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/imss/test_agents/test_tier1_new.py::TestMarketBotAgent -v`
Expected: FAIL — `marketbot` not in PERSONAS, MarketBotAgent does not exist

- [ ] **Step 3: Add MarketBot persona to PERSONAS dict**

In `imss/agents/tier1/personas.py`, add after the `"dr_lim"` entry:

```python
    "marketbot": PersonaConfig(
        key="marketbot",
        name="MarketBot",
        description="Automated market making algorithm. Provides liquidity, profits from bid-ask spread. No directional bias.",
        behavioral_rules=(
            "- You provide liquidity — you buy when others are selling, sell when others are buying\n"
            "- You have no directional opinion — your goal is to capture spread\n"
            "- You reduce activity during high volatility\n"
            "- You increase activity during calm, range-bound markets\n"
            "- You never hold large net positions — you rebalance toward neutral\n"
            "- You execute based on aggregate order flow, not events"
        ),
        risk_tolerance=0.2,
        max_allocation=20,
        stop_loss_pct=1,
        holding_period="intraday",
        initial_cash=20_000_000_000,
        initial_holdings={"BBRI": 200_000},
    ),
```

- [ ] **Step 4: Create MarketBotAgent class**

Create `imss/agents/tier1/marketbot.py`:

```python
"""MarketBot — hybrid market-maker agent with rule-based direction + LLM reasoning."""

from __future__ import annotations

from typing import Any

from imss.agents.base import AgentAction, default_hold_action, round_to_lot
from imss.agents.tier1.personas import Tier1Agent
from imss.llm.prompts.marketbot_decision import (
    build_marketbot_system_prompt,
    build_marketbot_user_prompt,
)


class MarketBotAgent(Tier1Agent):
    """Hybrid market maker: rule-based direction, LLM-reasoned sizing."""

    async def decide(
        self,
        market_state: dict[str, Any],
        events: list[dict[str, Any]],
        step: int,
    ) -> AgentAction:
        stock = market_state.get("symbol", "BBRI")
        imbalance = market_state.get("prev_aggregate_order_imbalance", 0.0)
        pct_change_5d = market_state.get("pct_change_5d", 0.0)

        # 1. Rule-based direction
        if imbalance > 0.1:
            direction = "SELL"
        elif imbalance < -0.1:
            direction = "BUY"
        else:
            return default_hold_action(self.id, stock, step)

        # 2. If no router, return with default quantity
        if self._router is None:
            return default_hold_action(self.id, stock, step)

        # 3. Format holdings
        parts = [f"{s}: {q:,} shares" for s, q in self.working_memory.holdings.items()]
        holdings_formatted = "; ".join(parts) if parts else "None"

        close_price = market_state.get("ohlcv", {}).get("close", 0)

        system_prompt = build_marketbot_system_prompt()
        user_prompt = build_marketbot_user_prompt(
            direction=direction,
            imbalance=imbalance,
            pct_change_5d=pct_change_5d,
            cash=self.working_memory.cash,
            holdings_formatted=holdings_formatted,
            stock_symbol=stock,
            close_price=close_price,
        )

        response = await self._router.call(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.3,
            max_tokens=512,
            tier=1,
        )

        if response.parsed_json is None:
            return default_hold_action(self.id, stock, step)

        data = response.parsed_json
        quantity = round_to_lot(int(data.get("quantity", 0)))

        # Volatility gate: reduce sizing by 50% in high volatility
        if abs(pct_change_5d) > 5:
            quantity = round_to_lot(quantity // 2)

        return AgentAction(
            agent_id=self.id,
            step=step,
            action=direction,  # Use rule-based direction, not LLM's
            stock=stock,
            quantity=quantity,
            confidence=float(data.get("confidence", 0.5)),
            reasoning=data.get("reasoning", ""),
            sentiment_update=float(data.get("sentiment_update", 0.0)),
        )
```

- [ ] **Step 5: Update factory function in personas.py**

In `imss/agents/tier1/personas.py`, modify `create_tier1_agent()` (line 193) to return MarketBotAgent for marketbot:

```python
def create_tier1_agent(persona_key: str) -> Tier1Agent:
    """Factory: create a Tier 1 agent from persona key."""
    config = PERSONAS[persona_key]
    if persona_key == "marketbot":
        from imss.agents.tier1.marketbot import MarketBotAgent
        return MarketBotAgent(
            id=config.key,
            name=config.name,
            persona_type=f"tier1_{config.key}",
            persona_config=config,
            working_memory=WorkingMemory(
                cash=config.initial_cash,
                holdings=dict(config.initial_holdings),
            ),
        )
    return Tier1Agent(
        id=config.key,
        name=config.name,
        persona_type=f"tier1_{config.key}",
        persona_config=config,
        working_memory=WorkingMemory(
            cash=config.initial_cash,
            holdings=dict(config.initial_holdings),
        ),
    )
```

- [ ] **Step 6: Update `imss/agents/tier1/__init__.py`**

```python
"""Tier 1 named agents."""

from imss.agents.tier1.marketbot import MarketBotAgent
from imss.agents.tier1.personas import PERSONAS, Tier1Agent, create_tier1_agent

__all__ = ["MarketBotAgent", "PERSONAS", "Tier1Agent", "create_tier1_agent"]
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `python3 -m pytest tests/imss/test_agents/test_tier1_new.py -v`
Expected: ALL 8 tests PASS

- [ ] **Step 8: Run all tests for regressions**

Run: `python3 -m pytest tests/imss/ -v`
Expected: ALL PASS (29 existing + 8 new)

- [ ] **Step 9: Commit**

```bash
git add -f imss/agents/tier1/marketbot.py imss/agents/tier1/personas.py imss/agents/tier1/__init__.py tests/imss/test_agents/test_tier1_new.py
git commit -m "feat(imss): add MarketBot hybrid market-maker agent"
```

---

## Chunk 3: Tier 2 Archetypes + Decision Latency

### Task 5: New Tier 2 Archetypes + Latency Filter

**Files:**
- Modify: `imss/agents/tier2/archetypes.py:30-58` (add archetypes), `:77-82` (add latency filter)
- Modify: `imss/config.py:65-69` (update default archetypes list)
- Test: `tests/imss/test_agents/test_tier2_new.py` (create)

- [ ] **Step 1: Write the failing tests**

Create `tests/imss/test_agents/test_tier2_new.py`:

```python
"""Tests for Phase 2B Tier 2 archetypes: dividend_holder and sector_rotator."""

import pytest
from imss.agents.tier2.archetypes import ARCHETYPES, create_tier2_agents, Tier2Agent


class TestNewArchetypes:
    def test_dividend_holder_exists(self):
        assert "dividend_holder" in ARCHETYPES

    def test_dividend_holder_config(self):
        cfg = ARCHETYPES["dividend_holder"]
        assert cfg.decision_latency == 2
        assert cfg.sentiment_bias_range == (0.1, 0.3)

    def test_sector_rotator_exists(self):
        assert "sector_rotator" in ARCHETYPES

    def test_sector_rotator_config(self):
        cfg = ARCHETYPES["sector_rotator"]
        assert cfg.decision_latency == 1
        assert cfg.sentiment_bias_range == (-0.1, 0.1)

    def test_all_five_archetypes_registered(self):
        expected = {"momentum_chaser", "panic_seller", "news_reactive", "dividend_holder", "sector_rotator"}
        assert set(ARCHETYPES.keys()) == expected

    def test_create_tier2_agents_dividend_holder(self):
        agents = create_tier2_agents("dividend_holder", 4)
        assert len(agents) == 4
        for a in agents:
            assert a.archetype_key == "dividend_holder"
            assert a.decision_latency == 2
            assert 100_000_000 <= a.working_memory.cash <= 500_000_000

    def test_create_tier2_agents_sector_rotator(self):
        agents = create_tier2_agents("sector_rotator", 4)
        assert len(agents) == 4
        for a in agents:
            assert a.archetype_key == "sector_rotator"
            assert a.decision_latency == 1
            assert 150_000_000 <= a.working_memory.cash <= 600_000_000


class TestDecisionLatency:
    @pytest.mark.asyncio
    async def test_latency_filters_recent_events(self):
        """Agent with decision_latency=2 only sees events where step - _step >= 2."""
        from unittest.mock import AsyncMock
        from imss.llm.router import LLMResponse

        agents = create_tier2_agents("dividend_holder", 1, seed=99)
        agent = agents[0]
        assert agent.decision_latency == 2

        mock_response = LLMResponse(
            content='{"action": "HOLD", "stock": "BBRI", "quantity": 0, "confidence": 0.5, "reasoning": "hold", "sentiment_update": 0.0}',
            parsed_json={"action": "HOLD", "stock": "BBRI", "quantity": 0, "confidence": 0.5, "reasoning": "hold", "sentiment_update": 0.0},
            input_tokens=50, output_tokens=25, model="glm-5", latency_ms=50,
        )
        mock_router = AsyncMock()
        mock_router.call = AsyncMock(return_value=mock_response)
        agent.set_router(mock_router)

        events = [
            {"title": "Old event", "category": "EARNINGS", "_step": 0, "sentiment_score": 0.5},
            {"title": "Recent event", "category": "NEWS", "_step": 2, "sentiment_score": -0.3},
        ]
        # At step 2: old event passes (2-0=2 >= 2), recent filtered (2-2=0 < 2)
        await agent.decide(
            {"symbol": "BBRI", "ohlcv": {"close": 5000}, "pct_change_1d": 0, "pct_change_5d": 0},
            events, step=2,
        )
        # Verify the LLM prompt only got the old event (events_brief should have "Old event")
        call_args = mock_router.call.call_args
        user_prompt = call_args.kwargs.get("user_prompt", call_args[1] if len(call_args) > 1 else "")
        assert "Old event" in user_prompt
        assert "Recent event" not in user_prompt

    @pytest.mark.asyncio
    async def test_zero_latency_passes_all_events(self):
        """Agent with decision_latency=0 receives all events (nothing filtered)."""
        from unittest.mock import AsyncMock
        from imss.llm.router import LLMResponse

        agents = create_tier2_agents("momentum_chaser", 1, seed=99)
        agent = agents[0]
        assert agent.decision_latency == 0

        mock_response = LLMResponse(
            content='{"action": "HOLD", "stock": "BBRI", "quantity": 0, "confidence": 0.5, "reasoning": "hold", "sentiment_update": 0.0}',
            parsed_json={"action": "HOLD", "stock": "BBRI", "quantity": 0, "confidence": 0.5, "reasoning": "hold", "sentiment_update": 0.0},
            input_tokens=50, output_tokens=25, model="glm-5", latency_ms=50,
        )
        mock_router = AsyncMock()
        mock_router.call = AsyncMock(return_value=mock_response)
        agent.set_router(mock_router)

        events = [
            {"title": "Event A", "category": "NEWS", "_step": 2, "sentiment_score": 0.1},
            {"title": "Event B", "category": "MACRO", "_step": 2, "sentiment_score": 0.2},
        ]
        await agent.decide(
            {"symbol": "BBRI", "ohlcv": {"close": 5000}, "pct_change_1d": 0, "pct_change_5d": 0},
            events, step=2,
        )
        call_args = mock_router.call.call_args
        user_prompt = call_args.kwargs.get("user_prompt", call_args[1] if len(call_args) > 1 else "")
        # Both events should be present since latency=0
        assert "Event A" in user_prompt
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/imss/test_agents/test_tier2_new.py -v`
Expected: FAIL — `dividend_holder` and `sector_rotator` not in ARCHETYPES

- [ ] **Step 3: Add new archetypes to ARCHETYPES dict**

In `imss/agents/tier2/archetypes.py`, add after the `"news_reactive"` entry (after line 57, before the closing `}`):

```python
    "dividend_holder": ArchetypeConfig(
        key="dividend_holder",
        name="Dividend Holder",
        one_liner="You buy high-dividend stocks and hold long term. You rarely sell unless dividends are cut.",
        risk_tolerance_range=(0.3, 0.5),
        sentiment_bias_range=(0.1, 0.3),
        decision_latency=2,
        initial_cash_range=(100_000_000, 500_000_000),
    ),
    "sector_rotator": ArchetypeConfig(
        key="sector_rotator",
        name="Sector Rotator",
        one_liner="You shift allocation based on economic cycle and macro data. Rate cuts mean buy banking, inflation means sell growth.",
        risk_tolerance_range=(0.4, 0.7),
        sentiment_bias_range=(-0.1, 0.1),
        decision_latency=1,
        initial_cash_range=(150_000_000, 600_000_000),
    ),
```

- [ ] **Step 4: Implement decision latency filter in Tier2Agent.decide()**

In `imss/agents/tier2/archetypes.py`, add latency filtering at the start of `decide()` method, right after line 82 (`market_state.get("symbol", "BBRI")`):

```python
        # Apply decision latency — filter out events too recent for this agent
        if self.decision_latency > 0:
            events = [e for e in events if step - e.get("_step", 0) >= self.decision_latency]
```

So lines 77-88 become:
```python
    async def decide(
        self,
        market_state: dict[str, Any],
        events: list[dict[str, Any]],
        step: int,
    ) -> AgentAction:
        # Apply decision latency BEFORE router check — filter too-recent events
        if self.decision_latency > 0:
            events = [e for e in events if step - e.get("_step", 0) >= self.decision_latency]

        if self._router is None:
            return default_hold_action(self.id, market_state.get("symbol", "BBRI"), step)

        stock = market_state.get("symbol", "BBRI")
```

- [ ] **Step 5: Update defaults in SimulationConfig**

In `imss/config.py`, change `tier1_personas` default (line 63) from:
```python
    tier1_personas: list[str] = ["pak_budi", "sarah", "andi"]
```
To:
```python
    tier1_personas: list[str] = ["pak_budi", "sarah", "andi", "dr_lim", "marketbot"]
```

And change `tier2_archetypes` (line 65-69) from:

```python
    tier2_archetypes: list[str] = [
        "momentum_chaser",
        "panic_seller",
        "news_reactive",
    ]
```

To:

```python
    tier2_archetypes: list[str] = [
        "momentum_chaser",
        "panic_seller",
        "news_reactive",
        "dividend_holder",
        "sector_rotator",
    ]
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `python3 -m pytest tests/imss/test_agents/test_tier2_new.py -v`
Expected: ALL 9 tests PASS

- [ ] **Step 7: Run all tests for regressions**

Run: `python3 -m pytest tests/imss/ -v`
Expected: ALL PASS

- [ ] **Step 8: Commit**

```bash
git add -f imss/agents/tier2/archetypes.py imss/config.py tests/imss/test_agents/test_tier2_new.py
git commit -m "feat(imss): add dividend_holder and sector_rotator archetypes with decision latency"
```

---

## Chunk 4: Loop Changes + CostTracker Snapshot + P&L Fix

### Task 6: Loop Changes (prev_aggregate_order_imbalance + _step tagging + fundamentals)

**Files:**
- Modify: `imss/simulation/loop.py:82-116` (add state carry-forward, fundamentals injection, _step tagging)

- [ ] **Step 1: Modify run_simulation_loop() signature and state**

In `imss/simulation/loop.py`, change the function signature (line 69) to accept `seed` and `fundamentals`:

```python
async def run_simulation_loop(
    agents: list[BaseAgent],
    market_data: MarketData,
    events_by_date: dict[date, list[dict[str, Any]]],
    trading_days: list[date],
    router: LLMRouter,
    batcher: LLMBatcher,
    on_step: Any = None,
    seed: int = 42,
    fundamentals: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
```

Change `rng = random.Random(42)` (line 82) to `rng = random.Random(seed)`.

Add `prev_aggregate_order_imbalance = 0.0` right after `step_logs: list[dict[str, Any]] = []` (line 83).

- [ ] **Step 2: Inject prev_aggregate_order_imbalance and fundamentals into market_state**

After building `market_state` dict (line 106-116), add two new keys:

```python
        market_state["prev_aggregate_order_imbalance"] = prev_aggregate_order_imbalance
        market_state["fundamentals"] = fundamentals
```

- [ ] **Step 3: Tag events with _step for all events (not just day_events)**

The current code at line 120-121 tags `day_events` with `_injection_step`. We need to also tag with `_step`. Change:

```python
        day_events = events_by_date.get(sim_date, [])
        for evt in day_events:
            evt["_injection_step"] = step
```

To:

```python
        day_events = events_by_date.get(sim_date, [])
        for evt in day_events:
            evt["_injection_step"] = step
            evt["_step"] = step
```

- [ ] **Step 4: Carry forward aggregate_order_imbalance**

After computing `order_imbalance` (line 154), add:

```python
        prev_aggregate_order_imbalance = order_imbalance
```

- [ ] **Step 5: Run all tests to verify no regressions**

Run: `python3 -m pytest tests/imss/ -v`
Expected: ALL PASS (the loop signature change adds optional params with defaults, so existing callers still work)

- [ ] **Step 6: Commit**

```bash
git add -f imss/simulation/loop.py
git commit -m "feat(imss): add loop state carry-forward, seed param, fundamentals injection"
```

---

### Task 7: CostTracker Snapshot + P&L Fix + run_number in SimulationResult

**Files:**
- Modify: `imss/llm/router.py:32-54` (add snapshot method)
- Modify: `imss/simulation/engine.py` (add snapshot, P&L, fundamentals loading, seed propagation, run_number)

- [ ] **Step 1: Add snapshot() to CostTracker**

In `imss/llm/router.py`, add method to `CostTracker` class after `json_parse_rate` property (after line 54):

```python
    def snapshot(self) -> dict:
        """Return current state for delta computation in multi-run."""
        return {
            "total_calls": self.total_calls,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "parse_successes": self.parse_successes,
            "parse_failures": self.parse_failures,
        }
```

- [ ] **Step 2: Add run_number to SimulationResult**

In `imss/simulation/engine.py`, add `run_number` field to `SimulationResult` (after line 48):

```python
class SimulationResult(BaseModel):
    simulation_id: str
    status: str
    total_steps: int
    agents_final: list[AgentSummary]
    total_llm_calls: int
    total_tokens_used: int
    estimated_cost_usd: float
    json_parse_success_rate: float
    step_count: int
    run_number: int = 0
```

- [ ] **Step 3: Add fundamentals loading and P&L tracking to run_single()**

In `imss/simulation/engine.py`, add `StockFundamentals` to the existing import (line 22 area):

```python
from imss.db.models import (
    AgentConfig,
    Base,
    Event,
    SimulationRun,
    SimulationStepLog,
    StockFundamentals,
    StockOHLCV,
)
```

Also add `timezone` to the datetime import at top of file:
```python
from datetime import date, datetime, timezone
```

After loading market data (after line 136, `market_data = MarketData(...)`), add fundamentals loading:

```python
        # 2b. Load fundamentals
        fundamentals = None
        async with session_factory() as session:
            stmt = (
                select(StockFundamentals)
                .where(StockFundamentals.symbol == config.target_stocks[0])
                .order_by(StockFundamentals.period.desc())
                .limit(1)
            )
            result = await session.execute(stmt)
            fund_row = result.scalars().first()
            if fund_row:
                fundamentals = {
                    "pe_ratio": fund_row.pe_ratio,
                    "pb_ratio": fund_row.pb_ratio,
                    "dividend_yield_pct": fund_row.dividend_yield_pct,
                    "roe_pct": fund_row.roe_pct,
                    "market_cap_trillion_idr": fund_row.market_cap_trillion_idr,
                }
```

- [ ] **Step 4: Add seed propagation to agent factories**

In `run_single()`, compute seed from run_number. After line 68 (`on_step: Callable | None = None,`), the method already has `run_number`. Add seed computation after the session_factory setup (after line 76):

```python
        seed = 42 + run_number
```

Change the Tier 2 agent creation (line 87) from:
```python
            t2_agents = create_tier2_agents(archetype, config.tier2_per_archetype)
```
To:
```python
            t2_agents = create_tier2_agents(archetype, config.tier2_per_archetype, seed=seed)
```

Change the Tier 3 agent creation (lines 92-95) from:
```python
        t3_agents = create_tier3_agents(
            total=config.tier3_total,
            distribution=config.tier3_distribution,
        )
```
To:
```python
        t3_agents = create_tier3_agents(
            total=config.tier3_total,
            distribution=config.tier3_distribution,
            seed=seed,
        )
```

- [ ] **Step 5: Pass seed and fundamentals to the simulation loop**

Change the `run_simulation_loop()` call (lines 190-198) from:
```python
        step_logs = await run_simulation_loop(
            agents=agents,
            market_data=market_data,
            events_by_date=events_by_date,
            trading_days=trading_days,
            router=self._router,
            batcher=self._batcher,
            on_step=on_step,
        )
```
To:
```python
        step_logs = await run_simulation_loop(
            agents=agents,
            market_data=market_data,
            events_by_date=events_by_date,
            trading_days=trading_days,
            router=self._router,
            batcher=self._batcher,
            on_step=on_step,
            seed=seed,
            fundamentals=fundamentals,
        )
```

- [ ] **Step 6: Add CostTracker snapshot for accurate per-run costs**

At the start of `run_single()` (right after the method opens, before DB engine creation), add:

```python
        seed = 42 + run_number
        cost_pre = self._router.cost_tracker.snapshot()
```

Also update the early-return FAILED path (around line 123-127) to use delta-based costs:

```python
        if not rows:
            logger.error("No price data found for %s", config.target_stocks[0])
            await engine.dispose()
            cost_post = self._router.cost_tracker.snapshot()
            return SimulationResult(
                simulation_id="", status="FAILED", total_steps=0,
                agents_final=[], total_llm_calls=cost_post["total_calls"] - cost_pre["total_calls"],
                total_tokens_used=0, estimated_cost_usd=0, json_parse_success_rate=0,
                step_count=0, run_number=run_number,
            )
```

In the finalize section (line 217), change from reading `cost_tracker` directly to computing deltas:

```python
        # 7. Finalize
        cost_post = self._router.cost_tracker.snapshot()
        run_llm_calls = cost_post["total_calls"] - cost_pre["total_calls"]
        run_input_tokens = cost_post["total_input_tokens"] - cost_pre["total_input_tokens"]
        run_output_tokens = cost_post["total_output_tokens"] - cost_pre["total_output_tokens"]
        run_tokens = run_input_tokens + run_output_tokens
        run_cost = (run_input_tokens * 0.001 + run_output_tokens * 0.002) / 1000
        run_parse_successes = cost_post["parse_successes"] - cost_pre["parse_successes"]
        run_parse_failures = cost_post["parse_failures"] - cost_pre["parse_failures"]
        run_parse_total = run_parse_successes + run_parse_failures
        run_parse_rate = run_parse_successes / run_parse_total if run_parse_total > 0 else 1.0
```

- [ ] **Step 7: Add P&L calculation**

Record initial portfolio values right after agent initialization (after line 96, `agents.extend(t3_agents)`):

```python
        # Record initial portfolio values for P&L calculation
        # Use first trading day's close price (loaded later), so store initial state
        initial_agent_state: dict[str, dict] = {}
        for agent in agents:
            initial_agent_state[agent.id] = {
                "cash": agent.working_memory.cash,
                "holdings": dict(agent.working_memory.holdings),
            }
```

In the finalize section, replace the agent_summaries loop (lines 220-228) with:

```python
        final_prices = {market_data.symbol: float(df.iloc[-1]["Close"])}
        first_prices = {market_data.symbol: float(df.iloc[0]["Close"])}

        agent_summaries = []
        for agent in agents:
            final_value = agent.working_memory.compute_portfolio_value(final_prices)
            init_state = initial_agent_state[agent.id]
            initial_value = init_state["cash"] + sum(
                qty * first_prices.get(sym, 0) for sym, qty in init_state["holdings"].items()
            )
            pnl_pct = ((final_value - initial_value) / initial_value * 100) if initial_value > 0 else 0.0
            agent_summaries.append(AgentSummary(
                id=agent.id, tier=agent.tier, persona_type=agent.persona_type,
                final_cash=agent.working_memory.cash,
                holdings=dict(agent.working_memory.holdings),
                pnl_pct=round(pnl_pct, 4),
            ))
```

- [ ] **Step 8: Update SimulationRun DB write and return statement to use deltas**

Replace the DB update (lines 230-242) with:

```python
        async with session_factory() as session:
            async with session.begin():
                await session.execute(
                    update(SimulationRun)
                    .where(SimulationRun.id == sim_id)
                    .values(
                        status="COMPLETED",
                        completed_at=datetime.now(tz=timezone.utc),
                        total_llm_calls=run_llm_calls,
                        total_tokens_used=run_tokens,
                        estimated_cost_usd=run_cost,
                    )
                )
```

Replace the return statement with:

```python
        await engine.dispose()

        return SimulationResult(
            simulation_id=sim_id,
            status="COMPLETED",
            total_steps=len(trading_days),
            agents_final=agent_summaries,
            total_llm_calls=run_llm_calls,
            total_tokens_used=run_tokens,
            estimated_cost_usd=run_cost,
            json_parse_success_rate=run_parse_rate,
            step_count=len(step_logs),
            run_number=run_number,
        )
```

- [ ] **Step 9: Run all tests**

Run: `python3 -m pytest tests/imss/ -v`
Expected: ALL PASS

- [ ] **Step 10: Commit**

```bash
git add -f imss/llm/router.py imss/simulation/engine.py
git commit -m "feat(imss): add CostTracker snapshots, P&L calculation, seed propagation, fundamentals loading"
```

---

## Chunk 5: Aggregator + Multi-Run + CLI + Final Tests

### Task 8: Aggregator Module

**Files:**
- Create: `imss/simulation/aggregator.py`
- Test: `tests/imss/test_simulation/test_aggregator.py` (create)

- [ ] **Step 1: Write the failing tests**

Create `tests/imss/test_simulation/test_aggregator.py`:

```python
"""Tests for multi-run aggregation."""

import pytest
from imss.simulation.aggregator import aggregate_runs, MultiRunResult, AgentRunStats
from imss.simulation.engine import SimulationResult, AgentSummary


def _make_result(run_number: int, pnl_offset: float = 0.0) -> SimulationResult:
    """Create a mock SimulationResult for testing."""
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

    def test_empty_results_raises(self):
        with pytest.raises(ValueError):
            aggregate_runs([])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/imss/test_simulation/test_aggregator.py -v`
Expected: FAIL — module does not exist

- [ ] **Step 3: Create aggregator module**

Create `imss/simulation/aggregator.py`:

```python
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

    # Compute per-type stats
    agent_stats: dict[str, AgentRunStats] = {}
    for persona_type, agent_data in agents_by_type.items():
        cash_values = [d["final_cash"] for d in agent_data]
        pnl_values = [d["pnl_pct"] for d in agent_data]
        n = len(agent_data)
        agent_stats[persona_type] = AgentRunStats(
            persona_type=persona_type,
            num_samples=n,
            mean_final_cash=float(np.mean(cash_values)),
            std_final_cash=float(np.std(cash_values)) if n > 1 else 0.0,
            mean_pnl_pct=float(np.mean(pnl_values)),
            std_pnl_pct=float(np.std(pnl_values)) if n > 1 else 0.0,
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/imss/test_simulation/test_aggregator.py -v`
Expected: ALL 5 PASS

- [ ] **Step 5: Commit**

```bash
git add -f imss/simulation/aggregator.py tests/imss/test_simulation/test_aggregator.py
git commit -m "feat(imss): add multi-run aggregation module"
```

---

### Task 9: run_multi() in SimulationEngine

**Files:**
- Modify: `imss/simulation/engine.py` (add run_multi method)
- Test: `tests/imss/test_simulation/test_multi_run.py` (create)

- [ ] **Step 1: Write the failing test**

Create `tests/imss/test_simulation/test_multi_run.py`:

```python
"""Integration test for multi-run simulation."""

import pytest
from unittest.mock import AsyncMock, patch
from datetime import datetime

from imss.config import SimulationConfig
from imss.simulation.engine import SimulationEngine
from imss.simulation.aggregator import MultiRunResult
from imss.llm.router import LLMResponse


MOCK_LLM_RESPONSE = LLMResponse(
    content='{"action": "HOLD", "stock": "BBRI", "quantity": 0, "confidence": 0.5, "reasoning": "test hold", "sentiment_update": 0.0}',
    parsed_json={"action": "HOLD", "stock": "BBRI", "quantity": 0, "confidence": 0.5, "reasoning": "test hold", "sentiment_update": 0.0},
    input_tokens=100, output_tokens=50, model="glm-5", latency_ms=100,
)


@pytest.mark.asyncio
async def test_run_multi_two_runs(tmp_path):
    """run_multi completes 2 runs and returns aggregated result."""
    db_url = f"sqlite+aiosqlite:///{tmp_path}/test.db"

    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
    from imss.db.models import Base, StockOHLCV

    engine_db = create_async_engine(db_url)
    async with engine_db.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    sf = async_sessionmaker(engine_db, expire_on_commit=False)
    async with sf() as session:
        async with session.begin():
            for d, c in [
                ("2024-07-01", 5150), ("2024-07-02", 5175), ("2024-07-03", 5200),
                ("2024-07-04", 5225), ("2024-07-05", 5250),
            ]:
                session.add(StockOHLCV(
                    symbol="BBRI", timestamp=datetime.fromisoformat(d),
                    open=c - 25, high=c + 25, low=c - 50, close=c,
                    volume=85_000_000, adjusted_close=c,
                ))
    await engine_db.dispose()

    config = SimulationConfig(
        target_stocks=["BBRI"],
        backtest_start="2024-07-01",
        backtest_end="2024-07-05",
        tier1_personas=["pak_budi"],
        tier2_per_archetype=0,
        tier2_archetypes=[],
        tier3_total=5,
        num_parallel_runs=2,
    )

    with patch.dict("os.environ", {"GLM_API_KEY": "test", "IMSS_DATABASE_URL": db_url}):
        sim_engine = SimulationEngine()
        sim_engine._router.call = AsyncMock(return_value=MOCK_LLM_RESPONSE)

        result = await sim_engine.run_multi(config)

    assert isinstance(result, MultiRunResult)
    assert result.num_runs == 2
    assert len(result.individual_results) == 2
    assert result.individual_results[0].run_number == 0
    assert result.individual_results[1].run_number == 1
    assert result.total_batch_cost_usd >= 0
    assert "tier1_pak_budi" in result.agent_stats


@pytest.mark.asyncio
async def test_different_seeds_produce_different_tier3(tmp_path):
    """Different run numbers produce different Tier 3 random agent behavior."""
    db_url = f"sqlite+aiosqlite:///{tmp_path}/test.db"

    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
    from imss.db.models import Base, StockOHLCV

    engine_db = create_async_engine(db_url)
    async with engine_db.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    sf = async_sessionmaker(engine_db, expire_on_commit=False)
    async with sf() as session:
        async with session.begin():
            for d, c in [
                ("2024-07-01", 5150), ("2024-07-02", 5175), ("2024-07-03", 5200),
            ]:
                session.add(StockOHLCV(
                    symbol="BBRI", timestamp=datetime.fromisoformat(d),
                    open=c - 25, high=c + 25, low=c - 50, close=c,
                    volume=85_000_000, adjusted_close=c,
                ))
    await engine_db.dispose()

    config = SimulationConfig(
        target_stocks=["BBRI"],
        backtest_start="2024-07-01",
        backtest_end="2024-07-03",
        tier1_personas=[],
        tier2_per_archetype=0,
        tier2_archetypes=[],
        tier3_total=10,
        num_parallel_runs=2,
    )

    with patch.dict("os.environ", {"GLM_API_KEY": "test", "IMSS_DATABASE_URL": db_url}):
        sim_engine = SimulationEngine()

        result = await sim_engine.run_multi(config)

    # Different seeds cause different initial cash for Tier 3 agents (via create_tier3_agents seed param).
    # This cascades into different BUY quantities (qty = cash * pct / close).
    # Note: RandomWalkAgent._get_rng() uses hash(self.id) not the run seed, so the
    # difference comes from initial cash, not from action randomness.
    r0_cash = {a.id: a.final_cash for a in result.individual_results[0].agents_final}
    r1_cash = {a.id: a.final_cash for a in result.individual_results[1].agents_final}
    differences = sum(1 for aid in r0_cash if abs(r0_cash.get(aid, 0) - r1_cash.get(aid, 0)) > 1.0)
    assert differences > 0, "Different seeds should produce different Tier 3 initial cash → different outcomes"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/imss/test_simulation/test_multi_run.py -v`
Expected: FAIL — `run_multi` does not exist

- [ ] **Step 3: Add run_multi() to SimulationEngine**

In `imss/simulation/engine.py`, add import for the aggregator at the top:

```python
from imss.simulation.aggregator import MultiRunResult, aggregate_runs
```

Add the `run_multi()` method to `SimulationEngine` class (after `run_single()`):

```python
    async def run_multi(
        self,
        config: SimulationConfig,
        on_step: Callable | None = None,
    ) -> MultiRunResult:
        """Execute N sequential simulation runs and aggregate results."""
        results = []
        for i in range(config.num_parallel_runs):
            result = await self.run_single(config, run_number=i, on_step=on_step)
            results.append(result)
        return aggregate_runs(results)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/imss/test_simulation/test_multi_run.py -v`
Expected: ALL 2 PASS

- [ ] **Step 5: Run full test suite**

Run: `python3 -m pytest tests/imss/ -v`
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
git add -f imss/simulation/engine.py tests/imss/test_simulation/test_multi_run.py
git commit -m "feat(imss): add run_multi() for multi-run parallel execution"
```

---

### Task 10: CLI Updates + Smoke Test

**Files:**
- Modify: `scripts/imss_run_backtest.py` (add --runs flag)
- Modify: `scripts/imss_smoke_test.py` (add multi-run test)

- [ ] **Step 1: Update CLI runner**

In `scripts/imss_run_backtest.py`, add `--runs` argument (after line 41):

```python
    parser.add_argument("--runs", type=int, default=1, help="Number of simulation runs")
```

Change the config creation (line 44) to include `num_parallel_runs`:

```python
    config = SimulationConfig(
        target_stocks=[args.stock],
        backtest_start=args.start,
        backtest_end=args.end,
        tier2_per_archetype=args.tier2_count,
        tier3_total=args.tier3_count,
        num_parallel_runs=args.runs,
    )
```

Replace the engine execution block (lines 56-74) with:

```python
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
        table.add_column("Runs")
        table.add_column("Mean Cash", justify="right")
        table.add_column("Std Cash", justify="right")
        table.add_column("Mean P&L%", justify="right")
        table.add_column("Std P&L%", justify="right")
        for persona_type, stats in sorted(multi_result.agent_stats.items()):
            table.add_row(
                persona_type, str(stats.num_runs),
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
```

- [ ] **Step 2: Update smoke test**

In `scripts/imss_smoke_test.py`, add a multi-run test case. Add after the existing `smoke_test()` function (after line 65):

```python
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
```

Update the `__main__` block:

```python
if __name__ == "__main__":
    ok = asyncio.run(smoke_test())
    if ok:
        ok = asyncio.run(smoke_test_multi_run())
    sys.exit(0 if ok else 1)
```

- [ ] **Step 3: Run full test suite**

Run: `python3 -m pytest tests/imss/ -v`
Expected: ALL PASS

- [ ] **Step 4: Commit**

```bash
git add -f scripts/imss_run_backtest.py scripts/imss_smoke_test.py
git commit -m "feat(imss): add --runs CLI flag and multi-run smoke test"
```

---

### Task 11: Final Verification

- [ ] **Step 1: Run complete test suite**

Run: `python3 -m pytest tests/imss/ -v --tb=short`
Expected: ALL tests pass (29 existing + ~24 new ≈ 53 total)

- [ ] **Step 2: Verify no import errors**

Run: `python3 -c "from imss.simulation.aggregator import MultiRunResult, aggregate_runs; from imss.agents.tier1.marketbot import MarketBotAgent; from imss.agents.tier1.personas import PERSONAS; print(f'Personas: {list(PERSONAS.keys())}'); print('All imports OK')"`
Expected: `Personas: ['pak_budi', 'sarah', 'andi', 'dr_lim', 'marketbot']` and `All imports OK`

- [ ] **Step 3: Verify archetype count**

Run: `python3 -c "from imss.agents.tier2.archetypes import ARCHETYPES; print(f'Archetypes: {list(ARCHETYPES.keys())}'); assert len(ARCHETYPES) == 5"`
Expected: `Archetypes: ['momentum_chaser', 'panic_seller', 'news_reactive', 'dividend_holder', 'sector_rotator']`

- [ ] **Step 4: Final commit (if any uncommitted changes)**

```bash
git status
# Only commit if there are changes
```
