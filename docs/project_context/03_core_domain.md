# Core Domain Context

## Core Module Responsibilities

## Data access and persistence

- `core/data/database.py`: SQLAlchemy models plus `DatabaseManager`
- `core/data/scraper.py`: primary market data scraping abstractions
- `core/data/foreign_flow.py`: foreign flow fetch and analysis
- `core/data/iqplus_broker_scraper.py`: IQPlus broker summary ingestion
- `core/data/broker_flow.py`: broker-flow scoring primitives
- `core/data/idx_enrichment.py`: IDX enrichment fetch logic
- `core/data/availability_guard.py`: data quality and availability validation
- `core/data/cache.py`: lightweight cache helpers

Important persistence entities in `database.py` include:

- `PriceHistory`
- `StockMetadata`
- `SentimentRecord`, `SentimentDaily`, `SentimentSector`
- `SimulationSession`, `SimulationTrade`
- `CommodityHistory`, `StockCommodityCorrelation`
- market summary tables for foreign, domestic, and industries
- broker and flow tables for IQPlus, Flow Klinik, and Invezgo snapshots

## Analysis

- `core/analysis/technical.py`: technical indicators and score generation
- `core/analysis/macro_correlation.py`: macro/commodity relationship analysis

## Signals

- `core/signals/signal_generator.py`: main signal generation engine and composite scoring
- `core/signals/forecast_enhanced_generator.py`: forecast-aware signal variant

These modules are central to screener behavior and should be treated as contract-sensitive for ranking/scoring changes.

## Risk

- `core/risk/risk_manager.py`: validation and risk checks
- `core/risk/position_sizer.py`: sizing calculations
- `core/risk/empirical_kelly.py`: empirical Kelly sizing
- `core/risk/pattern_matcher.py`: pattern recognition and historical analogs
- `core/risk/forecast_enhanced_risk.py`: forecast-aware risk extensions

## Portfolio and execution

- `core/portfolio/portfolio_manager.py`: portfolio state and trade history logic
- `core/execution/paper_trader.py`: virtual trading engine, orders, fills, equity, and replay behavior

`paper_trader.py` is a major behavior hotspot because it underpins both simulation APIs and dashboard virtual trading flows.

## ML prediction pipeline

- `core/ml_prediction/features.py`: feature engineering
- `core/ml_prediction/models.py`: model definitions and builders
- `core/ml_prediction/trainer.py`: artifact training and persistence
- `core/ml_prediction/predictor.py`: inference and walk-forward helpers
- `core/ml_prediction/service.py`: higher-level prediction service used by API
- `core/ml_prediction/utils.py`: support utilities

Important product constraints:

- model training is offline/batch-oriented
- inference depends on existing artifacts in `data/ml_artifacts`
- refresh means retraining/replacing artifacts, not online incremental learning

## Backtest Domain

The `backtest/` package complements `core/` and powers backtest API behavior:

- `backtest/engine.py`
- `backtest/simulator.py`
- `backtest/walk_forward.py`
- `backtest/monte_carlo_backtest.py`

If a backtest result shape changes, check:

- `api/routes/backtest.py`
- `dashboard/pages/08_backtesting.py`
- `dashboard/pages/backtesting_helpers.py`
- `tests/integration/test_backtest_api.py`
- `tests/dashboard/test_backtesting_page.py`

## Research Tooling

The standalone strategy-iteration workflow lives under `research/autoresearch/`.
It is adjacent to the main backtest domain, but intentionally isolated from API
and dashboard contracts.

Key files:

- `research/autoresearch/evaluator.py`: fixed evaluator for research-only experiments
- `research/autoresearch/runner.py`: LLM-guided strategy iteration loop
- `research/autoresearch/workspace.py`: scaffold and snapshot management

This feature uses Yahoo Finance daily data instead of the main app database, so
changes there should not be treated as production market-data changes unless the
integration boundary is explicitly widened.

## IMSS (Separate Simulation Domain)

The `imss/` package is a self-contained multi-agent market simulation engine that does NOT share code with `core/` or `backtest/`. It has its own database (`data/imss.db`), its own LLM client (Anthropic SDK → GLM-5), and its own agent/simulation architecture.

If the task involves IMSS agents, simulation runs, or LLM-driven trading decisions, see `docs/project_context/10_imss.md` instead of this file.

## Core Edit Guidance

- Keep domain math and business rules in `core/` or `backtest/`, not in API routes or Streamlit pages.
- If UI code starts transforming financial calculations or risk formulas, that logic likely belongs in `core/`.
- For bug fixes, identify whether the issue is a transport bug, response-shape bug, or domain-calculation bug before patching.
- Do not mix `imss/` imports into `core/` or vice versa — they are intentionally isolated.
