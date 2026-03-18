# Repo Map

## Top-Level Structure

- `api/`: FastAPI application, routes, schemas, cache, dependency wiring
- `dashboard/`: Streamlit app, pages, shared UI components, API client helpers
- `core/`: core trading logic, technical analysis, data fetchers, ML, risk, portfolio
- `backtest/`: backtesting engines and walk-forward logic
- `imss/`: IDX Market Swarm Simulator — self-contained multi-agent LLM simulation engine (3-tier agents, async SQLAlchemy, Anthropic SDK)
- `research/`: offline research tooling, Monte Carlo analysis, and autoresearch experiments
- `scripts/`: ingestion, enrichment, training, setup, and maintenance jobs
- `config/`: settings, constants, logging, trading modes
- `data/`: SQLite DB, artifacts, job history, cached market/fundamental data
- `tests/`: Python unit, integration, and dashboard tests
- `e2e/`: Playwright configuration, fixtures, helpers, browser tests
- `docs/`: product notes, plans, and this context set

## Main Entrypoints

- `api/main.py`: FastAPI app creation, CORS, exception handler, router registration
- `dashboard/app.py`: Streamlit landing page and home/dashboard summary
- `dashboard/pages/*.py`: Streamlit multipage feature modules
- `scripts/setup_database.py`: local DB bootstrapping path
- `e2e/playwright.config.ts`: browser test runner config
- `scripts/autoresearch.py`: CLI entrypoint for autonomous strategy-research workspaces
- `scripts/imss_run_backtest.py`: CLI entrypoint for IMSS simulation runs
- `scripts/imss_smoke_test.py`: IMSS end-to-end validation (single + multi-run)

## Important Configuration

- `config/settings.py`: central settings object used by API and dashboard
- `.env.example`: environment variable template
- `requirements.txt`: Python dependencies
- `pyproject.toml`: formatter, linter, mypy, pytest, coverage config
- `docker/docker-compose.yml`: containerized local stack definition

## Current Large Modules

These files are the main complexity hotspots and should be read carefully before edits:

- `core/data/database.py` (~2175 lines): SQLAlchemy models and DB manager
- `api/routes/prediction.py` (~1110 lines): ML-related API surface
- `dashboard/pages/07_ml_prediction.py` (~912 lines): largest current dashboard page
- `dashboard/pages/05_settings.py` (~808 lines): settings and model-ops page
- `dashboard/components/nextgen_styles.py` (~1335 lines): shared styling and theme helpers
- `core/execution/paper_trader.py` (~641 lines): simulation execution logic

## Recently Reduced UI Files

The UI separation refactor already shrank these pages:

- `dashboard/pages/02_stock_detail.py`: now delegates API access and the flow tab
- `dashboard/pages/06_market_overview.py`: now delegates normalization and rendering helpers
- `dashboard/pages/08_backtesting.py`: now delegates formatting/rendering helpers

Supporting extracted modules:

- `dashboard/services/api_client.py`
- `dashboard/pages/stock_detail_api.py`
- `dashboard/pages/stock_detail_sections.py`
- `dashboard/pages/market_overview_helpers.py`
- `dashboard/pages/backtesting_helpers.py`

## Feature-Oriented Navigation

- Screening: `dashboard/pages/01_screener.py`, `api/routes/signals.py`, `core/signals/`
- Stock detail: `dashboard/pages/02_stock_detail.py`, `api/routes/stocks.py`, `api/routes/analysis.py`
- Sentiment: `dashboard/pages/03_sentiment.py`, `api/routes/sentiment.py`
- Virtual trading: `dashboard/pages/04_virtual_trading.py`, `api/routes/simulation.py`, `core/execution/paper_trader.py`
- Settings and model ops: `dashboard/pages/05_settings.py`, `api/routes/prediction.py`
- Market overview: `dashboard/pages/06_market_overview.py`, `api/routes/stocks.py`, `api/routes/market_enrichment.py`
- ML prediction: `dashboard/pages/07_ml_prediction.py`, `api/routes/prediction.py`, `core/ml_prediction/`
- Backtesting: `dashboard/pages/08_backtesting.py`, `api/routes/backtest.py`, `backtest/`
- Autoresearch: `research/autoresearch/`, `scripts/autoresearch.py`, `docs/project_context/07_autoresearch.md`
- IMSS simulation: `imss/`, `scripts/imss_*.py`, `tests/imss/`, `docs/project_context/10_imss.md`
