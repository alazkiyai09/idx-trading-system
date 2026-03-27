# IDX Trading System

AI-assisted trading and research workspace for Indonesia Stock Exchange (IDX) equities, built with FastAPI + Streamlit and optimized for daily operations.

This repository now includes expanded production-style modules: asynchronous backtesting jobs, market enrichment analytics, persisted portfolio reconciliation, research preset promotion, and IMSS (IDX Market Swarm Simulator) run orchestration.

## Highlights

- Daily-data trading system with SQLite-backed market storage (`data/trading.db`)
- FastAPI backend with route groups for stocks, signals, sentiment, prediction, simulation, backtest, market enrichment, and IMSS
- Streamlit multipage dashboard with 12 feature pages and shared service layer
- Signal generation, technical/risk analysis, and paper-trading simulation
- Offline ML model training, artifact management, and prediction APIs
- Research workspaces (`autoscreener`, `automontecarlo`, `autoresearch`) with preset promotion flow
- IMSS multi-agent simulation package with background job lifecycle APIs and dashboard board/command builder
- Python unit/integration/dashboard tests plus Playwright E2E coverage

## Product Modules

### Dashboard pages

- `Home` (`dashboard/app.py`): health, freshness, and operator overview
- `01_screener.py`: filter-driven stock screening and signal scan workflows
- `02_stock_detail.py`: chart + technical + sentiment + flow + risk views
- `03_sentiment.py`: sentiment fetch, cleanup, and thematic views
- `04_virtual_trading.py`: simulation sessions and order/replay controls
- `05_settings.py`: model operations, training readiness, and artifact management
- `06_market_overview.py`: market breadth, leaders, and flow screener surfaces
- `07_ml_prediction.py`: prediction, correlation, Monte Carlo, and technical overlays
- `08_backtesting.py`: async backtest launch, polling, and results
- `09_research_presets.py`: promote accepted research candidates into durable presets
- `10_portfolio.py`: positions, trade history filters, and reconciliation analytics
- `11_market_enrichment.py`: market -> sector -> symbol enrichment workflow
- `12_imss.py`: background IMSS run launch, monitoring, logs, and summary inspection

### API route groups

- `/health`: system health, freshness, and manual update hooks
- `/stocks`: symbols, snapshot data, charts, foreign flow, broker datasets
- `/analysis`, `/fundamental`: technical, signal, risk, and LLM-assisted analysis
- `/signals`: scan and active signal feeds
- `/portfolio`: summary, positions, and trade history feeds
- `/simulation`: paper-trading session lifecycle and metrics
- `/prediction`: training readiness, artifact management, inference, correlation, Monte Carlo
- `/backtest`: synchronous runs and background backtest jobs
- `/market-enrichment`: summary, foreign/domestic, industries, brokers, symbol diagnostics
- `/imss`: background IMSS run lifecycle (`create`, `list`, `status`, `logs`, `summary`)

## Architecture

```text
Daily Jobs + SQLite (trading.db, imss.db)
        |                      |
        v                      v
   FastAPI Route Layer   IMSS Simulation Engine
        |                      |
        +----------> Streamlit Dashboard <----------+
                         (12 pages)
```

Core directories:

- `api/`: FastAPI app, routes, schemas, cache
- `dashboard/`: Streamlit pages, components, API client services
- `core/`: trading domain logic (analysis, signals, risk, execution, ML)
- `backtest/`: backtest engines and walk-forward logic
- `imss/`: self-contained multi-agent market simulator
- `research/`: autoresearch, autoscreener, automontecarlo tooling
- `scripts/`: ingestion, enrichment, training, and simulation job scripts
- `tests/`, `e2e/`: Python + Playwright test coverage
- `docs/project_context/`: maintainable context map and subsystem handoffs

## Quick Start

### 1. Install

```bash
git clone https://github.com/alazkiyai09/idx-trading-system.git
cd idx-trading-system

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Optional (for full TensorFlow-backed ensembles):

```bash
python3 -m pip install tensorflow-cpu
```

### 2. Configure environment

Create `.env` (or export env vars directly). Common runtime variables:

```bash
API_URL=http://127.0.0.1:8000
ENABLE_REAL_PREDICTION=true

# IMSS / GLM
GLM_API_KEY=your-key
IMSS_GLM_BASE_URL=https://api.z.ai/api/anthropic
IMSS_GLM_MODEL=glm-5
```

### 3. Run API and dashboard

```bash
ENABLE_REAL_PREDICTION=true python3 -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

```bash
API_URL=http://127.0.0.1:8000 streamlit run dashboard/app.py --server.port 8501
```

Open:

- Dashboard: `http://localhost:8501`
- API: `http://localhost:8000`
- OpenAPI: `http://localhost:8000/docs`

## Data and Enrichment Workflow

The platform is designed for daily refresh cycles (not intraday streaming).

Typical operator sequence:

1. Ingest/update prices and metadata
2. Refresh enrichment datasets (IDX, IQPlus, Flow Klinik, Invezgo)
3. Run optional ML retraining
4. Operate dashboard/API with cached read models

Useful scripts:

```bash
python3 scripts/ingest_all_stocks.py
python3 scripts/enrich_missing_metadata.py
python3 scripts/sync_idx_enrichment.py
python3 scripts/sync_flow_klinik_latest_snapshot.py --symbols ADRO,BBRI --snapshot-date 2026-03-11
python3 scripts/sync_invezgo_latest_snapshot.py --symbols ADRO,BRMS --snapshot-date 2026-03-10
python3 scripts/sync_iqplus_stock_broker_daily.py --symbols BBCA,BBRI
```

## ML and Research Workflows

### ML model lifecycle

- Train offline and store artifacts in `data/ml_artifacts/`
- Track jobs in `data/training_jobs/`
- Manage artifacts from `Settings -> Model Ops` or API

Example:

```bash
bash scripts/run_training_in_background.sh ADRO
LOOKBACK_DAYS=400 OVERWRITE=true bash scripts/run_training_in_background.sh ADRO BBCA TLKM
```

### Research presets and autonomous iteration

- `09_research_presets.py` promotes accepted candidates from:
  - `data/autoscreener*`
  - `data/automontecarlo*`
- `research/autoresearch/` supports LLM-driven strategy iteration

Example:

```bash
python3 scripts/autoresearch.py init --workspace data/autoresearch_glm --agent-name codex --provider-name glm --model-name glm-5
python3 scripts/autoresearch.py run --workspace data/autoresearch_glm --agent-name codex --provider glm --model glm-5 --data-backend sqlite --database-url sqlite:///data/trading.db --max-experiments 10
```

## IMSS (IDX Market Swarm Simulator)

IMSS is a separate simulation domain under `imss/` with its own database (`data/imss.db`) and background job orchestration.

Current capabilities:

- Three-tier agent architecture (Tier 1 personas, Tier 2 archetypes, Tier 3 heuristics)
- Single-run and multi-run simulation execution
- Background run lifecycle APIs under `/imss/runs*`
- Dashboard operator page (`12_imss.py`) for launch, board, logs, and summary

Common commands:

```bash
python3 scripts/imss_seed_data.py
python3 scripts/imss_run_backtest.py --stock BBRI --start 2024-07-01 --end 2024-09-30 --runs 2
python3 scripts/imss_smoke_test.py
```

## Testing

Python:

```bash
pytest -q
```

Focused suites:

```bash
pytest -q tests/integration/test_prediction_api.py
pytest -q tests/integration/test_backtest_api.py
pytest -q tests/integration/test_autoresearch_glm.py
pytest -q tests/imss/
```

Playwright:

```bash
cd e2e
npx playwright test
```

## Notes

- Virtual trading is paper-trading only (no broker-connected live execution)
- Market data assumptions are daily-cycle oriented
- ML inference depends on trained artifacts per symbol
- Model refresh is retrain-and-replace, not incremental online learning

## License

Private repository. All rights reserved.
