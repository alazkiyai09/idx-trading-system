# IDX Trading System

Trading and research workspace for Indonesia Stock Exchange equities, with a FastAPI backend, Streamlit dashboard, daily data pipeline, paper trading, screening, sentiment, and offline ML model management.

## What Is In This Repo

- FastAPI service for stock data, health, signals, simulation, sentiment, portfolio, and ML prediction routes
- Streamlit dashboard with top navigation and inline page controls
- SQLite-backed daily market data workflow
- Paper trading and replay-style virtual trading
- Offline ML training pipeline with artifact publishing and model management
- Automated test coverage across unit, integration, dashboard, and Playwright E2E layers

## Current Product Shape

Main dashboard modules:

- `Home`: system status, data freshness, desk actions, manual daily refresh
- `Screener`: inline filters and signal scanning
- `Stock Detail`: price, technicals, sentiment, flow, and analysis views
- `Market Overview`: market breadth and summary views
- `Sentiment`: article-based sentiment analysis
- `Virtual Trading`: beta paper trading and replay workflow
- `ML Prediction & Analysis`: inference, comparison, Monte Carlo, and technical overlays
- `Settings -> Model Ops`: training readiness, batch training, artifact inventory, upload/delete, and background training commands

Important current behavior:

- Market data is treated as daily, not real-time
- ML inference only works for symbols that already have trained artifacts in `data/ml_artifacts`
- Training is offline and batch-oriented; the dashboard launches background jobs but does not do inline training on the prediction page
- Model refresh is currently a retrain-and-replace workflow, not true incremental online learning

## Architecture

```text
SQLite/Data Jobs -> FastAPI Read Models -> Streamlit Dashboard
                     |                  |
                     |                  -> Paper trading / replay / ML inference UI
                     -> Signals / sentiment / prediction / health APIs
```

Core areas:

- [`api/`](/mnt/data/Project/idx-trading-system/api): FastAPI entrypoint, routes, schemas, cache
- [`dashboard/`](/mnt/data/Project/idx-trading-system/dashboard): Streamlit app, pages, shared UI components
- [`core/`](/mnt/data/Project/idx-trading-system/core): analysis, execution, portfolio, ML pipeline, data logic
- [`scripts/`](/mnt/data/Project/idx-trading-system/scripts): ingestion, metadata enrichment, training helpers
- [`tests/`](/mnt/data/Project/idx-trading-system/tests): unit, integration, and dashboard tests
- [`e2e/`](/mnt/data/Project/idx-trading-system/e2e): Playwright browser and API tests

## Quick Start

### 1. Install

```bash
git clone https://github.com/alazkiyai09/idx-trading-system.git
cd idx-trading-system

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

If you want the full ML ensemble with `LSTM` and `CNN-LSTM`, install TensorFlow in the environment:

```bash
python3 -m pip install tensorflow-cpu
```

### 2. Run the API

```bash
ENABLE_REAL_PREDICTION=true python3 -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### 3. Run the dashboard

```bash
API_URL=http://127.0.0.1:8000 streamlit run dashboard/app.py --server.port 8501
```

Open:

- Dashboard: `http://localhost:8501`
- API: `http://localhost:8000`
- API docs: `http://localhost:8000/docs`

## Data Workflow

The app is optimized for daily updates rather than intraday streaming.

- Price history is stored in SQLite under `data/trading.db`
- The dashboard exposes a manual refresh action for daily data
- Expected cadence is after market close or by midnight Asia/Jakarta
- Expensive stock list reads are served through cached API read models instead of full-table recalculation on every page load

Useful scripts:

```bash
python3 scripts/ingest_all_stocks.py
python3 scripts/enrich_missing_metadata.py
```

## ML Model Workflow

### Readiness

Use `Settings -> Model Ops -> Training` to inspect:

- available history rows
- minimum rows required
- recommended rows
- whether a symbol is `not_ready`, `limited`, or `ready`

Current rule of thumb:

- below minimum history: training blocked
- minimum to recommended range: training allowed but lower trust
- above recommended range: normal training

### Train models

Single or batch training is offline and writes artifacts to `data/ml_artifacts`.

Dashboard path:

- `Settings -> Model Ops -> Training -> Batch Training`

Shell path:

```bash
bash scripts/run_training_in_background.sh ADRO
LOOKBACK_DAYS=400 OVERWRITE=true bash scripts/run_training_in_background.sh ADRO BBCA TLKM
```

Artifacts produced:

- `data/ml_artifacts/{SYMBOL}_ensemble.pkl`
- `data/ml_artifacts/{SYMBOL}_ensemble.meta.json`

Training job status is tracked in:

- `data/training_jobs/model_training_status.json`
- `data/training_jobs/model_training_history.json`

### Manage models

Use `Settings -> Model Ops -> Model Management` to:

- list stored models
- upload artifacts
- delete artifacts
- copy background training commands

### Refresh models

Yes, models can be updated with newer daily data.

Current implementation:

- rerun training with overwrite enabled
- retrain from scratch on the latest available history
- replace the old artifact

This is not incremental online learning yet.

## Testing

Python tests:

```bash
pytest -q
```

Targeted examples:

```bash
pytest -q tests/integration/test_prediction_api.py
pytest -q tests/dashboard/test_components.py
```

Playwright:

```bash
cd e2e
npx playwright test
```

## Notes

- `Virtual Trading` is a beta paper-trading workflow, not a broker-connected execution system
- Some advanced modules still depend on external data quality and artifact availability
- Streamlit page transitions can still show brief shell repaint behavior because this is a multipage Streamlit app, not a SPA frontend

## License

Private repository. All rights reserved.
