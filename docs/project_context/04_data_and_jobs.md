# Data And Jobs Context

## Storage Model

The system is optimized for daily refreshes, not intraday streaming.

Main storage locations:

- `data/trading.db`: primary SQLite database (main trading system)
- `data/imss.db`: IMSS simulation database (separate, async SQLAlchemy)
- `data/seed_events/`: IMSS seed data (BBRI prices, events, fundamentals JSON)
- `data/ml_artifacts/`: saved prediction model artifacts
- `data/training_jobs/`: training status and history JSON files
- `data/market/`, `data/fundamental/`, `data/backtest/`: supporting generated data
- `logs/` and `output/`: run artifacts and diagnostics

## Database Bootstrap

Primary setup flow:

- `scripts/setup_database.py`
- `api/main.py` startup via `DatabaseManager(...).create_tables()`

The DB manager in `core/data/database.py` is broad: schema definitions, query helpers, and table lifecycle logic all live there.

## Daily Data Workflow

Typical daily workflow:

1. ingest or refresh stock price/history data
2. enrich missing metadata
3. sync market-enrichment summary tables
4. sync broker/flow snapshots
5. run optional model training
6. serve cached API reads to dashboard pages

Useful scripts:

- `scripts/ingest_all_stocks.py`
- `scripts/fetch_historical_data.py`
- `scripts/enrich_missing_metadata.py`
- `scripts/populate_metadata.py`
- `scripts/populate_all_yq.py`
- `scripts/populate_from_tradingview.py`
- `scripts/sync_idx_enrichment.py`
- `scripts/daily_scan.py`

## Broker And Flow Ingestion

Current broker/flow pipelines include:

- IQPlus:
  - `scripts/fetch_iqplus_broker_summary.py`
  - `scripts/sync_iqplus_stock_broker_daily.py`
  - `core/data/iqplus_broker_scraper.py`

- Flow Klinik:
  - `scripts/sync_flow_klinik_latest_snapshot.py`

- Invezgo:
  - `scripts/sync_invezgo_latest_snapshot.py`
  - `scripts/sync_invezgo_stock_broker_detail_daily.py`
  - browser helpers in `scripts/*.cjs`

Operational note:

- Invezgo ingestion depends on an authenticated browser session and is not a simple anonymous HTTP fetch.

## Training Workflow

Training is offline and artifact-based.

Key scripts:

- `scripts/train_models.py`
- `scripts/run_training_in_background.sh`

Artifacts and job tracking:

- `data/ml_artifacts/{SYMBOL}_ensemble.pkl`
- `data/ml_artifacts/{SYMBOL}_ensemble.meta.json`
- `data/training_jobs/model_training_status.json`
- `data/training_jobs/model_training_history.json`

## Freshness And Health

Health/freshness UI depends on:

- `GET /health/data`
- `GET /health/update-status`
- `POST /health/update-data`
- `GET /health/dashboard-summary`

If dashboard freshness metrics look wrong, inspect both:

- route logic in `api/routes/health.py`
- underlying DB timestamps and ingestion jobs

## Data Edit Guidance

- Be careful with scripts that write snapshots or training artifacts; their outputs are consumed by both API routes and tests.
- If adding a new market or flow dataset, document the storage table, sync script, and API surface together.
- Keep raw fetch/auth details in scripts or core data modules, not in Streamlit pages.
- IMSS data (`data/imss.db`, `data/seed_events/`) is independent of `data/trading.db`. See `10_imss.md` for IMSS-specific data flows.
