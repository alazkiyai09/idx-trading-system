# Testing And E2E Context

## Test Layers

## Unit tests

- Location: `tests/unit/`
- Focus:
  - indicators and technicals
  - signal generation
  - risk and sizing
  - database helpers
  - scrapers and sync jobs
  - paper trader and ML helpers

## Integration tests

- Location: `tests/integration/`
- Focus:
  - API endpoint behavior
  - prediction/backtest contracts
  - data refresh and fetch flows
  - cross-module behavior

## IMSS tests

- Location: `tests/imss/`
- Count: 58 tests
- Focus:
  - agent creation and behavior (Tier 1/2/3)
  - LLM router JSON parsing
  - simulation engine integration (mocked LLM)
  - multi-run aggregation and action rates
  - order resolution and lot enforcement
  - database table creation
- Run: `python3 -m pytest tests/imss/ -v`
- All tests mock LLM at the `router.call()` level — no live API needed
- Live smoke test: `GLM_API_KEY=... PYTHONPATH=. python3 scripts/imss_smoke_test.py`

## Dashboard tests

- Location: `tests/dashboard/`
- Focus:
  - Streamlit page structure
  - helper functions
  - UI behavior assumptions
  - some API-integrated dashboard flows

Important note:

- some dashboard tests are structure-oriented and inspect page source or explicit UI markers
- if code is extracted into helpers, these tests may need targeted updates rather than reverting the refactor

## Browser E2E

- Python-level E2E-style tests: `tests/e2e/`
- Playwright project: `e2e/`

Playwright files include:

- `e2e/playwright.config.ts`
- `e2e/tests/dashboard/*.spec.ts`
- `e2e/helpers/`
- `e2e/pages/`

## Runtime Expectations

Common local stack for live checks:

1. API on `127.0.0.1:8000`
2. Streamlit on `127.0.0.1:8501`

Example commands:

```bash
ENABLE_REAL_PREDICTION=true python3 -m uvicorn api.main:app --host 127.0.0.1 --port 8000
API_URL=http://127.0.0.1:8000 streamlit run dashboard/app.py --server.port 8501
```

Python tests:

```bash
pytest -q
pytest -q tests/dashboard/test_backtesting_page.py tests/dashboard/test_stock_detail.py
pytest -q tests/dashboard/test_market_overview.py
```

Playwright:

```bash
cd e2e
npx playwright test
```

## Current Practical Test Notes

- market-overview API-dependent dashboard tests should skip cleanly when no local backend is running
- E2E assertions around dashboard pages should reflect visible user-facing content, not hidden markdown/container internals
- contract-sensitive areas:
  - `/stocks`
  - stock detail page loading path
  - market overview leader summaries
  - simulation/backtest result structures
  - IMSS `SimulationResult` and `MultiRunResult` models (consumed by aggregator and CLI)

## Test Edit Guidance

- If you extract page code into helpers, keep page-level tests meaningful by asserting behavior or stable markers, not implementation trivia.
- If an API test assumes a live local service, gate it with an availability check instead of failing with connection refused in frontend-only runs.
- When a live smoke run fails, separate product bugs from brittle selector/test bugs before changing code.
