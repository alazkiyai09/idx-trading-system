# Project Context Index

This folder is a durable editing reference for the `idx-trading-system` repo.
It is split by responsibility so future work can load only the relevant context
instead of rebuilding a full mental model from scratch.

Use this index first, then open the specific file for the area being changed.

## Files

- [`00_repo_map.md`](./00_repo_map.md): high-level structure, entrypoints, and large-module hotspots
- [`01_backend_api.md`](./01_backend_api.md): FastAPI app shape, route groups, and backend contracts
- [`02_dashboard_ui.md`](./02_dashboard_ui.md): Streamlit architecture, page ownership, and current UI separation state
- [`03_core_domain.md`](./03_core_domain.md): domain logic in `core/`, `backtest/`, and supporting services
- [`04_data_and_jobs.md`](./04_data_and_jobs.md): database, ingestion scripts, and daily-update workflow
- [`05_testing_and_e2e.md`](./05_testing_and_e2e.md): test layers, what they cover, and how to run them
- [`06_change_hotspots.md`](./06_change_hotspots.md): practical guidance for maintainable edits and current refactor focus areas
- [`07_autoresearch.md`](./07_autoresearch.md): autonomous strategy-research workspace, runner, and safe extension points
- [`08_autoresearch_comparison.md`](./08_autoresearch_comparison.md): current live GLM state, prepared Codex workspace, and next-session comparison handoff
- [`09_research_presets_and_benchmarks.md`](./09_research_presets_and_benchmarks.md): autoscreener and automontecarlo benchmark state, promoted UI presets, browser-test status, and next-session recommendation
- [`10_imss.md`](./10_imss.md): IDX Market Swarm Simulator — multi-agent LLM simulation engine, three-tier agent architecture, multi-run execution, and Phase 1/2B implementation state

## Current Product Summary

The system is a daily-data IDX trading workspace with:

- FastAPI backend under `api/`
- Streamlit dashboard under `dashboard/`
- domain logic under `core/` and `backtest/`
- multi-agent market simulation under `imss/` (IMSS)
- SQLite-backed storage under `data/`
- ingestion and training jobs under `scripts/`
- Python tests under `tests/`
- Playwright E2E tests under `e2e/`

## Main Runtime Paths

- API entrypoint: `python3 -m uvicorn api.main:app --host 0.0.0.0 --port 8000`
- Dashboard entrypoint: `API_URL=http://127.0.0.1:8000 streamlit run dashboard/app.py --server.port 8501`
- Primary local URLs:
  - API: `http://127.0.0.1:8000`
  - Dashboard: `http://127.0.0.1:8501`
  - OpenAPI docs: `http://127.0.0.1:8000/docs`

## How To Use This Context

- If editing an API route or response shape, start with `01_backend_api.md`.
- If editing Streamlit page structure or maintainability, start with `02_dashboard_ui.md`.
- If changing indicators, risk, signals, ML, or simulations, open `03_core_domain.md`.
- If the task touches ingestion, snapshots, SQLite tables, or freshness, open `04_data_and_jobs.md`.
- If validating changes or fixing failing suites, open `05_testing_and_e2e.md`.
- If the task is a refactor, split, or cleanup, open `06_change_hotspots.md`.
- If the task touches autonomous strategy experiments, open `07_autoresearch.md`.
- If the task is continuing the GLM vs Codex benchmark, open `08_autoresearch_comparison.md`.
- If the task touches promoted research presets, autoscreener/automontecarlo benchmark outputs, or the current browser-test follow-up, open `09_research_presets_and_benchmarks.md`.
- If the task touches the multi-agent market simulation (IMSS), agent personas, LLM-driven trading decisions, or simulation runs, open `10_imss.md`.
