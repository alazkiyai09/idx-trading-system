"""Shared fixtures for IMSS tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def mock_prices() -> list[dict]:
    """5 days of BBRI OHLCV data."""
    return json.loads((FIXTURES_DIR / "mock_prices.json").read_text())


@pytest.fixture
def mock_events() -> list[dict]:
    """2 sample events."""
    return json.loads((FIXTURES_DIR / "mock_events.json").read_text())
