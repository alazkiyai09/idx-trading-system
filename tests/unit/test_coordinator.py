"""Tests for the Coordinator module."""

import pytest
from unittest.mock import MagicMock, patch
from datetime import date

from agents.coordinator import (
    Coordinator,
    CoordinatorConfig,
    ScanReport,
    ScanResult,
)
from config.trading_modes import TradingMode


class TestScanReport:
    """Tests for ScanReport dataclass."""

    def test_default_values(self):
        """Test default ScanReport values."""
        report = ScanReport(
            scan_date=date(2024, 1, 15),
            mode=TradingMode.SWING,
            result=ScanResult.SUCCESS,
        )
        assert report.signals_generated == 0
        assert report.signals_approved == 0
        assert report.symbols_scanned == 0
        assert report.errors == []
        assert report.signals == []

    def test_to_dict(self):
        """Test ScanReport conversion to dictionary."""
        report = ScanReport(
            scan_date=date(2024, 1, 15),
            mode=TradingMode.SWING,
            result=ScanResult.SUCCESS,
            signals_generated=5,
            signals_approved=3,
        )
        d = report.to_dict()
        assert d["signals_generated"] == 5
        assert d["signals_approved"] == 3


class TestCoordinatorConfig:
    """Tests for CoordinatorConfig."""

    def test_default_values(self):
        """Test default config values."""
        config = CoordinatorConfig()
        assert config.mode == TradingMode.SWING
        assert config.dry_run is True
        assert config.max_signals_per_day == 10
        assert config.min_signal_score == 60.0


class TestCoordinator:
    """Tests for Coordinator."""

    def test_init_default(self):
        """Test coordinator initialization with defaults."""
        coord = Coordinator(dry_run=True)
        assert coord.dry_run is True

    def test_init_with_mode(self):
        """Test coordinator initialization with specific mode."""
        coord = Coordinator(mode=TradingMode.INTRADAY, dry_run=True)
        assert coord.mode == TradingMode.INTRADAY

    def test_init_with_universe(self):
        """Test coordinator initialization with custom universe."""
        symbols = ["BBCA", "BMRI", "BBRI"]
        coord = Coordinator(universe=symbols, dry_run=True)
        assert coord.universe == symbols

    def test_get_portfolio_summary(self):
        """Test getting portfolio summary."""
        coord = Coordinator(dry_run=True)
        summary = coord.get_portfolio_summary()
        assert isinstance(summary, dict)

    def test_scan_report_result_enum(self):
        """Test ScanResult enum values."""
        assert ScanResult.SUCCESS.value == "success"
        assert ScanResult.PARTIAL.value == "partial"
        assert ScanResult.FAILED.value == "failed"
        assert ScanResult.NO_SIGNALS.value == "no_signals"
