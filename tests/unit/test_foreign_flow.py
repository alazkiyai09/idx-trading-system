"""Tests for foreign flow module."""

import pytest
from datetime import date, timedelta
from unittest.mock import patch

from core.data.foreign_flow import (
    FlowAnalysis,
    ForeignFlowFetcher,
)
from core.data.models import ForeignFlow, FlowSignal


class TestFlowAnalysis:
    """Tests for FlowAnalysis dataclass."""

    def test_flow_analysis_creation(self):
        """Test creating FlowAnalysis instance."""
        analysis = FlowAnalysis(
            symbol="BBCA",
            date=date(2024, 1, 15),
            today_net=5_000_000_000.0,
            five_day_net=15_000_000_000.0,
            twenty_day_net=40_000_000_000.0,
            consecutive_buy_days=3,
            consecutive_sell_days=0,
            signal=FlowSignal.BUY,
            signal_score=75.0,
            foreign_pct_of_volume=45.0,
            is_unusual_volume=False,
        )

        assert analysis.symbol == "BBCA"
        assert analysis.today_net == 5_000_000_000.0
        assert analysis.signal == FlowSignal.BUY
        assert analysis.signal_score == 75.0


class TestForeignFlowFetcher:
    """Tests for ForeignFlowFetcher class."""

    @pytest.fixture
    def fetcher(self):
        """Create ForeignFlowFetcher instance."""
        return ForeignFlowFetcher()

    @pytest.fixture
    def sample_flow_data(self):
        """Create sample foreign flow data for testing."""
        data = []
        base_date = date(2024, 1, 1)

        for i in range(20):
            flow = ForeignFlow(
                symbol="BBCA",
                date=base_date + timedelta(days=i),
                foreign_buy=100_000_000_000.0 + i * 1_000_000_000,
                foreign_sell=80_000_000_000.0 + i * 500_000_000,
                foreign_net=20_000_000_000.0 + i * 500_000_000,
                total_value=200_000_000_000.0,
                foreign_pct=40.0,
            )
            data.append(flow)

        return data

    def test_thresholds(self, fetcher):
        """Test threshold values are set correctly."""
        assert fetcher.STRONG_THRESHOLD == 10_000_000_000
        assert fetcher.MODERATE_THRESHOLD == 2_000_000_000
        assert fetcher.CONSECUTIVE_DAYS_BONUS == 3

    def test_generate_simulated_data(self, fetcher):
        """Test generating simulated flow data."""
        start_date = date(2024, 1, 1)
        end_date = date(2024, 1, 10)

        data = fetcher._generate_simulated_data("BBCA", start_date, end_date)

        # Should have ~8 days (excluding weekend)
        assert len(data) >= 6
        assert all(f.symbol == "BBCA" for f in data)

    def test_analyze_flow_empty_data(self, fetcher):
        """Test analyzing empty flow data."""
        result = fetcher.analyze_flow([])
        assert result is None

    def test_analyze_flow_positive(self, fetcher, sample_flow_data):
        """Test analyzing positive foreign flow."""
        # Make all flows positive
        for flow in sample_flow_data:
            flow.foreign_net = 5_000_000_000.0  # Strong positive

        result = fetcher.analyze_flow(sample_flow_data)

        assert result is not None
        assert result.symbol == "BBCA"
        assert result.signal in [FlowSignal.BUY, FlowSignal.STRONG_BUY]
        assert result.signal_score >= 50
        assert result.consecutive_buy_days > 0
        assert result.consecutive_sell_days == 0

    def test_analyze_flow_negative(self, fetcher, sample_flow_data):
        """Test analyzing negative foreign flow."""
        # Make all flows negative
        for flow in sample_flow_data:
            flow.foreign_net = -5_000_000_000.0  # Strong negative

        result = fetcher.analyze_flow(sample_flow_data)

        assert result is not None
        assert result.signal in [FlowSignal.SELL, FlowSignal.STRONG_SELL]
        assert result.signal_score <= 50
        assert result.consecutive_sell_days > 0
        assert result.consecutive_buy_days == 0

    def test_analyze_flow_neutral(self, fetcher, sample_flow_data):
        """Test analyzing neutral foreign flow."""
        # Make flows balanced
        for i, flow in enumerate(sample_flow_data):
            flow.foreign_net = 100_000_000.0 if i % 2 == 0 else -100_000_000.0

        result = fetcher.analyze_flow(sample_flow_data)

        assert result is not None
        # Signal could be neutral or slightly positive/negative

    def test_calculate_signal_strong_buy(self, fetcher):
        """Test signal calculation for strong buy."""
        signal, score = fetcher._calculate_signal(
            five_day_net=15_000_000_000.0,  # Above strong threshold
            twenty_day_net=50_000_000_000.0,
            consecutive_buy=5,
            consecutive_sell=0,
        )

        assert signal == FlowSignal.STRONG_BUY
        assert score >= 80

    def test_calculate_signal_strong_sell(self, fetcher):
        """Test signal calculation for strong sell."""
        signal, score = fetcher._calculate_signal(
            five_day_net=-15_000_000_000.0,  # Below strong threshold
            twenty_day_net=-50_000_000_000.0,
            consecutive_buy=0,
            consecutive_sell=5,
        )

        assert signal == FlowSignal.STRONG_SELL
        assert score <= 20

    def test_calculate_signal_neutral(self, fetcher):
        """Test signal calculation for neutral."""
        signal, score = fetcher._calculate_signal(
            five_day_net=500_000_000.0,  # Below moderate threshold
            twenty_day_net=1_000_000_000.0,
            consecutive_buy=1,
            consecutive_sell=0,
        )

        assert signal == FlowSignal.NEUTRAL
        assert 40 <= score <= 60

    def test_calculate_consecutive_days_buy(self, fetcher):
        """Test consecutive days calculation for buying."""
        flows = []
        for i in range(5):
            flows.append(
                ForeignFlow(
                    symbol="TEST",
                    date=date(2024, 1, 1 + i),
                    foreign_buy=100.0,
                    foreign_sell=50.0,
                    foreign_net=50.0,  # Positive
                    total_value=200.0,
                    foreign_pct=30.0,
                )
            )

        buy_days, sell_days = fetcher._calculate_consecutive_days(flows)
        assert buy_days == 5
        assert sell_days == 0

    def test_calculate_consecutive_days_mixed(self, fetcher):
        """Test consecutive days calculation with mixed flows."""
        flows = []
        for i in range(5):
            flows.append(
                ForeignFlow(
                    symbol="TEST",
                    date=date(2024, 1, 1 + i),
                    foreign_buy=100.0,
                    foreign_sell=50.0,
                    foreign_net=50.0 if i >= 3 else -50.0,  # Last 2 positive
                    total_value=200.0,
                    foreign_pct=30.0,
                )
            )

        buy_days, sell_days = fetcher._calculate_consecutive_days(flows)
        assert buy_days == 2
        assert sell_days == 0

    def test_cache_flow_data(self, fetcher):
        """Test caching flow data."""
        flows = [
            ForeignFlow(
                symbol="TEST",
                date=date(2024, 1, 1),
                foreign_buy=100.0,
                foreign_sell=50.0,
                foreign_net=50.0,
                total_value=200.0,
                foreign_pct=30.0,
            )
        ]

        fetcher.cache_flow_data("TEST", flows)
        cached = fetcher.get_cached_flow_data("TEST")

        assert cached is not None
        assert len(cached) == 1
        assert cached[0].symbol == "TEST"

    def test_get_flow_signal_for_symbol(self, fetcher):
        """Test convenience method for getting flow signal."""
        result = fetcher.get_flow_signal_for_symbol("BBCA", days=30)

        assert result is not None
        assert result.symbol == "BBCA"
        assert isinstance(result.signal, FlowSignal)

    def test_get_multiple_flow_signals(self, fetcher):
        """Test getting flow signals for multiple symbols."""
        symbols = ["BBCA", "TLKM", "ASII"]
        results = fetcher.get_multiple_flow_signals(symbols, days=30)

        assert len(results) == 3
        for symbol in symbols:
            assert symbol in results
            assert results[symbol] is not None
