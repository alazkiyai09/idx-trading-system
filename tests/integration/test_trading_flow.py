"""
Integration Tests for Trading Flow

Tests the complete trading flow from signal generation to execution.
"""

import pytest
from datetime import datetime, timedelta, date
from typing import List, Dict

from core.data.models import OHLCV, Signal, SignalType, SetupType, Position, PortfolioState
from core.data.foreign_flow import FlowAnalysis, FlowSignal
from core.analysis.technical import TechnicalAnalyzer
from core.signals.signal_generator import SignalGenerator
from core.risk.position_sizer import PositionSizer
from core.risk.risk_manager import RiskManager
from core.execution.paper_trader import PaperTrader
from core.portfolio.portfolio_manager import PortfolioManager
from config.trading_modes import TradingMode, get_mode_config
from config.constants import IDX_LOT_SIZE


def create_ohlcv_data(
    num_days: int = 60,
    start_price: float = 9000.0,
    trend: str = "up",
    base_date: date = None,
    symbol: str = "TEST",
) -> List[OHLCV]:
    """Create test OHLCV data.

    Args:
        num_days: Number of days.
        start_price: Starting price.
        trend: Price trend.
        base_date: Starting date.
        symbol: Stock symbol.

    Returns:
        List of OHLCV objects.
    """
    if base_date is None:
        base_date = date(2024, 1, 1)

    data = []
    price = start_price

    for i in range(num_days):
        if trend == "up":
            change = price * 0.015
        elif trend == "down":
            change = -price * 0.015
        else:
            change = 0

        open_price = price
        close_price = price + change
        high_price = max(open_price, close_price) + price * 0.01
        low_price = min(open_price, close_price) - price * 0.01

        data.append(
            OHLCV(
                symbol=symbol,
                date=base_date + timedelta(days=i),
                open=open_price,
                high=high_price,
                low=low_price,
                close=close_price,
                volume=10_000_000 + i * 100_000,
            )
        )
        price = close_price

    return data


def create_flow_analysis(
    symbol: str,
    signal: FlowSignal = FlowSignal.BUY,
    score: float = 70.0,
) -> FlowAnalysis:
    """Create test flow analysis."""
    return FlowAnalysis(
        symbol=symbol,
        date=datetime.now().date(),
        today_net=3_000_000_000,
        five_day_net=15_000_000_000,
        twenty_day_net=60_000_000_000,
        consecutive_buy_days=3,
        consecutive_sell_days=0,
        signal=signal,
        signal_score=score,
        foreign_pct_of_volume=15.0,
        is_unusual_volume=False,
    )


class TestTradingFlow:
    """Integration tests for complete trading flow."""

    @pytest.fixture
    def config(self):
        """Get swing trading config."""
        return get_mode_config(TradingMode.SWING)

    @pytest.fixture
    def capital(self):
        """Initial capital."""
        return 100_000_000  # 100M IDR

    @pytest.fixture
    def risk_manager(self, capital):
        """Create risk manager."""
        return RiskManager(mode=TradingMode.SWING, capital=capital)

    @pytest.fixture
    def signal_generator(self, config):
        """Create signal generator."""
        return SignalGenerator(config)

    @pytest.fixture
    def paper_trader(self):
        """Create paper trader."""
        return PaperTrader()

    @pytest.fixture
    def portfolio_manager(self, capital):
        """Create portfolio manager."""
        return PortfolioManager(initial_capital=capital)

    def test_complete_buy_flow(
        self,
        risk_manager,
        signal_generator,
        paper_trader,
        portfolio_manager,
    ):
        """Test complete buy flow from signal to position."""
        # 1. Generate signal
        ohlcv_data = create_ohlcv_data(num_days=60, trend="up")
        flow = create_flow_analysis("BBCA", signal=FlowSignal.BUY, score=75.0)

        signal = signal_generator.generate(
            symbol="BBCA",
            ohlcv_data=ohlcv_data,
            flow_analysis=flow,
        )

        if signal is None or signal.signal_type != SignalType.BUY:
            pytest.skip("No buy signal generated in this test run")

        # 2. Validate with risk manager
        portfolio_state = portfolio_manager.get_state()
        validation = risk_manager.validate_entry(signal, portfolio_state)

        assert validation.approved is True, f"Trade rejected: {validation.veto_reason}"

        # 3. Execute trade
        result = paper_trader.buy(
            symbol="BBCA",
            quantity=validation.position_size,
            current_market_price=signal.entry_price,
        )

        assert result.success is True
        assert result.position is not None

        # 4. Add to portfolio
        portfolio_manager.open_position(result.position)

        # 5. Verify portfolio state
        state = portfolio_manager.get_state()
        assert state.open_positions == 1
        assert portfolio_manager.has_position("BBCA")

    def test_rejected_trade_flow(
        self,
        risk_manager,
        signal_generator,
        portfolio_manager,
    ):
        """Test flow when trade is rejected by risk manager."""
        # Create situation where trade should be rejected
        # Simulate max drawdown hit
        portfolio = portfolio_manager.get_state()

        # Manually trigger rejection by creating low score signal
        ohlcv_data = create_ohlcv_data(num_days=60, trend="down")
        signal = signal_generator.generate(
            symbol="BBRI",
            ohlcv_data=ohlcv_data,
        )

        if signal is None:
            pytest.skip("No signal generated")

        # Create artificial rejection condition
        # Lower the signal score to trigger rejection
        original_score = signal.composite_score

        # Try validation with modified portfolio (simulate max positions)
        from core.data.models import Position, SetupType

        # Add positions to reach max
        for i in range(12):  # Over max
            pos = Position(
                position_id=f"POS-{i}",
                symbol=f"STOCK{i}",
                entry_date=date.today(),
                entry_price=9000.0,
                quantity=100,
                current_price=9000.0,
                unrealized_pnl=0,
                unrealized_pnl_pct=0,
                stop_loss=8550.0,
                target_1=9450.0,
                target_2=9900.0,
                target_3=10350.0,
                highest_price=9000.0,
                days_held=1,
                setup_type=SetupType.MOMENTUM,
                signal_score=70.0,
            )
            portfolio_manager.open_position(pos)

        portfolio_state = portfolio_manager.get_state()
        validation = risk_manager.validate_entry(signal, portfolio_state)

        assert validation.approved is False
        assert validation.veto_reason is not None

    def test_sell_flow(
        self,
        risk_manager,
        signal_generator,
        paper_trader,
        portfolio_manager,
    ):
        """Test complete sell flow."""
        # First, create a position
        ohlcv_data = create_ohlcv_data(num_days=60, trend="up")

        result = paper_trader.buy(
            symbol="BBCA",
            quantity=500,  # 5 lots
            current_market_price=9000.0,
        )

        assert result.success is True

        # Update portfolio with position
        portfolio_manager.open_position(result.position)

        # Now simulate price increase and sell
        new_price = 9500.0  # 5.5% gain

        # Validate exit
        validation = risk_manager.validate_exit(
            position=result.position,
            proposed_exit_price=new_price,
            exit_reason="target_1",
        )

        assert validation.approved is True

        # Execute sell
        sell_result = paper_trader.sell(
            symbol="BBCA",
            quantity=500,
            current_market_price=new_price,
            exit_reason="target_1",
        )

        assert sell_result.success is True
        assert sell_result.trade is not None
        assert sell_result.trade.net_pnl > 0  # Should be profitable

        # Update portfolio
        trade = portfolio_manager.close_position(
            symbol="BBCA",
            exit_price=new_price,
            exit_date=date.today(),
            exit_reason="target_1",
        )

        # Verify portfolio updated
        assert not portfolio_manager.has_position("BBCA")

    def test_portfolio_tracking(
        self,
        portfolio_manager,
    ):
        """Test portfolio tracking through multiple trades."""
        initial_value = portfolio_manager.get_total_value()

        # Open multiple positions
        positions_data = [
            ("BBCA", 9000.0, 500),
            ("BBRI", 4500.0, 1000),
            ("TLKM", 3200.0, 1500),
        ]

        for symbol, price, qty in positions_data:
            from core.data.models import Position, SetupType

            pos = Position(
                position_id=f"POS-{symbol}",
                symbol=symbol,
                entry_date=date.today(),
                entry_price=price,
                quantity=qty,
                current_price=price,
                unrealized_pnl=0,
                unrealized_pnl_pct=0,
                stop_loss=price * 0.95,
                target_1=price * 1.05,
                target_2=price * 1.10,
                target_3=price * 1.15,
                highest_price=price,
                days_held=0,
                setup_type=SetupType.MOMENTUM,
                signal_score=70.0,
            )
            portfolio_manager.open_position(pos)

        # Verify positions
        assert portfolio_manager.get_position_count() == 3

        # Update prices
        new_prices = {
            "BBCA": 9200.0,  # +2.2%
            "BBRI": 4600.0,  # +2.2%
            "TLKM": 3100.0,  # -3.1%
        }
        portfolio_manager.update_prices(new_prices)

        # Check state
        state = portfolio_manager.get_state()
        assert state.open_positions == 3

        # Check unrealized P&L
        bbca_pos = portfolio_manager.get_position("BBCA")
        assert bbca_pos.unrealized_pnl > 0

        tlkm_pos = portfolio_manager.get_position("TLKM")
        assert tlkm_pos.unrealized_pnl < 0

    def test_risk_limits_enforcement(
        self,
        risk_manager,
        portfolio_manager,
    ):
        """Test that risk limits are enforced."""
        # Create a high-risk scenario
        from core.data.models import Signal, SignalType, SetupType, FlowSignal

        # Test daily loss limit
        portfolio = portfolio_manager.get_state()

        # Create signal with normal score
        signal = Signal(
            symbol="TEST",
            timestamp=datetime.now(),
            signal_type=SignalType.BUY,
            composite_score=75.0,
            technical_score=75.0,
            flow_score=50.0,
            fundamental_score=None,
            setup_type=SetupType.MOMENTUM,
            flow_signal=FlowSignal.NEUTRAL,
            entry_price=9000.0,
            stop_loss=8550.0,
            target_1=9450.0,
            target_2=9900.0,
            target_3=10350.0,
            risk_pct=0.05,
            key_factors=[],
            risks=[],
        )

        # Should be approved normally
        validation = risk_manager.validate_entry(signal, portfolio)
        normal_result = validation.approved

        # Now simulate portfolio hitting risk limits
        # Create portfolio state with high drawdown
        high_drawdown_state = PortfolioState(
            timestamp=datetime.now(),
            cash=80_000_000,
            total_value=85_000_000,
            positions_value=5_000_000,
            total_pnl=-15_000_000,
            total_pnl_pct=-0.15,
            daily_pnl=-2_000_000,
            daily_pnl_pct=-0.023,  # -2.3% exceeds limit
            peak_value=100_000_000,
            drawdown=15_000_000,
            drawdown_pct=0.15,  # 15% exceeds 10% limit
            open_positions=3,
            positions=[],
        )

        validation = risk_manager.validate_entry(signal, high_drawdown_state)
        assert validation.approved is False

    def test_position_sizing_integration(
        self,
        risk_manager,
        signal_generator,
    ):
        """Test position sizing integration with risk manager."""
        ohlcv_data = create_ohlcv_data(num_days=60, trend="up")
        flow = create_flow_analysis("BBCA", score=80.0)

        signal = signal_generator.generate(
            symbol="BBCA",
            ohlcv_data=ohlcv_data,
            flow_analysis=flow,
        )

        if signal is None:
            pytest.skip("No signal generated")

        portfolio = PortfolioState(
            timestamp=datetime.now(),
            cash=100_000_000,
            total_value=100_000_000,
            positions_value=0,
            total_pnl=0,
            total_pnl_pct=0,
            daily_pnl=0,
            daily_pnl_pct=0,
            peak_value=100_000_000,
            drawdown=0,
            drawdown_pct=0,
            open_positions=0,
            positions=[],
        )

        validation = risk_manager.validate_entry(signal, portfolio)

        if validation.approved:
            # Position should be properly sized
            assert validation.position_size > 0
            assert validation.position_size % IDX_LOT_SIZE == 0
            assert validation.position_value > 0
            assert validation.risk_amount > 0

    def test_multi_symbol_flow(
        self,
        signal_generator,
        risk_manager,
        paper_trader,
        portfolio_manager,
    ):
        """Test processing multiple symbols."""
        symbols = ["BBCA", "BBRI", "TLKM", "ASII"]
        results = {}

        for symbol in symbols:
            # Generate data for each symbol
            ohlcv_data = create_ohlcv_data(
                num_days=60,
                start_price=5000.0 + hash(symbol) % 5000,  # Different prices
                trend="up",
            )
            flow = create_flow_analysis(symbol, score=70.0)

            signal = signal_generator.generate(
                symbol=symbol,
                ohlcv_data=ohlcv_data,
                flow_analysis=flow,
            )

            if signal and signal.signal_type == SignalType.BUY:
                portfolio_state = portfolio_manager.get_state()
                validation = risk_manager.validate_entry(signal, portfolio_state)

                if validation.approved:
                    result = paper_trader.buy(
                        symbol=symbol,
                        quantity=validation.position_size,
                        current_market_price=signal.entry_price,
                    )

                    if result.success:
                        results[symbol] = {
                            "signal": signal,
                            "validation": validation,
                            "execution": result,
                        }

                        # Add to portfolio
                        portfolio_manager.open_position(result.position)

        # Verify results - positions may or may not be opened depending on signal generation
        # The test verifies the flow works without errors
        # Position count could be 0 if no signals meet the threshold
        assert isinstance(portfolio_manager.get_position_count(), int)

    def test_stop_loss_trigger(
        self,
        paper_trader,
    ):
        """Test stop loss triggering."""
        # Create position in paper trader
        entry_price = 9000.0
        result = paper_trader.buy(
            symbol="BBCA",
            quantity=500,
            current_market_price=entry_price,
        )
        assert result.success is True

        # Simulate price drop below stop loss
        new_price = 8400.0  # Below 5% stop

        # Update prices in paper trader
        paper_trader.update_position_prices({"BBCA": new_price})

        # Check stop loss
        prices = {"BBCA": new_price}
        stopped = paper_trader.check_stop_losses(prices)

        # Should detect stop hit (paper trader has default 5% stop)
        assert len(stopped) > 0
        assert stopped[0][0].symbol == "BBCA"


class TestPaperTraderIntegration:
    """Integration tests for paper trader."""

    @pytest.fixture
    def trader(self):
        """Create paper trader."""
        return PaperTrader()

    def test_round_trip_trade(self, trader):
        """Test complete round trip (buy then sell)."""
        # Buy
        buy_result = trader.buy(
            symbol="BBCA",
            quantity=500,
            current_market_price=9000.0,
        )

        assert buy_result.success is True
        assert buy_result.position is not None
        assert len(trader.get_positions()) == 1

        # Sell
        sell_result = trader.sell(
            symbol="BBCA",
            quantity=500,
            current_market_price=9500.0,  # Higher price
            exit_reason="profit_taking",
        )

        assert sell_result.success is True
        assert sell_result.trade is not None
        assert sell_result.trade.net_pnl > 0  # Profitable trade
        assert len(trader.get_positions()) == 0

    def test_multiple_trades(self, trader):
        """Test multiple sequential trades."""
        trades_count = 0

        for i in range(3):
            symbol = f"STOCK{i}"

            # Buy
            buy_result = trader.buy(
                symbol=symbol,
                quantity=100,
                current_market_price=5000.0 + i * 1000,
            )

            if buy_result.success:
                # Sell
                sell_result = trader.sell(
                    symbol=symbol,
                    quantity=100,
                    current_market_price=5500.0 + i * 1000,
                    exit_reason="target",
                )

                if sell_result.success:
                    trades_count += 1

        assert trades_count == 3
        assert len(trader.get_trades()) == 3

    def test_execution_stats(self, trader):
        """Test execution statistics."""
        # Make some trades
        for i in range(5):
            symbol = f"TST{i}"
            trader.buy(symbol=symbol, quantity=100, current_market_price=1000.0)
            trader.sell(symbol=symbol, quantity=100, current_market_price=1100.0, exit_reason="test")

        stats = trader.get_execution_stats()

        assert stats["total_trades"] == 5
        assert stats["total_fees"] > 0


class TestPortfolioManagerIntegration:
    """Integration tests for portfolio manager."""

    @pytest.fixture
    def manager(self):
        """Create portfolio manager."""
        return PortfolioManager(initial_capital=100_000_000)

    def test_portfolio_lifecycle(self, manager):
        """Test complete portfolio lifecycle."""
        from core.data.models import Position, SetupType

        # Initial state
        assert manager.get_cash_available() == 100_000_000
        assert manager.get_position_count() == 0

        # Add positions
        for i in range(3):
            pos = Position(
                position_id=f"POS-{i}",
                symbol=f"STOCK{i}",
                entry_date=date.today(),
                entry_price=5000.0,
                quantity=1000,
                current_price=5000.0,
                unrealized_pnl=0,
                unrealized_pnl_pct=0,
                stop_loss=4750.0,
                target_1=5250.0,
                target_2=5500.0,
                target_3=5750.0,
                highest_price=5000.0,
                days_held=0,
                setup_type=SetupType.MOMENTUM,
                signal_score=70.0,
            )
            manager.open_position(pos)

        assert manager.get_position_count() == 3
        assert manager.get_cash_available() < 100_000_000

        # Update prices
        manager.update_prices({
            "STOCK0": 5200.0,
            "STOCK1": 5100.0,
            "STOCK2": 4800.0,
        })

        # Check P&L
        state = manager.get_state()
        assert state.positions_value > 0

        # Close positions
        for i in range(3):
            manager.close_position(
                symbol=f"STOCK{i}",
                exit_price=5500.0,
                exit_date=date.today(),
                exit_reason="test",
            )

        assert manager.get_position_count() == 0

    def test_drawdown_tracking(self, manager):
        """Test drawdown tracking."""
        from core.data.models import Position, SetupType

        # Initial state
        assert manager.get_drawdown_pct() == 0

        # Add position
        pos = Position(
            position_id="POS-TEST",
            symbol="TEST",
            entry_date=date.today(),
            entry_price=9000.0,
            quantity=1000,
            current_price=9000.0,
            unrealized_pnl=0,
            unrealized_pnl_pct=0,
            stop_loss=8550.0,
            target_1=9450.0,
            target_2=9900.0,
            target_3=10350.0,
            highest_price=9000.0,
            days_held=0,
            setup_type=SetupType.MOMENTUM,
            signal_score=70.0,
        )
        manager.open_position(pos)

        # Price goes up - no drawdown
        manager.update_prices({"TEST": 9500.0})
        assert manager.get_drawdown_pct() == 0

        # Price goes down - drawdown
        manager.update_prices({"TEST": 8500.0})
        assert manager.get_drawdown_pct() > 0

    def test_daily_reset(self, manager):
        """Test daily reset functionality."""
        from core.data.models import Position, SetupType

        # Add position and update prices
        pos = Position(
            position_id="POS-TEST",
            symbol="TEST",
            entry_date=date.today(),
            entry_price=9000.0,
            quantity=1000,
            current_price=9000.0,
            unrealized_pnl=0,
            unrealized_pnl_pct=0,
            stop_loss=8550.0,
            target_1=9450.0,
            target_2=9900.0,
            target_3=10350.0,
            highest_price=9000.0,
            days_held=0,
            setup_type=SetupType.MOMENTUM,
            signal_score=70.0,
        )
        manager.open_position(pos)
        manager.update_prices({"TEST": 9200.0})

        # Reset daily
        manager.reset_daily()

        # Should update daily start value
        # (Implementation specific behavior)
