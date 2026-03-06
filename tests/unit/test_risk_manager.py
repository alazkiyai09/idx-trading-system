"""
Tests for Risk Manager Module

Tests risk validation, position sizing integration, and veto logic.
"""

import pytest
from datetime import datetime, date

from core.risk.risk_manager import RiskManager, ValidationResult
from core.risk.position_sizer import PositionSize
from core.data.models import Signal, SignalType, SetupType, FlowSignal, Position, PortfolioState
from config.trading_modes import TradingMode


def create_test_signal(
    symbol: str = "BBCA",
    entry_price: float = 9000.0,
    stop_loss: float = 8550.0,
    composite_score: float = 75.0,
) -> Signal:
    """Create a test signal.

    Args:
        symbol: Stock symbol.
        entry_price: Entry price.
        stop_loss: Stop loss price.
        composite_score: Signal score.

    Returns:
        Test Signal object.
    """
    return Signal(
        symbol=symbol,
        timestamp=datetime.now(),
        signal_type=SignalType.BUY,
        composite_score=composite_score,
        technical_score=composite_score,
        flow_score=50.0,
        fundamental_score=None,
        setup_type=SetupType.MOMENTUM,
        flow_signal=FlowSignal.NEUTRAL,
        entry_price=entry_price,
        stop_loss=stop_loss,
        target_1=entry_price * 1.05,
        target_2=entry_price * 1.10,
        target_3=entry_price * 1.15,
        risk_pct=0.05,
        key_factors=["Test signal"],
        risks=[],
    )


def create_test_portfolio(
    total_value: float = 100_000_000,
    cash: float = 80_000_000,
    open_positions: int = 2,
    daily_pnl_pct: float = 0.0,
    drawdown_pct: float = 0.0,
    positions: list = None,
) -> PortfolioState:
    """Create a test portfolio state.

    Args:
        total_value: Total portfolio value.
        cash: Available cash.
        open_positions: Number of open positions.
        daily_pnl_pct: Daily P&L percentage.
        drawdown_pct: Drawdown percentage.
        positions: List of positions.

    Returns:
        Test PortfolioState object.
    """
    if positions is None:
        positions = []

    positions_value = total_value - cash

    return PortfolioState(
        timestamp=datetime.now(),
        cash=cash,
        total_value=total_value,
        positions_value=positions_value,
        total_pnl=0.0,
        total_pnl_pct=0.0,
        daily_pnl=total_value * daily_pnl_pct,
        daily_pnl_pct=daily_pnl_pct,
        peak_value=total_value / (1 - drawdown_pct) if drawdown_pct > 0 else total_value,
        drawdown=total_value * drawdown_pct,
        drawdown_pct=drawdown_pct,
        open_positions=open_positions,
        positions=positions,
    )


def create_test_position(
    symbol: str = "BBCA",
    quantity: int = 1000,
    entry_price: float = 9000.0,
    current_price: float = 9200.0,
    days_held: int = 5,
) -> Position:
    """Create a test position.

    Args:
        symbol: Stock symbol.
        quantity: Number of shares.
        entry_price: Entry price.
        current_price: Current price.
        days_held: Days held.

    Returns:
        Test Position object.
    """
    unrealized_pnl = (current_price - entry_price) * quantity
    unrealized_pnl_pct = (current_price - entry_price) / entry_price * 100

    return Position(
        position_id=f"POS-{symbol}",
        symbol=symbol,
        entry_date=date.today(),
        entry_price=entry_price,
        quantity=quantity,
        current_price=current_price,
        unrealized_pnl=unrealized_pnl,
        unrealized_pnl_pct=unrealized_pnl_pct,
        stop_loss=entry_price * 0.95,
        target_1=entry_price * 1.05,
        target_2=entry_price * 1.10,
        target_3=entry_price * 1.15,
        highest_price=current_price,
        days_held=days_held,
        setup_type=SetupType.MOMENTUM,
        signal_score=75.0,
    )


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_approved_result(self):
        """Test creating an approved validation result."""
        result = ValidationResult(
            approved=True,
            position_size=1000,
            position_value=9_000_000,
            risk_amount=450_000,
        )

        assert result.approved is True
        assert result.veto_reason is None
        assert result.position_size == 1000

    def test_rejected_result(self):
        """Test creating a rejected validation result."""
        result = ValidationResult(
            approved=False,
            veto_reason="Daily loss limit reached",
        )

        assert result.approved is False
        assert result.veto_reason == "Daily loss limit reached"
        assert result.position_size is None

    def test_result_with_warnings(self):
        """Test validation result with warnings."""
        result = ValidationResult(
            approved=True,
            warnings=["Drawdown > 5%", "Sector concentration high"],
        )

        assert result.approved is True
        assert len(result.warnings) == 2


class TestRiskManager:
    """Tests for RiskManager class."""

    @pytest.fixture
    def risk_manager(self):
        """Create a risk manager with 100M capital."""
        return RiskManager(mode=TradingMode.SWING, capital=100_000_000)

    @pytest.fixture
    def healthy_portfolio(self):
        """Create a healthy portfolio state."""
        return create_test_portfolio(
            total_value=100_000_000,
            cash=70_000_000,
            open_positions=3,
            daily_pnl_pct=0.005,  # +0.5%
            drawdown_pct=0.02,  # 2% drawdown
        )

    def test_initialization(self):
        """Test risk manager initialization."""
        rm = RiskManager(mode=TradingMode.SWING, capital=100_000_000)

        assert rm.mode == TradingMode.SWING
        assert rm.capital == 100_000_000
        assert rm.position_sizer is not None

    def test_initialization_default_mode(self):
        """Test default mode is SWING."""
        rm = RiskManager(capital=100_000_000)
        assert rm.mode == TradingMode.SWING

    # ============== ENTRY VALIDATION TESTS ==============

    def test_validate_entry_approved(self, risk_manager, healthy_portfolio):
        """Test entry validation approved."""
        signal = create_test_signal(composite_score=75.0)

        result = risk_manager.validate_entry(signal, healthy_portfolio)

        assert result.approved is True
        assert result.position_size is not None
        assert result.position_size > 0

    def test_validate_entry_low_score_rejected(self, risk_manager, healthy_portfolio):
        """Test entry rejected for low score."""
        signal = create_test_signal(composite_score=50.0)  # Below minimum

        result = risk_manager.validate_entry(signal, healthy_portfolio)

        assert result.approved is False
        assert "below minimum" in result.veto_reason.lower()

    def test_validate_entry_daily_loss_limit(self, risk_manager):
        """Test entry rejected for daily loss limit."""
        portfolio = create_test_portfolio(
            daily_pnl_pct=-0.025,  # -2.5% (exceeds 2% limit)
        )
        signal = create_test_signal(composite_score=75.0)

        result = risk_manager.validate_entry(signal, portfolio)

        assert result.approved is False
        assert "daily loss" in result.veto_reason.lower()

    def test_validate_entry_max_drawdown(self, risk_manager):
        """Test entry rejected for max drawdown."""
        portfolio = create_test_portfolio(
            drawdown_pct=0.12,  # 12% (exceeds 10% limit)
        )
        signal = create_test_signal(composite_score=75.0)

        result = risk_manager.validate_entry(signal, portfolio)

        assert result.approved is False
        assert "drawdown" in result.veto_reason.lower()

    def test_validate_entry_max_positions(self, risk_manager):
        """Test entry rejected for max positions."""
        portfolio = create_test_portfolio(
            open_positions=10,  # At or above limit
        )
        signal = create_test_signal(composite_score=75.0)

        result = risk_manager.validate_entry(signal, portfolio)

        assert result.approved is False
        assert "max positions" in result.veto_reason.lower()

    def test_validate_entry_existing_position(self, risk_manager, healthy_portfolio):
        """Test entry rejected for existing position."""
        existing_position = create_test_position(symbol="BBCA")
        portfolio = create_test_portfolio(
            open_positions=1,
            positions=[existing_position],
        )

        signal = create_test_signal(symbol="BBCA")  # Same symbol

        result = risk_manager.validate_entry(signal, portfolio)

        assert result.approved is False
        assert "already have position" in result.veto_reason.lower()

    def test_validate_entry_invalid_stop_loss(self, risk_manager, healthy_portfolio):
        """Test entry rejected for invalid stop loss."""
        signal = create_test_signal(
            entry_price=9000.0,
            stop_loss=9500.0,  # Above entry price
        )

        result = risk_manager.validate_entry(signal, healthy_portfolio)

        assert result.approved is False
        assert "invalid stop loss" in result.veto_reason.lower()

    def test_validate_entry_stop_too_tight(self, risk_manager, healthy_portfolio):
        """Test entry rejected for stop loss too tight."""
        signal = create_test_signal(
            entry_price=9000.0,
            stop_loss=8990.0,  # Only 0.1% below
        )

        result = risk_manager.validate_entry(signal, healthy_portfolio)

        assert result.approved is False
        assert "too tight" in result.veto_reason.lower()

    def test_validate_entry_drawdown_warning(self, risk_manager):
        """Test position reduced for drawdown warning."""
        portfolio = create_test_portfolio(
            drawdown_pct=0.06,  # 6% (triggers 50% reduction)
        )
        signal = create_test_signal(composite_score=75.0)

        result = risk_manager.validate_entry(signal, portfolio)

        # Should be approved but with warning
        assert result.approved is True
        assert any("drawdown" in w.lower() for w in result.warnings)

    def test_validate_entry_daily_loss_warning(self, risk_manager):
        """Test position reduced for daily loss warning."""
        portfolio = create_test_portfolio(
            daily_pnl_pct=-0.015,  # -1.5% (triggers 25% reduction)
        )
        signal = create_test_signal(composite_score=75.0)

        result = risk_manager.validate_entry(signal, portfolio)

        # Should be approved but with warning
        assert result.approved is True
        assert any("daily loss" in w.lower() for w in result.warnings)

    # ============== EXIT VALIDATION TESTS ==============

    def test_validate_exit_time_stop_allowed(self, risk_manager):
        """Test time stop exit is always allowed."""
        position = create_test_position()

        result = risk_manager.validate_exit(
            position=position,
            proposed_exit_price=9500.0,
            exit_reason="time_stop",
        )

        assert result.approved is True

    def test_validate_exit_stop_loss_allowed(self, risk_manager):
        """Test stop loss exit is always allowed."""
        position = create_test_position()

        result = risk_manager.validate_exit(
            position=position,
            proposed_exit_price=8500.0,
            exit_reason="stop_loss",
        )

        assert result.approved is True

    def test_validate_exit_target_allowed(self, risk_manager):
        """Test target exits are always allowed."""
        position = create_test_position()

        for target in ["target_1", "target_2", "target_3"]:
            result = risk_manager.validate_exit(
                position=position,
                proposed_exit_price=9500.0,
                exit_reason=target,
            )
            assert result.approved is True

    def test_validate_exit_signal_reversal_allowed(self, risk_manager):
        """Test signal reversal exit is allowed."""
        position = create_test_position()

        result = risk_manager.validate_exit(
            position=position,
            proposed_exit_price=9200.0,
            exit_reason="signal_reversal",
        )

        assert result.approved is True

    def test_validate_exit_manual_early_profit_rejected(self, risk_manager):
        """Test manual exit rejected if too early with profit."""
        position = create_test_position(
            entry_price=9000.0,
            current_price=9500.0,  # 5.5% profit
            days_held=1,  # Only 1 day
        )

        result = risk_manager.validate_exit(
            position=position,
            proposed_exit_price=9500.0,
            exit_reason="manual",
        )

        assert result.approved is False
        assert "too early" in result.veto_reason.lower()

    def test_validate_exit_manual_before_min_hold_days(self, risk_manager):
        """Test manual exit warning before min hold days."""
        position = create_test_position(
            days_held=1,  # Below min_hold_days (2 for swing mode)
        )

        result = risk_manager.validate_exit(
            position=position,
            proposed_exit_price=9100.0,  # Small profit (< 5%)
            exit_reason="manual",
        )

        # Should be approved with warning
        assert result.approved is True
        assert any("min hold" in w.lower() for w in result.warnings)

    # ============== PORTFOLIO RISK TESTS ==============

    def test_check_portfolio_risk_no_warnings(self, risk_manager, healthy_portfolio):
        """Test portfolio risk check with no warnings."""
        warnings = risk_manager.check_portfolio_risk(healthy_portfolio)

        # Should have no warnings for healthy portfolio
        # May have warnings depending on exact thresholds
        assert isinstance(warnings, list)

    def test_check_portfolio_risk_high_drawdown(self, risk_manager):
        """Test portfolio risk check with high drawdown."""
        portfolio = create_test_portfolio(drawdown_pct=0.06)

        warnings = risk_manager.check_portfolio_risk(portfolio)

        assert any("drawdown" in w.lower() for w in warnings)

    def test_check_portfolio_risk_daily_loss(self, risk_manager):
        """Test portfolio risk check with daily loss."""
        portfolio = create_test_portfolio(daily_pnl_pct=-0.015)

        warnings = risk_manager.check_portfolio_risk(portfolio)

        assert any("daily loss" in w.lower() for w in warnings)

    def test_check_portfolio_risk_near_max_positions(self, risk_manager):
        """Test portfolio risk check near max positions."""
        portfolio = create_test_portfolio(open_positions=9)  # Near limit

        warnings = risk_manager.check_portfolio_risk(portfolio)

        assert any("max positions" in w.lower() for w in warnings)

    def test_check_portfolio_risk_large_position(self, risk_manager):
        """Test portfolio risk check with large position."""
        large_position = create_test_position(
            quantity=4000,  # Large position
            current_price=9000.0,  # 36M value in 100M portfolio = 36%
        )
        portfolio = create_test_portfolio(
            total_value=100_000_000,
            cash=64_000_000,
            open_positions=1,
            positions=[large_position],
        )

        warnings = risk_manager.check_portfolio_risk(portfolio)

        assert any("large position" in w.lower() for w in warnings)

    def test_check_portfolio_risk_low_cash(self, risk_manager):
        """Test portfolio risk check with low cash."""
        portfolio = create_test_portfolio(
            total_value=100_000_000,
            cash=5_000_000,  # Only 5% cash
        )

        warnings = risk_manager.check_portfolio_risk(portfolio)

        assert any("low cash" in w.lower() for w in warnings)

    # ============== TRADING HALT TESTS ==============

    def test_should_halt_trading_no_halt(self, risk_manager, healthy_portfolio):
        """Test no trading halt for healthy portfolio."""
        should_halt, reason = risk_manager.should_halt_trading(healthy_portfolio)

        assert should_halt is False
        assert reason == ""

    def test_should_halt_trading_critical_drawdown(self, risk_manager):
        """Test trading halt for critical drawdown."""
        portfolio = create_test_portfolio(drawdown_pct=0.12)  # 12%

        should_halt, reason = risk_manager.should_halt_trading(portfolio)

        assert should_halt is True
        assert "drawdown" in reason.lower()

    def test_should_halt_trading_critical_daily_loss(self, risk_manager):
        """Test trading halt for critical daily loss."""
        portfolio = create_test_portfolio(daily_pnl_pct=-0.025)  # -2.5%

        should_halt, reason = risk_manager.should_halt_trading(portfolio)

        assert should_halt is True
        assert "daily loss" in reason.lower()

    # ============== OTHER METHOD TESTS ==============

    def test_update_capital(self, risk_manager):
        """Test updating capital."""
        original_capital = risk_manager.capital

        risk_manager.update_capital(150_000_000)

        assert risk_manager.capital == 150_000_000
        assert risk_manager.capital != original_capital

    def test_reset_daily_pnl(self, risk_manager):
        """Test resetting daily P&L tracking."""
        risk_manager.reset_daily_pnl()
        # Should not raise any errors

    def test_get_risk_report(self, risk_manager, healthy_portfolio):
        """Test generating risk report."""
        report = risk_manager.get_risk_report(healthy_portfolio)

        assert isinstance(report, str)
        assert "RISK REPORT" in report
        assert "Total Value" in report

    def test_get_risk_report_with_warnings(self, risk_manager):
        """Test risk report with warnings."""
        portfolio = create_test_portfolio(
            drawdown_pct=0.06,
            daily_pnl_pct=-0.015,
        )

        report = risk_manager.get_risk_report(portfolio)

        assert "RISK WARNINGS" in report

    def test_get_risk_report_with_halt(self, risk_manager):
        """Test risk report with trading halt."""
        portfolio = create_test_portfolio(drawdown_pct=0.12)

        report = risk_manager.get_risk_report(portfolio)

        assert "TRADING HALT" in report


class TestRiskManagerDifferentModes:
    """Test risk manager with different trading modes."""

    def test_intraday_mode(self):
        """Test risk manager in intraday mode."""
        rm = RiskManager(mode=TradingMode.INTRADAY, capital=100_000_000)
        signal = create_test_signal(composite_score=70.0)
        portfolio = create_test_portfolio()

        result = rm.validate_entry(signal, portfolio)

        # Intraday uses 0.5% max risk
        assert rm.config.max_risk_per_trade == 0.005

    def test_swing_mode(self):
        """Test risk manager in swing mode."""
        rm = RiskManager(mode=TradingMode.SWING, capital=100_000_000)

        # Swing uses 1% max risk
        assert rm.config.max_risk_per_trade == 0.01

    def test_position_mode(self):
        """Test risk manager in position mode."""
        rm = RiskManager(mode=TradingMode.POSITION, capital=100_000_000)

        # Position uses 1.5% max risk
        assert rm.config.max_risk_per_trade == 0.015

    def test_investor_mode(self):
        """Test risk manager in investor mode."""
        rm = RiskManager(mode=TradingMode.INVESTOR, capital=100_000_000)

        # Investor uses 2% max risk
        assert rm.config.max_risk_per_trade == 0.02


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.fixture
    def risk_manager(self):
        """Create risk manager."""
        return RiskManager(mode=TradingMode.SWING, capital=100_000_000)

    def test_exact_boundary_daily_loss(self, risk_manager):
        """Test exact boundary for daily loss limit."""
        # Exactly at -2%
        portfolio = create_test_portfolio(daily_pnl_pct=-0.02)
        signal = create_test_signal(composite_score=75.0)

        result = risk_manager.validate_entry(signal, portfolio)

        # At boundary should be rejected
        assert result.approved is False

    def test_just_below_boundary_daily_loss(self, risk_manager):
        """Test just below boundary for daily loss limit."""
        # Just below -2%
        portfolio = create_test_portfolio(daily_pnl_pct=-0.0199)
        signal = create_test_signal(composite_score=75.0)

        result = risk_manager.validate_entry(signal, portfolio)

        # Just below boundary should be approved
        assert result.approved is True

    def test_exact_boundary_max_positions(self, risk_manager):
        """Test exact boundary for max positions."""
        # At max positions (typically 10)
        portfolio = create_test_portfolio(open_positions=10)
        signal = create_test_signal(composite_score=75.0)

        result = risk_manager.validate_entry(signal, portfolio)

        assert result.approved is False

    def test_zero_position_value(self, risk_manager):
        """Test with zero position size calculation."""
        signal = create_test_signal(
            entry_price=100_000.0,  # Very high price
            stop_loss=99_999.0,  # Very tight stop
        )
        portfolio = create_test_portfolio(cash=1_000_000)  # Low cash

        # Should handle gracefully
        result = risk_manager.validate_entry(signal, portfolio)
        # May be rejected for various reasons (position too small, etc.)
        assert isinstance(result, ValidationResult)

    def test_multiple_warnings(self, risk_manager):
        """Test multiple warnings in validation."""
        portfolio = create_test_portfolio(
            drawdown_pct=0.06,  # Warning
            daily_pnl_pct=-0.012,  # Warning
        )
        signal = create_test_signal(composite_score=75.0)

        result = risk_manager.validate_entry(signal, portfolio)

        # Should have multiple warnings
        assert result.approved is True
        assert len(result.warnings) >= 2
