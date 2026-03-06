"""
Backtest Engine Module

Event-driven backtest engine that processes price bars chronologically,
generates signals, manages risk, executes trades, and tracks performance.
"""

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from enum import Enum

from config.settings import settings
from config.trading_modes import TradingMode, MODE_CONFIGS
from core.data.models import (
    OHLCV,
    Signal,
    Trade,
    Position,
    OrderSide,
    SignalType,
    SetupType,
    FlowSignal,
)
from core.analysis.technical import TechnicalAnalyzer
from core.signals.signal_generator import SignalGenerator
from core.risk.risk_manager import RiskManager
from core.risk.position_sizer import PositionSizer
from core.risk.empirical_kelly import EmpiricalKelly
from core.risk.pattern_matcher import PatternMatcher
from core.execution.paper_trader import PaperTrader
from core.portfolio.portfolio_manager import PortfolioManager

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Configuration for backtest run.

    Attributes:
        start_date: Backtest start date.
        end_date: Backtest end date.
        initial_capital: Starting capital in IDR.
        trading_mode: Trading mode (intraday, swing, position, investor).
        commission_buy: Buy commission rate.
        commission_sell: Sell commission rate.
        slippage: Default slippage rate.
        position_sizing: Position sizing method ('fixed', 'kelly', 'empirical_kelly').
        max_positions: Maximum concurrent positions.
        risk_per_trade: Risk per trade as decimal.
    """

    start_date: date
    end_date: date
    initial_capital: float = 100_000_000
    trading_mode: TradingMode = TradingMode.SWING
    commission_buy: float = 0.0015  # 0.15%
    commission_sell: float = 0.0025  # 0.25%
    slippage: float = 0.001  # 0.1%
    position_sizing: str = "empirical_kelly"
    max_positions: int = 5
    risk_per_trade: float = 0.01  # 1%
    use_calibration: bool = True
    use_pattern_matching: bool = True


@dataclass
class BacktestResult:
    """Result of a backtest run.

    Attributes:
        config: Backtest configuration used.
        trades: List of all completed trades.
        equity_curve: Daily equity values.
        metrics: Calculated performance metrics.
        signals: List of all generated signals.
        positions_history: History of position changes.
    """

    config: BacktestConfig
    trades: List[Trade] = field(default_factory=list)
    equity_curve: List[Dict[str, Any]] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    signals: List[Signal] = field(default_factory=list)
    positions_history: List[Dict[str, Any]] = field(default_factory=list)
    final_capital: float = 0.0
    total_return_pct: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "config": {
                "start_date": self.config.start_date.isoformat(),
                "end_date": self.config.end_date.isoformat(),
                "initial_capital": self.config.initial_capital,
                "trading_mode": self.config.trading_mode.value,
                "position_sizing": self.config.position_sizing,
            },
            "metrics": self.metrics,
            "final_capital": self.final_capital,
            "total_return_pct": self.total_return_pct,
            "total_trades": len(self.trades),
            "winning_trades": len([t for t in self.trades if t.net_pnl > 0]),
            "losing_trades": len([t for t in self.trades if t.net_pnl <= 0]),
        }


class BacktestEngine:
    """Event-driven backtest engine.

    Processes historical data chronologically, generating signals,
    managing risk, executing trades, and tracking performance.

    Example:
        engine = BacktestEngine(config)
        result = engine.run(bars_dict)
        print(f"Total return: {result.total_return_pct:.2f}%")
    """

    def __init__(self, config: BacktestConfig) -> None:
        """Initialize backtest engine.

        Args:
            config: Backtest configuration.
        """
        self.config = config

        # Initialize components
        self.technical_analyzer = TechnicalAnalyzer()

        # Trading mode config - get this first for signal generator
        self.mode_config = MODE_CONFIGS.get(config.trading_mode, MODE_CONFIGS[TradingMode.SWING])

        self.signal_generator = SignalGenerator(config=self.mode_config)
        self.risk_manager = RiskManager()
        self.position_sizer = PositionSizer(
            capital=config.initial_capital,
            config=self.mode_config,
        )
        self.kelly = EmpiricalKelly()
        self.pattern_matcher = PatternMatcher()
        self.portfolio = PortfolioManager(initial_capital=config.initial_capital)
        self.trader = PaperTrader(slippage=config.slippage)

        # Data storage
        self.bars_by_symbol: Dict[str, List[Bar]] = {}
        self.current_date: Optional[date] = None

        # Historical trades for pattern matching
        self.historical_trades: List[Trade] = []

        # Results
        self.trades: List[Trade] = []
        self.signals: List[Signal] = []
        self.equity_curve: List[Dict[str, Any]] = []

    def run(
        self,
        bars_by_symbol: Dict[str, List[OHLCV]],
        historical_trades: Optional[List[Trade]] = None,
    ) -> BacktestResult:
        """Run backtest on historical data.

        Args:
            bars_by_symbol: Dictionary of symbol to list of bars.
            historical_trades: Optional historical trades for pattern matching.

        Returns:
            BacktestResult with all trades and metrics.
        """
        self.bars_by_symbol = bars_by_symbol
        self.historical_trades = historical_trades or []

        # Load historical trades into pattern matcher
        if self.historical_trades:
            for trade in self.historical_trades:
                self.pattern_matcher.add_trade(trade)

        # Get all unique dates
        all_dates = set()
        for bars in bars_by_symbol.values():
            for bar in bars:
                all_dates.add(bar.date)

        sorted_dates = sorted(all_dates)

        # Filter to date range
        sorted_dates = [
            d for d in sorted_dates
            if self.config.start_date <= d <= self.config.end_date
        ]

        logger.info(
            f"Running backtest from {self.config.start_date} to {self.config.end_date} "
            f"({len(sorted_dates)} trading days, {len(bars_by_symbol)} symbols)"
        )

        # Process each date
        for current_date in sorted_dates:
            self.current_date = current_date
            self._process_day(current_date)

        # Calculate final metrics
        result = self._build_result()

        logger.info(
            f"Backtest complete: {len(self.trades)} trades, "
            f"Final capital: {result.final_capital:,.0f}, "
            f"Return: {result.total_return_pct:.2f}%"
        )

        return result

    def _process_day(self, current_date: date) -> None:
        """Process a single trading day.

        Args:
            current_date: Date to process.
        """
        # Get OHLCV for today
        today_bars: Dict[str, OHLCV] = {}
        for symbol, bars in self.bars_by_symbol.items():
            for bar in bars:
                if bar.date == current_date:
                    today_bars[symbol] = bar
                    break

        if not today_bars:
            return

        # Update portfolio prices
        prices = {symbol: bar.close for symbol, bar in today_bars.items()}
        self.portfolio.update_prices(prices)
        self.trader.update_position_prices(prices)

        # Record equity
        self._record_equity(current_date)

        # Check exits for existing positions
        self._check_exits(current_date, today_bars)

        # Generate new signals
        if self.portfolio.get_position_count() < self.config.max_positions:
            self._generate_signals(current_date, today_bars)

    def _record_equity(self, current_date: date) -> None:
        """Record daily equity.

        Args:
            current_date: Current date.
        """
        state = self.portfolio.get_state()
        self.equity_curve.append({
            "date": current_date.isoformat(),
            "equity": state.total_value,
            "cash": state.cash,
            "positions_value": state.positions_value,
            "drawdown": state.drawdown,
            "drawdown_pct": state.drawdown_pct,
            "positions": state.open_positions,
        })

    def _check_exits(self, current_date: date, today_bars: Dict[str, OHLCV]) -> None:
        """Check exit conditions for existing positions.

        Args:
            current_date: Current date.
            today_bars: Today's bars by symbol.
        """
        # Get list of positions (PortfolioManager stores in self.positions dict)
        positions = list(self.portfolio.positions.values())

        for position in positions:
            symbol = position.symbol
            if symbol not in today_bars:
                continue

            bar = today_bars[symbol]
            should_exit, reason = self._should_exit_position(position, bar)

            if should_exit:
                self._execute_exit(symbol, bar.close, current_date, reason)

    def _should_exit_position(self, position: Position, bar: OHLCV) -> tuple:
        """Determine if position should be exited.

        Args:
            position: Position to check.
            bar: Current bar.

        Returns:
            Tuple of (should_exit, reason).
        """
        # Stop loss check
        if bar.low <= position.stop_loss:
            return True, f"Stop loss hit at {position.stop_loss:.0f}"

        # Target checks
        if bar.high >= position.target_1:
            # Partial exit at target 1 (for now, full exit)
            return True, f"Target 1 hit at {position.target_1:.0f}"

        # Max holding days check
        if position.days_held >= self.mode_config.max_hold_days:
            return True, f"Max holding days ({self.mode_config.max_hold_days}) reached"

        # Calibration-based exit (if enabled)
        if self.config.use_calibration and hasattr(position, 'signal_score'):
            from research.calibration import CalibrationSurface
            # Simplified check - in real system would use built calibration surface
            if position.days_held >= 5 and position.unrealized_pnl_pct > 3:
                return True, "Calibration-based exit: edge decay"

        return False, ""

    def _generate_signals(self, current_date: date, today_bars: Dict[str, OHLCV]) -> None:
        """Generate trading signals for today.

        Args:
            current_date: Current date.
            today_bars: Today's bars by symbol.
        """
        for symbol, bar in today_bars.items():
            # Skip if already have position
            if self.portfolio.has_position(symbol):
                continue

            # Get historical bars for this symbol
            symbol_bars = self.bars_by_symbol.get(symbol, [])

            # Need enough bars for analysis
            if len(symbol_bars) < 50:
                continue

            # Filter to bars up to current date
            historical = [b for b in symbol_bars if b.date <= current_date]
            if len(historical) < 50:
                continue

            # Generate signal
            signal = self._generate_signal(symbol, historical, bar)
            if signal and signal.signal_type == SignalType.BUY:
                self.signals.append(signal)
                self._execute_entry(signal, bar, current_date)

    def _generate_signal(
        self,
        symbol: str,
        historical_bars: List[OHLCV],
        current_bar: OHLCV,
    ) -> Optional[Signal]:
        """Generate a trading signal.

        Args:
            symbol: Stock symbol.
            historical_bars: Historical bars up to current.
            current_bar: Current bar.

        Returns:
            Signal if generated, None otherwise.
        """
        # Technical analysis
        tech_list = self.technical_analyzer.calculate(historical_bars)
        if not tech_list:
            return None

        # Get latest indicators
        tech = tech_list[-1]

        # Calculate score
        score_result = self.technical_analyzer.calculate_score(tech)
        score = score_result.score

        # Score check
        if score < self.mode_config.min_score:
            return None

        # RSI check
        if tech.rsi is not None:
            if tech.rsi > self.mode_config.rsi_overbought:
                return None

        # Volume check
        if tech.volume_ratio is not None:
            if tech.volume_ratio < self.mode_config.min_volume_ratio:
                return None

        # Create signal
        signal = Signal(
            symbol=symbol,
            signal_type=SignalType.BUY,
            composite_score=score,
            entry_price=current_bar.close,
            timestamp=datetime.combine(current_bar.date, datetime.min.time()),
            setup_type=SetupType.MOMENTUM,
        )

        return signal

    def _execute_entry(self, signal: Signal, bar: OHLCV, current_date: date) -> None:
        """Execute entry trade.

        Args:
            signal: Entry signal.
            bar: Current bar.
            current_date: Current date.
        """
        # Calculate position size
        position_size = self._calculate_position_size(signal, bar)

        if position_size <= 0:
            return

        # Check buying power
        buying_power = self.portfolio.get_buying_power()
        position_value = position_size * bar.close

        if position_value > buying_power:
            # Reduce size
            position_size = int(buying_power / bar.close / 100) * 100
            if position_size < 100:
                return

        # Execute trade
        result = self.trader.buy(
            symbol=signal.symbol,
            quantity=position_size,
            current_market_price=bar.close,
        )

        if result.success and result.position:
            # Add to portfolio
            self.portfolio.open_position(result.position)

            logger.debug(
                f"BUY {signal.symbol}: {position_size} shares @ {bar.close:,.0f} "
                f"(Score: {signal.signal_score})"
            )

    def _execute_exit(
        self,
        symbol: str,
        price: float,
        date: date,
        reason: str,
    ) -> None:
        """Execute exit trade.

        Args:
            symbol: Symbol to exit.
            price: Exit price.
            date: Exit date.
            reason: Exit reason.
        """
        position = self.portfolio.get_position(symbol)
        if not position:
            return

        result = self.trader.sell(
            symbol=symbol,
            quantity=position.quantity,
            current_market_price=price,
            exit_reason=reason,
        )

        if result.success and result.trade:
            # Remove from portfolio
            self.portfolio.close_position(symbol, price, date, reason)

            # Record trade
            self.trades.append(result.trade)

            # Add to pattern matcher for future reference
            self.pattern_matcher.add_trade(result.trade)

            logger.debug(
                f"SELL {symbol}: {position.quantity} shares @ {price:,.0f} "
                f"P&L: {result.trade.net_pnl:,.0f} ({result.trade.return_pct:.2f}%) - {reason}"
            )

    def _calculate_position_size(self, signal: Signal, bar: OHLCV) -> int:
        """Calculate position size based on config.

        Args:
            signal: Entry signal.
            bar: Current bar.

        Returns:
            Number of shares to trade.
        """
        if self.config.position_sizing == "fixed":
            # Fixed percentage of capital
            capital = self.portfolio.get_total_value()
            risk_amount = capital * self.config.risk_per_trade
            position_value = capital * self.mode_config.max_position_pct

        elif self.config.position_sizing == "kelly":
            # Standard Kelly (use fixed for safety)
            capital = self.portfolio.get_total_value()
            position_value = capital * 0.10  # 10% max

        elif self.config.position_sizing == "empirical_kelly":
            # Empirical Kelly with pattern matching
            if self.config.use_pattern_matching and self.historical_trades:
                # Find similar patterns
                pattern_result = self.pattern_matcher.get_best_matches(
                    signal_score=signal.signal_score,
                    min_matches=10,
                )

                if pattern_result and pattern_result.is_significant:
                    # Calculate Kelly from pattern
                    kelly_result = self.kelly.calculate_from_pattern(pattern_result)

                    if kelly_result.is_valid:
                        position_value = kelly_result.get_position_pct(
                            self.portfolio.get_total_value()
                        )
                    else:
                        # Conservative fallback
                        position_value = self.portfolio.get_total_value() * 0.05
                else:
                    # Not enough pattern data
                    position_value = self.portfolio.get_total_value() * 0.05
            else:
                # No pattern matching
                position_value = self.portfolio.get_total_value() * 0.05

        else:
            # Default to fixed
            capital = self.portfolio.get_total_value()
            position_value = capital * self.mode_config.max_position_pct

        # Calculate shares (must be multiple of lot size)
        shares = int(position_value / bar.close / 100) * 100

        # Ensure minimum lot
        if shares < 100:
            shares = 0

        return shares

    def _build_result(self) -> BacktestResult:
        """Build final backtest result.

        Returns:
            BacktestResult with all data and metrics.
        """
        # Calculate metrics
        from backtest.metrics import calculate_metrics

        metrics = calculate_metrics(
            trades=self.trades,
            equity_curve=self.equity_curve,
            initial_capital=self.config.initial_capital,
        )

        # Get final values
        final_capital = self.config.initial_capital
        if self.equity_curve:
            final_capital = self.equity_curve[-1]["equity"]

        total_return = (final_capital - self.config.initial_capital) / self.config.initial_capital * 100

        result = BacktestResult(
            config=self.config,
            trades=self.trades,
            equity_curve=self.equity_curve,
            metrics=metrics,
            signals=self.signals,
            final_capital=final_capital,
            total_return_pct=total_return,
        )

        return result


def run_backtest(
    bars_by_symbol: Dict[str, List[OHLCV]],
    start_date: date,
    end_date: date,
    initial_capital: float = 100_000_000,
    trading_mode: TradingMode = TradingMode.SWING,
    **kwargs,
) -> BacktestResult:
    """Convenience function to run a backtest.

    Args:
        bars_by_symbol: Dictionary of symbol to bars.
        start_date: Start date.
        end_date: End date.
        initial_capital: Initial capital.
        trading_mode: Trading mode.
        **kwargs: Additional config options.

    Returns:
        BacktestResult.
    """
    config = BacktestConfig(
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
        trading_mode=trading_mode,
        **kwargs,
    )

    engine = BacktestEngine(config)
    return engine.run(bars_by_symbol)
