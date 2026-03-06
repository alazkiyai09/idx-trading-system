"""
Coordinator Module

Main orchestrator for the IDX Trading System.
Coordinates all components for daily scanning, signal generation,
and trade execution workflows.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, date
from enum import Enum
from typing import Dict, List, Optional, Any, Callable

from config.settings import settings
from config.trading_modes import TradingMode, ModeConfig

logger = logging.getLogger(__name__)


class ScanResult(Enum):
    """Result of a scan operation."""
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    NO_SIGNALS = "no_signals"


@dataclass
class ScanReport:
    """Report from a daily scan."""
    scan_date: date
    mode: TradingMode
    result: ScanResult
    signals_generated: int = 0
    signals_approved: int = 0
    symbols_scanned: int = 0
    errors: List[str] = field(default_factory=list)
    signals: List[Dict[str, Any]] = field(default_factory=list)
    execution_time_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "scan_date": self.scan_date.isoformat(),
            "mode": self.mode.value,
            "result": self.result.value,
            "signals_generated": self.signals_generated,
            "signals_approved": self.signals_approved,
            "symbols_scanned": self.symbols_scanned,
            "errors": self.errors,
            "signals": self.signals,
            "execution_time_seconds": self.execution_time_seconds,
        }


@dataclass
class CoordinatorConfig:
    """Configuration for the coordinator."""
    universe: List[str] = field(default_factory=list)
    mode: TradingMode = TradingMode.SWING
    dry_run: bool = True
    max_signals_per_day: int = 10
    min_signal_score: float = 60.0
    enable_notifications: bool = False


class Coordinator:
    """Main orchestrator for the trading system.

    Coordinates all components:
    - Data fetching and updates
    - Technical analysis
    - Signal generation
    - Risk validation
    - Portfolio management
    - Notifications

    Example:
        coordinator = Coordinator(mode=TradingMode.SWING)
        report = coordinator.run_daily_scan()
        print(f"Generated {report.signals_generated} signals")
    """

    def __init__(
        self,
        mode: TradingMode = TradingMode.SWING,
        universe: Optional[List[str]] = None,
        dry_run: bool = True,
        config: Optional[CoordinatorConfig] = None,
    ) -> None:
        """Initialize coordinator.

        Args:
            mode: Trading mode (intraday, swing, position, investor).
            universe: List of symbols to scan. If None, uses LQ45.
            dry_run: If True, no actual trades are executed.
            config: Optional configuration object.
        """
        self.config = config or CoordinatorConfig(
            universe=universe or [],
            mode=mode,
            dry_run=dry_run,
        )
        self.mode = mode
        self.dry_run = dry_run

        # Initialize components (lazy loading to avoid circular imports)
        self._data_manager = None
        self._technical_analyzer = None
        self._flow_analyzer = None
        self._signal_generator = None
        self._risk_manager = None
        self._portfolio_manager = None
        self._notifier = None

        # State
        self.approved_signals: List[Dict[str, Any]] = []
        self._initialized = False

        logger.info(f"Coordinator initialized: mode={mode.value}, dry_run={dry_run}")

    def _initialize_components(self) -> None:
        """Initialize all components (lazy loading)."""
        if self._initialized:
            return

        try:
            # Import and initialize components
            from core.data.scraper import IDXScraper
            from core.data.foreign_flow import ForeignFlowAnalyzer
            from core.analysis.technical import TechnicalAnalyzer
            from core.signals.signal_generator import SignalGenerator
            from core.risk.risk_manager import RiskManager

            self._data_manager = IDXScraper()
            self._flow_analyzer = ForeignFlowAnalyzer()
            self._technical_analyzer = TechnicalAnalyzer()
            self._signal_generator = SignalGenerator(
                trading_mode=self.config.mode,
                min_score=self.config.min_signal_score,
            )
            self._risk_manager = RiskManager(trading_mode=self.config.mode)

            # Try to initialize notifier if available
            try:
                from notifications.telegram_bot import TelegramNotifier
                if self.config.enable_notifications:
                    self._notifier = TelegramNotifier()
            except ImportError:
                logger.warning("Telegram notifier not available")

            self._initialized = True
            logger.info("All components initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise

    @property
    def universe(self) -> List[str]:
        """Get the trading universe."""
        if self.config.universe:
            return self.config.universe
        # Default to LQ45
        return self._get_lq45_symbols()

    def _get_lq45_symbols(self) -> List[str]:
        """Get LQ45 symbol list."""
        # Standard LQ45 symbols
        return [
            "AALI", "ADRO", "ANTM", "ASII", "BBCA", "BBNI", "BBRI", "BMRI",
            "CPIN", "EXCL", "ICBP", "INCO", "INDF", "INTP", "ITMG", "JSMR",
            "KLBF", "MNCN", "PGAS", "PTBA", "PTPP", "SIDO", "SMGR", "SRTG",
            "TLKM", "UNTR", "UNVR", "WIKA", "WSKT", "BRIS", "BUKA", "EMTK",
            "GGRM", "HMSP", "INDF", "INKP", "MDKA", "MIKA", "TBIG", "TOWR",
        ]

    def run_daily_scan(
        self,
        symbols: Optional[List[str]] = None,
        mode: Optional[TradingMode] = None,
    ) -> ScanReport:
        """Run a complete daily scan.

        This is the main entry point for daily operations:
        1. Fetch latest price data
        2. Update foreign flow data
        3. Calculate technical indicators
        4. Generate signals for each symbol
        5. Validate with risk manager
        6. Return approved signals

        Args:
            symbols: Optional list of symbols to scan. Uses universe if None.
            mode: Optional trading mode override.

        Returns:
            ScanReport with results and approved signals.
        """
        start_time = datetime.now()
        scan_mode = mode or self.config.mode
        scan_symbols = symbols or self.universe

        report = ScanReport(
            scan_date=date.today(),
            mode=scan_mode,
            result=ScanResult.SUCCESS,
            symbols_scanned=len(scan_symbols),
        )

        try:
            self._initialize_components()
            self.approved_signals = []

            logger.info(f"Starting daily scan: mode={scan_mode.value}, symbols={len(scan_symbols)}")

            # Step 1: Update data
            self._update_data(scan_symbols, report)

            # Step 2: Scan each symbol
            for symbol in scan_symbols:
                try:
                    signal = self._scan_symbol(symbol, scan_mode)
                    if signal:
                        report.signals_generated += 1

                        # Validate with risk manager
                        if self._validate_signal(signal, report):
                            report.signals_approved += 1
                            report.signals.append(signal)
                            self.approved_signals.append(signal)

                except Exception as e:
                    error_msg = f"Error scanning {symbol}: {e}"
                    logger.error(error_msg)
                    report.errors.append(error_msg)

            # Step 3: Limit signals
            if len(report.signals) > self.config.max_signals_per_day:
                report.signals = report.signals[:self.config.max_signals_per_day]
                report.signals_approved = len(report.signals)

            # Step 4: Determine result
            if report.errors and not report.signals:
                report.result = ScanResult.FAILED
            elif report.errors:
                report.result = ScanResult.PARTIAL
            elif not report.signals:
                report.result = ScanResult.NO_SIGNALS

            # Step 5: Send notifications
            if self._notifier and report.signals:
                self._send_notifications(report)

        except Exception as e:
            logger.error(f"Scan failed: {e}")
            report.result = ScanResult.FAILED
            report.errors.append(str(e))

        report.execution_time_seconds = (datetime.now() - start_time).total_seconds()
        logger.info(f"Scan complete: result={report.result.value}, signals={report.signals_approved}")

        return report

    def _update_data(
        self,
        symbols: List[str],
        report: ScanReport
    ) -> None:
        """Update data for all symbols."""
        logger.info("Updating price and flow data...")

        try:
            # Fetch latest prices
            if self._data_manager:
                self._data_manager.fetch_current(symbols)
        except Exception as e:
            report.errors.append(f"Data update failed: {e}")
            logger.error(f"Data update failed: {e}")

    def _scan_symbol(
        self,
        symbol: str,
        mode: TradingMode
    ) -> Optional[Dict[str, Any]]:
        """Scan a single symbol for signals.

        Args:
            symbol: Symbol to scan.
            mode: Trading mode.

        Returns:
            Signal dictionary if generated, None otherwise.
        """
        try:
            # Get historical data
            if self._data_manager:
                history = self._data_manager.fetch_historical(symbol, period="3mo")
            else:
                # Use mock data if data manager not available
                history = self._get_mock_data(symbol)

            if history is None or len(history) < 20:
                logger.debug(f"Insufficient data for {symbol}")
                return None

            # Calculate technical indicators
            indicators = self._technical_analyzer.calculate(history)

            # Analyze foreign flow
            flow_data = None
            if self._flow_analyzer:
                try:
                    flow_data = self._flow_analyzer.analyze(symbol)
                except Exception:
                    pass  # Flow analysis is optional

            # Generate signal
            signal = self._signal_generator.generate(
                symbol=symbol,
                data=history,
                indicators=indicators,
                flow_data=flow_data,
            )

            return signal

        except Exception as e:
            logger.error(f"Error scanning {symbol}: {e}")
            raise

    def _validate_signal(
        self,
        signal: Dict[str, Any],
        report: ScanReport
    ) -> bool:
        """Validate a signal with risk manager.

        Args:
            signal: Signal to validate.
            report: Scan report for error tracking.

        Returns:
            True if signal is approved, False otherwise.
        """
        if not self._risk_manager:
            return True  # Approve if no risk manager

        try:
            # Create a mock portfolio for validation
            portfolio_state = {
                "total_value": 1_000_000_000,  # 1 billion IDR
                "cash": 500_000_000,
                "positions": [],
                "daily_pnl": 0,
            }

            result = self._risk_manager.validate_entry(
                signal=signal,
                portfolio=portfolio_state,
            )

            if result.approved:
                logger.info(f"Signal approved: {signal['symbol']}")
                return True
            else:
                logger.info(f"Signal rejected: {signal['symbol']} - {result.veto_reason}")
                return False

        except Exception as e:
            report.errors.append(f"Risk validation error: {e}")
            logger.error(f"Risk validation error: {e}")
            return False

    def _send_notifications(self, report: ScanReport) -> None:
        """Send notifications for approved signals."""
        if not self._notifier:
            return

        try:
            self._notifier.send_signals(report.signals)
            logger.info("Notifications sent successfully")
        except Exception as e:
            logger.error(f"Failed to send notifications: {e}")

    def _get_mock_data(self, symbol: str) -> List[Dict[str, Any]]:
        """Get mock data for testing.

        Args:
            symbol: Symbol name.

        Returns:
            List of mock OHLCV data.
        """
        import random
        from datetime import timedelta

        base_price = random.uniform(1000, 10000)
        data = []

        # Use a safe starting date in the past
        end_date = date.today()

        for i in range(100):
            change = random.uniform(-0.03, 0.03)
            base_price *= (1 + change)

            # Calculate date safely (go back from today)
            days_ago = 100 - i
            try:
                data_date = end_date - timedelta(days=days_ago)
            except ValueError:
                # Fallback if date calculation fails
                data_date = end_date - timedelta(days=i)

            data.append({
                "date": data_date,
                "open": base_price * random.uniform(0.98, 1.02),
                "high": base_price * random.uniform(1.00, 1.05),
                "low": base_price * random.uniform(0.95, 1.00),
                "close": base_price,
                "volume": random.randint(100000, 10000000),
            })

        return data

    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get current portfolio summary.

        Returns:
            Portfolio summary dictionary.
        """
        if self._portfolio_manager:
            return self._portfolio_manager.get_summary()

        return {
            "total_value": 0,
            "cash": 0,
            "positions": [],
            "daily_pnl": 0,
            "unrealized_pnl": 0,
        }

    def execute_signals(
        self,
        signals: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Execute approved signals.

        Args:
            signals: List of approved signals to execute.

        Returns:
            List of execution results.
        """
        if self.dry_run:
            logger.info("Dry run mode - signals not executed")
            return [{"status": "dry_run", "signal": s} for s in signals]

        results = []
        for signal in signals:
            try:
                result = self._execute_single(signal)
                results.append(result)
            except Exception as e:
                logger.error(f"Execution failed for {signal['symbol']}: {e}")
                results.append({
                    "status": "failed",
                    "signal": signal,
                    "error": str(e),
                })

        return results

    def _execute_single(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single signal."""
        # This would integrate with paper trader or live execution
        logger.info(f"Executing signal: {signal['symbol']} {signal['signal_type']}")

        return {
            "status": "executed",
            "signal": signal,
            "execution_time": datetime.now().isoformat(),
        }

    def run_end_of_day(self) -> Dict[str, Any]:
        """Run end-of-day workflow.

        Returns:
            EOD report dictionary.
        """
        logger.info("Running end-of-day workflow")

        report = {
            "date": date.today().isoformat(),
            "portfolio_summary": self.get_portfolio_summary(),
            "positions_closed": 0,
            "risk_metrics": {},
        }

        # Check risk limits
        if self._risk_manager:
            portfolio = self.get_portfolio_summary()
            risk_report = self._risk_manager.get_risk_report(portfolio)
            report["risk_metrics"] = risk_report

        return report
