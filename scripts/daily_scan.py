#!/usr/bin/env python3
"""
Daily Scan Script

CLI tool for running daily market scans.
Scans the market for trading signals based on technical analysis
and foreign flow patterns.

Usage:
    python scripts/daily_scan.py --mode swing
    python scripts/daily_scan.py --mode position --symbols BBCA,BBRI,TLKM
    python scripts/daily_scan.py --dry-run
"""

import argparse
import logging
import sys
from datetime import date
from pathlib import Path
from typing import List, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.logging_config import setup_logging
from config.trading_modes import TradingMode
from agents.coordinator import Coordinator, ScanReport
from notifications.telegram_bot import TelegramNotifier, ConsoleNotifier


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run daily market scan for IDX Trading System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run swing mode scan on LQ45 universe
  python scripts/daily_scan.py --mode swing

  # Run position mode scan on specific symbols
  python scripts/daily_scan.py --mode position --symbols BBCA,BBRI,TLKM

  # Dry run (no notifications)
  python scripts/daily_scan.py --mode swing --dry-run

  # Enable verbose output
  python scripts/daily_scan.py --mode swing -v
        """,
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["intraday", "swing", "position", "investor"],
        default="swing",
        help="Trading mode (default: swing)",
    )

    parser.add_argument(
        "--symbols",
        type=str,
        help="Comma-separated list of symbols to scan (default: LQ45)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without sending notifications or executing trades",
    )

    parser.add_argument(
        "--notify",
        action="store_true",
        help="Send Telegram notifications",
    )

    parser.add_argument(
        "--output",
        type=str,
        choices=["console", "json", "markdown"],
        default="console",
        help="Output format (default: console)",
    )

    parser.add_argument(
        "--output-file",
        type=str,
        help="Write output to file",
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    parser.add_argument(
        "--min-score",
        type=float,
        default=60.0,
        help="Minimum signal score (default: 60)",
    )

    parser.add_argument(
        "--max-signals",
        type=int,
        default=10,
        help="Maximum signals per scan (default: 10)",
    )

    return parser.parse_args()


def get_trading_mode(mode_str: str) -> TradingMode:
    """Convert mode string to TradingMode enum."""
    mode_map = {
        "intraday": TradingMode.INTRADAY,
        "swing": TradingMode.SWING,
        "position": TradingMode.POSITION,
        "investor": TradingMode.INVESTOR,
    }
    return mode_map[mode_str]


def format_console_output(report: ScanReport) -> str:
    """Format scan report for console output."""
    lines = [
        "=" * 60,
        f"IDX TRADING SYSTEM - Daily Scan Report",
        "=" * 60,
        f"Date: {report.scan_date}",
        f"Mode: {report.mode.value}",
        f"Result: {report.result.value}",
        f"Execution Time: {report.execution_time_seconds:.2f}s",
        "",
        f"Symbols Scanned: {report.symbols_scanned}",
        f"Signals Generated: {report.signals_generated}",
        f"Signals Approved: {report.signals_approved}",
        "",
    ]

    if report.errors:
        lines.append("Errors:")
        for error in report.errors:
            lines.append(f"  ❌ {error}")
        lines.append("")

    if report.signals:
        lines.append("-" * 60)
        lines.append("APPROVED SIGNALS:")
        lines.append("-" * 60)

        for i, signal in enumerate(report.signals, 1):
            symbol = signal.get("symbol", "UNKNOWN")
            signal_type = signal.get("signal_type", "UNKNOWN")
            entry_price = signal.get("entry_price", signal.get("price", 0))
            score = signal.get("score", signal.get("composite_score", 0))
            stop_loss = signal.get("stop_loss", 0)

            lines.append(f"\n{i}. {symbol} - {signal_type}")
            lines.append(f"   Entry Price: Rp {entry_price:,.0f}")
            lines.append(f"   Score: {score:.1f}")
            lines.append(f"   Stop Loss: Rp {stop_loss:,.0f}")

            # Targets
            targets = signal.get("targets", [])
            if targets:
                lines.append("   Targets:")
                for t in targets[:3]:
                    target_price = t.get("price", 0)
                    target_pct = t.get("pct", 0)
                    lines.append(f"     • Rp {target_price:,.0f} ({target_pct:+.1f}%)")

    else:
        lines.append("-" * 60)
        lines.append("No signals generated.")
        lines.append("-" * 60)

    lines.append("\n" + "=" * 60)
    return "\n".join(lines)


def format_json_output(report: ScanReport) -> str:
    """Format scan report as JSON."""
    import json
    return json.dumps(report.to_dict(), indent=2, default=str)


def format_markdown_output(report: ScanReport) -> str:
    """Format scan report as Markdown."""
    lines = [
        f"# Daily Scan Report - {report.scan_date}",
        "",
        f"**Mode:** {report.mode.value}",
        f"**Result:** {report.result.value}",
        f"**Execution Time:** {report.execution_time_seconds:.2f}s",
        "",
        "## Summary",
        "",
        f"- Symbols Scanned: {report.symbols_scanned}",
        f"- Signals Generated: {report.signals_generated}",
        f"- Signals Approved: {report.signals_approved}",
        "",
    ]

    if report.errors:
        lines.append("## Errors")
        lines.append("")
        for error in report.errors:
            lines.append(f"- {error}")
        lines.append("")

    if report.signals:
        lines.append("## Approved Signals")
        lines.append("")
        lines.append("| # | Symbol | Type | Entry | Score | Stop Loss |")
        lines.append("|---|--------|------|-------|-------|-----------|")

        for i, signal in enumerate(report.signals, 1):
            symbol = signal.get("symbol", "UNKNOWN")
            signal_type = signal.get("signal_type", "UNKNOWN")
            entry_price = signal.get("entry_price", signal.get("price", 0))
            score = signal.get("score", signal.get("composite_score", 0))
            stop_loss = signal.get("stop_loss", 0)

            lines.append(
                f"| {i} | {symbol} | {signal_type} | "
                f"Rp {entry_price:,.0f} | {score:.0f} | "
                f"Rp {stop_loss:,.0f} |"
            )

    return "\n".join(lines)


def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(log_level=log_level)
    logger = logging.getLogger(__name__)

    logger.info(f"Starting daily scan: mode={args.mode}, dry_run={args.dry_run}")

    # Parse symbols
    symbols: Optional[List[str]] = None
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",")]
        logger.info(f"Scanning {len(symbols)} symbols: {symbols}")

    # Get trading mode
    mode = get_trading_mode(args.mode)

    # Create coordinator
    coordinator = Coordinator(
        mode=mode,
        universe=symbols,
        dry_run=args.dry_run,
    )

    # Override config
    coordinator.config.min_signal_score = args.min_score
    coordinator.config.max_signals_per_day = args.max_signals
    coordinator.config.enable_notifications = args.notify and not args.dry_run

    # Run scan
    try:
        report = coordinator.run_daily_scan()
    except Exception as e:
        logger.error(f"Scan failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

    # Format output
    if args.output == "json":
        output = format_json_output(report)
    elif args.output == "markdown":
        output = format_markdown_output(report)
    else:
        output = format_console_output(report)

    # Write to file or console
    if args.output_file:
        with open(args.output_file, "w") as f:
            f.write(output)
        logger.info(f"Output written to {args.output_file}")
    else:
        print(output)

    # Send notifications if enabled
    if args.notify and report.signals:
        try:
            notifier = TelegramNotifier()
            if notifier.config.enabled:
                notifier.send_signals(report.signals)
                logger.info("Telegram notifications sent")
            else:
                logger.warning("Telegram not configured - skipping notifications")
        except Exception as e:
            logger.error(f"Failed to send notifications: {e}")

    # Return exit code based on result
    if report.result.value in ("failed", "partial"):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
