"""
Macro correlation analysis module.

Analyzes correlations between stock prices and commodity prices (Gold, Silver, Oil).
Provides signal generation based on macro correlations with statistical significance checks.
"""

import logging
from datetime import datetime, date, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

from core.data.database import DatabaseManager

logger = logging.getLogger(__name__)


# Constants for signal generation
MIN_CORRELATION_THRESHOLD = 0.3  # Minimum absolute correlation to consider significant
MIN_PRICE_CHANGE_PCT = 1.0  # Minimum 1% price change to generate signal
COMMODITY_STALENESS_DAYS = 3  # Consider data stale if older than 3 days
MIN_OVERLAPPING_DAYS = 20  # Minimum overlapping trading days for valid correlation


class MacroCorrelationAnalyzer:
    """Analyze correlations between stocks and commodities."""

    COMMODITIES = ["GOLD", "SILVER", "OIL"]

    # ⚠️ DEPRECATION WARNING: These are HEURISTIC estimates with NO empirical basis.
    # These values are theoretical and should NOT be used for trading decisions.
    # TODO: Replace with computed correlations from historical data.
    # For now, only use for qualitative understanding, not quantitative signals.
    SECTOR_COMMODITY_IMPACT = {
        # Mining sector positively impacted by Gold
        "MINING": {"GOLD": 0.7, "OIL": 0.3, "SILVER": 0.1},
        # Energy sector positively impacted by Oil
        "ENERGY": {"GOLD": -0.2, "OIL": 0.8, "SILVER": 0.0},
        # Banking sector often inversely impacted by commodities
        "BANKING": {"GOLD": -0.3, "OIL": -0.4, "SILVER": -0.2},
        # Consumer goods often inversely impacted by commodities
        "CONSUMER": {"GOLD": -0.2, "OIL": -0.3, "SILVER": -0.1},
        # Infrastructure positively impacted by commodities
        "INFRASTRUCTURE": {"GOLD": 0.1, "OIL": 0.2, "SILVER": 0.0},
        # Default for unknown sectors - neutral impact
        "DEFAULT": {"GOLD": 0.0, "OIL": 0.0, "SILVER": 0.0},
    }

    def __init__(self, db_manager: DatabaseManager):
        """Initialize the correlation analyzer.

        Args:
            db_manager: DatabaseManager instance
        """
        self.db = db_manager
        self._correlations_cache: Dict[str, Tuple[float, float, datetime]] = {}
        self._cache_timestamps: Dict[str, datetime] = {}  # Per-key timestamps
        self._cache_ttl_seconds = 3600  # 1 hour

    def _get_cache_key(self, symbol: str, commodity: str) -> Optional[Tuple[float, float]]:
        """Get cached correlation data if not expired."""
        cache_key = f"{symbol}_{commodity}"

        if cache_key not in self._cache_timestamps:
            return None

        cached_time = self._cache_timestamps[cache_key]
        if datetime.now(timezone.utc) - cached_time < timedelta(seconds=self._cache_ttl_seconds):
            cached = self._correlations_cache.get(cache_key)
            if cached:
                return (cached[0], cached[1])  # Return only correlations, not timestamp
        return None

    def _set_cache(
        self, symbol: str, commodity: str, corr_30d: float, corr_90d: float
    ) -> None:
        """Cache correlation data with individual timestamp."""
        cache_key = f"{symbol}_{commodity}"
        now = datetime.now(timezone.utc)
        self._correlations_cache[cache_key] = (corr_30d, corr_90d, now)
        self._cache_timestamps[cache_key] = now
        logger.debug(f"Cached correlation for {symbol}-{commodity}: 30d={corr_30d:.3f}, 90d={corr_90d:.3f}")

    def _check_commodity_staleness(self, commodity: str, prices: List) -> Tuple[bool, Optional[date]]:
        """Check if commodity data is stale.

        Args:
            commodity: Commodity name
            prices: List of commodity price records

        Returns:
            Tuple of (is_stale, latest_date)
        """
        if not prices:
            return True, None

        latest_date = prices[-1].date
        days_old = (date.today() - latest_date).days

        is_stale = days_old > COMMODITY_STALENESS_DAYS
        if is_stale:
            logger.warning(
                f"{commodity} data is stale: latest date {latest_date} is {days_old} days old "
                f"(threshold: {COMMODITY_STALENESS_DAYS} days)"
            )

        return is_stale, latest_date

    def compute_correlations(
        self,
        symbol: str,
        window_days: int = 90,
    ) -> Dict[str, Any]:
        """Compute rolling correlations between stock and commodities.

        Args:
            symbol: Stock symbol
            window_days: Rolling window in days (default 90)

        Returns:
            Dict mapping commodity to correlation value and metadata
        """
        result = {}

        # Check cache first
        all_cached = True
        for commodity in self.COMMODITIES:
            cached = self._get_cache_key(symbol, commodity)
            if cached:
                result[commodity] = {
                    "correlation_30d": cached[0],
                    "correlation_90d": cached[1],
                    "is_significant": abs(cached[0]) >= MIN_CORRELATION_THRESHOLD,
                    "from_cache": True
                }
            else:
                all_cached = False

        # If all cached, return early
        if all_cached:
            return result

        # Get stock prices
        stock_prices = self._get_stock_prices(symbol, days=window_days)
        if not stock_prices:
            logger.warning(f"No stock data for {symbol}")
            return {
                commodity: {
                    "correlation_30d": None,
                    "correlation_90d": None,
                    "error": "No stock data available"
                }
                for commodity in self.COMMODITIES
            }

        # Get commodity prices
        start_date = date.today() - timedelta(days=window_days + 30)  # Extra buffer for alignment

        for commodity in self.COMMODITIES:
            # Skip if already cached
            if commodity in result:
                continue

            commodity_prices = self.db.get_commodity_prices(commodity, start_date=start_date)
            if not commodity_prices:
                logger.warning(f"No {commodity} data available")
                result[commodity] = {
                    "correlation_30d": None,
                    "correlation_90d": None,
                    "error": "No commodity data available"
                }
                continue

            # Check staleness
            is_stale, latest_date = self._check_commodity_staleness(commodity, commodity_prices)

            # Align dates
            stock_dates = set(p["date"] for p in stock_prices)
            commodity_dates = set(p.date for p in commodity_prices)
            common_dates = stock_dates.intersection(commodity_dates)

            if len(common_dates) < MIN_OVERLAPPING_DAYS:
                logger.warning(
                    f"Insufficient overlapping dates for {symbol} and {commodity}: "
                    f"{len(common_dates)} (need {MIN_OVERLAPPING_DAYS})"
                )
                result[commodity] = {
                    "correlation_30d": None,
                    "correlation_90d": None,
                    "error": f"Insufficient overlapping data ({len(common_dates)} days)",
                    "is_stale": is_stale
                }
                continue

            # Filter to common dates and sort
            stock_dict = {p["date"]: p["close"] for p in stock_prices}
            commodity_dict = {p.date: p.close for p in commodity_prices}

            common_dates_sorted = sorted(common_dates)
            aligned_stock = [stock_dict[d] for d in common_dates_sorted]
            aligned_commodity = [commodity_dict[d] for d in common_dates_sorted]

            # Compute returns on aligned data
            stock_series = pd.Series(aligned_stock)
            commodity_series = pd.Series(aligned_commodity)

            stock_returns = stock_series.pct_change().dropna()
            commodity_returns = commodity_series.pct_change().dropna()

            # Ensure same length after pct_change
            min_len = min(len(stock_returns), len(commodity_returns))
            if min_len < 10:
                logger.warning(f"Insufficient return data for {symbol} and {commodity}: {min_len} points")
                result[commodity] = {
                    "correlation_30d": None,
                    "correlation_90d": None,
                    "error": "Insufficient return data"
                }
                continue

            stock_returns = stock_returns.iloc[-min_len:]
            commodity_returns = commodity_returns.iloc[-min_len:]

            # Compute correlations for different windows
            # 30-day correlation (last 30 trading days)
            window_30d = min(30, len(stock_returns))
            corr_30d = stock_returns.iloc[-window_30d:].corr(commodity_returns.iloc[-window_30d:])

            # 90-day correlation (all available data in window)
            corr_90d = stock_returns.corr(commodity_returns)

            # Handle NaN
            corr_30d = float(corr_30d) if not pd.isna(corr_30d) else 0.0
            corr_90d = float(corr_90d) if not pd.isna(corr_90d) else 0.0

            result[commodity] = {
                "correlation_30d": corr_30d,
                "correlation_90d": corr_90d,
                "is_significant": abs(corr_30d) >= MIN_CORRELATION_THRESHOLD,
                "is_stale": is_stale,
                "latest_commodity_date": str(latest_date) if latest_date else None,
                "overlapping_days": len(common_dates),
                "from_cache": False
            }

            self._set_cache(symbol, commodity, corr_30d, corr_90d)

            logger.info(
                f"Computed {commodity} correlation for {symbol}: "
                f"30d={corr_30d:.3f}, 90d={corr_90d:.3f}, "
                f"significant={result[commodity]['is_significant']}"
            )

        return result

    def _get_stock_prices(self, symbol: str, days: int = 90) -> List[Dict]:
        """Get historical stock prices from database."""
        session = self.db.get_session()
        try:
            from core.data.database import PriceHistory

            cutoff = date.today() - timedelta(days=days)
            records = (
                session.query(PriceHistory)
                .filter(PriceHistory.symbol == symbol)
                .filter(PriceHistory.date >= cutoff)
                .order_by(PriceHistory.date)
                .all()
            )
            return [{"date": r.date, "close": r.close} for r in records]
        except Exception as e:
            logger.error(f"Error fetching stock prices for {symbol}: {e}")
            return []
        finally:
            session.close()

    def get_commodity_impact(self, sector: str) -> Dict:
        """Analyze which commodities most impact a sector.

        ⚠️ WARNING: These are THEORETICAL values with NO empirical basis.
        Do NOT use for trading decisions. For demonstration purposes only.

        Args:
            sector: Stock sector name

        Returns:
            Dict with impact analysis per commodity
        """
        sector_upper = (sector or "DEFAULT").upper()
        impact = self.SECTOR_COMMODITY_IMPACT.get(
            sector_upper,
            self.SECTOR_COMMODITY_IMPACT["DEFAULT"]
        )
        return {
            "sector": sector,
            "commodity_impacts": impact,
            "is_theoretical": True,
            "warning": "⚠️ THEORETICAL VALUES - Not based on empirical data. For qualitative understanding only.",
            "note": "Impact values are heuristic estimates. Use computed correlations from compute_correlations() for actual signals."
        }

    def generate_signals(self, symbol: str, sector: str = None) -> Dict:
        """Generate trading signals based on macro correlations.

        Signals are only generated when:
        1. Correlation is statistically significant (|r| >= 0.3)
        2. Commodity price change is meaningful (>= 1%)
        3. Commodity data is not stale

        Args:
            symbol: Stock symbol
            sector: Stock sector (optional)

        Returns:
            Dict with signals and impact analysis
        """
        signals = {
            "symbol": symbol,
            "signals": {},
            "warnings": [],
        }

        # Get correlations
        correlations = self.compute_correlations(symbol)
        if not correlations:
            signals["error"] = f"No correlation data for {symbol}"
            return signals

        # Get sector impact
        if sector:
            signals["sector_impact"] = self.get_commodity_impact(sector)

        # Get latest commodity price changes
        start_date = date.today() - timedelta(days=7)  # Look back a week for latest data
        latest_commodity_changes = {}

        for commodity in self.COMMODITIES:
            prices = self.db.get_commodity_prices(commodity, start_date=start_date)
            if prices and len(prices) >= 2:
                # Access ORM attributes
                latest_price = prices[-1].close
                prev_price = prices[-2].close
                change_pct = (latest_price - prev_price) / prev_price * 100
                latest_date = prices[-1].date

                latest_commodity_changes[commodity] = {
                    "price": latest_price,
                    "change_pct": change_pct,
                    "date": str(latest_date),
                }

                # Check staleness
                days_old = (date.today() - latest_date).days
                if days_old > COMMODITY_STALENESS_DAYS:
                    signals["warnings"].append(
                        f"{commodity} data is {days_old} days old (stale threshold: {COMMODITY_STALENESS_DAYS})"
                    )
            else:
                latest_commodity_changes[commodity] = {
                    "price": 0.0,
                    "change_pct": 0.0,
                    "date": None,
                }
                signals["warnings"].append(f"No recent {commodity} data available")

        # Generate signals based on correlations and price changes
        bullish_count = 0
        bearish_count = 0

        for commodity in self.COMMODITIES:
            if commodity not in correlations or correlations[commodity] is None:
                continue

            corr_data = correlations[commodity]
            if isinstance(corr_data, dict) and corr_data.get("error"):
                continue

            corr_30d = corr_data.get("correlation_30d", 0)
            is_significant = corr_data.get("is_significant", False)
            is_stale = corr_data.get("is_stale", False)

            commodity_change = latest_commodity_changes.get(commodity, {"price": 0, "change_pct": 0})
            change_pct = commodity_change["change_pct"]

            # Skip if correlation not significant
            if not is_significant:
                signals["signals"][f"{commodity}_signal"] = {
                    "direction": "NEUTRAL",
                    "correlation": corr_30d,
                    "commodity_change_pct": change_pct,
                    "strength": 0.0,
                    "reason": f"Correlation {corr_30d:.2f} below threshold {MIN_CORRELATION_THRESHOLD}"
                }
                continue

            # Skip if commodity data is stale
            if is_stale:
                signals["signals"][f"{commodity}_signal"] = {
                    "direction": "NEUTRAL",
                    "correlation": corr_30d,
                    "commodity_change_pct": change_pct,
                    "strength": 0.0,
                    "reason": "Commodity data is stale"
                }
                signals["warnings"].append(f"{commodity} signal skipped - stale data")
                continue

            # Skip if price change is too small
            if abs(change_pct) < MIN_PRICE_CHANGE_PCT:
                signals["signals"][f"{commodity}_signal"] = {
                    "direction": "NEUTRAL",
                    "correlation": corr_30d,
                    "commodity_change_pct": change_pct,
                    "strength": 0.0,
                    "reason": f"Price change {change_pct:.2f}% below threshold {MIN_PRICE_CHANGE_PCT}%"
                }
                continue

            # Generate signal based on correlation direction and commodity movement
            # Positive correlation: commodity up -> stock up, commodity down -> stock down
            # Negative correlation: commodity up -> stock down, commodity down -> stock up
            if corr_30d > 0:
                if change_pct > 0:
                    signal = "BULLISH"
                    bullish_count += 1
                else:
                    signal = "BEARISH"
                    bearish_count += 1
            else:  # Negative correlation
                if change_pct > 0:
                    signal = "BEARISH"
                    bearish_count += 1
                else:
                    signal = "BULLISH"
                    bullish_count += 1

            # Calculate signal strength
            strength = abs(corr_30d) * abs(change_pct) / 100

            signals["signals"][f"{commodity}_signal"] = {
                "direction": signal,
                "correlation": corr_30d,
                "commodity_change_pct": change_pct,
                "strength": strength,
                "reason": f"Significant {'positive' if corr_30d > 0 else 'negative'} correlation with {change_pct:.1f}% commodity move"
            }

        signals["bullish_count"] = bullish_count
        signals["bearish_count"] = bearish_count

        # Overall signal - require at least 2 agreeing signals for conviction
        if bullish_count >= 2 and bullish_count > bearish_count:
            signals["overall_signal"] = "BULLISH"
            signals["conviction"] = "HIGH" if bullish_count >= 2 else "MEDIUM"
        elif bearish_count >= 2 and bearish_count > bullish_count:
            signals["overall_signal"] = "BEARISH"
            signals["conviction"] = "HIGH" if bearish_count >= 2 else "MEDIUM"
        else:
            signals["overall_signal"] = "NEUTRAL"
            signals["conviction"] = "LOW"

        return signals
