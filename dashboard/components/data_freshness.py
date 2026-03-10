"""
Data Freshness Components for IDX Trading System Dashboard

This module provides reusable components for displaying data freshness:
- Data freshness badges with relative time
- System status cards with real data from API
- Last refresh indicators
"""

import streamlit as st
from datetime import datetime, timezone
from typing import Optional, Dict, Any
import requests
import logging

logger = logging.getLogger(__name__)

# IDX timezone is Asia/Jakarta (WIB, UTC+7)
try:
    from zoneinfo import ZoneInfo
    WIB = ZoneInfo("Asia/Jakarta")
except ImportError:
    # Fallback for Python < 3.9
    from datetime import timedelta
    WIB = timezone(timedelta(hours=7))


def get_relative_time(dt: datetime) -> str:
    """
    Convert datetime to relative time string (e.g., "5 min ago").
    Uses WIB (Asia/Jakarta) timezone for IDX market.

    Args:
        dt: Datetime object to convert

    Returns:
        Human-readable relative time string
    """
    if dt is None:
        return "Unknown"

    now = datetime.now(WIB)

    # Ensure dt is timezone-aware
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=WIB)
    else:
        dt = dt.astimezone(WIB)

    diff = now - dt
    seconds = diff.total_seconds()

    # Handle future dates
    if seconds < 0:
        return "In the future"
    elif seconds < 60:
        return "Just now"
    elif seconds < 3600:
        minutes = int(seconds / 60)
        return f"{minutes} min ago"
    elif seconds < 86400:
        hours = int(seconds / 3600)
        return f"{hours}h ago"
    elif seconds < 604800:
        days = int(seconds / 86400)
        return f"{days}d ago"
    else:
        return dt.strftime("%Y-%m-%d %H:%M")


def data_freshness_badge(
    last_updated: Optional[datetime],
    label: str = "Data",
    max_age_hours: float = 24.0
) -> bool:
    """
    Show data freshness badge with relative time.
    Uses WIB (Asia/Jakarta) timezone for IDX market.

    Args:
        last_updated: Datetime of last update
        label: Label for the data type
        max_age_hours: Maximum age in hours before showing stale warning

    Returns:
        True if data is fresh, False if stale
    """
    if last_updated is None:
        st.caption(f"🕐 {label}: Unknown")
        return False

    relative_time = get_relative_time(last_updated)

    # Check freshness
    now = datetime.now(WIB)
    if last_updated.tzinfo is None:
        last_updated = last_updated.replace(tzinfo=WIB)
    else:
        last_updated = last_updated.astimezone(WIB)

    age_hours = (now - last_updated).total_seconds() / 3600

    if age_hours <= max_age_hours:
        st.caption(f"🕐 {label}: {relative_time}")
        return True
    else:
        st.caption(f"⚠️ {label}: {relative_time} (may be stale)")
        return False


def system_status_card(api_url: str = "http://localhost:8000") -> Dict[str, Any]:
    """
    Show system status with real data from API.

    Args:
        api_url: Base URL for the API

    Returns:
        Dictionary with system status data
    """
    status_data = {
        "status": "unknown",
        "stock_count": 0,
        "record_count": 0,
        "last_update": None,
        "components": {}
    }

    try:
        # Fetch detailed health
        resp = requests.get(f"{api_url}/health/detailed", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            status_data["status"] = data.get("status", "unknown")
            status_data["components"] = data.get("components", {})

            # Parse timestamp
            ts = data.get("timestamp")
            if ts:
                try:
                    status_data["last_update"] = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                except:
                    pass
    except Exception:
        status_data["status"] = "offline"

    try:
        # Fetch stock count
        resp = requests.get(f"{api_url}/stocks", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            stocks = data.get("stocks", data) if isinstance(data, dict) else data
            status_data["stock_count"] = len(stocks)
    except Exception:
        pass

    # Display status
    if status_data["status"] == "ok":
        st.success("🟢 System Online")
    elif status_data["status"] == "degraded":
        st.warning("🟡 System Degraded")
    else:
        st.error("🔴 System Offline")

    return status_data


def last_refresh_indicator(
    api_endpoint: str,
    label: str = "Data",
    show_spinner: bool = True
) -> Optional[datetime]:
    """
    Show when data was last refreshed from API.

    Args:
        api_endpoint: Full URL to check data freshness
        label: Label for the data type
        show_spinner: Whether to show a spinner while loading

    Returns:
        Datetime of last refresh or None
    """
    last_refresh = None

    try:
        if show_spinner:
            with st.spinner(f"Loading {label.lower()} freshness..."):
                resp = requests.get(api_endpoint, timeout=5)
        else:
            resp = requests.get(api_endpoint, timeout=5)

        if resp.status_code == 200:
            data = resp.json()

            # Try various timestamp fields
            ts = (
                data.get("last_updated") or
                data.get("timestamp") or
                data.get("last_price_update") or
                data.get("latest", {}).get("date")
            )

            if ts:
                try:
                    if isinstance(ts, str):
                        last_refresh = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                except:
                    pass
    except Exception:
        pass

    if last_refresh:
        relative = get_relative_time(last_refresh)
        st.caption(f"🔄 Last {label}: {relative}")
    else:
        st.caption(f"🔄 {label} freshness: Unknown")

    return last_refresh


def data_metrics_row(
    stock_count: int,
    record_count: Optional[int] = None,
    signal_count: Optional[int] = None,
    last_update: Optional[datetime] = None
) -> None:
    """
    Display a row of data metrics with freshness indicator.

    Args:
        stock_count: Number of stocks tracked
        record_count: Total price records (optional)
        signal_count: Active signals (optional)
        last_update: Last data update time
    """
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Stocks Tracked", f"{stock_count:,}")

    with col2:
        if record_count is not None:
            if record_count >= 1_000_000:
                st.metric("Price Records", f"{record_count/1_000_000:.2f}M")
            elif record_count >= 1_000:
                st.metric("Price Records", f"{record_count/1_000:.1f}K")
            else:
                st.metric("Price Records", f"{record_count:,}")
        else:
            st.metric("Price Records", "—")

    with col3:
        if signal_count is not None:
            st.metric("Active Signals", signal_count)
        else:
            st.metric("Active Signals", "—")

    with col4:
        if last_update:
            relative = get_relative_time(last_update)
            st.metric("Last Update", relative)
        else:
            st.metric("Last Update", "—")


def commodity_freshness_indicator(
    commodity_data: Dict[str, Any],
    commodity_name: str = "Commodity"
) -> bool:
    """
    Show freshness indicator for commodity data.

    Args:
        commodity_data: Dictionary with commodity price data
        commodity_name: Name of the commodity

    Returns:
        True if data is fresh (within 24 hours)
    """
    latest = commodity_data.get("latest", {})
    date_str = latest.get("date")

    if not date_str:
        st.caption(f"⚠️ {commodity_name}: No data")
        return False

    try:
        last_date = datetime.fromisoformat(date_str.replace("Z", "+00:00")).date()
        today = datetime.now(timezone.utc).date()
        days_old = (today - last_date).days

        if days_old == 0:
            st.caption(f"✅ {commodity_name}: Today")
            return True
        elif days_old == 1:
            st.caption(f"🕐 {commodity_name}: Yesterday")
            return True
        else:
            st.caption(f"⚠️ {commodity_name}: {days_old} days old")
            return False
    except:
        st.caption(f"⚠️ {commodity_name}: Unknown")
        return False


def progress_with_status(
    current: int,
    total: int,
    status_message: str = "",
    show_percentage: bool = True
) -> None:
    """
    Show progress bar with status message.

    Args:
        current: Current progress value
        total: Total value for completion
        status_message: Message to display
        show_percentage: Whether to show percentage
    """
    progress = current / total if total > 0 else 0
    pct = f" ({progress*100:.0f}%)" if show_percentage else ""
    text = f"{status_message}{pct}" if status_message else f"Processing{pct}"
    st.progress(progress, text=text)
