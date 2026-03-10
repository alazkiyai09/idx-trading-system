"""
IDX Trading System Dashboard - Main Entry Point

This is the main entry point for the Streamlit dashboard.
Navigate to different modules using the sidebar.

Design: NextGen-style dark theme with Zinc/Emerald color palette.
"""
import streamlit as st
import requests
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
import logging
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logger = logging.getLogger(__name__)

# Get API URL from settings or environment
try:
    from config.settings import settings
    API_URL = os.environ.get("API_URL", settings.api_url)
except Exception:
    API_URL = os.environ.get("API_URL", "http://localhost:8000")  # Fallback

# Import NextGen styling
from dashboard.components.nextgen_styles import get_nextgen_css, COLORS, render_live_badge
from dashboard.components.top_nav import render_top_nav, render_status_strip

# Import dashboard components
from dashboard.components.ux_components import trading_hours_indicator

# Run this to set wide mode as the default layout
st.set_page_config(
    page_title="IDX Trading System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================================
# API FUNCTIONS
# ============================================================================

@st.cache_data(ttl=30, show_spinner=False)
def get_system_status() -> Dict[str, Any]:
    """Fetch system status from API. Cached for 30 seconds."""
    try:
        resp = requests.get(f"{API_URL}/health/detailed", timeout=5)
        if resp.status_code == 200:
            return resp.json()
    except requests.exceptions.Timeout:
        logger.warning(f"Timeout fetching system status from {API_URL}")
    except requests.exceptions.ConnectionError as e:
        logger.warning(f"Cannot connect to API: {e}")
    except Exception as e:
        logger.error(f"Unexpected error fetching system status: {e}")
    return {"status": "offline", "components": {}}


@st.cache_data(ttl=15, show_spinner=False)
def get_dashboard_summary() -> Dict[str, Any]:
    """Fetch compact homepage summary payload."""
    try:
        resp = requests.get(f"{API_URL}/health/dashboard-summary", timeout=5)
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        logger.error(f"Error fetching dashboard summary: {e}")
    return {}


@st.cache_data(ttl=60, show_spinner=False)
def get_stocks() -> List[Dict[str, Any]]:
    """Fetch all stocks from API. Cached for 60 seconds."""
    try:
        resp = requests.get(f"{API_URL}/stocks", timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            return data.get("stocks", data) if isinstance(data, dict) else data
    except Exception as e:
        logger.error(f"Error fetching stocks: {e}")
    return []


@st.cache_data(ttl=30, show_spinner=False)
def get_data_freshness() -> Dict[str, Any]:
    """Fetch data freshness info from API. Cached for 30 seconds."""
    try:
        resp = requests.get(f"{API_URL}/health/data", timeout=5)
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        logger.error(f"Error fetching data freshness: {e}")
    return {}


@st.cache_data(ttl=10, show_spinner=False)
def get_update_status() -> Dict[str, Any]:
    """Fetch current update policy and manual refresh state."""
    try:
        resp = requests.get(f"{API_URL}/health/update-status", timeout=5)
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        logger.error(f"Error fetching update status: {e}")
    return {}


def trigger_manual_data_refresh() -> Dict[str, Any]:
    """Trigger a background daily price refresh job."""
    try:
        resp = requests.post(f"{API_URL}/health/update-data", timeout=10)
        payload = resp.json() if resp.content else {}
        payload["_status_code"] = resp.status_code
        return payload
    except Exception as e:
        return {"_status_code": 0, "detail": str(e)}


@st.cache_data(ttl=30, show_spinner=False)
def get_active_signals_count() -> int:
    """Fetch active signals count from API. Cached for 30 seconds."""
    try:
        resp = requests.get(f"{API_URL}/signals", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            if isinstance(data, list):
                return len(data)
            elif isinstance(data, dict) and "signals" in data:
                return len(data["signals"])
            return data.get("total", 0)
    except Exception as e:
        logger.error(f"Error fetching signals count: {e}")
    return 0


# ============================================================================
# MAIN PAGE
# ============================================================================

def render_home_metrics() -> None:
    """Render the home page metrics."""
    with st.spinner("Loading dashboard metrics..."):
        summary = get_dashboard_summary()

    # Main metrics row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Stocks", f"{summary.get('stock_count', 0):,}")

    with col2:
        record_count = summary.get("record_count", "—")
        st.metric("Price Records", "1.86M" if record_count == "—" else record_count)

    with col3:
        signals_count = summary.get("signal_count", 0)
        st.metric("Active Signals", signals_count, delta=None if signals_count == 0 else f"+{signals_count}")

    with col4:
        price_age = summary.get("price_data_age_hours")
        if price_age is not None:
            if price_age < 1:
                age_str = f"{price_age*60:.0f} min ago"
            elif price_age < 24:
                age_str = f"{price_age:.1f}h ago"
            else:
                age_str = f"{price_age/24:.1f}d ago"
            st.metric("Last Update", age_str)
        else:
            st.metric("Last Update", "Unknown")


def render_command_center() -> None:
    """Render the landing-page command center summary."""
    stocks = get_stocks()
    freshness = get_data_freshness()
    status_data = get_system_status()

    gainers = [s for s in stocks if s.get("change_pct", 0) > 0]
    losers = [s for s in stocks if s.get("change_pct", 0) < 0]
    top_mover = max(stocks, key=lambda s: abs(s.get("change_pct", 0)), default={})

    price_age = freshness.get("price_data_age_hours")
    if price_age is None:
        refresh_label = "Unknown"
    elif price_age < 1:
        refresh_label = f"{price_age * 60:.0f} min"
    elif price_age < 24:
        refresh_label = f"{price_age:.1f}h"
    else:
        refresh_label = f"{price_age / 24:.1f}d"

    components = status_data.get("components", {})
    degraded = [name for name, details in components.items() if details.get("status") != "ok"]
    system_note = "All systems nominal" if not degraded else f"Check {', '.join(degraded[:2])}"

    col1, col2, col3 = st.columns([1.6, 1.2, 1.2])

    with col1:
        st.markdown(f"""
        <div class="nextgen-card" style="min-height: 240px; background:
            radial-gradient(circle at top right, rgba(16, 185, 129, 0.22), transparent 42%),
            linear-gradient(135deg, rgba(24, 24, 27, 0.96), rgba(9, 9, 11, 1));">
            <div style="display: flex; justify-content: space-between; align-items: start; gap: 16px;">
                <div>
                    <div style="font-size: 0.72rem; letter-spacing: 0.14em; color: {COLORS['muted_foreground']};">
                        COMMAND CENTER
                    </div>
                    <div style="font-size: 2rem; font-weight: 700; margin-top: 10px;">
                        Focus the desk before the open.
                    </div>
                    <p style="margin: 14px 0 0 0; max-width: 44ch; color: {COLORS['muted_foreground']}; line-height: 1.6;">
                        Review market breadth, data freshness, and the strongest mover before launching scans or simulations.
                    </p>
                </div>
                <div style="padding: 10px 12px; border: 1px solid {COLORS['border_light']}; border-radius: 999px; background: rgba(16, 185, 129, 0.08);">
                    <span style="font-size: 0.75rem; color: {COLORS['primary_light']};">LIVE DESK</span>
                </div>
            </div>
            <div style="display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; margin-top: 20px;">
                <div style="padding: 14px; border-radius: 10px; border: 1px solid {COLORS['border']}; background: rgba(255,255,255,0.02);">
                    <div style="font-size: 0.72rem; color: {COLORS['muted_foreground']};">ADVANCERS</div>
                    <div style="font-size: 1.5rem; font-weight: 600; color: {COLORS['primary']};">{len(gainers)}</div>
                </div>
                <div style="padding: 14px; border-radius: 10px; border: 1px solid {COLORS['border']}; background: rgba(255,255,255,0.02);">
                    <div style="font-size: 0.72rem; color: {COLORS['muted_foreground']};">DECLINERS</div>
                    <div style="font-size: 1.5rem; font-weight: 600; color: {COLORS['destructive']};">{len(losers)}</div>
                </div>
                <div style="padding: 14px; border-radius: 10px; border: 1px solid {COLORS['border']}; background: rgba(255,255,255,0.02);">
                    <div style="font-size: 0.72rem; color: {COLORS['muted_foreground']};">DATA LATENCY</div>
                    <div style="font-size: 1.5rem; font-weight: 600;">{refresh_label}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        mover_symbol = top_mover.get("symbol", "N/A")
        mover_change = top_mover.get("change_pct", 0.0)
        mover_price = top_mover.get("close", 0.0)
        mover_color = COLORS["primary"] if mover_change >= 0 else COLORS["destructive"]
        st.markdown(f"""
        <div class="nextgen-card" style="min-height: 240px;">
            <div style="font-size: 0.72rem; letter-spacing: 0.14em; color: {COLORS['muted_foreground']};">MARKET PULSE</div>
            <div style="margin-top: 18px; font-size: 0.9rem; color: {COLORS['muted_foreground']};">Top mover today</div>
            <div style="display: flex; justify-content: space-between; align-items: end; margin-top: 8px;">
                <div style="font-size: 2rem; font-weight: 700;">{mover_symbol}</div>
                <div style="font-size: 1rem; font-family: 'JetBrains Mono', monospace;">Rp {mover_price:,.0f}</div>
            </div>
            <div style="margin-top: 8px; font-size: 1.2rem; font-weight: 600; color: {mover_color};">
                {mover_change:+.2f}%
            </div>
            <div style="margin-top: 18px; padding-top: 16px; border-top: 1px solid {COLORS['border']};">
                <div style="font-size: 0.78rem; color: {COLORS['muted_foreground']};">Desk note</div>
                <div style="margin-top: 6px; line-height: 1.6;">{system_note}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        signal_count = get_active_signals_count()
        st.markdown(f"""
        <div class="nextgen-card" style="min-height: 240px;">
            <div style="font-size: 0.72rem; letter-spacing: 0.14em; color: {COLORS['muted_foreground']};">TRADE DESK PRIORITIES</div>
            <div style="display: flex; flex-direction: column; gap: 12px; margin-top: 18px;">
                <div style="padding: 12px; border-radius: 10px; border: 1px solid {COLORS['border']}; background: rgba(16, 185, 129, 0.06);">
                    <div style="font-size: 0.72rem; color: {COLORS['muted_foreground']};">ACTIVE SIGNALS</div>
                    <div style="font-size: 1.45rem; font-weight: 600;">{signal_count}</div>
                </div>
                <div style="padding: 12px; border-radius: 10px; border: 1px solid {COLORS['border']};">
                    <div style="font-size: 0.72rem; color: {COLORS['muted_foreground']};">NEXT ACTION</div>
                    <div style="margin-top: 4px;">Run screener and review top-ranked setups.</div>
                </div>
                <div style="padding: 12px; border-radius: 10px; border: 1px solid {COLORS['border']};">
                    <div style="font-size: 0.72rem; color: {COLORS['muted_foreground']};">SIM CHECK</div>
                    <div style="margin-top: 4px;">Validate position sizing in paper trading before live use.</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)


def render_quick_actions() -> None:
    """Render quick action buttons."""
    st.markdown("### Desk Actions")
    st.caption("Navigation shortcuts for the main modules. These do not run refresh or analysis jobs on their own.")
    st.markdown(
        """
        <div class="idx-link-grid">
            <a class="idx-link-pill" href="/screener" target="_self">🔍 Run Screener</a>
            <a class="idx-link-pill" href="/virtual_trading" target="_self">💹 Virtual Trading</a>
            <a class="idx-link-pill" href="/ml_prediction" target="_self">🤖 ML Prediction</a>
            <a class="idx-link-pill" href="/settings" target="_self">⚙️ Settings</a>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_module_cards() -> None:
    """Render module navigation cards."""
    st.markdown("### Modules")

    col1, col2, col3 = st.columns(3)

    modules = [
        ("🌐 Market Overview", "Browse 657 IDX stocks with interactive sector heatmap", "/market_overview"),
        ("🔍 Stock Screener", "Filter and score stocks with real-time technical analysis", "/screener"),
        ("📰 Sentiment Dashboard", "Monitor market news and sector-wide sentiment trends", "/sentiment"),
        ("💹 Virtual Trading", "Simulate trades with paper trading or historical replay", "/virtual_trading"),
        ("🤖 ML Prediction", "Train models and generate price forecasts", "/ml_prediction"),
        ("⚙️ Settings", "Configure LLM providers, risk limits, and notifications", "/settings"),
    ]

    for i, (title, desc, page) in enumerate(modules):
        col = [col1, col2, col3][i % 3]
        with col:
            st.markdown(
                f"""
                <div class="nextgen-card">
                    <h4 style="margin: 0 0 8px 0; color: {COLORS['foreground']};">{title}</h4>
                    <p style="margin: 0 0 14px 0; font-size: 0.85rem; color: {COLORS['muted_foreground']};">{desc}</p>
                    <a class="idx-module-link" href="{page}" target="_self">Open {title}</a>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_system_status_card() -> None:
    """Render detailed system status card."""
    status_data = get_system_status()
    status = status_data.get("status", "unknown")
    components = status_data.get("components", {})

    if status == "ok":
        st.success("✅ All services operational")
    elif status == "degraded":
        degraded = [k for k, v in components.items() if v.get("status") != "ok"]
        if degraded:
            st.warning(f"⚠️ Degraded: {', '.join(degraded)}")
        else:
            st.warning("⚠️ System Degraded")
    else:
        st.error("🔴 System offline or unreachable")


def render_data_update_status() -> None:
    """Render update cadence, freshness, and manual refresh controls."""
    update = get_update_status()
    data_status = update.get("data_status", {})
    manual = update.get("manual_refresh", {})
    policy = update.get("refresh_policy", {})

    col1, col2 = st.columns([1.5, 1])

    with col1:
        last_price = data_status.get("last_price_update") or "Unknown"
        last_flow = data_status.get("last_flow_update") or "Unavailable"
        st.markdown(f"""
        <div class="nextgen-card">
            <div style="font-size: 0.72rem; letter-spacing: 0.14em; color: {COLORS['muted_foreground']};">DATA UPDATE STATUS</div>
            <div style="display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; margin-top: 14px;">
                <div>
                    <div style="font-size: 0.72rem; color: {COLORS['muted_foreground']};">PRICE DATA</div>
                    <div style="font-size: 1.15rem; font-weight: 600;">{data_status.get('price_status', 'unknown').upper()}</div>
                    <div style="margin-top: 6px; color: {COLORS['muted_foreground']};">Last update: {last_price}</div>
                </div>
                <div>
                    <div style="font-size: 0.72rem; color: {COLORS['muted_foreground']};">EXPECTED CADENCE</div>
                    <div style="font-size: 1.15rem; font-weight: 600;">{policy.get('expected_frequency', 'daily').title()}</div>
                    <div style="margin-top: 6px; color: {COLORS['muted_foreground']};">{policy.get('expected_window', 'Not configured')}</div>
                </div>
            </div>
            <div style="margin-top: 16px; padding-top: 14px; border-top: 1px solid {COLORS['border']}; color: {COLORS['muted_foreground']};">
                Foreign flow last update: {last_flow}<br/>
                Price rows: {data_status.get('price_record_count', 0):,} | Flow rows: {data_status.get('flow_record_count', 0):,}
            </div>
        </div>
        """, unsafe_allow_html=True)
        for warning in data_status.get("warnings", [])[:3]:
            st.warning(warning)

    with col2:
        st.markdown("### Update Controls")
        if manual.get("is_running"):
            st.info(f"Refresh job running since {manual.get('started_at', 'unknown')}")
        else:
            st.caption("Manual refresh is for missed daily updates. The normal path should be a scheduled job after market close or by midnight.")

        if st.button("Refresh Daily Data Now", type="primary", use_container_width=True):
            result = trigger_manual_data_refresh()
            code = result.get("_status_code")
            if code == 200:
                st.success(result.get("message", "Refresh started"))
                st.cache_data.clear()
                st.rerun()
            elif code == 409:
                st.warning(result.get("detail", "Refresh already running"))
            else:
                st.error(result.get("detail", result.get("message", "Failed to start refresh")))

        if manual.get("log_tail"):
            st.caption("Latest job output")
            st.code("\n".join(manual["log_tail"][-8:]), language="text")


def render_desk_notes() -> None:
    """Render high-level workflow notes for operators."""
    notes = [
        ("1", "Scan", "Start with the screener to narrow the universe before opening detail pages."),
        ("2", "Validate", "Use technical and ML analysis together. Do not treat either in isolation."),
        ("3", "Simulate", "Route every actionable setup through paper trading and risk checks."),
    ]
    cols = st.columns(3)
    for col, (step, title, body) in zip(cols, notes):
        with col:
            st.markdown(f"""
            <div class="nextgen-card">
                <div style="display: inline-flex; width: 28px; height: 28px; border-radius: 999px; align-items: center; justify-content: center; background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.3); color: {COLORS['primary']}; font-weight: 700;">
                    {step}
                </div>
                <h4 style="margin: 14px 0 8px 0;">{title}</h4>
                <p style="margin: 0; color: {COLORS['muted_foreground']}; line-height: 1.6;">{body}</p>
            </div>
            """, unsafe_allow_html=True)


def main():
    # Apply NextGen CSS
    st.markdown(get_nextgen_css(), unsafe_allow_html=True)

    # Main content
    st.markdown(f"""
    <div style="display: flex; align-items: baseline; gap: 16px; margin-bottom: 8px;">
        <h1 style="margin: 0;">IDX Trading Dashboard</h1>
        {render_live_badge("NEXTGEN")}
    </div>
    <p style="color: {COLORS['muted_foreground']}; margin-bottom: 24px;">
        Institutional-grade trading platform for the Indonesia Stock Exchange
    </p>
    """, unsafe_allow_html=True)

    render_top_nav("home")

    summary = get_dashboard_summary()
    update_status = summary.get("update_status", {})
    refresh_policy = summary.get("refresh_policy", {})
    freshness_value = summary.get("freshness_status", "unknown").upper()
    if summary.get("last_price_update"):
        freshness_value = f"{freshness_value} | {summary['last_price_update'][:10]}"
    refresh_job = "Running" if update_status.get("is_running") else "Idle"
    render_status_strip([
        ("System", get_system_status().get("status", "unknown").upper()),
        ("Price Data", freshness_value),
        ("Refresh Job", refresh_job),
        ("Cadence", refresh_policy.get("expected_frequency", "daily").title()),
    ])

    st.markdown("---")

    render_command_center()

    st.markdown("---")

    # Metrics
    render_home_metrics()

    st.markdown("---")

    # System status
    render_system_status_card()

    st.markdown("---")

    render_data_update_status()

    st.markdown("---")

    # Quick actions
    render_quick_actions()

    st.markdown("---")

    render_desk_notes()

    st.markdown("---")

    # Module cards
    render_module_cards()


if __name__ == "__main__":
    main()
