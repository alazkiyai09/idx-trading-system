"""
Stock Screener Page - Filter and analyze IDX stocks.

Design: NextGen-style with advanced screener panel and real-time results.
"""
import streamlit as st
import pandas as pd
import requests
import sys
import os
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from dashboard.components.filters import (
    render_classification_filters, render_price_filters,
    render_technical_filters, render_volume_filters,
    render_performance_filters, render_analysis_settings,
)
from dashboard.components.ux_components import (
    quick_filter_buttons, trading_hours_indicator,
    success_with_details, error_with_recovery, progress_indicator,
    QUICK_FILTER_PRESETS
)
from dashboard.components.nextgen_styles import (
    get_nextgen_css, COLORS, render_live_badge, render_signal_badge
)
from dashboard.components.top_nav import render_top_nav

# Configure logging
logger = logging.getLogger(__name__)

# Get API URL from settings or environment
try:
    from config.settings import settings
    API_URL = os.environ.get("API_URL", settings.api_url)
except Exception:
    API_URL = os.environ.get("API_URL", "http://localhost:8000")

# Default capital for analysis
DEFAULT_CAPITAL = 100_000_000  # 100M IDR

st.set_page_config(page_title="Stock Screener | IDX", page_icon="🔍", layout="wide", initial_sidebar_state="collapsed")

# Apply NextGen CSS
st.markdown(get_nextgen_css(), unsafe_allow_html=True)

# --- Header ---
st.markdown(f"""
<div style="display: flex; align-items: baseline; gap: 16px; margin-bottom: 8px;">
    <h1 style="margin: 0;">Advanced Screener</h1>
    {render_live_badge("NEXTGEN")}
</div>
<p style="color: {COLORS['muted_foreground']}; margin-bottom: 16px;">
    Screen 657 IDX stocks using technical, volume, and performance filters
</p>
""", unsafe_allow_html=True)

render_top_nav("screener")
st.markdown("---")

# Market hours indicator
trading_hours_indicator()

# --- Data Fetch ---
def get_stocks() -> list:
    """Fetch all stocks from API with proper error handling."""
    try:
        resp = requests.get(f"{API_URL}/stocks?limit=500", timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            return data.get("stocks", data) if isinstance(data, dict) else data
        logger.warning(f"API returned status {resp.status_code}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching stocks: {e}")
    return []


stocks_data = get_stocks()
if not stocks_data:
    st.warning("No stock data available. Is the API running?")
    if st.button("Retry loading stock data", type="primary", use_container_width=False):
        st.cache_data.clear()
        st.rerun()
    st.stop()

df = pd.DataFrame(stocks_data)

# Ensure symbol column exists (may be named differently by API)
if 'symbol' not in df.columns:
    # Try common alternatives
    if 'ticker' in df.columns:
        df['symbol'] = df['ticker']
    elif 'code' in df.columns:
        df['symbol'] = df['code']
    elif len(df.columns) > 0:
        # Use first column as fallback
        df['symbol'] = df.iloc[:, 0]
    else:
        st.error("No symbol column found in stock data")
        st.stop()

# --- Quick Filter Presets (Collapsible) ---
with st.expander("🎯 Quick Filters", expanded=False):
    st.markdown('<div class="section-header">PRESET SCREENS</div>', unsafe_allow_html=True)

    def apply_quick_filter(filter_data: dict):
        """Apply quick filter preset to session state."""
        st.session_state["quick_filter_active"] = filter_data
        st.session_state.analysis_results = None

    quick_filter_buttons(on_select=apply_quick_filter, key_prefix="screener")

# --- Advanced Screener Panel (Collapsible) ---
with st.expander("⚙️ Advanced Screener", expanded=True):
    st.markdown('<div class="section-header">FILTER CONFIGURATION</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class="nextgen-card">
            <h4 style="margin: 0 0 12px 0; font-size: 0.8rem; color: {COLORS['muted_foreground']};">UNIVERSE FILTERS</h4>
        """, unsafe_allow_html=True)
        cls_filters = render_classification_filters(df, container=st)
        price_filters = render_price_filters(container=st)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="nextgen-card">
            <h4 style="margin: 0 0 12px 0; font-size: 0.8rem; color: {COLORS['muted_foreground']};">TECHNICAL FILTERS</h4>
        """, unsafe_allow_html=True)
        tech_filters = render_technical_filters(container=st)
        st.markdown("</div>", unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="nextgen-card">
            <h4 style="margin: 0 0 12px 0; font-size: 0.8rem; color: {COLORS['muted_foreground']};">VOLUME & PERFORMANCE</h4>
        """, unsafe_allow_html=True)
        vol_filters = render_volume_filters(container=st)
        perf_filters = render_performance_filters(container=st)
        st.markdown("</div>", unsafe_allow_html=True)

with st.expander("Strategy Settings", expanded=False):
    analysis_cfg = render_analysis_settings(container=st)

# --- Apply Filters ---
filtered_df = df.copy()

# Apply quick filter if active
if "quick_filter_active" in st.session_state:
    qf = st.session_state.pop("quick_filter_active", None)
    if qf:
        if qf.get("lq45") and 'is_lq45' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['is_lq45'].fillna(False) == True]
        if qf.get("min_market_cap") and 'market_cap' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['market_cap'] >= qf["min_market_cap"] * 1e9]
        if qf.get("min_volume") and 'volume' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['volume'] >= qf["min_volume"]]

# Apply classification filters
if cls_filters["lq45"] and 'is_lq45' in filtered_df.columns:
    filtered_df = filtered_df[filtered_df['is_lq45'].fillna(False) == True]
if cls_filters["idx30"] and 'is_idx30' in filtered_df.columns:
    filtered_df = filtered_df[filtered_df['is_idx30'].fillna(False) == True]
if cls_filters["sector"] != "All" and 'sector' in filtered_df.columns:
    filtered_df = filtered_df[filtered_df['sector'] == cls_filters["sector"]]
if cls_filters["sub_sector"] != "All" and 'sub_sector' in filtered_df.columns:
    filtered_df = filtered_df[filtered_df['sub_sector'] == cls_filters["sub_sector"]]

# Apply price filters
if price_filters["min_market_cap"] > 0 and 'market_cap' in filtered_df.columns:
    filtered_df = filtered_df[filtered_df['market_cap'] >= price_filters["min_market_cap"] * 1e9]

# Apply technical filters (RSI range)
if 'rsi' in filtered_df.columns:
    rsi_min, rsi_max = tech_filters.get("rsi_range", (0, 100))
    if rsi_min > 0:
        filtered_df = filtered_df[filtered_df['rsi'] >= rsi_min]
    if rsi_max < 100:
        filtered_df = filtered_df[filtered_df['rsi'] <= rsi_max]

# Apply volume filters
if vol_filters.get("min_volume", 0) > 0 and 'volume' in filtered_df.columns:
    filtered_df = filtered_df[filtered_df['volume'] >= vol_filters["min_volume"]]

# Apply performance filters
if perf_filters.get("perf_1w") and 'perf_1w' in filtered_df.columns:
    filtered_df = filtered_df[filtered_df['perf_1w'] > 0]

# --- Scan Context ---
active_filters = []
if cls_filters["lq45"]:
    active_filters.append("LQ45")
if cls_filters["idx30"]:
    active_filters.append("IDX30")
if cls_filters["sector"] != "All":
    active_filters.append(f"Sector: {cls_filters['sector']}")
if cls_filters["sub_sector"] != "All":
    active_filters.append(f"Sub-sector: {cls_filters['sub_sector']}")
if price_filters["min_market_cap"] > 0:
    active_filters.append(f"Min cap: Rp {price_filters['min_market_cap']:.0f}B")
if vol_filters.get("min_volume", 0) > 0:
    active_filters.append(f"Min vol: {vol_filters['min_volume']:,}")
if perf_filters.get("perf_1w"):
    active_filters.append("Positive 1W performance")

market_coverage = (len(filtered_df) / len(df) * 100) if len(df) else 0
watchlist_count = len(st.session_state.get("watchlist", []))

col_ctx1, col_ctx2 = st.columns([2, 1])
with col_ctx1:
    chips = " | ".join(active_filters[:5]) if active_filters else "No filters active. Current view reflects the broad market universe."
    st.markdown(f"""
    <div class="nextgen-card">
        <div style="font-size: 0.72rem; letter-spacing: 0.14em; color: {COLORS['muted_foreground']};">SCAN CONTEXT</div>
        <div style="margin-top: 12px; font-size: 1.15rem; font-weight: 600;">{len(filtered_df)} candidates remain in the working set</div>
        <p style="margin: 10px 0 0 0; color: {COLORS['muted_foreground']}; line-height: 1.6;">
            {chips}
        </p>
    </div>
    """, unsafe_allow_html=True)
with col_ctx2:
    st.markdown(f"""
    <div class="nextgen-card">
        <div style="display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px;">
            <div>
                <div style="font-size: 0.72rem; color: {COLORS['muted_foreground']};">Coverage</div>
                <div style="font-size: 1.5rem; font-weight: 600;">{market_coverage:.1f}%</div>
            </div>
            <div>
                <div style="font-size: 0.72rem; color: {COLORS['muted_foreground']};">Watchlist</div>
                <div style="font-size: 1.5rem; font-weight: 600;">{watchlist_count}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# --- Results Summary ---
st.markdown("---")

col_stats, col_action = st.columns([3, 1])

with col_stats:
    if len(filtered_df) > 0:
        lq45_count = len(filtered_df[filtered_df['is_lq45'] == True]) if 'is_lq45' in filtered_df.columns else 0
        sector_count = filtered_df['sector'].nunique() if 'sector' in filtered_df.columns else 0

        st.markdown(f"""
        <div class="nextgen-card" style="display: flex; gap: 24px; align-items: center;">
            <div>
                <div style="font-size: 0.75rem; color: {COLORS['muted_foreground']};">MATCHING STOCKS</div>
                <div style="font-size: 1.5rem; font-weight: 600; color: {COLORS['primary']};">{len(filtered_df)}</div>
            </div>
            <div>
                <div style="font-size: 0.75rem; color: {COLORS['muted_foreground']};">LQ45 STOCKS</div>
                <div style="font-size: 1.25rem; font-weight: 500;">{lq45_count}</div>
            </div>
            <div>
                <div style="font-size: 0.75rem; color: {COLORS['muted_foreground']};">SECTORS</div>
                <div style="font-size: 1.25rem; font-weight: 500;">{sector_count}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("No stocks match your criteria. Try adjusting filters.")

with col_action:
    trading_mode = analysis_cfg.get("mode", "SWING") if analysis_cfg else "SWING"
    run_clicked = st.button("🚀 Run Scan", type="primary", use_container_width=True)

# --- Analysis Execution ---
scan_errors = []

if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None

if run_clicked:
    scan_errors = []
    st.session_state.analysis_results = None
    symbols_to_scan = filtered_df['symbol'].tolist()[:100]  # Limit to 100 for performance

    if len(symbols_to_scan) > 20:
        st.info(f"Analyzing {len(symbols_to_scan)} stocks via batch endpoint...")

    # Check API health
    try:
        health = requests.get(f"{API_URL}/health", timeout=5)
        if health.status_code != 200:
            st.error("API server is not healthy.")
            st.stop()
    except requests.exceptions.RequestException as e:
        st.error(f"Cannot connect to API: {e}")
        st.stop()

    progress_bar = st.progress(0, text="Running batch scan...")

    symbol_lookup = filtered_df.set_index('symbol')[['name', 'sector']].to_dict('index')
    capital = analysis_cfg.get("capital", DEFAULT_CAPITAL) if analysis_cfg else DEFAULT_CAPITAL

    try:
        # Use batch scan endpoint for efficiency
        scan_resp = requests.post(
            f"{API_URL}/signals/scan",
            json={"mode": trading_mode.lower(), "symbols": symbols_to_scan, "dry_run": True},
            timeout=120,
        )

        if scan_resp.status_code == 200:
            scan_data = scan_resp.json()
            signals = scan_data.get('signals', [])

            results = []
            for sig in signals:
                sym = sig.get('symbol', '')
                sym_data = symbol_lookup.get(sym, {})
                score_value = sig.get('score', sig.get('composite_score', 0))
                results.append({
                    "Symbol": sym,
                    "Name": sym_data.get('name', sig.get('name', 'N/A')),
                    "Sector": sym_data.get('sector', 'N/A'),
                    "Tech Score": score_value,
                    "Trend": sig.get('trend', 'N/A'),
                    "Signal": sig.get('signal_type', 'N/A'),
                    "Action": sig.get('type', 'HOLD'),
                    "Setup": sig.get('setup', 'N/A'),
                    "R/R": round(sig.get('risk_reward', 0), 2),
                })

            progress_bar.progress(100, text="Complete!")
            progress_bar.empty()

            st.session_state.analysis_results = pd.DataFrame(results)
            st.session_state.scan_errors = scan_errors

            if results:
                buy_signals = sum(1 for r in results if r.get("Action") == "BUY")
                valid_scores = [r.get('Tech Score') for r in results if r.get('Tech Score') is not None]
                avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
                st.success(f"✅ Scan complete! {len(results)} stocks analyzed, {buy_signals} BUY signals, avg score: {avg_score:.1f}")
        else:
            progress_bar.empty()
            st.session_state.analysis_results = None
            st.error(f"Scan failed: {scan_resp.status_code} - {scan_resp.text[:200]}")

    except requests.exceptions.RequestException as e:
        progress_bar.empty()
        st.session_state.analysis_results = None
        st.error(f"Scan error: {e}")

# --- Display Results ---
st.markdown("---")
st.markdown('<div class="section-header">SCAN RESULTS</div>', unsafe_allow_html=True)

if st.session_state.analysis_results is not None and not st.session_state.analysis_results.empty:
    res_df = st.session_state.analysis_results.copy()

    focus_df = res_df.sort_values(["Tech Score", "R/R"], ascending=[False, False], na_position="last")
    focus = focus_df.iloc[0].to_dict()
    action_color = COLORS["primary"] if focus.get("Action") == "BUY" else COLORS["destructive"] if focus.get("Action") == "SELL" else COLORS["muted_foreground"]

    spotlight_col, breakdown_col = st.columns([1.6, 1.4])
    with spotlight_col:
        st.markdown(f"""
        <div class="nextgen-card" style="background:
            radial-gradient(circle at top right, rgba(16, 185, 129, 0.14), transparent 40%),
            linear-gradient(135deg, rgba(24, 24, 27, 0.96), rgba(9, 9, 11, 1));">
            <div style="font-size: 0.72rem; letter-spacing: 0.14em; color: {COLORS['muted_foreground']};">OPPORTUNITY SPOTLIGHT</div>
            <div style="display: flex; justify-content: space-between; align-items: end; margin-top: 14px;">
                <div>
                    <div style="font-size: 2rem; font-weight: 700;">{focus.get('Symbol', 'N/A')}</div>
                    <div style="color: {COLORS['muted_foreground']};">{focus.get('Name', 'N/A')}</div>
                </div>
                <div style="font-size: 1rem; color: {action_color}; font-weight: 600;">{focus.get('Action', 'HOLD')}</div>
            </div>
            <div style="display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin-top: 18px;">
                <div><div style="font-size: 0.72rem; color: {COLORS['muted_foreground']};">Score</div><div style="font-size: 1.35rem; font-weight: 600;">{focus.get('Tech Score', 0):.0f}</div></div>
                <div><div style="font-size: 0.72rem; color: {COLORS['muted_foreground']};">R/R</div><div style="font-size: 1.35rem; font-weight: 600;">{focus.get('R/R', 0):.2f}</div></div>
                <div><div style="font-size: 0.72rem; color: {COLORS['muted_foreground']};">Trend</div><div style="font-size: 1rem; font-weight: 600;">{focus.get('Trend', 'N/A')}</div></div>
                <div><div style="font-size: 0.72rem; color: {COLORS['muted_foreground']};">Setup</div><div style="font-size: 1rem; font-weight: 600;">{focus.get('Setup', 'N/A')}</div></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with breakdown_col:
        buy_count = int((res_df["Action"] == "BUY").sum()) if "Action" in res_df.columns else 0
        sell_count = int((res_df["Action"] == "SELL").sum()) if "Action" in res_df.columns else 0
        hold_count = int((res_df["Action"] == "HOLD").sum()) if "Action" in res_df.columns else 0
        st.markdown(f"""
        <div class="nextgen-card">
            <div style="font-size: 0.72rem; letter-spacing: 0.14em; color: {COLORS['muted_foreground']};">SIGNAL MIX</div>
            <div style="display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; margin-top: 16px;">
                <div style="padding: 12px; border-radius: 10px; border: 1px solid {COLORS['border']};">
                    <div style="font-size: 0.72rem; color: {COLORS['muted_foreground']};">BUY</div>
                    <div style="font-size: 1.35rem; font-weight: 600; color: {COLORS['primary']};">{buy_count}</div>
                </div>
                <div style="padding: 12px; border-radius: 10px; border: 1px solid {COLORS['border']};">
                    <div style="font-size: 0.72rem; color: {COLORS['muted_foreground']};">SELL</div>
                    <div style="font-size: 1.35rem; font-weight: 600; color: {COLORS['destructive']};">{sell_count}</div>
                </div>
                <div style="padding: 12px; border-radius: 10px; border: 1px solid {COLORS['border']};">
                    <div style="font-size: 0.72rem; color: {COLORS['muted_foreground']};">HOLD</div>
                    <div style="font-size: 1.35rem; font-weight: 600;">{hold_count}</div>
                </div>
            </div>
            <p style="margin: 16px 0 0 0; color: {COLORS['muted_foreground']}; line-height: 1.6;">
                Use the spotlight candidate for a deep dive, then validate sizing in Virtual Trading before execution.
            </p>
        </div>
        """, unsafe_allow_html=True)

    # Sort controls
    col_sort, col_dir = st.columns([2, 1])
    with col_sort:
        sort_col = st.selectbox("Sort by", ["Symbol", "Tech Score", "R/R", "Action"], index=1, label_visibility="collapsed")
    with col_dir:
        ascending = st.checkbox("Ascending", value=False)

    if sort_col in res_df.columns:
        res_df = res_df.sort_values(sort_col, ascending=ascending, na_position='last')

    # Style the dataframe
    def style_action(val):
        if val == "BUY":
            return f'color: {COLORS["primary"]}; font-weight: 600;'
        elif val == "SELL":
            return f'color: {COLORS["destructive"]}; font-weight: 600;'
        return ''

    styled_df = res_df.style.map(style_action, subset=['Action']) if 'Action' in res_df.columns else res_df.style
    st.dataframe(
        styled_df,
        column_config={
            "Tech Score": st.column_config.ProgressColumn("Score", format="%.0f", min_value=0, max_value=100),
            "R/R": st.column_config.NumberColumn("R/R", format="%.2f"),
        },
        use_container_width=True,
        hide_index=True,
    )

    # Quick actions
    st.markdown("### Quick Actions")
    col_sym, col_act1, col_act2, col_act3 = st.columns([2, 1, 1, 1])
    with col_sym:
        action_symbol = st.selectbox("Select:", res_df['Symbol'].tolist(), key="action_sym", label_visibility="collapsed")
    with col_act1:
        if st.button("📊 Details", use_container_width=True):
            st.session_state.selected_symbol_detail = action_symbol
            st.switch_page("pages/02_stock_detail.py")
    with col_act2:
        if st.button("💹 Trade", use_container_width=True):
            st.session_state.prefill_symbol = action_symbol
            st.switch_page("pages/04_virtual_trading.py")
    with col_act3:
        if st.button("⭐ Watch", use_container_width=True):
            if 'watchlist' not in st.session_state:
                st.session_state.watchlist = []
            if action_symbol not in st.session_state.watchlist:
                st.session_state.watchlist.append(action_symbol)
                st.toast(f"Added {action_symbol} to watchlist", icon="✅")

else:
    # Display filtered stocks without analysis
    st.info("Refine the universe, then run a batch scan to rank actionable setups and surface the highest-conviction candidate.")
    display_columns = ['symbol', 'name', 'sector', 'sub_sector', 'is_lq45']
    available_columns = [col for col in display_columns if col in filtered_df.columns]
    display_df = filtered_df[available_columns].copy()

    st.dataframe(display_df, use_container_width=True, hide_index=True)

# --- Deep Dive ---
st.markdown("---")
col_c, col_d = st.columns([3, 1])
with col_c:
    # Defensive: ensure we have symbols to display
    symbols = df['symbol'].dropna().unique().tolist() if 'symbol' in df.columns else []
    if symbols:
        selected = st.selectbox("Deep dive:", sorted(symbols), label_visibility="collapsed")
    else:
        selected = None
        st.info("No symbols available for deep dive")
with col_d:
    if selected and st.button("Open Stock Details ➡️", use_container_width=True):
        st.session_state.selected_symbol_detail = selected
        st.switch_page("pages/02_stock_detail.py")
