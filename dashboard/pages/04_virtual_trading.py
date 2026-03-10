"""
Paper Trading Page - Practice trading with virtual capital and track performance.

This page provides:
- Active trading sessions management
- Session creation with different modes (live/replay)
- Order entry with validation
- Performance analytics and equity curves

Design: NextGen-style with order terminal and risk analytics.
"""
import streamlit as st
import pandas as pd
import requests
import sys
import os
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from dashboard.components.charts import build_equity_curve
from dashboard.components.metrics import render_portfolio_metrics
from dashboard.components.ux_components import (
    idx_quantity_input, idx_symbol_input, confirm_trade,
    trading_hours_indicator, error_with_recovery, success_with_details,
    api_error_handler, IDX_FEES, IDX_LOT_SIZE
)
from dashboard.components.nextgen_styles import (
    get_nextgen_css, COLORS, render_live_badge
)
from dashboard.components.top_nav import render_top_nav

# Configure logging
logger = logging.getLogger(__name__)

# Get API URL from settings or environment
try:
    from config.settings import settings
    API_URL = os.environ.get("API_URL", settings.api_url)
except (ImportError, AttributeError):
    API_URL = os.environ.get("API_URL", "http://localhost:8000")

# Constants
REQUEST_TIMEOUT = 10
LONG_REQUEST_TIMEOUT = 30

st.set_page_config(page_title="Paper Trading | IDX", page_icon="💹", layout="wide", initial_sidebar_state="collapsed")

# Apply NextGen CSS
st.markdown(get_nextgen_css(), unsafe_allow_html=True)

# --- Header ---
st.markdown(f"""
<div style="display: flex; align-items: baseline; gap: 16px; margin-bottom: 8px;">
    <h1 style="margin: 0;">Virtual Trading</h1>
    {render_live_badge("PAPER")}
</div>
<p style="color: {COLORS['muted_foreground']}; margin-bottom: 16px;">
    Practice trading with virtual capital and track performance
</p>
""", unsafe_allow_html=True)

render_top_nav("trading")
st.markdown("---")

trading_hours_indicator()
st.warning("Simulation is `beta`: session cash, positions, replay cursor, and realized P&L are persisted, but advanced risk analytics are not available in this build.")

# --- Session Helpers ---
@st.cache_data(ttl=1, show_spinner=False)
def get_sessions():
    """Fetch trading sessions from API."""
    try:
        resp = requests.get(f"{API_URL}/simulation/", timeout=REQUEST_TIMEOUT)
        if resp.status_code == 200:
            return resp.json()
    except requests.exceptions.RequestException as e:
        logger.warning(f"Failed to fetch sessions: {e}")
    return []


sessions = get_sessions()

# --- Tabs ---
tab1, tab2, tab3, tab4 = st.tabs([
    "Sessions", "New", "Trading", "Performance"
])

# === TAB 1: Active Sessions ===
with tab1:
    st.markdown('<div class="section-header">YOUR TRADING SESSIONS</div>', unsafe_allow_html=True)

    if not sessions:
        st.markdown(f"""
        <div class="nextgen-card" style="text-align: center; padding: 40px;">
            <div style="font-size: 2rem; margin-bottom: 8px;">📊</div>
            <div style="color: {COLORS['muted_foreground']};">No active sessions</div>
            <div style="font-size: 0.85rem; color: {COLORS['muted_foreground']};">Create one in the 'New' tab</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        for row in sessions:
            pnl = row.get('total_pnl', 0)
            pnl_color = COLORS['primary'] if pnl >= 0 else COLORS['destructive']
            pnl_sign = "+" if pnl >= 0 else ""

            st.markdown(f"""
            <div class="nextgen-card">
                <div style="display: flex; justify-content: space-between; align-items: start;">
                    <div>
                        <h4 style="margin: 0; color: {COLORS['foreground']};">{row.get('name', 'Unnamed')}</h4>
                        <div style="font-size: 0.75rem; color: {COLORS['muted_foreground']}; font-family: monospace;">{row['session_id']}</div>
                    </div>
                    <span class="signal-badge {'bullish' if row['status'] == 'active' else 'neutral'}">{row['status'].upper()}</span>
                </div>
                <div style="display: flex; gap: 24px; margin-top: 12px;">
                    <div>
                        <div style="font-size: 0.75rem; color: {COLORS['muted_foreground']};">MODE</div>
                        <div style="font-weight: 500;">{row['mode'].upper()} / {row['trading_mode'].title()}</div>
                    </div>
                    <div>
                        <div style="font-size: 0.75rem; color: {COLORS['muted_foreground']};">CAPITAL</div>
                        <div style="font-weight: 500;">Rp {row.get('current_capital', 0):,.0f}</div>
                    </div>
                    <div>
                        <div style="font-size: 0.75rem; color: {COLORS['muted_foreground']};">P&L</div>
                        <div style="font-weight: 600; color: {pnl_color};">{pnl_sign}Rp {pnl:,.0f}</div>
                    </div>
                    <div>
                        <div style="font-size: 0.75rem; color: {COLORS['muted_foreground']};">WIN RATE</div>
                        <div style="font-weight: 500;">{row.get('win_rate', 0) * 100:.1f}%</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            if st.button("Load Session", key=f"load_{row['session_id']}", use_container_width=True):
                st.session_state['active_session'] = row['session_id']
                st.toast(f"Loaded session {row['session_id']}", icon="✅")

# === TAB 2: New Simulation ===
with tab2:
    st.markdown('<div class="section-header">CREATE NEW SIMULATION</div>', unsafe_allow_html=True)

    with st.container():
        col1, col2 = st.columns(2)

        with col1:
            name = st.text_input("Simulation Name", "My Test Strategy")
            mode = st.selectbox("Mode", ["live", "replay"])
            capital = st.number_input("Initial Capital (IDR)", min_value=1_000_000, value=100_000_000, step=1_000_000)

        with col2:
            trading_mode = st.selectbox("Trading Mode", ["swing", "intraday", "position", "investor"])
            start_date = st.date_input("Start Date (replay only)", disabled=mode != "replay")

        st.markdown("---")

        if st.button("🚀 Create Session", type="primary", use_container_width=True):
            payload = {"name": name, "mode": mode, "trading_mode": trading_mode, "initial_capital": capital}
            if mode == "replay":
                payload["start_date"] = start_date.isoformat()
            try:
                resp = requests.post(f"{API_URL}/simulation/create", json=payload, timeout=REQUEST_TIMEOUT)
                if resp.status_code == 200:
                    sid = resp.json().get('session_id')
                    st.success(f"✅ Created! Session ID: {sid}")
                    st.session_state['active_session'] = sid
                    st.cache_data.clear()
                else:
                    st.error(f"Failed to create session: {resp.text}")
            except requests.exceptions.RequestException as e:
                st.error(f"Error: {e}")

# === TAB 3: Trading Panel ===
with tab3:
    active_sid = st.session_state.get('active_session')

    if not active_sid:
        st.warning("Please load or create a session first.")
    else:
        # Find session info
        session_info = next((s for s in sessions if s.get('session_id') == active_sid), {})
        is_replay = session_info.get('mode', '').lower() == 'replay'

        st.markdown(f"""
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;">
            <div class="section-header" style="margin: 0;">TRADING TERMINAL</div>
            <div style="font-size: 0.75rem; color: {COLORS['muted_foreground']}; font-family: monospace;">{active_sid}</div>
        </div>
        """, unsafe_allow_html=True)

        # --- Replay Controls ---
        if is_replay:
            st.markdown('<div class="section-header">⏯ REPLAY CONTROLS</div>', unsafe_allow_html=True)

            rc1, rc2, rc3 = st.columns([1, 1, 2])
            with rc1:
                if st.button("⏭ Advance 1 Day", type="primary", use_container_width=True):
                    try:
                        resp = requests.post(f"{API_URL}/simulation/{active_sid}/step", timeout=REQUEST_TIMEOUT)
                        if resp.status_code == 200:
                            st.toast(f"Advanced to {resp.json().get('current_date', 'next day')}", icon="⏭")
                            st.rerun()
                    except requests.exceptions.RequestException:
                        st.error("Could not advance.")
            with rc2:
                speed = st.selectbox("Speed", ["Manual", "1 day/sec", "1 day/5sec"])
            with rc3:
                st.metric("Current Date", session_info.get('current_date', 'N/A'))

        # --- Portfolio Summary ---
        port_data = {"capital": 0, "pnl": 0, "positions": []}
        try:
            port_resp = requests.get(f"{API_URL}/simulation/{active_sid}/portfolio", timeout=REQUEST_TIMEOUT)
            if port_resp.status_code == 200:
                port_data = port_resp.json()
                if port_data.get("feature_state") in {"demo", "beta"}:
                    st.caption(f"Portfolio source: {port_data.get('data_source', 'unknown')} | State: {port_data.get('feature_state')}")
        except requests.exceptions.RequestException as e:
            logger.warning(f"Failed to fetch portfolio: {e}")

        col_cap, col_pnl, col_trades = st.columns(3)
        with col_cap:
            st.metric("Current Capital", f"Rp {port_data.get('capital', 0):,.0f}")
        with col_pnl:
            pnl = port_data.get('pnl', 0)
            capital = port_data.get('capital', 1)
            delta_pct = f"{pnl/capital*100:+.1f}%" if capital > 0 else "N/A"
            st.metric("Open P&L", f"Rp {pnl:,.0f}", delta=delta_pct)
        with col_trades:
            pos_count = len(port_data.get('positions', []))
            st.metric("Open Positions", pos_count)

        st.markdown("---")

        # --- Order Entry Panel ---
        col_order, col_risk = st.columns([1, 2])

        with col_order:
            st.markdown('<div class="section-header">ORDER ENTRY</div>', unsafe_allow_html=True)

            prefill_sym = st.session_state.get('prefill_symbol', '')

            with st.form("order_entry"):
                symbol = st.text_input("Symbol", value=prefill_sym, max_chars=4, placeholder="e.g., BBCA").upper()
                side = st.selectbox("Side", ["BUY", "SELL"],
                                    index=0 if st.session_state.get('prefill_side', 'BUY') == 'BUY' else 1)

                qty_lots = st.number_input("Quantity (Lots)", min_value=1, value=100, step=10)
                qty = qty_lots * IDX_LOT_SIZE  # Convert to shares

                order_type = st.selectbox("Order Type", ["MARKET", "LIMIT"])
                price = st.number_input("Limit Price", min_value=0.0,
                                        value=float(st.session_state.get('prefill_price', 0)))

                # Validation
                is_valid = len(symbol) == 4 and symbol.isalpha()

                if not is_valid and symbol:
                    st.error("⚠️ Symbol must be 4 letters")

                # Estimated cost
                if price > 0 and qty > 0:
                    est_total = qty * price
                    fee_rate = IDX_FEES["buy"] if side == "BUY" else IDX_FEES["sell"]
                    est_fee = est_total * fee_rate
                    st.info(f"💰 Est. Total: Rp {est_total + est_fee:,.0f} (incl. fees)")

                col_buy, col_sell = st.columns(2)
                with col_buy:
                    buy_btn = st.form_submit_button("BUY", type="primary", use_container_width=True)
                with col_sell:
                    sell_btn = st.form_submit_button("SELL", use_container_width=True)

                if (buy_btn or sell_btn) and is_valid:
                    actual_side = "BUY" if buy_btn else "SELL"
                    payload = {
                        "symbol": symbol,
                        "side": actual_side,
                        "quantity": qty,
                        "order_type": order_type,
                        "price": float(price),
                        "targets": [],
                    }
                    try:
                        resp = requests.post(f"{API_URL}/simulation/{active_sid}/order", json=payload, timeout=LONG_REQUEST_TIMEOUT)
                        if resp.status_code == 200:
                            st.success(f"✅ {actual_side} order executed for {symbol}")
                            st.session_state.pop('prefill_symbol', None)
                            st.session_state.pop('prefill_side', None)
                            st.session_state.pop('prefill_price', None)
                        else:
                            st.error(f"Order failed: {resp.text}")
                    except requests.exceptions.RequestException as e:
                        st.error(f"Error: {e}")

        with col_risk:
            st.markdown('<div class="section-header">QUANTITATIVE RISK CHECK</div>', unsafe_allow_html=True)

            risk_col1, risk_col2, risk_col3, risk_col4 = st.columns(4)

            with risk_col1:
                st.markdown(f"""
                <div class="nextgen-card" style="text-align: center; opacity: 0.7;">
                    <div style="font-size: 0.75rem; color: {COLORS['muted_foreground']};">Empirical Kelly Optimal</div>
                    <div style="font-size: 0.9rem; font-weight: 500; color: {COLORS['muted_foreground']};">Not available in this build</div>
                    <div style="font-size: 0.7rem; color: {COLORS['muted_foreground']};">Suggested Max Allocation</div>
                </div>
                """, unsafe_allow_html=True)

            with risk_col2:
                st.markdown(f"""
                <div class="nextgen-card" style="text-align: center; opacity: 0.7;">
                    <div style="font-size: 0.75rem; color: {COLORS['muted_foreground']};">Portfolio Fit</div>
                    <div style="font-size: 0.9rem; font-weight: 500; color: {COLORS['muted_foreground']};">Not available in this build</div>
                    <div style="font-size: 0.7rem; color: {COLORS['muted_foreground']};">Sector concentration</div>
                </div>
                """, unsafe_allow_html=True)

            with risk_col3:
                st.markdown(f"""
                <div class="nextgen-card" style="text-align: center; opacity: 0.7;">
                    <div style="font-size: 0.75rem; color: {COLORS['muted_foreground']};">Monte Carlo CVaR</div>
                    <div style="font-size: 0.9rem; font-weight: 500; color: {COLORS['muted_foreground']};">Not available in this build</div>
                    <div style="font-size: 0.7rem; color: {COLORS['muted_foreground']};">Expected tail loss (95%)</div>
                </div>
                """, unsafe_allow_html=True)

            with risk_col4:
                st.markdown(f"""
                <div class="nextgen-card" style="text-align: center; opacity: 0.7;">
                    <div style="font-size: 0.75rem; color: {COLORS['muted_foreground']};">Current Volatility</div>
                    <div style="font-size: 0.9rem; font-weight: 500; color: {COLORS['muted_foreground']};">Not available in this build</div>
                    <div style="font-size: 0.7rem; color: {COLORS['muted_foreground']};">ATR status</div>
                </div>
                """, unsafe_allow_html=True)

            # Active Signals
            st.markdown('<div class="section-header" style="margin-top: 16px;">📡 ACTIVE SIGNALS</div>', unsafe_allow_html=True)

            try:
                sig_resp = requests.get(f"{API_URL}/signals/active", timeout=REQUEST_TIMEOUT)
                if sig_resp.status_code == 200:
                    signals = sig_resp.json()
                    if signals:
                        for sig in signals[:3]:
                            action = sig.get('type', 'N/A')
                            badge_class = "bullish" if action == "BUY" else "bearish" if action == "SELL" else "neutral"
                            st.markdown(f"""
                            <div class="nextgen-card" style="display: flex; justify-content: space-between; align-items: center;">
                                <div>
                                    <span style="font-weight: 600;">{sig.get('symbol', '')}</span>
                                    <span style="font-size: 0.85rem; color: {COLORS['muted_foreground']};"> @ Rp {sig.get('entry_price', 0):,.0f}</span>
                                </div>
                                <span class="signal-badge {badge_class}">{action}</span>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.info("No active signals")
            except requests.exceptions.RequestException:
                st.info("Signals unavailable")

        st.markdown("---")

        # --- Open Positions Table ---
        st.markdown('<div class="section-header">OPEN POSITIONS</div>', unsafe_allow_html=True)

        positions = port_data.get("positions", [])
        if positions:
            pos_df = pd.DataFrame(positions)
            st.dataframe(pos_df, use_container_width=True, hide_index=True)
        else:
            st.info("No open positions")

        # --- Trade History ---
        st.markdown('<div class="section-header">TRADE HISTORY</div>', unsafe_allow_html=True)

        try:
            hist_resp = requests.get(f"{API_URL}/simulation/{active_sid}/history", timeout=REQUEST_TIMEOUT)
            if hist_resp.status_code == 200:
                hist = hist_resp.json()
                if hist:
                    st.dataframe(pd.DataFrame(hist), use_container_width=True, hide_index=True)
                else:
                    st.info("No trades yet")
        except requests.exceptions.RequestException:
            st.info("Could not fetch history")

# === TAB 4: Performance ===
with tab4:
    active_sid = st.session_state.get('active_session')

    if not active_sid:
        st.warning("Please load or create a session first.")
    else:
        st.markdown(f"""
        <div class="section-header">PERFORMANCE REPORT — {active_sid}</div>
        """, unsafe_allow_html=True)

        # Fetch metrics
        try:
            met_resp = requests.get(f"{API_URL}/simulation/{active_sid}/metrics", timeout=REQUEST_TIMEOUT)
            if met_resp.status_code == 200:
                metrics = met_resp.json()
                render_portfolio_metrics(
                    capital=metrics.get('current_capital', 0),
                    pnl=metrics.get('total_pnl', 0),
                    win_rate=metrics.get('win_rate', 0),
                    total_trades=metrics.get('total_trades', 0),
                    sharpe=metrics.get('sharpe_ratio'),
                    max_dd=metrics.get('max_drawdown'),
                    profit_factor=metrics.get('profit_factor'),
                )
            else:
                session_info = next((s for s in sessions if s.get('session_id') == active_sid), {})
                render_portfolio_metrics(
                    capital=session_info.get('current_capital', 100_000_000),
                    pnl=session_info.get('total_pnl', 0),
                    win_rate=session_info.get('win_rate', 0),
                    total_trades=session_info.get('total_trades', 0),
                )
        except requests.exceptions.RequestException:
            st.info("Metrics unavailable")

        st.markdown("---")

        # Equity Curve
        st.markdown('<div class="section-header">📈 EQUITY CURVE</div>', unsafe_allow_html=True)

        try:
            eq_resp = requests.get(f"{API_URL}/simulation/{active_sid}/equity-curve", timeout=REQUEST_TIMEOUT)
            if eq_resp.status_code == 200:
                eq_data = eq_resp.json()
                if eq_data:
                    fig = build_equity_curve(
                        [d['date'] for d in eq_data],
                        [d['value'] for d in eq_data],
                        title="Portfolio Value",
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Not enough data for equity curve")
            else:
                st.info("Equity curve not available")
        except requests.exceptions.RequestException:
            st.info("Could not fetch equity curve")
