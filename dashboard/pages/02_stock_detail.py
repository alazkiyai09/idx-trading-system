"""
Stock Detail Page - Comprehensive single-stock analysis dashboard.

This page provides a multi-tab interface for analyzing individual IDX stocks:
- Overview: Company information and price sparkline
- Price Chart: Interactive candlestick with technical overlays and ML predictions
- Analysis: Technical analysis and signal generation
- Risk Validation: Risk manager trade validation
- News: Sentiment analysis from news articles
- Foreign Flow: Foreign investor activity tracking
- AI Analysis: Multi-agent LLM fundamental analysis

Design: NextGen-style with three-panel layout.
"""
import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
import logging
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from dashboard.components.charts import build_candlestick_chart
from dashboard.components.metrics import render_stock_info_card, render_signal_card
from dashboard.components.ux_components import (
    trading_hours_indicator, info_card, collapsible_section,
    help_tooltip, error_with_recovery, success_with_details,
    api_error_handler, render_stock_selector
)
from dashboard.components.nextgen_styles import (
    get_nextgen_css, COLORS, render_live_badge, render_signal_badge,
    get_chart_colors
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
REQUEST_TIMEOUT = 10  # seconds
LONG_REQUEST_TIMEOUT = 120  # seconds for AI analysis
MIN_CHART_DAYS = 30
MAX_CHART_DAYS = 500
DEFAULT_CHART_DAYS = 200
CHART_DAYS_STEP = 10
DEFAULT_CAPITAL = 100_000_000.0  # 100 million IDR
MIN_CAPITAL = 10_000_000.0  # 10 million IDR
CAPITAL_STEP = 10_000_000.0

st.set_page_config(page_title="Stock Analysis | IDX", page_icon="📈", layout="wide", initial_sidebar_state="collapsed")

# Apply NextGen CSS
st.markdown(get_nextgen_css(), unsafe_allow_html=True)


@st.cache_data(ttl=300, show_spinner=False)
def get_stock_list() -> list:
    """Fetch stock list from API with caching."""
    try:
        resp = requests.get(f"{API_URL}/stocks/symbols", timeout=REQUEST_TIMEOUT)
        if resp.status_code == 200:
            data = resp.json()
            return data.get("symbols", []) or ["BBCA"]
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching stock list: {e}")
    return ["BBCA"]


@st.cache_data(ttl=60, show_spinner=False)
def get_stock_details(symbol: str) -> dict:
    """Fetch stock details from API."""
    try:
        resp = requests.get(f"{API_URL}/stocks/{symbol}", timeout=REQUEST_TIMEOUT)
        if resp.status_code == 200:
            return resp.json()
    except requests.exceptions.RequestException as e:
        logger.warning(f"Failed to fetch stock details: {e}")
    return {}


# --- Symbol Selection ---
all_stocks = get_stock_list()
default_symbol = st.session_state.get('selected_symbol_detail', '')

# --- Header with Symbol Selection ---
col_header, col_symbol = st.columns([3, 1])
with col_symbol:
    selected_symbol = render_stock_selector(
        label="Symbol",
        default_symbol=default_symbol,
        api_url=API_URL,
        label_visibility="collapsed",
        allow_empty=True,
    )

# Update session state when selection changes
if selected_symbol != default_symbol:
    st.session_state['selected_symbol_detail'] = selected_symbol

if not selected_symbol:
    st.info("Select a stock to view details.")
    st.stop()

# Fetch stock details
details = get_stock_details(selected_symbol)
stock_name = details.get('name', selected_symbol)
stock_sector = details.get('sector', 'N/A')
latest_price = details.get('latest_price', {})
current_price = latest_price.get('close', 0) if isinstance(latest_price, dict) else 0

# --- Header Section ---
st.markdown(f"""
<div style="display: flex; align-items: baseline; gap: 16px; margin-bottom: 8px;">
    <h1 style="margin: 0;">{selected_symbol}</h1>
    <span style="color: {COLORS['muted_foreground']}; font-size: 0.9rem;">{stock_name}</span>
    {render_live_badge("LIVE")}
</div>
<p style="color: {COLORS['muted_foreground']}; margin-bottom: 16px;">
    {stock_sector} | {details.get('sub_sector', 'N/A')}
</p>
""", unsafe_allow_html=True)

render_top_nav("detail")
st.markdown("---")

# --- Metrics Row ---
met1, met2, met3, met4, met5 = st.columns(5)
with met1:
    st.metric("Latest Close", f"Rp {current_price:,.0f}" if current_price else "—")
with met2:
    change_pct = latest_price.get('change_pct', 0) if isinstance(latest_price, dict) else 0
    st.metric("Change", f"{change_pct:+.2f}%")
with met3:
    st.metric("Volume", f"{latest_price.get('volume', 0)/1e6:.1f}M" if isinstance(latest_price, dict) else "—")
with met4:
    if details.get('is_lq45'):
        st.metric("Index", "LQ45")
    elif details.get('is_idx30'):
        st.metric("Index", "IDX30")
    else:
        st.metric("Index", "—")
with met5:
    mcap = details.get('market_cap', 0)
    st.metric("Market Cap", f"Rp {mcap/1e12:.2f}T" if mcap > 0 else "—")

st.markdown("---")

# --- Chart Controls ---
with st.expander("Chart Controls", expanded=False):
    ctl1, ctl2, ctl3 = st.columns(3)
    with ctl1:
        days_history = st.slider("Chart Days", MIN_CHART_DAYS, MAX_CHART_DAYS, DEFAULT_CHART_DAYS, CHART_DAYS_STEP)
    with ctl2:
        show_volume = st.checkbox("Volume Bars", True)
        show_ma = st.checkbox("MA 20/50", True)
        show_bbands = st.checkbox("Bollinger Bands", False)
    with ctl3:
        show_rsi = st.checkbox("RSI Subplot", False)
        show_macd = st.checkbox("MACD Subplot", False)
        show_prediction = st.checkbox("Show ML Prediction", True)
        show_confidence_band = st.checkbox("Confidence Band", False)

trading_hours_indicator()

# --- Main Content: Two Column Layout ---
col_chart, col_analysis = st.columns([2, 1])

with col_chart:
    st.markdown('<div class="section-header">📈 PRICE CHART</div>', unsafe_allow_html=True)

    # Fetch and render chart
    try:
        chart_resp = requests.get(f"{API_URL}/stocks/{selected_symbol}/chart?days={days_history}", timeout=REQUEST_TIMEOUT)
        if chart_resp.status_code == 200:
            chart_data = chart_resp.json()
            if chart_data:
                df = pd.DataFrame(chart_data)
                df['date'] = pd.to_datetime(df['date'])

                # Fetch prediction if enabled
                pred_data = None
                if show_prediction:
                    try:
                        pred_resp = requests.get(f"{API_URL}/prediction/ensemble/{selected_symbol}", timeout=REQUEST_TIMEOUT)
                        if pred_resp.status_code == 200:
                            pred_data = pred_resp.json()
                    except requests.exceptions.RequestException:
                        pass

                fig = build_candlestick_chart(
                    df, title="",
                    show_volume=show_volume, show_ma=show_ma,
                    show_rsi=show_rsi, show_macd=show_macd,
                    show_bbands=show_bbands,
                    prediction_data=pred_data,
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No price data available.")
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to load chart: {e}")

with col_analysis:
    # --- Intelligence Panel ---
    st.markdown(f"""
    <div class="section-header">
        <span style="color: {COLORS['primary']};">🧠</span> INTELLIGENCE
    </div>
    """, unsafe_allow_html=True)

    # Tabs for analysis
    tab_overview, tab_tech, tab_fundamental, tab_sentiment, tab_flow = st.tabs([
        "Overview", "Technicals", "AI", "Sentiment", "Flow"
    ])

    with tab_overview:
        # Conviction Score Card
        st.markdown(f"""
        <div class="nextgen-card">
            <div class="section-header">CONVICTION SCORE</div>
            <div style="display: flex; align-items: center; gap: 16px;">
                <div class="conviction-score" style="color: {COLORS['primary']};">—</div>
                <div style="flex: 1;">
                    <div class="conviction-bar">
                        <div class="fill" style="width: 0%;"></div>
                    </div>
                    <div style="font-size: 0.75rem; color: {COLORS['muted_foreground']}; margin-top: 8px;">
                        Run analysis to generate score
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Quick Actions
        st.markdown("#### Quick Actions")
        if st.button("▶ Run Full Analysis", use_container_width=True):
            st.toast("Starting analysis...", icon="🔄")

    with tab_tech:
        st.markdown('<div class="section-header">📊 TECHNICAL INDICATORS</div>', unsafe_allow_html=True)

        # Technical indicator grid
        st.markdown(f"""
        <div class="indicator-grid">
            <div class="indicator-card">
                <div class="label">RSI (14)</div>
                <div class="value">—</div>
            </div>
            <div class="indicator-card">
                <div class="label">MACD</div>
                <div class="value">—</div>
            </div>
            <div class="indicator-card">
                <div class="label">EMA 20</div>
                <div class="value">—</div>
            </div>
            <div class="indicator-card">
                <div class="label">SMA 200</div>
                <div class="value">—</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("Run Technical Analysis", type="primary", use_container_width=True):
            with st.spinner("Analyzing..."):
                try:
                    resp = requests.post(f"{API_URL}/analysis/technical/{selected_symbol}", timeout=REQUEST_TIMEOUT)
                    if resp.status_code == 200:
                        data = resp.json()
                        score = data.get('score', {})
                        indicators = data.get('indicators', {})

                        # Update display with results
                        st.success(f"Score: {score.get('total', 0):.1f}/100")
                        st.markdown(f"**Trend:** {score.get('trend', 'N/A').upper()}")

                        col1, col2 = st.columns(2)
                        col1.metric("Support", f"Rp {indicators.get('support', 0):,.0f}")
                        col2.metric("Resistance", f"Rp {indicators.get('resistance', 0):,.0f}")
                        st.metric("RSI", f"{indicators.get('rsi', 0):.1f}")
                    else:
                        st.error("Analysis failed")
                except requests.exceptions.RequestException as e:
                    st.error(f"Error: {e}")

        # Support/Resistance levels
        st.markdown('<div class="section-header" style="margin-top: 16px;">SUPPORT & RESISTANCE</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="nextgen-card" style="padding: 0;">
            <div class="sr-level"><span>Resistance 2</span><span class="price-mono">—</span></div>
            <div class="sr-level"><span>Resistance 1</span><span class="price-mono">—</span></div>
            <div class="sr-level current"><span>Current Price</span><span class="price-mono">{current_price:,.0f}</span></div>
            <div class="sr-level"><span>Support 1</span><span class="price-mono">—</span></div>
            <div class="sr-level"><span>Support 2</span><span class="price-mono">—</span></div>
        </div>
        """, unsafe_allow_html=True)

    with tab_fundamental:
        st.markdown('<div class="section-header">🤖 MULTI-AGENT AI ANALYSIS</div>', unsafe_allow_html=True)

        if st.button("🧠 Trigger Full PDF Financial Audit", use_container_width=True):
            with st.spinner("Running multi-agent analysis (30-60s)..."):
                try:
                    resp = requests.post(
                        f"{API_URL}/fundamental/analyze",
                        json={"symbol": selected_symbol},
                        timeout=LONG_REQUEST_TIMEOUT
                    )
                    if resp.status_code == 200:
                        result = resp.json()
                        for agent_name, output in result.items():
                            is_synthesis = agent_name.lower() == "synthesis"
                            agent_class = "synthesizer" if is_synthesis else "auditor"
                            with st.expander(f"📝 {agent_name}", expanded=is_synthesis):
                                st.markdown(f"""
                                <div class="agent-card {agent_class}">
                                    <p style="margin: 0; font-size: 0.85rem;">{output}</p>
                                </div>
                                """, unsafe_allow_html=True)
                    else:
                        st.warning("Fundamental analysis service unavailable.")
                except requests.exceptions.RequestException as e:
                    st.warning(f"Service unavailable: {e}")

        # Agent cards placeholder
        st.markdown(f"""
        <div class="agent-card auditor">
            <h4 style="margin: 0 0 8px 0; font-size: 0.8rem; color: {COLORS['foreground']};">🔍 Auditor Agent</h4>
            <p style="margin: 0; font-size: 0.75rem; color: {COLORS['muted_foreground']};">
                Run analysis to detect accounting irregularities...
            </p>
        </div>
        <div class="agent-card growth">
            <h4 style="margin: 0 0 8px 0; font-size: 0.8rem; color: {COLORS['foreground']};">📈 Value & Growth Agents</h4>
            <p style="margin: 0; font-size: 0.75rem; color: {COLORS['muted_foreground']};">
                P/E, revenue growth, and intrinsic value analysis...
            </p>
        </div>
        <div class="agent-card synthesizer">
            <h4 style="margin: 0 0 8px 0; font-size: 0.8rem; color: {COLORS['primary']};">⭐ Synthesizer Consensus</h4>
            <p style="margin: 0; font-size: 0.75rem; color: {COLORS['muted_foreground']};">
                Final verdict pending analysis...
            </p>
        </div>
        """, unsafe_allow_html=True)

    with tab_sentiment:
        st.markdown('<div class="section-header">📰 NEWS & THEME SENTIMENT</div>', unsafe_allow_html=True)

        # Sentiment metrics
        col_bull, col_bear = st.columns(2)
        with col_bull:
            st.markdown(f"""
            <div class="nextgen-card" style="text-align: center; background: rgba(16, 185, 129, 0.1); border-color: rgba(16, 185, 129, 0.3);">
                <div style="font-size: 0.75rem; color: {COLORS['muted_foreground']};">Bullish Sources</div>
                <div style="font-size: 1.5rem; font-weight: 600; color: {COLORS['primary']};">—</div>
            </div>
            """, unsafe_allow_html=True)
        with col_bear:
            st.markdown(f"""
            <div class="nextgen-card" style="text-align: center;">
                <div style="font-size: 0.75rem; color: {COLORS['muted_foreground']};">Bearish Sources</div>
                <div style="font-size: 1.5rem; font-weight: 600; color: {COLORS['muted_foreground']};">—</div>
            </div>
            """, unsafe_allow_html=True)

        if st.button("Fetch Latest Headlines", use_container_width=True):
            with st.spinner("Fetching news..."):
                try:
                    resp = requests.post(f"{API_URL}/sentiment/fetch/{selected_symbol}", timeout=REQUEST_TIMEOUT)
                    if resp.status_code == 200:
                        st.success("News fetched successfully")
                        st.rerun()
                except requests.exceptions.RequestException as e:
                    st.error(f"Error: {e}")

    with tab_flow:
        st.markdown('<div class="section-header">💱 FOREIGN ACCUMULATION</div>', unsafe_allow_html=True)

        # Foreign flow summary card
        st.markdown(f"""
        <div class="flow-card">
            <div style="font-size: 0.75rem; color: {COLORS['muted_foreground']}; margin-bottom: 4px;">Net Foreign Flow (5-Day)</div>
            <div class="value">— <span style="font-size: 0.9rem; color: {COLORS['muted_foreground']};">IDR</span></div>
        </div>
        """, unsafe_allow_html=True)

        # Fetch actual flow data
        try:
            flow_resp = requests.get(f"{API_URL}/stocks/{selected_symbol}/foreign-flow", timeout=REQUEST_TIMEOUT)
            if flow_resp.status_code == 200:
                flow_data = flow_resp.json()
                if flow_data:
                    flow_df = pd.DataFrame(flow_data)
                    if 'foreign_net' in flow_df.columns:
                        net_5d = flow_df['foreign_net'].tail(5).sum()
                        net_color = COLORS['primary'] if net_5d >= 0 else COLORS['destructive']
                        st.metric("5-Day Net Flow", f"Rp {net_5d/1e9:.1f}B")
        except requests.exceptions.RequestException:
            pass

# --- Bottom Panel: Order Entry & Risk ---
st.markdown("---")
st.markdown('<div class="section-header">💹 TERMINAL / ORDER ENTRY</div>', unsafe_allow_html=True)

col_order, col_risk = st.columns([1, 2])

with col_order:
    st.markdown(f"""
    <div style="font-size: 0.75rem; color: {COLORS['muted_foreground']}; margin-bottom: 8px;">
        ORDER ENTRY
    </div>
    """, unsafe_allow_html=True)

    order_mode = st.selectbox("Trading Mode", ["SWING", "INTRADAY", "POSITION"], key="order_mode")
    qty_lots = st.number_input("Quantity (Lots)", min_value=1, value=1, step=1, help="1 Lot = 100 shares")
    limit_price = st.text_input("Limit Price", placeholder="Market")

    col_buy, col_sell = st.columns(2)
    with col_buy:
        if st.button("BUY", type="primary", use_container_width=True):
            st.toast(f"Preparing BUY order for {selected_symbol}", icon="✅")
    with col_sell:
        if st.button("SELL", use_container_width=True):
            st.toast(f"Preparing SELL order for {selected_symbol}", icon="⚠️")

with col_risk:
    st.markdown('<div class="section-header">QUANTITATIVE RISK CHECK</div>', unsafe_allow_html=True)

    risk_col1, risk_col2, risk_col3, risk_col4 = st.columns(4)

    with risk_col1:
        st.markdown(f"""
        <div class="nextgen-card" style="text-align: center;">
            <div style="font-size: 0.75rem; color: {COLORS['muted_foreground']};">Empirical Kelly Optimal</div>
            <div style="font-size: 1.1rem; font-weight: 600; color: {COLORS['primary']};">—%</div>
            <div style="font-size: 0.75rem; color: {COLORS['muted_foreground']};">Suggested Max Allocation</div>
        </div>
        """, unsafe_allow_html=True)

    with risk_col2:
        st.markdown(f"""
        <div class="nextgen-card" style="text-align: center;">
            <div style="font-size: 0.75rem; color: {COLORS['muted_foreground']};">Portfolio Fit</div>
            <div style="font-size: 1.1rem; font-weight: 600; color: {COLORS['primary']};">—</div>
            <div style="font-size: 0.75rem; color: {COLORS['muted_foreground']};">Sector concentration check</div>
        </div>
        """, unsafe_allow_html=True)

    with risk_col3:
        st.markdown(f"""
        <div class="nextgen-card" style="text-align: center;">
            <div style="font-size: 0.75rem; color: {COLORS['muted_foreground']};">Monte Carlo CVaR</div>
            <div style="font-size: 1.1rem; font-weight: 600;">—%</div>
            <div style="font-size: 0.75rem; color: {COLORS['muted_foreground']};">Expected tail loss (95%)</div>
        </div>
        """, unsafe_allow_html=True)

    with risk_col4:
        st.markdown(f"""
        <div class="nextgen-card" style="text-align: center;">
            <div style="font-size: 0.75rem; color: {COLORS['muted_foreground']};">Current Volatility</div>
            <div style="font-size: 1.1rem; font-weight: 600; color: {COLORS['warning']};">—</div>
            <div style="font-size: 0.75rem; color: {COLORS['muted_foreground']};">ATR status</div>
        </div>
        """, unsafe_allow_html=True)

    # Risk validation button
    if st.button("Validate Trade", use_container_width=True):
        with st.spinner("Validating..."):
            try:
                resp = requests.post(
                    f"{API_URL}/analysis/risk-check/{selected_symbol}",
                    json={"mode": order_mode, "capital": DEFAULT_CAPITAL},
                    timeout=REQUEST_TIMEOUT,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("approved"):
                        st.success(f"✅ Approved - Size: Rp {data.get('position_size', 0):,.0f}")
                    else:
                        st.error(f"❌ Vetoed: {', '.join(data.get('reasons', ['Unknown']))}")
            except requests.exceptions.RequestException as e:
                st.error(f"Validation error: {e}")
