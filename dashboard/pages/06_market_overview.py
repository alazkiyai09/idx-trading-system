"""
Market Overview Page - Sector heatmap and stock universe browser.

Design: NextGen-style with interactive treemap and stock list.
"""
import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import sys
import os
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Configure logging
logger = logging.getLogger(__name__)

# Get API URL
try:
    from config.settings import settings
    API_URL = os.environ.get("API_URL", settings.api_url)
except Exception:
    API_URL = os.environ.get("API_URL", "http://localhost:8000")

from dashboard.components.ux_components import trading_hours_indicator
from dashboard.components.nextgen_styles import get_nextgen_css, COLORS, render_live_badge
from dashboard.components.top_nav import render_top_nav

# Constants
REQUEST_TIMEOUT = 10

st.set_page_config(page_title="Market Overview | IDX", page_icon="🌐", layout="wide", initial_sidebar_state="collapsed")

# Apply NextGen CSS
st.markdown(get_nextgen_css(), unsafe_allow_html=True)

# --- Header ---
st.markdown(f"""
<div style="display: flex; align-items: baseline; gap: 16px; margin-bottom: 8px;">
    <h1 style="margin: 0;">Market Overview</h1>
    {render_live_badge("657 STOCKS")}
</div>
<p style="color: {COLORS['muted_foreground']}; margin-bottom: 16px;">
    Visualize market with sector heatmap and browse all IDX stocks
</p>
""", unsafe_allow_html=True)

render_top_nav("overview")
st.markdown("---")

trading_hours_indicator()


def get_all_stocks():
    """Fetch all stocks from API with caching."""
    try:
        response = requests.get(f"{API_URL}/stocks", timeout=REQUEST_TIMEOUT)
        if response.status_code == 200:
            data = response.json()
            return data.get("stocks", data) if isinstance(data, dict) else data
        logger.warning(f"Failed to fetch stocks: status {response.status_code}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching stocks: {e}")
    return []


stocks_data = get_all_stocks()

if not stocks_data:
    st.warning("No stock data available. Is the API running?")
    if st.button("Retry loading market data", type="primary", use_container_width=False):
        st.cache_data.clear()
        st.rerun()
    st.stop()

df = pd.DataFrame(stocks_data)

# Ensure data quality
df['market_cap'] = pd.to_numeric(df['market_cap'], errors='coerce').fillna(0)
df['sector'] = df['sector'].fillna('Unknown')
df['sub_sector'] = df['sub_sector'].fillna('Unknown')
if 'is_lq45' not in df.columns:
    df['is_lq45'] = False
if 'change_pct' not in df.columns:
    df['change_pct'] = 0.0
df['change_pct'] = pd.to_numeric(df['change_pct'], errors='coerce').fillna(0)

# --- Summary Metrics ---
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Stocks", f"{len(df):,}")
with col2:
    lq45_count = len(df[df['is_lq45'] == True]) if 'is_lq45' in df.columns else 0
    st.metric("LQ45 Stocks", lq45_count)
with col3:
    sectors = df['sector'].nunique()
    st.metric("Sectors", sectors)
with col4:
    total_mcap = df['market_cap'].sum() / 1e12  # Trillions
    st.metric("Total Market Cap", f"Rp {total_mcap:.1f}T")

st.markdown("---")

# --- Tabs ---
tab1, tab2 = st.tabs(["📊 Sector Heatmap", "📋 Stock List"])

with tab1:
    st.markdown('<div class="section-header">MARKET HEATMAP BY SECTOR</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <p style="color: {COLORS['muted_foreground']}; font-size: 0.8rem; margin-bottom: 16px;">
        Block size represents market capitalization. Color indicates daily price change.
        <span style="color: {COLORS['primary']};">Green = Gain</span> |
        <span style="color: {COLORS['destructive']};">Red = Loss</span>
    </p>
    """, unsafe_allow_html=True)

    # Create Treemap (filter out zero market cap)
    treemap_df = df[df['market_cap'] > 0].copy()
    fig = px.treemap(
        treemap_df,
        path=[px.Constant("IDX Market"), 'sector', 'sub_sector', 'symbol'],
        values='market_cap',
        color='change_pct',
        color_continuous_scale=[COLORS['destructive'], COLORS['muted_foreground'], COLORS['primary']],
        color_continuous_midpoint=0,
        hover_data=['name', 'market_cap', 'change_pct'],
    )

    fig.update_traces(
        root_color=COLORS['muted'],
        textfont=dict(color=COLORS['foreground']),
        hovertemplate="<b>%{label}</b><br>Market Cap: %{customdata[1]:.2e}<br>Change: %{customdata[2]:.2f}%<extra></extra>",
    )
    fig.update_layout(
        paper_bgcolor=COLORS['background'],
        plot_bgcolor=COLORS['background'],
        font=dict(color=COLORS['foreground']),
        margin=dict(t=10, l=10, r=10, b=10),
        height=600,
        coloraxis_colorbar=dict(title="% Change", tickfont=dict(color=COLORS['muted_foreground'])),
    )

    st.plotly_chart(fig, use_container_width=True)

    # Sector summary cards
    st.markdown('<div class="section-header">SECTOR PERFORMANCE</div>', unsafe_allow_html=True)

    sector_summary = df.groupby('sector').agg({
        'change_pct': 'mean',
        'market_cap': 'sum',
        'symbol': 'count'
    }).reset_index()
    sector_summary.columns = ['Sector', 'Avg Change', 'Market Cap', 'Stocks']
    sector_summary = sector_summary.sort_values('Market Cap', ascending=False)

    cols = st.columns(min(len(sector_summary), 4))
    for idx, (_, row) in enumerate(sector_summary.head(8).iterrows()):
        col_idx = idx % 4
        with cols[col_idx]:
            change = row['Avg Change']
            change_color = COLORS['primary'] if change >= 0 else COLORS['destructive']
            st.markdown(f"""
            <div class="nextgen-card" style="text-align: center;">
                <div style="font-size: 0.75rem; color: {COLORS['muted_foreground']};">{row['Sector']}</div>
                <div style="font-size: 1.1rem; font-weight: 600; color: {change_color};">{change:+.2f}%</div>
                <div style="font-size: 0.75rem; color: {COLORS['muted_foreground']};">{row['Stocks']} stocks</div>
            </div>
            """, unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="section-header">STOCK UNIVERSE</div>', unsafe_allow_html=True)

    # Filters
    col_f1, col_f2, col_f3 = st.columns([2, 2, 1])
    with col_f1:
        sector_filter = st.selectbox("Sector", ["All"] + sorted(df['sector'].unique().tolist()))
    with col_f2:
        lq45_only = st.checkbox("LQ45 Only")
    with col_f3:
        search = st.text_input("Search", placeholder="Symbol...")

    # Apply filters
    filtered_df = df.copy()
    if sector_filter != "All":
        filtered_df = filtered_df[filtered_df['sector'] == sector_filter]
    if lq45_only and 'is_lq45' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['is_lq45'] == True]
    if search:
        filtered_df = filtered_df[filtered_df['symbol'].str.contains(search.upper(), na=False)]

    st.markdown(f"**{len(filtered_df)}** stocks found")

    # Display table
    display_df = filtered_df[['symbol', 'name', 'sector', 'sub_sector', 'is_lq45', 'market_cap', 'change_pct']].copy()
    display_df['market_cap_bn'] = (display_df['market_cap'] / 1e9).round(2)
    display_df = display_df.drop('market_cap', axis=1)

    def highlight_change(val):
        if pd.isna(val):
            return ''
        color = COLORS['primary'] if val >= 0 else COLORS['destructive']
        return f'color: {color}; font-weight: 500;'

    st.dataframe(
        display_df.style.map(highlight_change, subset=['change_pct']),
        column_config={
            "symbol": st.column_config.TextColumn("Symbol", width="small"),
            "name": st.column_config.TextColumn("Company", width="large"),
            "sector": st.column_config.TextColumn("Sector", width="medium"),
            "sub_sector": st.column_config.TextColumn("Sub-Sector", width="medium"),
            "is_lq45": st.column_config.CheckboxColumn("LQ45"),
            "market_cap_bn": st.column_config.NumberColumn("M.Cap (B)", format="%.2f"),
            "change_pct": st.column_config.NumberColumn("Change", format="%.2f%%"),
        },
        use_container_width=True,
        hide_index=True,
    )

    # Quick action
    st.markdown("---")
    col_c, col_d = st.columns([3, 1])
    with col_c:
        selected = st.selectbox("Select for details:", sorted(filtered_df['symbol'].unique()), label_visibility="collapsed")
    with col_d:
        if st.button("Open Stock Details ➡️", use_container_width=True):
            st.session_state.selected_symbol_detail = selected
            st.switch_page("pages/02_stock_detail.py")
