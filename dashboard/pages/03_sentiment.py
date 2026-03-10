"""
Sentiment Analysis Page - AI-powered news sentiment with sector trend analysis.

Design: NextGen-style with sentiment gauge and news feed.
"""
import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import sys
import os
import logging
import html
from urllib.parse import urlparse
from typing import Optional, Any
import concurrent.futures

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from dashboard.components.charts import build_sentiment_gauge
from dashboard.components.ux_components import trading_hours_indicator
from dashboard.components.nextgen_styles import get_nextgen_css, COLORS, render_live_badge
from dashboard.components.top_nav import render_top_nav

# Configure logging
logger = logging.getLogger(__name__)

# Get API URL
try:
    from config.settings import settings
    API_URL = os.environ.get("API_URL", settings.api_url)
except Exception:
    API_URL = os.environ.get("API_URL", "http://localhost:8000")

# Constants
REQUEST_TIMEOUT = 10
LONG_REQUEST_TIMEOUT = 30
ALLOWED_URL_SCHEMES = {'http', 'https'}

st.set_page_config(page_title="Sentiment Analysis | IDX", page_icon="📰", layout="wide", initial_sidebar_state="collapsed")

# Apply NextGen CSS
st.markdown(get_nextgen_css(), unsafe_allow_html=True)

# --- Header ---
st.markdown(f"""
<div style="display: flex; align-items: baseline; gap: 16px; margin-bottom: 8px;">
    <h1 style="margin: 0;">Sentiment Analysis</h1>
    {render_live_badge("AI")}
</div>
<p style="color: {COLORS['muted_foreground']}; margin-bottom: 16px;">
    AI-powered news sentiment with sector trend analysis
</p>
""", unsafe_allow_html=True)

render_top_nav("sentiment")
st.markdown("---")

trading_hours_indicator()


def is_safe_url(url: str) -> bool:
    """Validate URL to prevent XSS attacks."""
    if not url:
        return False
    try:
        parsed = urlparse(url)
        return parsed.scheme.lower() in ALLOWED_URL_SCHEMES
    except Exception:
        return False


# --- Fetch Functions ---
@st.cache_data(ttl=300, show_spinner=False)
def _fetch_api(endpoint: str) -> Optional[Any]:
    """Generic API fetch helper."""
    try:
        resp = requests.get(f"{API_URL}{endpoint}", timeout=REQUEST_TIMEOUT)
        if resp.status_code == 200:
            return resp.json()
    except requests.exceptions.RequestException as e:
        logger.warning(f"Error fetching {endpoint}: {e}")
    return None

def get_sector_sentiment():
    return _fetch_api("/sentiment/sector")

def get_themes():
    return _fetch_api("/sentiment/themes")

def get_latest_articles():
    return _fetch_api("/sentiment/latest")


articles_data = get_latest_articles()
if articles_data and isinstance(articles_data, dict):
    articles = articles_data.get('articles', [])
elif articles_data and isinstance(articles_data, list):
    articles = articles_data
else:
    articles = []


# --- Main Layout ---
col_gauge, col_actions = st.columns([2, 1])

with col_gauge:
    st.markdown('<div class="section-header">MARKET-WIDE SENTIMENT</div>', unsafe_allow_html=True)

    sector_data = get_sector_sentiment()
    if sector_data:
        avg_score = sum(s.get('avg_score', 50) for s in sector_data) / max(len(sector_data), 1)
    else:
        avg_score = 50

    fig = build_sentiment_gauge(avg_score, "Market Sentiment Score")
    st.plotly_chart(fig, use_container_width=True)

with col_actions:
    st.markdown('<div class="section-header">QUICK ACTIONS</div>', unsafe_allow_html=True)

    def fetch_single_symbol(sym: str) -> tuple:
        try:
            resp = requests.post(f"{API_URL}/sentiment/fetch/{sym}", timeout=LONG_REQUEST_TIMEOUT)
            return (sym, resp.status_code == 200)
        except requests.exceptions.RequestException:
            return (sym, False)

    if st.button("📰 Fetch All News", type="primary", use_container_width=True):
        watchlist = st.session_state.get('watchlist', ['BBCA', 'BBRI', 'TLKM', 'ASII'])
        prog = st.progress(0, text=f"Fetching 0/{len(watchlist)}...")

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(fetch_single_symbol, sym): sym for sym in watchlist}
            success_count = 0
            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                sym, success = future.result()
                if success:
                    success_count += 1
                prog.progress((i + 1) / len(watchlist), text=f"Completed {sym}...")

        prog.empty()
        st.success(f"Fetched {success_count}/{len(watchlist)} symbols!")
        st.cache_data.clear()

    st.markdown("---")

    st.markdown('<div class="section-header">DATA MANAGEMENT</div>', unsafe_allow_html=True)
    days_keep = st.slider("Delete articles older than (days)", 1, 90, 30,
                          help="Articles older than this will be permanently deleted")

    # Confirmation flow for destructive action
    if st.button("🗑️ Clear Old Data", use_container_width=True):
        st.session_state["confirm_cleanup"] = True

    if st.session_state.get("confirm_cleanup"):
        st.warning(f"⚠️ Click again to confirm deletion of articles older than {days_keep} days")
        if st.button("✅ Confirm Delete", type="primary", use_container_width=True):
            try:
                resp = requests.delete(f"{API_URL}/sentiment/cleanup?days={days_keep}", timeout=REQUEST_TIMEOUT)
                if resp.status_code == 200:
                    d = resp.json().get('deleted', {})
                    st.success(f"Cleaned: {d.get('records_deleted', 0)} articles")
                    st.cache_data.clear()
                else:
                    st.error("Cleanup failed")
            except requests.exceptions.RequestException:
                st.error("Cleanup request failed")
            finally:
                st.session_state["confirm_cleanup"] = False

st.markdown("---")

# --- Data Transparency Panel ---
with st.expander("📊 How Sentiment Scores Are Calculated", expanded=False):
    st.markdown("""
    ### Data Sources

    | Source | Type | Update Frequency |
    |--------|------|------------------|
    | Google News RSS | News Headlines | Real-time |
    | LLM Analysis | Sentiment Scoring | On-demand |

    ### Scoring Methodology

    1. **Article Collection**: News articles are fetched using queries like `"BBCA saham Indonesia"`
    2. **LLM Analysis**: Each article is analyzed by Claude/GPT for:
       - Sentiment score (0-100, where 50 is neutral)
       - Confidence level (0-1)
       - Key themes mentioned
    3. **Aggregation**: Daily scores are weighted by:
       - Recency (72-hour half-life decay)
       - Confidence (higher confidence = more weight)
    4. **Signal Generation**:
       - **Bullish**: Score ≥ 65
       - **Bearish**: Score ≤ 35
       - **Neutral**: 35-65

    ### Data Retention

    - Articles are retained for 30 days by default
    - Use "Clear Old Data" to manually clean up old records

    ### Limitations

    - News sentiment is one input among many for trading decisions
    - LLM sentiment analysis may miss nuance or context
    - Real-time news may have reporting delays
    """)

    # Show current data status
    st.markdown("### Current Data Status")
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        article_count = len(articles) if articles else 0
        st.metric("Articles in Feed", article_count)
    with col_s2:
        symbol_count = len(set(a.get('symbol', '') for a in articles)) if articles else 0
        st.metric("Symbols with Data", symbol_count)

st.markdown("---")

# --- Sector Heatmap ---
st.markdown('<div class="section-header">SECTOR SENTIMENT HEATMAP</div>', unsafe_allow_html=True)

if sector_data:
    df = pd.DataFrame(sector_data)
else:
    st.info("No sector sentiment data available. Click 'Fetch All News' to populate data from news sources.")
    df = None

if df is not None and len(df) > 0 and 'avg_score' in df.columns:
    if 'article_count' not in df.columns or df['article_count'].sum() == 0:
        df['article_count'] = 1

    fig = px.treemap(
        df,
        path=[px.Constant("IDX"), 'sector'],
        values='article_count',
        color='avg_score',
        color_continuous_scale=[COLORS['destructive'], COLORS['primary']],
        color_continuous_midpoint=50,
        title="30-Day Aggregate Sentiment",
    )
    fig.update_layout(
        paper_bgcolor=COLORS['background'],
        plot_bgcolor=COLORS['background'],
        font=dict(color=COLORS['foreground']),
        margin=dict(t=40, l=10, r=10, b=10),
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# --- Articles Feed ---
st.markdown('<div class="section-header">📰 RECENT ARTICLES</div>', unsafe_allow_html=True)

if articles:
    for art in articles[:10]:
        score = art.get('sentiment_score', 50)
        if score > 60:
            badge = f'<span class="signal-badge bullish">BULLISH</span>'
        elif score < 40:
            badge = f'<span class="signal-badge bearish">BEARISH</span>'
        else:
            badge = f'<span class="signal-badge neutral">NEUTRAL</span>'

        title = html.escape(art.get('article_title', 'Untitled'))
        url = art.get('url', '')

        # Make title clickable if URL is safe
        if url and is_safe_url(url):
            title_html = f'<a href="{html.escape(url)}" target="_blank" rel="noopener noreferrer" style="color: {COLORS["foreground"]}; text-decoration: none;">{title}</a>'
        else:
            title_html = title

        st.markdown(f"""
        <div class="nextgen-card">
            <div style="display: flex; justify-content: space-between; align-items: start;">
                <div style="flex: 1;">
                    <div style="font-weight: 500; margin-bottom: 4px;">{title_html}</div>
                    <div style="font-size: 0.75rem; color: {COLORS['muted_foreground']};">
                        {html.escape(str(art.get('source', 'N/A')))} | {html.escape(str(art.get('symbol', '')))} | {html.escape(str(art.get('published_at', '')))}
                    </div>
                </div>
                <div style="text-align: right; margin-left: 16px;">
                    {badge}
                    <div style="font-size: 1.1rem; font-weight: 600; margin-top: 4px;">{score:.0f}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
else:
    st.info("No articles available. Use 'Fetch All News' to populate sentiment data.")

st.markdown("---")

# --- Theme Mapping ---
st.markdown('<div class="section-header">🔗 THEME → SECTOR MAPPING</div>', unsafe_allow_html=True)

themes_data = get_themes()
if themes_data and isinstance(themes_data, list) and len(themes_data) > 0:
    cols = st.columns(min(len(themes_data), 3))
    for i, theme in enumerate(themes_data[:6]):
        with cols[i % 3]:
            impact = theme.get('impact_direction', 'neutral')
            if impact == 'positive':
                border_color = COLORS['primary']
                icon = "📈"
            elif impact == 'negative':
                border_color = COLORS['destructive']
                icon = "📉"
            else:
                border_color = COLORS['warning']
                icon = "🔄"

            st.markdown(f"""
            <div class="nextgen-card" style="border-left: 3px solid {border_color};">
                <div style="font-weight: 600;">{icon} {theme.get('theme', 'Unknown')}</div>
                <div style="font-size: 0.85rem; color: {COLORS['muted_foreground']}; margin-top: 8px;">
                    <div><strong>Impact:</strong> {impact.title()}</div>
                    <div><strong>Sector:</strong> {theme.get('sector', 'N/A')}</div>
                    <div><strong>Stocks:</strong> {', '.join(theme.get('stocks', [])[:3])}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
else:
    st.info("No theme mappings available. Fetch news data first to see market themes.")
