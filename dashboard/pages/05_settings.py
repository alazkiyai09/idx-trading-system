"""
Configuration Page - Manage API keys, trading modes, and notification preferences.

Design: NextGen-style with card-based settings layout.
"""
import streamlit as st
import requests
import sys
import os
import logging
import html
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Configure logging
logger = logging.getLogger(__name__)

# Get API URL
try:
    from config.settings import settings
    API_URL = os.environ.get("API_URL", settings.api_url)
except ImportError:
    API_URL = os.environ.get("API_URL", "http://localhost:8000")

from dashboard.components.ux_components import trading_hours_indicator, render_stock_selector, get_stock_options
from dashboard.components.nextgen_styles import get_nextgen_css, COLORS, render_live_badge
from dashboard.components.top_nav import render_top_nav

# Constants
REQUEST_TIMEOUT = 10
TRAINING_TIMEOUT = 60
SUPPORTED_PROVIDERS = ["Claude (Anthropic)", "GLM (Z.AI)"]

st.set_page_config(page_title="Configuration | IDX", page_icon="⚙️", layout="wide", initial_sidebar_state="collapsed")

# Apply NextGen CSS
st.markdown(get_nextgen_css(), unsafe_allow_html=True)

# --- Header ---
st.markdown(f"""
<div style="display: flex; align-items: baseline; gap: 16px; margin-bottom: 8px;">
    <h1 style="margin: 0;">Settings</h1>
</div>
<p style="color: {COLORS['muted_foreground']}; margin-bottom: 16px;">
    Manage API keys, trading modes, and notification preferences
</p>
""", unsafe_allow_html=True)

render_top_nav("settings")
st.markdown("---")

trading_hours_indicator()


def get_error_detail(response: requests.Response) -> str:
    """Extract a concise error message from an API response."""
    try:
        payload = response.json()
        detail = payload.get("detail") or payload.get("message")
        if detail:
            return str(detail)
    except ValueError:
        pass
    return response.text[:200] or f"HTTP {response.status_code}"


def render_copyable_command(label: str, command: str, key: str) -> None:
    escaped_label = html.escape(label)
    escaped_command = html.escape(command)
    js_command = json.dumps(command)
    st.markdown(
        f"""
        <div class="command-card">
            <div class="command-card__header">
                <span class="command-card__label">{escaped_label}</span>
                <button class="command-card__copy" onclick='navigator.clipboard.writeText({js_command})'>
                    Copy
                </button>
            </div>
            <div class="command-card__body" id="{key}">{escaped_command}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(ttl=10, show_spinner=False)
def fetch_training_status(base_url: str) -> dict:
    resp = requests.get(f"{base_url}/prediction/training/status", timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


@st.cache_data(ttl=10, show_spinner=False)
def fetch_model_inventory(base_url: str) -> dict:
    resp = requests.get(f"{base_url}/prediction/models", timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


@st.cache_data(ttl=10, show_spinner=False)
def fetch_training_readiness(base_url: str, symbol: str, lookback_days: int) -> dict:
    resp = requests.get(
        f"{base_url}/prediction/training/readiness/{symbol}",
        params={"lookback_days": lookback_days},
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()

# Initialize session state
if 'settings' not in st.session_state:
    st.session_state.settings = {
        'primary_provider': 'Claude (Anthropic)',
        'fallback_provider': 'GLM (Z.AI)',
        'max_daily_loss': 2.0,
        'kelly_fraction': 0.5,
        'intraday_risk': 0.5,
        'swing_risk': 1.0,
        'position_risk': 1.5,
        'investor_risk': 2.0,
        'telegram_enabled': True,
        'telegram_chat_id': '',
        'email_enabled': True,
        'email_address': '',
        'email_frequency': 'Daily',
    }

tab1, tab2, tab3, tab4 = st.tabs(["🤖 LLM Providers", "📊 Trading Modes", "🔔 Notifications", "🧠 Model Ops"])

# === TAB 1: LLM Providers ===
with tab1:
    st.markdown('<div class="section-header">LLM PROVIDER CONFIGURATION</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="nextgen-card">
        <p style="margin: 0; font-size: 0.85rem; color: {COLORS['muted_foreground']};">
            The system automatically fails over from Claude to GLM if the primary provider is unavailable.
            Both providers support JSON mode for structured outputs.
        </p>
    </div>
    """, unsafe_allow_html=True)

    with st.form("llm_settings"):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"""
            <div class="nextgen-card">
                <h4 style="margin: 0 0 12px 0; font-size: 0.85rem;">PRIMARY PROVIDER</h4>
            """, unsafe_allow_html=True)
            primary_idx = SUPPORTED_PROVIDERS.index(st.session_state.settings['primary_provider']) \
                if st.session_state.settings['primary_provider'] in SUPPORTED_PROVIDERS else 0
            primary = st.selectbox(
                "Primary Provider",
                SUPPORTED_PROVIDERS,
                index=primary_idx,
                key="llm_primary",
                label_visibility="collapsed",
            )

        with col2:
            st.markdown(f"""
            <div class="nextgen-card">
                <h4 style="margin: 0 0 12px 0; font-size: 0.85rem;">FALLBACK PROVIDER</h4>
            """, unsafe_allow_html=True)
            fallback_idx = SUPPORTED_PROVIDERS.index(st.session_state.settings['fallback_provider']) \
                if st.session_state.settings['fallback_provider'] in SUPPORTED_PROVIDERS else 1
            fallback = st.selectbox(
                "Fallback Provider",
                SUPPORTED_PROVIDERS,
                index=fallback_idx,
                key="llm_fallback",
                label_visibility="collapsed",
            )

        st.markdown("---")
        st.markdown('<div class="section-header">RISK MANAGEMENT</div>', unsafe_allow_html=True)

        col_r1, col_r2 = st.columns(2)
        with col_r1:
            max_daily_loss = st.slider(
                "Max Daily Loss Limit (%)",
                1.0, 5.0,
                st.session_state.settings['max_daily_loss'],
                0.5,
                key="risk_max_daily_loss",
                help="Stop trading when daily loss exceeds this percentage"
            )
        with col_r2:
            kelly_fraction = st.slider(
                "Kelly Fraction (f)",
                0.1, 1.0,
                st.session_state.settings['kelly_fraction'],
                0.1,
                key="risk_kelly",
                help="0.5 = Half Kelly (recommended for conservative sizing)"
            )

        if st.form_submit_button("💾 Save LLM Settings", type="primary", use_container_width=True):
            st.session_state.settings['primary_provider'] = primary
            st.session_state.settings['fallback_provider'] = fallback
            st.session_state.settings['max_daily_loss'] = max_daily_loss
            st.session_state.settings['kelly_fraction'] = kelly_fraction
            st.toast("LLM settings saved!", icon="✅")

# === TAB 2: Trading Modes ===
with tab2:
    st.markdown('<div class="section-header">GLOBAL TRADING PARAMETERS</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    modes = [
        ("Intraday", "intraday_risk", "Same-day trades, quick momentum", 0.5),
        ("Swing", "swing_risk", "2-7 day hold, foreign flow focus", 1.0),
        ("Position", "position_risk", "1-4 week hold, trend following", 1.5),
        ("Investor", "investor_risk", "Months hold, fundamentals focus", 2.0),
    ]

    with col1:
        for name, key, desc, default in modes[:2]:
            st.markdown(f"""
            <div class="nextgen-card">
                <h4 style="margin: 0; font-size: 0.9rem; color: {COLORS['primary']};">{name.upper()}</h4>
                <p style="margin: 4px 0 12px 0; font-size: 0.75rem; color: {COLORS['muted_foreground']};">{desc}</p>
            """, unsafe_allow_html=True)
            val = st.number_input(
                f"{name} Risk per Trade (%)",
                value=st.session_state.settings[key],
                key=f"mode_{key}",
                min_value=0.1,
                max_value=5.0,
                step=0.1,
                label_visibility="collapsed",
            )
            st.session_state.settings[key] = val

    with col2:
        for name, key, desc, default in modes[2:]:
            st.markdown(f"""
            <div class="nextgen-card">
                <h4 style="margin: 0; font-size: 0.9rem; color: {COLORS['primary']};">{name.upper()}</h4>
                <p style="margin: 4px 0 12px 0; font-size: 0.75rem; color: {COLORS['muted_foreground']};">{desc}</p>
            """, unsafe_allow_html=True)
            val = st.number_input(
                f"{name} Risk per Trade (%)",
                value=st.session_state.settings[key],
                key=f"mode_{key}",
                min_value=0.1,
                max_value=5.0,
                step=0.1,
                label_visibility="collapsed",
            )
            st.session_state.settings[key] = val

    st.markdown("---")

    # Mode comparison table
    st.markdown('<div class="section-header">MODE COMPARISON</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="nextgen-card" style="padding: 0;">
        <table style="width: 100%; border-collapse: collapse;">
            <thead>
                <tr style="background: {COLORS['muted']};">
                    <th style="padding: 12px; text-align: left; font-size: 0.7rem; color: {COLORS['muted_foreground']};">MODE</th>
                    <th style="padding: 12px; text-align: center; font-size: 0.7rem; color: {COLORS['muted_foreground']};">HOLD PERIOD</th>
                    <th style="padding: 12px; text-align: center; font-size: 0.7rem; color: {COLORS['muted_foreground']};">RISK/TRADE</th>
                    <th style="padding: 12px; text-align: left; font-size: 0.7rem; color: {COLORS['muted_foreground']};">KEY FOCUS</th>
                </tr>
            </thead>
            <tbody>
                <tr style="border-bottom: 1px solid {COLORS['border']};">
                    <td style="padding: 12px; font-weight: 600;">Intraday</td>
                    <td style="padding: 12px; text-align: center;">Same day</td>
                    <td style="padding: 12px; text-align: center; color: {COLORS['primary']};">0.5%</td>
                    <td style="padding: 12px;">Quick momentum</td>
                </tr>
                <tr style="border-bottom: 1px solid {COLORS['border']};">
                    <td style="padding: 12px; font-weight: 600;">Swing</td>
                    <td style="padding: 12px; text-align: center;">2-7 days</td>
                    <td style="padding: 12px; text-align: center; color: {COLORS['primary']};">1.0%</td>
                    <td style="padding: 12px;">Foreign flow + technical</td>
                </tr>
                <tr style="border-bottom: 1px solid {COLORS['border']};">
                    <td style="padding: 12px; font-weight: 600;">Position</td>
                    <td style="padding: 12px; text-align: center;">1-4 weeks</td>
                    <td style="padding: 12px; text-align: center; color: {COLORS['primary']};">1.5%</td>
                    <td style="padding: 12px;">Trend following</td>
                </tr>
                <tr>
                    <td style="padding: 12px; font-weight: 600;">Investor</td>
                    <td style="padding: 12px; text-align: center;">Months</td>
                    <td style="padding: 12px; text-align: center; color: {COLORS['primary']};">2.0%</td>
                    <td style="padding: 12px;">Fundamentals</td>
                </tr>
            </tbody>
        </table>
    </div>
    """, unsafe_allow_html=True)

# === TAB 3: Notifications ===
with tab3:
    st.markdown('<div class="section-header">NOTIFICATION PREFERENCES</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"""
        <div class="nextgen-card">
            <h4 style="margin: 0 0 12px 0; font-size: 0.9rem;">📱 TELEGRAM</h4>
        """, unsafe_allow_html=True)

        telegram_enabled = st.checkbox(
            "Enable Telegram Alerts",
            value=st.session_state.settings['telegram_enabled'],
            key="notif_telegram_enabled",
        )
        if telegram_enabled:
            telegram_chat_id = st.text_input(
                "Chat ID",
                value=st.session_state.settings['telegram_chat_id'],
                key="notif_telegram_chat",
                type="password",
                placeholder="Your Telegram chat ID",
            )
        else:
            telegram_chat_id = st.session_state.settings['telegram_chat_id']

    with col2:
        st.markdown(f"""
        <div class="nextgen-card">
            <h4 style="margin: 0 0 12px 0; font-size: 0.9rem;">📧 EMAIL</h4>
        """, unsafe_allow_html=True)

        email_enabled = st.checkbox(
            "Enable Email Reports",
            value=st.session_state.settings['email_enabled'],
            key="notif_email_enabled",
        )

        if email_enabled:
            email_address = st.text_input(
                "Email Address",
                value=st.session_state.settings['email_address'],
                key="notif_email_address",
                placeholder="your@email.com",
            )
            email_frequency = st.selectbox(
                "Frequency",
                ["Daily", "Weekly"],
                index=0 if st.session_state.settings['email_frequency'] == 'Daily' else 1,
                key="notif_email_freq",
            )
        else:
            email_address = st.session_state.settings['email_address']
            email_frequency = st.session_state.settings['email_frequency']

    if st.button("💾 Save Notification Settings", type="primary", use_container_width=True):
        if email_enabled and email_address:
            import re

            email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            if not re.match(email_pattern, email_address):
                st.error("Please enter a valid email address.")
            else:
                st.session_state.settings['telegram_enabled'] = telegram_enabled
                st.session_state.settings['telegram_chat_id'] = telegram_chat_id
                st.session_state.settings['email_enabled'] = email_enabled
                st.session_state.settings['email_address'] = email_address
                st.session_state.settings['email_frequency'] = email_frequency
                st.toast("Notification settings saved!", icon="✅")
        else:
            st.session_state.settings['telegram_enabled'] = telegram_enabled
            st.session_state.settings['telegram_chat_id'] = telegram_chat_id
            st.session_state.settings['email_enabled'] = email_enabled
            st.session_state.settings['email_address'] = email_address
            st.session_state.settings['email_frequency'] = email_frequency
            st.toast("Notification settings saved!", icon="✅")

# === TAB 4: Model Ops ===
with tab4:
    ops_tab1, ops_tab2 = st.tabs(["Training", "Model Management"])

with ops_tab1:
    st.markdown('<div class="section-header">MODEL OPS</div>', unsafe_allow_html=True)
    st.caption(
        "Inference uses published artifacts. Training runs offline in the background and writes new artifacts to "
        "`data/ml_artifacts`. To update a model with newer daily data, rerun training with overwrite enabled."
    )

    training_status = None
    try:
        training_status = fetch_training_status(API_URL)
    except requests.exceptions.RequestException as e:
        st.warning(f"Could not load training status: {e}")

    if training_status:
        artifacts = training_status.get("artifacts", {})
        job = training_status.get("job", {})
        if st.button("Refresh Training Status", key="refresh_training_status"):
            fetch_training_status.clear()
            st.rerun()
        metric_cols = st.columns(4)
        metric_cols[0].metric("Trained Models", artifacts.get("total_models", 0))
        metric_cols[1].metric("Up To Date", artifacts.get("up_to_date_models", 0))
        metric_cols[2].metric("Latest Market Date", training_status.get("latest_market_date") or "n/a")
        metric_cols[3].metric("Batch Cap", training_status.get("batch_limit", settings.model_training_batch_limit))

        st.markdown(
            f"""
            <div class="nextgen-card">
                <div style="font-size: 0.78rem; color: {COLORS['muted_foreground']};">CURRENT TRAINING JOB</div>
                <div style="margin-top: 8px; font-size: 1.1rem; font-weight: 600;">{job.get('status', 'idle').replace('_', ' ').title()}</div>
                <div style="margin-top: 6px; color: {COLORS['muted_foreground']};">
                    Symbols: {', '.join(job.get('symbols', [])) or 'None'}<br/>
                    Current: {job.get('current_symbol') or 'Idle'}<br/>
                    Started: {job.get('started_at') or 'n/a'}<br/>
                    Finished: {job.get('finished_at') or 'n/a'}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if job.get("total_symbols"):
            st.progress(min(max((job.get("progress_pct", 0.0) / 100.0), 0.0), 1.0))
            st.caption(
                f"Progress: {job.get('finished_count', 0)}/{job.get('total_symbols', 0)} "
                f"completed | trained {job.get('completed_count', 0)} | "
                f"skipped {job.get('skipped_count', 0)} | failed {job.get('failed_count', 0)}"
            )

            symbol_statuses = job.get("symbol_statuses", {})
            if symbol_statuses:
                st.markdown("**Current Batch Detail**")
                st.dataframe(
                    [
                        {
                            "Symbol": symbol,
                            "Status": detail.get("status", "pending"),
                            "Started At": detail.get("started_at"),
                            "Finished At": detail.get("finished_at"),
                            "Source Latest Date": detail.get("source_latest_date"),
                            "Reason": detail.get("reason") or detail.get("error"),
                        }
                        for symbol, detail in symbol_statuses.items()
                    ],
                    use_container_width=True,
                    hide_index=True,
                )

    st.markdown("---")
    st.markdown('<div class="section-header">TRAINING READINESS</div>', unsafe_allow_html=True)

    col_train1, col_train2 = st.columns([1, 2])

    with col_train1:
        default_train_symbol = st.session_state.get("last_train_symbol", "")
        train_symbol = render_stock_selector(
            label="Train Symbol",
            default_symbol=default_train_symbol,
            api_url=API_URL,
            allow_empty=True,
        )
        if train_symbol:
            st.session_state["last_train_symbol"] = train_symbol

        lookback = st.slider("Lookback Days", 200, 1000, 400, 50, key="settings_training_lookback")
        test_size = st.slider("Test Size %", 10, 40, 20, 5, key="settings_training_test_size")

        train_button = st.button(
            "Check Training Readiness",
            type="primary",
            use_container_width=True,
            disabled=not train_symbol,
            key="settings_check_training_readiness",
        )

    with col_train2:
        if not train_symbol:
            st.info("Select a stock to check readiness and artifact availability.")
        else:
            try:
                readiness = fetch_training_readiness(API_URL, train_symbol, lookback)
                readiness_status = readiness.get("status", "unknown")
                if readiness_status == "not_ready":
                    st.error(readiness.get("message", "Not enough history for ML training yet."))
                elif readiness_status == "limited":
                    st.warning(readiness.get("message", "Training can run, but trust will be limited."))
                else:
                    st.success(readiness.get("message", "History is sufficient for ML training."))

                st.markdown(
                    f"""
                    <div class="nextgen-card">
                        <div style="font-size: 0.78rem; color: {COLORS['muted_foreground']};">READINESS SNAPSHOT</div>
                        <div style="margin-top: 8px; color: {COLORS['foreground']};">
                            Available rows: <strong>{readiness.get('data_rows', 'n/a')}</strong><br/>
                            Minimum rows: <strong>{readiness.get('minimum_rows', 'n/a')}</strong><br/>
                            Recommended rows: <strong>{readiness.get('recommended_rows', 'n/a')}</strong><br/>
                            Max recent window: <strong>{readiness.get('lookback_days', 'n/a')}</strong>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.caption(
                    "Newly listed stocks below the minimum row threshold are blocked from ML training. "
                    "Stocks between the minimum and recommended range can train, but trust will be lower."
                )
            except requests.exceptions.RequestException as e:
                st.warning(f"Could not load readiness snapshot: {e}")

        if train_button:
            try:
                resp = requests.post(
                    f"{API_URL}/prediction/train/{train_symbol}",
                    json={
                        "lookback_days": lookback,
                        "test_size": test_size / 100,
                        "use_exogenous": False,
                    },
                    timeout=TRAINING_TIMEOUT,
                )

                result = resp.json()
                st.session_state.training_result = result
                if resp.status_code in (200, 202):
                    st.success(f"Readiness check accepted for {train_symbol}")
                elif resp.status_code in (501, 503):
                    st.warning(result.get("message", get_error_detail(resp)))
                    st.caption(result.get("next_step", "Use the offline training flow below."))
                elif resp.status_code == 429:
                    st.warning(result.get("detail", "Rate limited"))
                else:
                    st.error(f"Training failed: {resp.status_code} - {get_error_detail(resp)}")
            except requests.exceptions.RequestException as e:
                st.error(f"API error: {e}")

        elif "training_result" in st.session_state:
            result = st.session_state.training_result
            requested_config = result.get("requested_config", {})
            st.markdown('<div class="nextgen-card">', unsafe_allow_html=True)
            st.markdown("**Last Readiness Check**")
            st.markdown(
                f"""
                - Symbol: `{result.get('symbol', 'Unknown')}`
                - Status: `{result.get('status', 'unknown')}`
                - Data rows: {result.get('data_rows', 'N/A')}
                - Lookback days: {requested_config.get('lookback_days', 'N/A')}
                - Next step: {result.get('next_step', 'n/a')}
                """
            )
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-header">BATCH TRAINING</div>', unsafe_allow_html=True)
    st.caption(
        f"Select up to {settings.model_training_batch_limit} symbols. "
        "Use overwrite to refresh existing models with the latest available daily data."
    )

    option_map = get_stock_options(API_URL)
    labels = list(option_map.keys())
    selected_labels = st.multiselect(
        "Symbols to train",
        options=labels,
        default=[],
        max_selections=settings.model_training_batch_limit,
        help="Type to search. Streamlit multiselect supports search by symbol or company name.",
    )
    selected_symbols = [option_map[label] for label in selected_labels]
    overwrite_existing = st.checkbox(
        "Overwrite existing artifacts",
        value=False,
        help="Enable this when you want to refresh a model with newer data. The current implementation retrains from scratch and replaces the artifact.",
    )

    if st.button(
        "Start Offline Batch Training",
        type="primary",
        disabled=not selected_symbols,
        key="settings_start_batch_training",
    ):
        try:
            resp = requests.post(
                f"{API_URL}/prediction/training/run",
                json={
                    "symbols": selected_symbols,
                    "lookback_days": lookback,
                    "overwrite": overwrite_existing,
                },
                timeout=REQUEST_TIMEOUT,
            )
            if resp.status_code == 200:
                st.success(f"Batch training started for {', '.join(selected_symbols)}")
                fetch_training_status.clear()
            else:
                st.error(f"Batch launch failed: {resp.status_code} - {get_error_detail(resp)}")
        except requests.exceptions.RequestException as e:
            st.error(f"Could not start training job: {e}")

    if training_status:
        recent_models = training_status.get("artifacts", {}).get("recent_models", [])
        if recent_models:
            st.markdown("**Recent Artifacts**")
            st.dataframe(
                [
                    {
                        "Symbol": item.get("symbol"),
                        "Trained At": item.get("trained_at"),
                        "Source Latest Date": item.get("source_latest_date"),
                        "Rows": item.get("source_row_count"),
                        "Horizon": item.get("trained_horizon"),
                    }
                    for item in recent_models
                ],
                use_container_width=True,
                hide_index=True,
            )

        history = training_status.get("history", [])
        if history:
            st.markdown("**Training History**")
            st.dataframe(
                [
                    {
                        "Job ID": item.get("job_id"),
                        "Status": item.get("status"),
                        "Started At": item.get("started_at"),
                        "Finished At": item.get("finished_at"),
                        "Symbols": ", ".join(item.get("symbols", [])),
                        "Completed": item.get("completed_count", 0),
                        "Skipped": item.get("skipped_count", 0),
                        "Failed": item.get("failed_count", 0),
                        "Overwrite": item.get("overwrite", False),
                    }
                    for item in history
                ],
                use_container_width=True,
                hide_index=True,
            )

        if training_status.get("job", {}).get("log_tail"):
            with st.expander("Training Log Tail", expanded=False):
                st.code("\n".join(training_status["job"]["log_tail"]), language="text")

with ops_tab2:
    st.markdown('<div class="section-header">MODEL MANAGEMENT</div>', unsafe_allow_html=True)
    st.caption(
        "Review stored artifacts, delete obsolete models, or upload externally trained artifacts."
    )
    with st.expander("Background Training Commands", expanded=False):
        render_copyable_command(
            "Single symbol",
            "bash scripts/run_training_in_background.sh ADRO",
            "bg_train_single_cmd",
        )
        render_copyable_command(
            "Batch with overwrite",
            "LOOKBACK_DAYS=400 OVERWRITE=true bash scripts/run_training_in_background.sh ADRO BBCA TLKM",
            "bg_train_batch_cmd",
        )
        st.caption(
            "These commands start `scripts/train_models.py` under `nohup` and update the same status/history files "
            "shown in Model Ops. Click inside the field and copy the full command."
        )

    try:
        model_inventory = fetch_model_inventory(API_URL)
    except requests.exceptions.RequestException as e:
        model_inventory = None
        st.warning(f"Could not load model inventory: {e}")

    if st.button("Refresh Model Inventory", key="refresh_model_inventory"):
        fetch_model_inventory.clear()
        st.rerun()

    if model_inventory:
        models = model_inventory.get("models", [])
        st.metric("Stored Models", model_inventory.get("total_models", 0))
        if models:
            st.dataframe(
                [
                    {
                        "Symbol": item.get("symbol"),
                        "Trained At": item.get("trained_at"),
                        "Source Latest Date": item.get("source_latest_date"),
                        "Type": item.get("artifact_type"),
                        "Horizon": item.get("trained_horizon"),
                        "Size (MB)": round((item.get("artifact_size_bytes") or 0) / (1024 * 1024), 2),
                    }
                    for item in models
                ],
                use_container_width=True,
                hide_index=True,
            )

            delete_symbol = st.selectbox(
                "Delete Stored Model",
                options=[""] + [item.get("symbol") for item in models if item.get("symbol")],
                index=0,
                help="Select a stored artifact to delete.",
            )
            if st.button("Delete Selected Model", disabled=not delete_symbol, key="delete_selected_model"):
                try:
                    resp = requests.delete(f"{API_URL}/prediction/models/{delete_symbol}", timeout=REQUEST_TIMEOUT)
                    if resp.status_code == 200:
                        st.success(f"Deleted stored model for {delete_symbol}")
                        fetch_model_inventory.clear()
                        fetch_training_status.clear()
                        st.rerun()
                    else:
                        st.error(f"Delete failed: {resp.status_code} - {get_error_detail(resp)}")
                except requests.exceptions.RequestException as e:
                    st.error(f"Delete request failed: {e}")
        else:
            st.info("No stored model artifacts found.")

    st.markdown("---")
    st.markdown("**Upload Trained Artifact**")
    upload_symbol = render_stock_selector(
        label="Upload Symbol",
        default_symbol=st.session_state.get("upload_model_symbol", ""),
        api_url=API_URL,
        allow_empty=True,
    )
    if upload_symbol:
        st.session_state["upload_model_symbol"] = upload_symbol
    upload_artifact = st.file_uploader("Artifact (.pkl)", type=["pkl"], key="upload_artifact_file")
    upload_metadata = st.file_uploader("Metadata (.json, optional)", type=["json"], key="upload_metadata_file")

    if st.button("Upload Model", disabled=upload_artifact is None, key="upload_model_button"):
        files = {"artifact_file": (upload_artifact.name, upload_artifact.getvalue(), "application/octet-stream")}
        if upload_metadata is not None:
            files["metadata_file"] = (upload_metadata.name, upload_metadata.getvalue(), "application/json")
        data = {}
        if upload_symbol.strip():
            data["symbol"] = upload_symbol.strip().upper()
        try:
            resp = requests.post(
                f"{API_URL}/prediction/models/upload",
                files=files,
                data=data,
                timeout=TRAINING_TIMEOUT,
            )
            if resp.status_code == 200:
                payload = resp.json()
                st.success(f"Uploaded model for {payload.get('symbol')}")
                fetch_model_inventory.clear()
                fetch_training_status.clear()
                st.rerun()
            else:
                st.error(f"Upload failed: {resp.status_code} - {get_error_detail(resp)}")
        except requests.exceptions.RequestException as e:
            st.error(f"Upload request failed: {e}")

st.markdown("---")

# --- Settings Management ---
st.markdown('<div class="section-header">SETTINGS MANAGEMENT</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown(f"""
    <div class="nextgen-card">
        <h4 style="margin: 0 0 12px 0; font-size: 0.85rem;">📤 EXPORT SETTINGS</h4>
    """, unsafe_allow_html=True)
    import json
    settings_json = json.dumps(st.session_state.settings, indent=2)
    st.download_button(
        label="Download Settings JSON",
        data=settings_json,
        file_name="idx_trading_settings.json",
        mime="application/json",
        use_container_width=True,
    )

with col2:
    st.markdown(f"""
    <div class="nextgen-card">
        <h4 style="margin: 0 0 12px 0; font-size: 0.85rem;">🔄 RESET TO DEFAULTS</h4>
    """, unsafe_allow_html=True)
    if st.button("Reset All Settings", use_container_width=True):
            st.session_state.settings = {
                'primary_provider': 'Claude (Anthropic)',
                'fallback_provider': 'GLM (Z.AI)',
                'max_daily_loss': 2.0,
                'kelly_fraction': 0.5,
                'intraday_risk': 0.5,
                'swing_risk': 1.0,
                'position_risk': 1.5,
                'investor_risk': 2.0,
                'telegram_enabled': True,
                'telegram_chat_id': '',
                'email_enabled': True,
                'email_address': '',
                'email_frequency': 'Daily',
            }
            st.session_state["confirm_reset"] = False
            st.toast("Settings reset to defaults!", icon="✅")
            st.rerun()

st.markdown("---")
st.caption("ℹ️ Settings are stored in browser session state. For persistent storage, backend API integration is required.")
