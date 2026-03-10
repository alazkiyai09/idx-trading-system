"""
ML Prediction & Analysis Dashboard Page.

Provides comprehensive analysis interface with:
- Technical Analysis: RSI, MACD, Bollinger Bands, Support/Resistance
- ML Ensemble Prediction: LSTM + CNN + SVR ensemble with confidence
- Monte Carlo Simulation: VaR, CVaR, probability distributions
- Comparison View: All analyses side-by-side

Design: NextGen-style with analysis panels.
"""
import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timezone
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

# Constants
REQUEST_TIMEOUT = 10
LONG_REQUEST_TIMEOUT = 90
MONTE_CARLO_TIMEOUT = 60

# Import components
from dashboard.components.ux_components import (
    trading_hours_indicator, progress_indicator, success_with_details,
    error_with_recovery, api_error_handler, render_stock_selector,
)
from dashboard.components.nextgen_styles import (
    get_nextgen_css, COLORS, render_live_badge, get_chart_colors
)
from dashboard.components.top_nav import render_top_nav

st.set_page_config(page_title="ML Prediction & Analysis", page_icon="🤖", layout="wide", initial_sidebar_state="collapsed")

# Apply NextGen CSS
st.markdown(get_nextgen_css(), unsafe_allow_html=True)

# --- Header ---
st.markdown(f"""
<div style="display: flex; align-items: baseline; gap: 16px; margin-bottom: 8px;">
    <h1 style="margin: 0;">ML Prediction & Analysis</h1>
    {render_live_badge("ENSEMBLE")}
</div>
<p style="color: {COLORS['muted_foreground']}; margin-bottom: 16px;">
    Comprehensive analysis: Technical indicators, ML predictions, Monte Carlo simulation
</p>
""", unsafe_allow_html=True)

render_top_nav("ml")
st.markdown("---")

trading_hours_indicator()
st.warning("Prediction contracts are explicit in this build: unsupported horizons are rejected, uncertainty bands may be unavailable, and training is offline-only unless artifacts are produced by the offline pipeline.")

chart_colors = get_chart_colors()


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

# --- Symbol Input ---
default_symbol = st.session_state.get("last_symbol", "")
symbol = render_stock_selector(
    label="Select Stock",
    default_symbol=default_symbol,
    api_url=API_URL,
    allow_empty=True,
)
st.session_state["last_symbol"] = symbol

if not symbol:
    st.info("Select a stock to start analysis.")
    st.stop()

st.markdown(f"""
<div class="nextgen-card" style="margin-bottom: 18px; background:
    radial-gradient(circle at top right, rgba(16, 185, 129, 0.14), transparent 40%),
    linear-gradient(135deg, rgba(24, 24, 27, 0.96), rgba(9, 9, 11, 1));">
    <div style="display: grid; grid-template-columns: 1.6fr 1fr; gap: 18px;">
        <div>
            <div style="font-size: 0.72rem; letter-spacing: 0.14em; color: {COLORS['muted_foreground']};">ANALYSIS COCKPIT</div>
            <div style="margin-top: 10px; font-size: 1.4rem; font-weight: 600;">{symbol} workbench</div>
            <p style="margin: 10px 0 0 0; color: {COLORS['muted_foreground']}; line-height: 1.6;">
                Run technical analysis first, compare it with the ensemble forecast, then stress the idea with Monte Carlo before making a trading decision.
            </p>
        </div>
        <div style="display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px;">
            <div style="padding: 12px; border-radius: 10px; border: 1px solid {COLORS['border']};">
                <div style="font-size: 0.72rem; color: {COLORS['muted_foreground']};">Selected Symbol</div>
                <div style="font-size: 1.35rem; font-weight: 600;">{symbol}</div>
            </div>
            <div style="padding: 12px; border-radius: 10px; border: 1px solid {COLORS['border']};">
                <div style="font-size: 0.72rem; color: {COLORS['muted_foreground']};">Mode</div>
                <div style="font-size: 1.35rem; font-weight: 600;">Multi-Model</div>
            </div>
            <div style="padding: 12px; border-radius: 10px; border: 1px solid {COLORS['border']};">
                <div style="font-size: 0.72rem; color: {COLORS['muted_foreground']};">Checklist</div>
                <div style="font-size: 0.95rem;">Technical -> Forecast</div>
            </div>
            <div style="padding: 12px; border-radius: 10px; border: 1px solid {COLORS['border']};">
                <div style="font-size: 0.72rem; color: {COLORS['muted_foreground']};">Decision Gate</div>
                <div style="font-size: 0.95rem;">Scenario stress test</div>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# --- Tabs ---
tab1, tab2, tab3, tab4 = st.tabs(["📊 Technical Analysis", "🎯 ML Prediction", "🎲 Monte Carlo", "📈 Comparison"])

# ============================================================================
# TAB 1: TECHNICAL ANALYSIS
# ============================================================================
with tab1:
    st.markdown('<div class="section-header">TECHNICAL INDICATORS</div>', unsafe_allow_html=True)

    col_t1, col_t2 = st.columns([2, 1])

    with col_t2:
        if st.button("📊 Run Technical Analysis", type="primary", use_container_width=True):
            with st.spinner("Calculating indicators..."):
                try:
                    resp = requests.post(f"{API_URL}/analysis/technical/{symbol}", timeout=REQUEST_TIMEOUT)
                    if resp.status_code == 200:
                        st.session_state.technical_data = resp.json()
                        st.session_state.technical_symbol = symbol
                        st.toast(f"Technical analysis complete for {symbol}", icon="✅")
                    else:
                        st.error(f"Analysis failed: {resp.status_code}")
                except requests.exceptions.RequestException as e:
                    st.error(f"API error: {e}")

        if 'technical_data' in st.session_state and st.session_state.technical_symbol == symbol:
            tech = st.session_state.technical_data

            # Score Card
            score = tech.get('score', {})
            total_score = score.get('total', 0)
            signal = score.get('signal', 'NEUTRAL')

            signal_color = COLORS['primary'] if signal == 'BUY' else COLORS['destructive'] if signal == 'SELL' else COLORS['muted_foreground']

            st.markdown(f"""
            <div class="nextgen-card" style="text-align: center;">
                <div style="font-size: 0.75rem; color: {COLORS['muted_foreground']};">COMPOSITE SCORE</div>
                <div style="font-size: 2.5rem; font-weight: 600; color: {signal_color};">{total_score:.0f}</div>
                <div style="font-size: 0.9rem; color: {signal_color}; font-weight: 500;">{signal}</div>
            </div>
            """, unsafe_allow_html=True)

            # Score breakdown
            st.markdown('<div class="section-header" style="margin-top: 16px;">SCORE BREAKDOWN</div>', unsafe_allow_html=True)
            for name, val in [
                ("Trend", score.get('trend_score', 0)),
                ("Momentum", score.get('momentum_score', 0)),
                ("Volume", score.get('volume_score', 0)),
                ("Volatility", score.get('volatility_score', 0)),
            ]:
                color = COLORS['primary'] if val >= 50 else COLORS['destructive']
                st.markdown(f"""
                <div style="display: flex; justify-content: space-between; padding: 4px 8px; border-bottom: 1px solid {COLORS['border']};">
                    <span>{name}</span>
                    <span style="color: {color}; font-weight: 600;">{val:.0f}</span>
                </div>
                """, unsafe_allow_html=True)

            # Trend direction
            trend = score.get('trend', 'Neutral')
            st.markdown(f"""
            <div class="nextgen-card" style="margin-top: 16px;">
                <div style="font-size: 0.75rem; color: {COLORS['muted_foreground']};">TREND DIRECTION</div>
                <div style="font-size: 1.2rem; font-weight: 600;">{trend}</div>
            </div>
            """, unsafe_allow_html=True)

    with col_t1:
        if 'technical_data' in st.session_state and st.session_state.technical_symbol == symbol:
            tech = st.session_state.technical_data
            ind = tech.get('indicators', {})

            # Indicator Grid
            st.markdown('<div class="section-header">INDICATOR VALUES</div>', unsafe_allow_html=True)

            # Row 1
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                rsi = ind.get('rsi', 50)
                rsi_color = COLORS['primary'] if 30 <= rsi <= 70 else COLORS['destructive']
                st.metric("RSI", f"{rsi:.1f}")
            with c2:
                macd = ind.get('macd', 0)
                macd_sig = ind.get('macd_signal', 0)
                st.metric("MACD", f"{macd:.2f}", f"{macd - macd_sig:.2f}")
            with c3:
                st.metric("ATR", f"{ind.get('atr', 0):.0f}")
            with c4:
                close = ind.get('close', 0)
                st.metric("Close", f"Rp {close:,.0f}")

            # Row 2
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                ema20 = ind.get('ema20', 0)
                st.metric("EMA 20", f"Rp {ema20:,.0f}")
            with c2:
                ema50 = ind.get('ema50', 0)
                st.metric("EMA 50", f"Rp {ema50:,.0f}")
            with c3:
                bb_upper = ind.get('bb_upper', 0)
                st.metric("BB Upper", f"Rp {bb_upper:,.0f}")
            with c4:
                bb_lower = ind.get('bb_lower', 0)
                st.metric("BB Lower", f"Rp {bb_lower:,.0f}")

            # Support/Resistance
            st.markdown('<div class="section-header" style="margin-top: 16px;">SUPPORT & RESISTANCE</div>', unsafe_allow_html=True)
            support = ind.get('support', 0)
            resistance = ind.get('resistance', 0)

            col_s, col_r = st.columns(2)
            with col_s:
                st.metric("Support", f"Rp {support:,.0f}", f"{((support/close)-1)*100:.1f}%")
            with col_r:
                st.metric("Resistance", f"Rp {resistance:,.0f}", f"{((resistance/close)-1)*100:+.1f}%")

            # RSI Gauge
            st.markdown('<div class="section-header" style="margin-top: 16px;">RSI GAUGE</div>', unsafe_allow_html=True)
            fig_rsi = go.Figure()
            fig_rsi.add_trace(go.Indicator(
                mode="gauge+number",
                value=rsi,
                title={'text': "RSI", 'font': {'color': COLORS['foreground']}},
                gauge={
                    'axis': {'range': [0, 100], 'tickfont': {'color': COLORS['muted_foreground']}},
                    'bar': {'color': rsi_color},
                    'steps': [
                        {'range': [0, 30], 'color': 'rgba(16, 185, 129, 0.3)'},
                        {'range': [30, 70], 'color': 'rgba(161, 161, 170, 0.2)'},
                        {'range': [70, 100], 'color': 'rgba(239, 68, 68, 0.3)'},
                    ],
                }
            ))
            fig_rsi.update_layout(
                paper_bgcolor=COLORS['background'],
                height=200,
                margin=dict(l=20, r=20, t=40, b=20),
            )
            st.plotly_chart(fig_rsi, use_container_width=True)

            # --- ML Prediction Summary in Technical Tab ---
            st.markdown('<div class="section-header" style="margin-top: 16px;">🤖 ML PREDICTION SUMMARY</div>', unsafe_allow_html=True)

            if 'prediction_data' in st.session_state and st.session_state.prediction_symbol == symbol:
                pred = st.session_state.prediction_data
                preds = pred.get('predictions', [])

                if preds:
                    current = pred.get('current_price', 0)
                    final_price = preds[-1].get('predicted_price', 0)
                    change_pct = ((final_price / current) - 1) * 100 if current > 0 else 0

                    pred_col1, pred_col2, pred_col3 = st.columns(3)
                    with pred_col1:
                        change_color = COLORS['primary'] if change_pct >= 0 else COLORS['destructive']
                        st.metric("7-Day Forecast", f"Rp {final_price:,.0f}", f"{change_pct:+.2f}%")
                    with pred_col2:
                        horizon = len(preds)
                        st.metric("Prediction Horizon", f"{horizon} days")
                    with pred_col3:
                        st.metric("Model Status", pred.get("model_status", "unknown").replace("_", " ").title())
            else:
                st.info("💡 Run ML Prediction (Tab 2) to see prediction summary here")
        else:
            st.info("Click 'Run Technical Analysis' to see indicator values.")

# ============================================================================
# TAB 2: ML ENSEMBLE PREDICTION
# ============================================================================
with tab2:
    st.markdown('<div class="section-header">ENSEMBLE PREDICTION</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])

    with col1:
        if st.button("🎯 Get Prediction", type="primary", use_container_width=True):
            with st.spinner("Running ensemble prediction..."):
                try:
                    resp = requests.get(f"{API_URL}/prediction/ensemble/{symbol}", timeout=LONG_REQUEST_TIMEOUT)
                    if resp.status_code == 200:
                        st.session_state.prediction_data = resp.json()
                        st.session_state.prediction_symbol = symbol
                        st.toast(f"Prediction loaded for {symbol}", icon="✅")
                        st.rerun()
                    elif resp.status_code == 404:
                        st.error(f"No trained model artifact for {symbol}")
                        st.info(
                            "Model inference only works for symbols that already have published artifacts in "
                            "`data/ml_artifacts`. This build does not train models from the ML page."
                        )
                        st.caption(
                            "Use Settings -> Model Ops to check readiness. If the symbol is ready, the actual "
                            "training/publish step still needs to be run offline before live prediction will work."
                        )
                    elif resp.status_code == 503:
                        st.error(get_error_detail(resp))
                        st.info("Real inference is enabled, but the model artifacts or feature pipeline are not ready for this symbol.")
                    else:
                        st.error(f"Prediction failed: {resp.status_code} - {get_error_detail(resp)}")
                except requests.exceptions.ReadTimeout:
                    st.error(
                        "Prediction timed out while loading the full ensemble artifact. "
                        "Try again once more; the first TensorFlow-backed request can take longer."
                    )
                except requests.exceptions.RequestException as e:
                    st.error(f"API error: {e}")

        # Prediction result cards
        if 'prediction_data' in st.session_state and st.session_state.prediction_symbol == symbol:
            pred = st.session_state.prediction_data
            preds = pred.get('predictions', [])

            if preds:
                current = pred.get('current_price', 0)
                final = preds[-1].get('predicted_price', 0)
                horizon_days = len(preds)
                change_pct = ((final / current) - 1) * 100 if current > 0 else 0
                change_color = COLORS['primary'] if change_pct >= 0 else COLORS['destructive']

                st.markdown(f"""
                <div class="nextgen-card" style="text-align: center;">
                    <div style="font-size: 0.75rem; color: {COLORS['muted_foreground']};">{horizon_days}-DAY FORECAST</div>
                    <div style="font-size: 2rem; font-weight: 600; color: {change_color};">Rp {final:,.0f}</div>
                    <div style="font-size: 0.9rem; color: {change_color};">{change_pct:+.2f}%</div>
                    <div style="font-size: 0.75rem; color: {COLORS['muted_foreground']}; margin-top: 8px;">
                        Current: Rp {current:,.0f}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.success("✅ ML ensemble prediction loaded")
                if pred.get("feature_state"):
                    st.caption(f"State: {pred.get('feature_state')} | Source: {pred.get('data_source', 'unknown')}")
                uncertainty = pred.get("uncertainty", {})
                if uncertainty.get("status") == "not_available":
                    st.info(uncertainty.get("message", "Uncertainty bands are unavailable for this artifact."))

                # Model contributions
                if 'model_contributions' in pred:
                    st.markdown('<div class="section-header">MODEL WEIGHTS</div>', unsafe_allow_html=True)
                    for model, weight in pred['model_contributions'].items():
                        # Handle weight as string or float
                        try:
                            weight_val = float(weight) * 100 if isinstance(weight, (int, float)) else float(weight) * 100 if weight else 0
                        except (ValueError, TypeError):
                            weight_val = 0
                        st.markdown(f"""
                        <div style="display: flex; justify-content: space-between; padding: 4px 8px; border-bottom: 1px solid {COLORS['border']};">
                            <span>{model.upper()}</span>
                            <span style="color: {COLORS['primary']};">{weight_val:.0f}%</span>
                        </div>
                        """, unsafe_allow_html=True)

                # Model trust panel
                st.markdown('<div class="section-header" style="margin-top: 16px;">MODEL TRUST</div>', unsafe_allow_html=True)
                rmse = pred.get('rmse', 'N/A')
                mae = pred.get('mae', 'N/A')
                r2 = pred.get('r2', 'N/A')
                artifact = pred.get("artifact_metadata", {})
                validation_mae = artifact.get("validation_mae", {}) if artifact else {}

                if rmse == 'N/A' and mae == 'N/A' and r2 == 'N/A' and validation_mae:
                    st.markdown(
                        f"""
                        <div class="nextgen-card">
                            <div style="font-size: 0.78rem; color: {COLORS['muted_foreground']}; margin-bottom: 8px;">
                                Cross-validation MAE from the saved artifact
                            </div>
                            {"".join(
                                f"<div style='display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid {COLORS['border']};'>"
                                f"<span>{model.upper()}</span><span style='font-weight:600;'>{value:.4f}</span></div>"
                                for model, value in validation_mae.items()
                            )}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.caption("RMSE / MAE / R² are not published by this runtime. Showing saved cross-validation MAE by base model instead.")
                else:
                    st.markdown(f"""
                    <div class="nextgen-card" style="display: flex; gap: 16px; text-align: center;">
                        <div style="flex: 1;">
                            <div style="font-size: 0.75rem; color: {COLORS['muted_foreground']};">RMSE</div>
                            <div style="font-weight: 600;">{rmse}</div>
                        </div>
                        <div style="flex: 1;">
                            <div style="font-size: 0.75rem; color: {COLORS['muted_foreground']};">MAE</div>
                            <div style="font-weight: 600;">{mae}</div>
                        </div>
                        <div style="flex: 1;">
                            <div style="font-size: 0.75rem; color: {COLORS['muted_foreground']};">R²</div>
                            <div style="font-weight: 600; color: {COLORS['primary']};">{r2}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                if artifact:
                    st.markdown('<div class="section-header" style="margin-top: 16px;">ARTIFACT METADATA</div>', unsafe_allow_html=True)
                    st.caption(
                        f"Trained horizon: {artifact.get('trained_horizon', 'n/a')} days | "
                        f"Lookback: {artifact.get('lookback_window', 'n/a')} | "
                        f"Features: {artifact.get('feature_count', 'n/a')}"
                    )
                    if artifact.get("trained_at"):
                        st.caption(f"Artifact timestamp: {artifact['trained_at']}")
                    if pred.get("base_model_predictions"):
                        st.caption(
                            "Available algorithms in this artifact: "
                            + ", ".join(model.upper() for model in pred["base_model_predictions"].keys())
                        )

    with col2:
        if 'prediction_data' in st.session_state and st.session_state.prediction_symbol == symbol:
            pred = st.session_state.prediction_data
            preds = pred.get('predictions', [])

            if preds:
                df = pd.DataFrame(preds)
                df['date'] = pd.to_datetime(df['date'])

                current = pred.get('current_price', 0)
                first_date = datetime.now()
                first_price = current

                dates = [first_date] + df['date'].tolist()
                prices = [first_price] + df['predicted_price'].tolist()

                fig = go.Figure()

                # Confidence band is shown only when the API returns calibrated bands.
                if 'upper_band' in df.columns and 'lower_band' in df.columns:
                    fig.add_trace(go.Scatter(
                        x=df['date'], y=df['upper_band'],
                        line=dict(width=0), showlegend=False
                    ))
                    fig.add_trace(go.Scatter(
                        x=df['date'], y=df['lower_band'],
                        fill='tonexty', fillcolor='rgba(16, 185, 129, 0.1)',
                        line=dict(width=0), name='Confidence Band'
                    ))

                # Prediction line
                fig.add_trace(go.Scatter(
                    x=dates, y=prices,
                    line=dict(color=COLORS['primary'], width=2, dash='dash'),
                    name='Ensemble Forecast'
                ))

                # Current price marker
                fig.add_trace(go.Scatter(
                    x=[first_date], y=[current],
                    mode='markers', marker=dict(size=10, color=COLORS['warning']),
                    name='Current Price'
                ))

                fig.update_layout(
                    title=f"{symbol} {len(preds)}-Day Price Forecast",
                    paper_bgcolor=COLORS['background'],
                    plot_bgcolor=COLORS['background'],
                    font=dict(color=COLORS['foreground']),
                    xaxis=dict(gridcolor=COLORS['border']),
                    yaxis=dict(gridcolor=COLORS['border'], title='Price (IDR)'),
                    height=400,
                    margin=dict(l=60, r=20, t=50, b=40),
                    legend=dict(orientation="h", y=1.02),
                )

                st.plotly_chart(fig, use_container_width=True)

                base_model_predictions = pred.get("base_model_predictions", {})
                if base_model_predictions:
                    st.markdown('<div class="section-header" style="margin-top: 16px;">ALGORITHM VS ENSEMBLE</div>', unsafe_allow_html=True)

                    compare_fig = go.Figure()
                    compare_fig.add_trace(
                        go.Scatter(
                            x=dates,
                            y=prices,
                            line=dict(color=COLORS['primary'], width=3),
                            name='Ensemble',
                        )
                    )

                    palette = [COLORS['warning'], COLORS['destructive'], COLORS['primary'], '#60a5fa']
                    comparison_rows = []
                    for idx, (model_name, model_preds) in enumerate(base_model_predictions.items()):
                        model_df = pd.DataFrame(model_preds)
                        model_df['date'] = pd.to_datetime(model_df['date'])
                        model_dates = [first_date] + model_df['date'].tolist()
                        model_prices = [current] + model_df['predicted_price'].tolist()
                        compare_fig.add_trace(
                            go.Scatter(
                                x=model_dates,
                                y=model_prices,
                                line=dict(color=palette[idx % len(palette)], width=2, dash='dot'),
                                name=model_name.upper(),
                            )
                        )
                        comparison_rows.append(
                            {
                                "Model": model_name.upper(),
                                "Final Price": float(model_df['predicted_price'].iloc[-1]),
                                "Final Return %": float(model_df['predicted_return'].sum() * 100),
                            }
                        )

                    compare_fig.update_layout(
                        title=dict(text='Forecast Comparison', font=dict(color=COLORS['foreground'])),
                        paper_bgcolor=COLORS['background'],
                        plot_bgcolor=COLORS['background'],
                        font=dict(color=COLORS['foreground']),
                        legend=dict(font=dict(color=COLORS['foreground'])),
                        xaxis=dict(
                            gridcolor=COLORS['border'],
                            tickfont=dict(color=COLORS['foreground']),
                            title=dict(text='Date', font=dict(color=COLORS['foreground'])),
                        ),
                        yaxis=dict(
                            gridcolor=COLORS['border'],
                            tickfont=dict(color=COLORS['foreground']),
                            title=dict(text='Price', font=dict(color=COLORS['foreground'])),
                        ),
                        height=360,
                        margin=dict(l=50, r=20, t=40, b=40),
                    )
                    st.plotly_chart(compare_fig, use_container_width=True)
                    st.caption(
                        "The ensemble is not a simple average of the visible lines. "
                        "This build first combines base-model outputs with horizon-specific meta-learners, "
                        "then aggregates across seeds. Because of that two-stage weighting, the ensemble path "
                        "can finish above or below individual model paths."
                    )

                    comparison_rows.insert(
                        0,
                        {
                            "Model": "ENSEMBLE",
                            "Final Price": float(df['predicted_price'].iloc[-1]),
                            "Final Return %": float(df['predicted_return'].sum() * 100),
                        },
                    )
                    st.dataframe(comparison_rows, use_container_width=True, hide_index=True)
        else:
            st.info("Enter a symbol and click 'Get Prediction' to see the forecast.")

# ============================================================================
# TAB 3: MONTE CARLO SIMULATION
# ============================================================================
with tab3:
    st.markdown('<div class="section-header">MONTE CARLO SIMULATION</div>', unsafe_allow_html=True)

    col_mc1, col_mc2 = st.columns([1, 2])

    with col_mc1:
        n_sims = st.slider("Number of Simulations", 100, 10000, 1000, 100)
        horizon = st.slider("Horizon (Days)", 7, 90, 30, 7)

        if st.button("🎲 Run Monte Carlo", type="primary", use_container_width=True):
            with st.spinner(f"Running {n_sims:,} simulations..."):
                try:
                    resp = requests.get(
                        f"{API_URL}/prediction/monte-carlo/{symbol}",
                        params={"n_simulations": n_sims, "horizon_days": horizon},
                        timeout=MONTE_CARLO_TIMEOUT
                    )
                    if resp.status_code == 200:
                        st.session_state.monte_carlo_data = resp.json()
                        st.session_state.monte_carlo_symbol = symbol
                        st.toast("Monte Carlo simulation complete!", icon="✅")
                    else:
                        st.error(f"Simulation failed: {resp.status_code}")
                except requests.exceptions.RequestException as e:
                    st.error(f"API error: {e}")

        if 'monte_carlo_data' in st.session_state and st.session_state.monte_carlo_symbol == symbol:
            mc = st.session_state.monte_carlo_data
            stats = mc.get('statistics', {})

            # VaR / CVaR Cards
            st.markdown('<div class="section-header">RISK METRICS</div>', unsafe_allow_html=True)

            var_95 = stats.get('var_95_pct', 0)
            cvar_95 = stats.get('cvar_95_pct', 0)

            var_color = COLORS['destructive'] if var_95 < -5 else COLORS['warning']
            cvar_color = COLORS['destructive'] if cvar_95 < -7 else COLORS['warning']

            st.markdown(f"""
            <div class="nextgen-card">
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px;">
                    <div style="text-align: center;">
                        <div style="font-size: 0.75rem; color: {COLORS['muted_foreground']};">VaR (95%)</div>
                        <div style="font-size: 1.5rem; font-weight: 600; color: {var_color};">{var_95:.2f}%</div>
                        <div style="font-size: 0.7rem; color: {COLORS['muted_foreground']};">Worst 5% scenario</div>
                    </div>
                    <div style="text-align: center;">
                        <div style="font-size: 0.75rem; color: {COLORS['muted_foreground']};">CVaR (95%)</div>
                        <div style="font-size: 1.5rem; font-weight: 600; color: {cvar_color};">{cvar_95:.2f}%</div>
                        <div style="font-size: 0.7rem; color: {COLORS['muted_foreground']};">Expected Shortfall</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Probability metrics
            prob_loss = stats.get('prob_loss_pct', 50)
            prob_gain_10 = stats.get('prob_gain_10pct_pct', 0)

            st.metric("Probability of Loss", f"{prob_loss:.1f}%")
            st.metric("Probability of +10% Gain", f"{prob_gain_10:.1f}%")

            # Model parameters
            model = mc.get('model', {})
            st.markdown('<div class="section-header" style="margin-top: 16px;">MODEL PARAMETERS</div>', unsafe_allow_html=True)
            st.caption(f"Drift (daily): {model.get('drift_daily', 0):.6f}")
            st.caption(f"Volatility (daily): {model.get('volatility_daily', 0):.4f}")
            st.caption(f"Data points used: {model.get('data_points_used', 0)}")

    with col_mc2:
        if 'monte_carlo_data' in st.session_state and st.session_state.monte_carlo_symbol == symbol:
            mc = st.session_state.monte_carlo_data

            # Sample paths visualization
            paths = mc.get('sample_paths', [])
            if paths:
                fig = go.Figure()

                # Plot sample paths (limit to 50 for performance)
                for i, path in enumerate(paths[:50]):
                    alpha = 0.1 if i < 30 else 0.05
                    fig.add_trace(go.Scatter(
                        y=path,
                        mode='lines',
                        line=dict(color=COLORS['primary'], width=0.5),
                        opacity=alpha,
                        showlegend=False
                    ))

                # Current price line
                current = mc.get('current_price', 0)
                fig.add_hline(y=current, line_dash="dash", line_color=COLORS['warning'],
                              annotation_text="Current")

                fig.update_layout(
                    title=f"Monte Carlo Paths ({horizon} days)",
                    paper_bgcolor=COLORS['background'],
                    plot_bgcolor=COLORS['background'],
                    font=dict(color=COLORS['foreground']),
                    xaxis=dict(gridcolor=COLORS['border'], title='Days'),
                    yaxis=dict(gridcolor=COLORS['border'], title='Price (IDR)'),
                    height=300,
                    margin=dict(l=60, r=20, t=40, b=40),
                )
                st.plotly_chart(fig, use_container_width=True)

            # Distribution histogram
            final_prices = mc.get('final_prices_sample', [])
            if final_prices:
                fig2 = go.Figure()
                fig2.add_trace(go.Histogram(
                    x=final_prices,
                    nbinsx=50,
                    marker_color=COLORS['primary'],
                    opacity=0.7,
                ))

                # Current price line
                current = mc.get('current_price', 0)
                fig2.add_vline(x=current, line_dash="dash", line_color=COLORS['warning'])

                fig2.update_layout(
                    title="Final Price Distribution",
                    paper_bgcolor=COLORS['background'],
                    plot_bgcolor=COLORS['background'],
                    font=dict(color=COLORS['foreground']),
                    xaxis=dict(gridcolor=COLORS['border'], title='Final Price'),
                    yaxis=dict(gridcolor=COLORS['border'], title='Frequency'),
                    height=250,
                    margin=dict(l=60, r=20, t=40, b=40),
                )
                st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Click 'Run Monte Carlo' to simulate price distributions.")

# ============================================================================
# TAB 4: COMPARISON VIEW
# ============================================================================
with tab4:
    st.markdown('<div class="section-header">ALL ANALYSES COMPARISON</div>', unsafe_allow_html=True)
    st.caption(
        "Use this view when you want one combined pass: technical score, ensemble forecast, and Monte Carlo "
        "stress test for the selected symbol."
    )

    run_full_analysis = st.button(
        "Run Full Analysis",
        type="primary",
        use_container_width=False,
        key="run_full_analysis",
    )

    if run_full_analysis:
        with st.spinner("Running all analyses..."):
            # Fetch all data
            tech_data = None
            pred_data = None
            mc_data = None
            analysis_errors = []

            try:
                # Technical Analysis
                tech_resp = requests.post(f"{API_URL}/analysis/technical/{symbol}", timeout=REQUEST_TIMEOUT)
                if tech_resp.status_code == 200:
                    tech_data = tech_resp.json()
                    st.session_state.technical_data = tech_data
                    st.session_state.technical_symbol = symbol
                else:
                    analysis_errors.append(f"Technical: {get_error_detail(tech_resp)}")
            except requests.exceptions.RequestException as exc:
                analysis_errors.append(f"Technical: {exc}")

            try:
                # ML Prediction
                pred_resp = requests.get(f"{API_URL}/prediction/ensemble/{symbol}", timeout=LONG_REQUEST_TIMEOUT)
                if pred_resp.status_code == 200:
                    pred_data = pred_resp.json()
                    st.session_state.prediction_data = pred_data
                    st.session_state.prediction_symbol = symbol
                else:
                    analysis_errors.append(f"Prediction: {get_error_detail(pred_resp)}")
            except requests.exceptions.RequestException as exc:
                analysis_errors.append(f"Prediction: {exc}")

            try:
                # Monte Carlo
                mc_resp = requests.get(f"{API_URL}/prediction/monte-carlo/{symbol}",
                                       params={"n_simulations": 1000, "horizon_days": 30},
                                       timeout=MONTE_CARLO_TIMEOUT)
                if mc_resp.status_code == 200:
                    mc_data = mc_resp.json()
                    st.session_state.monte_carlo_data = mc_data
                    st.session_state.monte_carlo_symbol = symbol
                else:
                    analysis_errors.append(f"Monte Carlo: {get_error_detail(mc_resp)}")
            except requests.exceptions.RequestException as exc:
                analysis_errors.append(f"Monte Carlo: {exc}")

            if analysis_errors:
                st.warning("Analyze All completed with partial failures.")
                for error in analysis_errors:
                    st.caption(error)
            else:
                st.toast("All analyses complete!", icon="✅")

    # Comparison Display
    col_c1, col_c2, col_c3 = st.columns(3)

    with col_c1:
        st.markdown(f"""
        <div class="nextgen-card">
            <h4 style="margin: 0 0 12px 0; color: {COLORS['primary']};">📊 TECHNICAL</h4>
        """, unsafe_allow_html=True)

        if 'technical_data' in st.session_state and st.session_state.technical_symbol == symbol:
            tech = st.session_state.technical_data
            score = tech.get('score', {})
            st.metric("Score", f"{score.get('total', 0):.0f}")
            st.metric("Signal", score.get('signal', 'N/A'))
            st.metric("Trend", score.get('trend', 'N/A'))
        else:
            st.caption("Run Technical Analysis first")

    with col_c2:
        st.markdown(f"""
        <div class="nextgen-card">
            <h4 style="margin: 0 0 12px 0; color: {COLORS['primary']};">🎯 ML PREDICTION</h4>
        """, unsafe_allow_html=True)

        if 'prediction_data' in st.session_state and st.session_state.prediction_symbol == symbol:
            pred = st.session_state.prediction_data
            preds = pred.get('predictions', [])
            if preds:
                current = pred.get('current_price', 0)
                final = preds[-1].get('predicted_price', 0)
                change = ((final / current) - 1) * 100 if current > 0 else 0
                st.metric("7-Day Forecast", f"Rp {final:,.0f}")
                st.metric("Expected Change", f"{change:+.2f}%")
                if pred.get('model_status') != 'trained':
                    st.warning("Prediction unavailable")
        else:
            st.caption("Run ML Prediction first")

    with col_c3:
        st.markdown(f"""
        <div class="nextgen-card">
            <h4 style="margin: 0 0 12px 0; color: {COLORS['primary']};">🎲 MONTE CARLO</h4>
        """, unsafe_allow_html=True)

        if 'monte_carlo_data' in st.session_state and st.session_state.monte_carlo_symbol == symbol:
            mc = st.session_state.monte_carlo_data
            stats = mc.get('statistics', {})
            st.metric("VaR (95%)", f"{stats.get('var_95_pct', 0):.2f}%")
            st.metric("CVaR (95%)", f"{stats.get('cvar_95_pct', 0):.2f}%")
            st.metric("Prob. Loss", f"{stats.get('prob_loss_pct', 0):.1f}%")
        else:
            st.caption("Run Monte Carlo first")

    # Signal Convergence
    st.markdown("---")
    st.markdown('<div class="section-header">SIGNAL CONVERGENCE</div>', unsafe_allow_html=True)

    tech_signal = "N/A"
    ml_signal = "N/A"
    mc_signal = "N/A"

    if 'technical_data' in st.session_state and st.session_state.technical_symbol == symbol:
        tech_signal = st.session_state.technical_data.get('score', {}).get('signal', 'N/A')

    if 'prediction_data' in st.session_state and st.session_state.prediction_symbol == symbol:
        pred = st.session_state.prediction_data
        if pred.get('predictions'):
            current = pred.get('current_price', 0)
            final = pred['predictions'][-1].get('predicted_price', 0)
            change = ((final / current) - 1) * 100 if current > 0 else 0
            if change > 3:
                ml_signal = "BUY"
            elif change < -3:
                ml_signal = "SELL"
            else:
                ml_signal = "HOLD"

    if 'monte_carlo_data' in st.session_state and st.session_state.monte_carlo_symbol == symbol:
        stats = st.session_state.monte_carlo_data.get('statistics', {})
        prob_loss = stats.get('prob_loss_pct', 50)
        if prob_loss < 40:
            mc_signal = "BUY"
        elif prob_loss > 60:
            mc_signal = "SELL"
        else:
            mc_signal = "HOLD"

    # Display convergence
    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    with col_s1:
        st.metric("Technical", tech_signal)
    with col_s2:
        st.metric("ML Prediction", ml_signal)
    with col_s3:
        st.metric("Monte Carlo", mc_signal)
    with col_s4:
        # Check agreement
        signals = [tech_signal, ml_signal, mc_signal]
        buy_count = signals.count("BUY")
        sell_count = signals.count("SELL")

        if buy_count >= 2:
            consensus = "STRONG BUY"
            consensus_color = COLORS['primary']
        elif sell_count >= 2:
            consensus = "STRONG SELL"
            consensus_color = COLORS['destructive']
        elif buy_count == 1 and sell_count == 0:
            consensus = "WEAK BUY"
            consensus_color = COLORS['primary']
        elif sell_count == 1 and buy_count == 0:
            consensus = "WEAK SELL"
            consensus_color = COLORS['destructive']
        else:
            consensus = "NEUTRAL"
            consensus_color = COLORS['muted_foreground']

        st.markdown(f"""
        <div style="text-align: center; padding: 12px; background: {COLORS['muted']}; border-radius: 8px;">
            <div style="font-size: 0.75rem; color: {COLORS['muted_foreground']};">CONSENSUS</div>
            <div style="font-size: 1.2rem; font-weight: 600; color: {consensus_color};">{consensus}</div>
        </div>
        """, unsafe_allow_html=True)
