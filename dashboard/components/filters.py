"""
Reusable Filter Widgets for the IDX Trading Dashboard Screener.
"""

import streamlit as st


def _ui_scope(container=None):
    """Return the Streamlit container to render into."""
    return container or st.sidebar


def render_classification_filters(df, container=None):
    """Render classification filter widgets.

    Returns dict of filter selections.
    """
    ui = _ui_scope(container)
    use_sidebar_defaults = container is None
    ui.markdown("#### 📋 Classification")
    col1, col2 = ui.columns(2)
    with col1:
        show_lq45 = ui.checkbox("LQ45 Only") if use_sidebar_defaults else col1.checkbox("LQ45 Only")
    with col2:
        show_idx30 = ui.checkbox("IDX30 Only") if use_sidebar_defaults else col2.checkbox("IDX30 Only")

    # Defensive: Check if 'sector' column exists
    if 'sector' in df.columns:
        all_sectors = ["All"] + sorted([s for s in df['sector'].dropna().unique() if s])
    else:
        all_sectors = ["All"]
    selected_sector = ui.selectbox("Sector", all_sectors)

    # Defensive: Check if 'sub_sector' column exists
    if 'sub_sector' in df.columns:
        if selected_sector == "All":
            sub_sectors = df['sub_sector'].dropna().unique().tolist()
        else:
            sub_sectors = df[df['sector'] == selected_sector]['sub_sector'].dropna().unique().tolist()
        all_sub = ["All"] + sorted([s for s in sub_sectors if s])
    else:
        all_sub = ["All"]
    selected_sub = ui.selectbox("Sub-Sector", all_sub)

    boards = ["All", "Main", "Development", "Acceleration"]
    selected_board = ui.selectbox("Board", boards)

    return {
        "lq45": show_lq45,
        "idx30": show_idx30,
        "sector": selected_sector,
        "sub_sector": selected_sub,
        "board": selected_board,
    }


def render_price_filters(container=None):
    """Render price-related filters. Returns dict."""
    ui = _ui_scope(container)
    use_sidebar_defaults = container is None
    ui.markdown("#### 💰 Price")
    col1, col2 = ui.columns(2)
    with col1:
        min_price = ui.number_input("Min Price", min_value=0, value=0, step=50) if use_sidebar_defaults else col1.number_input("Min Price", min_value=0, value=0, step=50)
    with col2:
        max_price = ui.number_input("Max Price", min_value=0, value=0, step=50,
                                    help="0 = no limit") if use_sidebar_defaults else col2.number_input("Max Price", min_value=0, value=0, step=50, help="0 = no limit")
    min_mcap = ui.number_input("Min Market Cap (Bn IDR)", min_value=0, value=0)
    return {"min_price": min_price, "max_price": max_price, "min_market_cap": min_mcap}


def render_technical_filters(container=None):
    """Render technical indicator filters. Returns dict."""
    ui = _ui_scope(container)
    ui.markdown("#### 📊 Technical")
    rsi_range = ui.slider("RSI Range", 0, 100, (0, 100))
    macd_signal = ui.selectbox("MACD", ["Any", "Bullish Cross", "Bearish Cross"])
    ma_position = ui.multiselect("MA Position", ["Above MA20", "Above MA50", "Above MA200"])
    return {"rsi_range": rsi_range, "macd_signal": macd_signal, "ma_position": ma_position}


def render_volume_filters(container=None):
    """Render volume filters. Returns dict."""
    ui = _ui_scope(container)
    ui.markdown("#### 📈 Volume")
    min_vol = ui.number_input("Min Avg Daily Volume", min_value=0, value=0, step=100000)
    vol_spike = ui.checkbox("Volume Spike (>2x avg)")
    return {"min_volume": min_vol, "volume_spike": vol_spike}


def render_performance_filters(container=None):
    """Render performance return filters. Returns dict."""
    ui = _ui_scope(container)
    use_sidebar_defaults = container is None
    ui.markdown("#### 🏆 Performance")
    period = ui.selectbox("Return Period", ["1D", "1W", "1M", "3M", "6M", "1Y"])
    col1, col2 = ui.columns(2)
    with col1:
        min_ret = ui.number_input("Min Return %", value=-100.0, step=1.0) if use_sidebar_defaults else col1.number_input("Min Return %", value=-100.0, step=1.0)
    with col2:
        max_ret = ui.number_input("Max Return %", value=100.0, step=1.0) if use_sidebar_defaults else col2.number_input("Max Return %", value=100.0, step=1.0)
    return {"period": period, "min_return": min_ret, "max_return": max_ret}


def render_analysis_settings(container=None):
    """Render analysis settings panel. Returns dict."""
    ui = _ui_scope(container)
    with ui.expander("⚙️ Analysis Settings", expanded=False):
        st.markdown("**Trading Mode**")
        mode = st.selectbox("Mode", ["SWING", "POSITION", "INTRADAY", "INVESTOR"], index=0)

        st.markdown("**Technical Parameters**")
        rsi_period = st.slider("RSI Period", 7, 28, 14)
        ma_type = st.selectbox("MA Type", ["EMA", "SMA"])

        st.markdown("**Signal Weights**")
        tech_w = st.slider("Technical Weight", 0.0, 1.0, 0.5, 0.1)
        flow_w = st.slider("Flow Weight", 0.0, 1.0, 0.3, 0.1)
        fund_w = st.slider("Fundamental Weight", 0.0, 1.0, 0.2, 0.1)

        min_score = st.slider("Min Composite Score", 0, 100, 40)

        st.markdown("**Setup Types**")
        setups = st.multiselect("Scan for", ["Breakout", "Pullback", "Mean Reversion", "Momentum"],
                                default=["Breakout", "Pullback", "Momentum"])

        st.markdown("**Prediction**")
        pred_horizon = st.selectbox("Horizon", ["7d", "14d", "30d"])
        pred_confidence = st.slider("Min Confidence", 0.0, 1.0, 0.6, 0.1)

        st.markdown("**Risk Filters**")
        max_alloc = st.slider("Max Allocation %", 1, 30, 10)
        kelly_max = st.slider("Kelly Fraction Cap", 0.1, 1.0, 0.5, 0.1)

    return {
        "mode": mode,
        "rsi_period": rsi_period, "ma_type": ma_type,
        "tech_weight": tech_w, "flow_weight": flow_w, "fund_weight": fund_w,
        "min_score": min_score, "setups": setups,
        "pred_horizon": pred_horizon, "pred_confidence": pred_confidence,
        "max_allocation": max_alloc, "kelly_max": kelly_max,
    }
