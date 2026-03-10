"""
KPI Cards and Metric Components for the IDX Trading Dashboard.

Design: NextGen-style metric cards with Emerald/Zinc color scheme.
"""

import streamlit as st
from dashboard.components.nextgen_styles import COLORS


def render_portfolio_metrics(capital, pnl, win_rate, total_trades, sharpe=None, max_dd=None, profit_factor=None):
    """Render a row of portfolio performance KPI cards."""
    cols = st.columns(4)

    with cols[0]:
        st.metric("💰 Capital", f"Rp {capital:,.0f}")

    with cols[1]:
        delta_pct = f"{pnl/max(capital,1)*100:.1f}%" if capital > 0 else "0%"
        st.metric("📊 Total P&L", f"Rp {pnl:,.0f}", delta=delta_pct)

    with cols[2]:
        st.metric("🎯 Win Rate", f"{win_rate*100:.1f}%")

    with cols[3]:
        st.metric("📈 Total Trades", f"{total_trades}")

    if sharpe is not None or max_dd is not None or profit_factor is not None:
        cols2 = st.columns(3)
        if sharpe is not None:
            with cols2[0]:
                st.metric("📐 Sharpe Ratio", f"{sharpe:.2f}")
        if max_dd is not None:
            with cols2[1]:
                st.metric("📉 Max Drawdown", f"{max_dd*100:.1f}%")
        if profit_factor is not None:
            with cols2[2]:
                st.metric("⚖️ Profit Factor", f"{profit_factor:.2f}")


def render_stock_info_card(details: dict):
    """Render a stock overview info card with NextGen styling."""
    st.markdown(f"""
    <div class="nextgen-card">
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 24px;">
            <div>
                <div style="font-size: 0.7rem; color: {COLORS['muted_foreground']}; margin-bottom: 8px;">COMPANY INFO</div>
                <div style="font-weight: 600; margin-bottom: 4px;">{details.get('name', 'N/A')}</div>
                <div style="font-size: 0.85rem;">Symbol: <span class="price-mono">{details.get('symbol', 'N/A')}</span></div>
                <div style="font-size: 0.85rem;">Sector: {details.get('sector', 'N/A')}</div>
            </div>
            <div>
                <div style="font-size: 0.7rem; color: {COLORS['muted_foreground']}; margin-bottom: 8px;">CLASSIFICATION</div>
                <div style="font-size: 0.85rem;">Sub-Sector: {details.get('sub_sector', 'N/A')}</div>
                <div style="font-size: 0.85rem;">Board: {details.get('board', 'N/A')}</div>
                <div style="font-size: 0.85rem;">
                    Market Cap: Rp {details.get('market_cap', 0)/1e12:.2f} T
                </div>
            </div>
            <div>
                <div style="font-size: 0.7rem; color: {COLORS['muted_foreground']}; margin-bottom: 8px;">INDEX MEMBERSHIP</div>
                <div style="display: flex; gap: 8px; flex-wrap: wrap;">
                    {'<span class="signal-badge bullish">LQ45</span>' if details.get('is_lq45') else ''}
                    {'<span class="signal-badge bullish">IDX30</span>' if details.get('is_idx30') else ''}
                    {'<span class="signal-badge neutral">—</span>' if not (details.get('is_lq45') or details.get('is_idx30')) else ''}
                </div>
                <div style="margin-top: 12px;">
                    Latest Close: <span class="price-mono" style="font-weight: 600;">Rp {details.get('latest_price', {}).get('close', 0):,.0f}</span>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_signal_card(signal_data: dict):
    """Render a signal result card with NextGen styling."""
    if signal_data.get("signal") == "None":
        st.info(signal_data.get("message", "No signal generated."))
        return

    action = signal_data.get('type', 'N/A')
    if action == "BUY":
        badge = f'<span class="signal-badge bullish">🟢 {action}</span>'
    elif action == "SELL":
        badge = f'<span class="signal-badge bearish">🔴 {action}</span>'
    else:
        badge = f'<span class="signal-badge neutral">🟡 {action}</span>'

    st.markdown(f"""
    <div class="nextgen-card">
        <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 16px;">
            <div>
                {badge}
                <div style="font-size: 1.1rem; font-weight: 600; margin-top: 8px;">
                    Setup: {signal_data.get('setup', 'N/A')}
                </div>
            </div>
            <div style="text-align: right;">
                <div style="font-size: 0.7rem; color: {COLORS['muted_foreground']};">SCORE</div>
                <div style="font-size: 1.5rem; font-weight: 600; color: {COLORS['primary']};">
                    {signal_data.get('score', 0):.1f}/100
                </div>
            </div>
        </div>
        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 16px;">
            <div>
                <div style="font-size: 0.7rem; color: {COLORS['muted_foreground']};">Entry</div>
                <div class="price-mono" style="font-weight: 500;">Rp {signal_data.get('entry_price', 0):,.0f}</div>
            </div>
            <div>
                <div style="font-size: 0.7rem; color: {COLORS['muted_foreground']};">Stop Loss</div>
                <div class="price-mono" style="font-weight: 500; color: {COLORS['destructive']};">Rp {signal_data.get('stop_loss', 0):,.0f}</div>
            </div>
            <div>
                <div style="font-size: 0.7rem; color: {COLORS['muted_foreground']};">Target(s)</div>
                <div class="price-mono" style="font-weight: 500; color: {COLORS['primary']};">
                    {', '.join([f"Rp {t:,.0f}" for t in signal_data.get('targets', [])])}
                </div>
            </div>
            <div>
                <div style="font-size: 0.7rem; color: {COLORS['muted_foreground']};">Risk/Reward</div>
                <div style="font-weight: 600; color: {COLORS['primary']};">1:{signal_data.get('risk_reward', 0):.1f}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_metric_row(metrics: list):
    """Render a row of metric cards.

    Args:
        metrics: List of dicts with keys: label, value, delta, delta_color
    """
    cols = st.columns(len(metrics))
    for i, m in enumerate(metrics):
        with cols[i]:
            delta = m.get('delta')
            st.metric(
                m.get('label', ''),
                m.get('value', ''),
                delta=delta,
            )


def render_conviction_score(score: float, rating: str = None):
    """Render a conviction score display with progress bar."""
    if score >= 80:
        rating = rating or "Strong Buy"
        color = COLORS['primary']
    elif score >= 60:
        rating = rating or "Buy"
        color = COLORS['primary_light']
    elif score >= 40:
        rating = rating or "Neutral"
        color = COLORS['warning']
    else:
        rating = rating or "Sell"
        color = COLORS['destructive']

    st.markdown(f"""
    <div class="nextgen-card">
        <div style="display: flex; align-items: center; gap: 16px;">
            <div class="conviction-score" style="color: {color};">{score:.0f}</div>
            <div style="flex: 1;">
                <div class="conviction-bar">
                    <div class="fill" style="width: {score}%; background: {color};"></div>
                </div>
                <div style="font-size: 0.8rem; color: {COLORS['muted_foreground']}; margin-top: 8px;">
                    {rating} — Component Alignment
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
