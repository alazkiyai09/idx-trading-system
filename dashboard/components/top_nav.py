"""Top navigation for the Streamlit dashboard."""

import streamlit as st

NAV_ITEMS = [
    ("home", "Home", "/"),
    ("overview", "Market", "/market_overview"),
    ("screener", "Screener", "/screener"),
    ("detail", "Stock", "/stock_detail"),
    ("sentiment", "Sentiment", "/sentiment"),
    ("trading", "Trading", "/virtual_trading"),
    ("ml", "ML", "/ml_prediction"),
    ("settings", "Settings", "/settings"),
]

def render_top_nav(current_key: str) -> None:
    """Render HTML top navigation without Streamlit widget chrome."""
    st.markdown('<div class="idx-top-nav-anchor"></div>', unsafe_allow_html=True)
    links = []
    for key, label, href in NAV_ITEMS:
        active_class = " idx-top-nav-link-active" if key == current_key else ""
        active_attr = ' data-testid="idx-top-nav-active"' if key == current_key else ""
        links.append(
            f'<a class="idx-top-nav-link{active_class}" href="{href}" target="_self"{active_attr}>{label}</a>'
        )
    st.markdown(
        (
            '<div class="idx-top-nav-shell">'
            f"<div class=\"idx-top-nav-links\">{''.join(links)}</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def render_status_strip(items: list[tuple[str, str]]) -> None:
    """Render compact shell status strip."""
    st.markdown('<div class="idx-status-strip-anchor"></div>', unsafe_allow_html=True)
    cols = st.columns(len(items))
    for col, (label, value) in zip(cols, items):
        with col:
            st.markdown(
                f"""
                <div class="idx-status-chip">
                    <div class="idx-status-label">{label}</div>
                    <div class="idx-status-value">{value}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
