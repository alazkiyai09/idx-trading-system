"""
UX Components for IDX Trading System Dashboard

This module provides reusable UX components that implement:
- Progressive disclosure patterns
- Inline validation hints
- Contextual help tooltips
- Confirmation dialogs
- Quick filter presets
"""

import streamlit as st
from typing import Optional, List, Dict, Any, Callable
from datetime import datetime, time


# ============================================================================
# IDX-SPECIFIC CONSTANTS
# ============================================================================

IDX_LOT_SIZE =100  # Shares per lot
IDX_TRADING_HOURS = (time(9,0), time(17,10))  # WIB
IDX_PRICE_LIMIT =0.07  # ±7% daily limit
IDX_FEES = {"buy": 0.0015, "sell": 0.0025}  # 0.15% buy, 0.25% sell


# ============================================================================
# INLINE VALIDATION COMPONENTS
# ============================================================================

def validated_number_input(
    label: str,
    min_value: float =0,
    max_value: Optional[float] = None,
    value: float =0,
    step: float =1,
    key: str = None,
    help_text: str = None,
    validation_rule: str = None,
    show_hint: bool = True
) -> tuple:
    """
    Number input with inline validation and IDX-specific hints.

    Returns:
        tuple: (value, is_valid, error_message)
    """
    hint_html = ""
    if show_hint and validation_rule:
        hint_html = f'<small style="color: gray;">💡{validation_rule}</small>'

    # Add hint above input
    if hint_html:
        st.markdown(hint_html, unsafe_allow_html=True)

    input_value = st.number_input(
        label,
        min_value=min_value,
        max_value=max_value,
        value=value,
        step=step,
        key=key,
        help=help_text
    )

    # Validation logic
    is_valid = True
    error_msg = None

    if validation_rule == "lot_size":
        if input_value % IDX_LOT_SIZE != 0:
            is_valid = False
            error_msg = f"⚠️ Must be multiple of {IDX_LOT_SIZE} (IDX lot size)"
            st.warning(error_msg)

    elif validation_rule == "price_limit":
        # This would need reference price to validate
        pass

    return input_value, is_valid, error_msg


def idx_quantity_input(
    label: str = "Quantity (shares)",
    value: int =100,
    key: str = None,
    show_presets: bool = True
) -> tuple:
    """
    IDX-specific quantity input with lot size validation.

    Returns:
        tuple: (quantity, is_valid)
    """
    st.markdown(f"**{label}**")

    if show_presets:
        st.markdown('<small style="color: gray;">💡Quick select:</small>', unsafe_allow_html=True)
        preset_cols = st.columns(5)
        presets = [100, 500, 1000, 5000, 10000]
        selected_preset = None

        for i, preset in enumerate(presets):
            with preset_cols[i]:
                if st.button(f"{preset:,}", key=f"preset_{preset}_{key}"):
                    selected_preset = preset

        if selected_preset:
            value = selected_preset

    qty = st.number_input(
        "Quantity",
        min_value=100,
        value=value,
        step=100,
        key=key,
        label_visibility="collapsed"
    )

    # Validate lot size
    is_valid = qty % IDX_LOT_SIZE == 0
    if not is_valid:
        st.error(f"⚠️ Quantity must be multiple of {IDX_LOT_SIZE} shares (IDX lot size)")
    else:
        lots = qty // IDX_LOT_SIZE
        st.success(f"✅ {lots:,} lots ({qty:,} shares)")

    return qty, is_valid


def idx_symbol_input(
    label: str = "Symbol",
    key: str = None,
    suggestions: List[str] = None
) -> tuple:
    """
    IDX symbol input with validation.

    Returns:
        tuple: (symbol, is_valid)
    """
    symbol = st.text_input(
        label,
        key=key,
        max_chars=4,
        placeholder="e.g., BBCA"
    ).upper().strip()

    is_valid = len(symbol) == 4 and symbol.isalpha()

    if symbol and not is_valid:
        st.error("⚠️ IDX symbols must be 4 letters (e.g., BBCA, TLKM)")
    elif symbol and is_valid:
        st.success(f"✅ {symbol}")

    return symbol, is_valid


# ============================================================================
# PROGRESSIVE DISCLOSURE COMPONENTS
# ============================================================================

def collapsible_section(
    title: str,
    content_func: Callable,
    default_expanded: bool = False,
    icon: str = None,
    badge: str = None
) -> None:
    """
    Collapsible section for progressive disclosure.

    Args:
        title: Section title
        content_func: Function to render section content
        default_expanded: Whether section starts expanded
        icon: Optional emoji icon
        badge: Optional badge text (e.g., "NEW", "PRO")
    """
    display_title = f"{icon}{title}" if icon else title
    if badge:
        display_title += f" `{badge}`"

    with st.expander(display_title, expanded=default_expanded):
        content_func()


def tab_with_badge(
    tab_label: str,
    badge_count: int = None,
    badge_color: str = "red"
) -> None:
    """
    Create a tab with optional notification badge.

    Note: Streamlit doesn't support native badges, so this is a visual workaround.
    """
    if badge_count and badge_count > 0:
        return f"{tab_label} 🔴{badge_count}"
    return tab_label


# ============================================================================
# CONFIRMATION DIALOGS
# ============================================================================

def confirm_dialog(
    title: str,
    message: str,
    confirm_text: str = "Confirm",
    cancel_text: str = "Cancel",
    key: str = None,
    danger: bool = False
) -> Optional[bool]:
    """
    Confirmation dialog for critical actions.

    Returns:
        bool: True if confirmed, False if cancelled, None if not shown
    """
    dialog_key = f"confirm_{key}"

    if dialog_key not in st.session_state:
        st.session_state[dialog_key] = None

    # Show dialog
    if st.session_state[dialog_key] is None:
        st.markdown("---")
        if danger:
            st.warning(f"⚠️ **{title}**")
        else:
            st.info(f"ℹ️ **{title}**")
        st.markdown(message)

        col1, col2, col3 = st.columns([2, 1, 1])
        with col2:
            if st.button(confirm_text, type="primary" if not danger else "secondary", use_container_width=True):
                st.session_state[dialog_key] = True
                st.rerun()
        with col3:
            if st.button(cancel_text, use_container_width=True):
                st.session_state[dialog_key] = False
                st.rerun()

        return None

    result = st.session_state[dialog_key]
    # Reset for next use
    st.session_state[dialog_key] = None
    return result


def confirm_trade(
    symbol: str,
    side: str,
    quantity: int,
    price: float,
    key: str = None
) -> Optional[bool]:
    """
    Trade confirmation dialog with order summary.
    """
    total = quantity * price
    buy_fee = total * IDX_FEES["buy"]
    sell_fee = total * IDX_FEES["sell"]
    fee = buy_fee if side == "BUY" else sell_fee
    total_with_fees = total + fee if side == "BUY" else total - fee

    message = f"""
    | Item | Value |
    |------|-------|
    | **Symbol** | {symbol} |
    | **Side** | {side} |
    | **Quantity** | {quantity:,} shares ({quantity//100:,} lots) |
    | **Price** | Rp {price:,.0f} |
    | **Subtotal** | Rp {total:,.0f} |
    | **Fee** ({IDX_FEES['buy']*100 if side=='BUY' else IDX_FEES['sell']*100:.2f}%) | Rp {fee:,.0f} |
    | **Total** | Rp {total_with_fees:,.0f} |
    """

    return confirm_dialog(
        title=f"Confirm {side} Order",
        message=message,
        confirm_text=f"Execute {side}",
        key=f"trade_{symbol}_{side}_{key}",
        danger=(side == "SELL")
    )


# ============================================================================
# CONTEXTUAL HELP COMPONENTS
# ============================================================================

def help_tooltip(
    text: str,
    icon: str = "❓",
    position: str = "right"
) -> None:
    """
    Contextual help tooltip.

    Note: Streamlit doesn't support native tooltips, so this uses markdown.
    """
    st.markdown(
        f'<span title="{text}" style="cursor: help;">{icon}</span>',
        unsafe_allow_html=True
    )


def info_card(
    title: str,
    content: str,
    icon: str = "ℹ️",
    collapsible: bool = False
) -> None:
    """
    Information card with optional collapse.
    """
    if collapsible:
        with st.expander(f"{icon} {title}"):
            st.markdown(content)
    else:
        st.info(f"{icon} **{title}**\n\n{content}")


def trading_hours_indicator() -> None:
    """
    Display current trading hours status.
    Uses Asia/Jakarta (WIB) timezone for IDX market.
    Checks for weekends (market closed on Saturday and Sunday).
    """
    try:
        from zoneinfo import ZoneInfo
        wib = ZoneInfo("Asia/Jakarta")
        now = datetime.now(wib)
    except ImportError:
        # Fallback for Python < 3.9
        from datetime import timedelta
        now = datetime.now() + timedelta(hours=7)  # Approximate WIB

    current_time = now.time()
    weekday = now.weekday()  # Monday=0, Sunday=6

    # IDX is closed on Saturday (5) and Sunday (6)
    is_weekend = weekday >= 5

    is_trading_hours = IDX_TRADING_HOURS[0] <= current_time <= IDX_TRADING_HOURS[1]

    if is_weekend:
        day_name = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][weekday]
        st.warning(f"📅 **Market Closed** - {day_name} (Weekend) | Trading hours: Mon-Fri {IDX_TRADING_HOURS[0].strftime('%H:%M')} - {IDX_TRADING_HOURS[1].strftime('%H:%M')} WIB")
    elif is_trading_hours:
        st.success(f"🟢 **Market Open** | Trading hours: {IDX_TRADING_HOURS[0].strftime('%H:%M')} - {IDX_TRADING_HOURS[1].strftime('%H:%M')} WIB")
    else:
        st.info(f"🔴 **Market Closed** | Trading hours: {IDX_TRADING_HOURS[0].strftime('%H:%M')} - {IDX_TRADING_HOURS[1].strftime('%H:%M')} WIB")


# ============================================================================
# QUICK FILTER PRESETS
# ============================================================================

QUICK_FILTER_PRESETS = {
    "🔥 Hot Stocks": {
        "description": "High momentum stocks with strong technical signals",
        "filters": {
            "lq45": True,
            "min_volume": 1000000,
            "rsi_min": 50,
            "rsi_max": 70
        }
    },
    "💎 Value Picks": {
        "description": "Undervalued stocks with good fundamentals",
        "filters": {
            "min_market_cap": 5,# Trillion IDR
            "rsi_max": 40
        }
    },
    "🚀 Momentum": {
        "description": "Stocks breaking out with strong volume",
        "filters": {
            "min_volume": 500000,
            "rsi_min": 60,
            "perf_1w": True
        }
    },
    "💰 Dividend": {
        "description": "Blue-chip stocks with dividend yield",
        "filters": {
            "lq45": True,
            "min_market_cap": 10
        }
    },
    "📊 Technical Breakout": {
        "description": "Stocks showing technical breakout patterns",
        "filters": {
            "rsi_min": 55,
            "rsi_max": 80,
            "min_volume": 500000
        }
    }
}


def quick_filter_buttons(
    on_select: Callable[[Dict[str, Any]], None],
    key_prefix: str = "quick_filter"
) -> None:
    """
    Render quick filter preset buttons.
    """
    st.markdown("### 🎯 Quick Filters")
    st.markdown('<small style="color: gray;">Click a preset to quickly apply common filter combinations</small>', unsafe_allow_html=True)

    cols = st.columns(len(QUICK_FILTER_PRESETS))

    for i, (preset_name, preset_data) in enumerate(QUICK_FILTER_PRESETS.items()):
        with cols[i]:
            if st.button(
                preset_name,
                key=f"{key_prefix}_{i}",
                use_container_width=True
            ):
                on_select(preset_data["filters"])
                st.toast(f"Applied: {preset_name}", icon="✅")


# ============================================================================
# ERROR HANDLING COMPONENTS
# ============================================================================

def error_with_recovery(
    error_message: str,
    recovery_suggestions: List[str],
    key: str = None
) -> None:
    """
    Display error message with recovery suggestions.
    """
    st.error(f"❌ {error_message}")

    st.markdown("**💡 Suggestions:**")
    for suggestion in recovery_suggestions:
        st.markdown(f"- {suggestion}")


def api_error_handler(
    error: Exception,
    context: str = "operation"
) -> None:
    """
    Handle API errors with user-friendly messages.
    """
    error_str = str(error).lower()

    if "connection" in error_str or "timeout" in error_str:
        error_with_recovery(
            "Unable to connect to the server",
            [
                "Check if the API server is running on port 8000",
                "Verify your network connection",
                "Try refreshing the page"
            ]
        )
    elif "401" in error_str or "unauthorized" in error_str:
        error_with_recovery(
            "Authentication required",
            [
                "Check your API credentials",
                "Log in again if your session expired"
            ]
        )
    elif "404" in error_str or "not found" in error_str:
        error_with_recovery(
            f"Resource not found during {context}",
            [
                "The requested data may have been removed",
                "Try refreshing or selecting a different option"
            ]
        )
    elif "400" in error_str or "bad request" in error_str:
        error_with_recovery(
            f"Invalid request for {context}",
            [
                "Check your input values",
                "Ensure all required fields are filled"
            ]
        )
    else:
        error_with_recovery(
            f"An error occurred during {context}",
            [
                "Try again in a few moments",
                "Contact support if the problem persists"
            ]
        )


# ============================================================================
# LOADING STATES
# ============================================================================

def loading_placeholder(message: str = "Loading..."):
    """
    Create a loading placeholder with estimated time.
    """
    return st.empty()


def progress_indicator(
    current: int,
    total: int,
    message: str = None
) -> None:
    """
    Show progress indicator for long operations.
    """
    progress = current / total if total > 0 else 0
    st.progress(progress, text=f"{message or 'Processing'} ({current}/{total})")


# ============================================================================
# SUCCESS FEEDBACK
# ============================================================================

def success_with_details(
    title: str,
    details: Dict[str, Any],
    show_confetti: bool = False
) -> None:
    """
    Display success message with details.
    """
    st.success(f"✅ {title}")

    with st.expander("View Details"):
        for key, value in details.items():
            st.markdown(f"**{key}:** {value}")

    if show_confetti:
        st.balloons()


# ============================================================================
# SEARCHABLE STOCK SELECTOR
# ============================================================================

def render_stock_selector(
    label: str = "Select Stock",
    default_symbol: str = "",
    api_url: str = None,
    label_visibility: str = "visible",
    allow_empty: bool = False,
) -> str:
    """
    Render a searchable stock selector dropdown.

    Fetches all stocks from API and displays as searchable selectbox
    with symbol and name for easy discovery.

    Args:
        label: Label for the selectbox
        default_symbol: Default symbol to select
        api_url: API base URL (optional, will try to get from settings)
        label_visibility: Visibility of the label ("visible", "hidden", "collapsed")

    Returns:
        Selected stock symbol (e.g., "BBCA")
    """
    import os

    # Get API URL from settings or environment
    if api_url is None:
        try:
            from config.settings import settings
            api_url = os.environ.get("API_URL", settings.api_url)
        except Exception:
            api_url = os.environ.get("API_URL", "http://localhost:8000")

    options = get_stock_options(api_url)
    option_labels = list(options.keys())
    if allow_empty:
        option_labels = ["Search by symbol or company name..."] + option_labels

    # Find default index
    default_idx = None if allow_empty and not default_symbol else 0
    if default_symbol:
        for i, (lbl, sym) in enumerate(options.items()):
            if sym == default_symbol:
                default_idx = i + (1 if allow_empty else 0)
                break

    # Create unique key based on label to avoid conflicts
    key = f"stock_selector_{label.replace(' ', '_').lower()}"

    selected = st.selectbox(
        label,
        options=option_labels,
        index=default_idx,
        key=key,
        help="Type to search for a stock by symbol or company name",
        label_visibility=label_visibility,
        placeholder="Search by symbol or company name",
    )

    if not selected:
        return ""

    if allow_empty and selected == "Search by symbol or company name...":
        return ""

    return options.get(selected, "")


@st.cache_data(ttl=300, show_spinner=False)
def get_stock_options(base_url: str) -> Dict[str, str]:
    """Fetch the full searchable stock universe as {label: symbol}."""
    import requests

    try:
        option_map: Dict[str, str] = {}
        all_stocks = []
        skip = 0
        limit = 500

        while True:
            resp = requests.get(f"{base_url}/stocks?skip={skip}&limit={limit}", timeout=10)
            if resp.status_code != 200:
                break
            data = resp.json()
            stocks = data.get("stocks", data) if isinstance(data, dict) else data
            if not stocks:
                break
            all_stocks.extend(
                s for s in stocks if isinstance(s, dict) and s.get("symbol")
            )
            total = data.get("total", len(all_stocks)) if isinstance(data, dict) else len(all_stocks)
            skip += len(stocks)
            if skip >= total or len(stocks) < limit:
                break

        if all_stocks:
            for stock in all_stocks:
                symbol = stock.get("symbol", "UNK")
                name = stock.get("name", "Unknown")
                option_map[f"{symbol} - {name[:40]}"] = symbol

        resp = requests.get(f"{base_url}/stocks/symbols", timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            symbols = data.get("symbols", []) if isinstance(data, dict) else data
            labels = data.get("labels", {}) if isinstance(data, dict) else {}
            if symbols:
                for symbol in sorted(symbols):
                    if symbol not in option_map.values():
                        option_map[labels.get(symbol, symbol)] = symbol

        if option_map:
            return dict(sorted(option_map.items(), key=lambda item: item[1]))
    except Exception:
        pass

    return {
        "BBCA - Bank Central Asia": "BBCA",
        "BBRI - Bank Rakyat Indonesia": "BBRI",
        "TLKM - Telkom Indonesia": "TLKM",
        "ASII - Astra International": "ASII",
    }
