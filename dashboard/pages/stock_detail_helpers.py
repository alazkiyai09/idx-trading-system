"""
Helper functions for Stock Detail page - extracted for testability.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta


def process_price_history(
    price_data: List[Dict],
    days: int = 200
) -> pd.DataFrame:
    """Process raw price data into DataFrame.

    Args:
        price_data: List of OHLCV dicts
        days: Number of days to include

    Returns:
        DataFrame with date index and OHLCV columns
    """
    if not price_data:
        return pd.DataFrame()

    df = pd.DataFrame(price_data)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').tail(days)

    return df


def calculate_price_change(
    current_price: float,
    previous_price: float
) -> Dict[str, float]:
    """Calculate price change metrics.

    Args:
        current_price: Current price
        previous_price: Previous price

    Returns:
        Dict with change metrics
    """
    if previous_price == 0:
        return {"change": 0, "change_pct": 0}

    change = current_price - previous_price
    change_pct = (change / previous_price) * 100

    return {
        "change": round(change, 2),
        "change_pct": round(change_pct, 2),
    }


def calculate_position_metrics(
    entry_price: float,
    current_price: float,
    quantity: int
) -> Dict[str, float]:
    """Calculate position metrics.

    Args:
        entry_price: Entry price
        current_price: Current price
        quantity: Number of shares

    Returns:
        Dict with position metrics
    """
    position_value = current_price * quantity
    cost_basis = entry_price * quantity
    pnl = position_value - cost_basis
    pnl_pct = ((current_price - entry_price) / entry_price) * 100 if entry_price > 0 else 0

    return {
        "position_value": round(position_value, 2),
        "cost_basis": round(cost_basis, 2),
        "pnl": round(pnl, 2),
        "pnl_pct": round(pnl_pct, 2),
    }


def validate_trade_params(
    symbol: str,
    side: str,
    quantity: int,
    price: float,
    capital: float
) -> Dict[str, Any]:
    """Validate trade parameters.

    Args:
        symbol: Stock symbol
        side: BUY or SELL
        quantity: Number of shares
        price: Trade price
        capital: Available capital

    Returns:
        Dict with validation result
    """
    errors = []

    if not symbol or len(symbol) < 2:
        errors.append("Invalid symbol")

    if side not in ["BUY", "SELL"]:
        errors.append("Side must be BUY or SELL")

    if quantity <= 0 or quantity % 100 != 0:
        errors.append("Quantity must be positive multiple of 100 (lot size)")

    if price <= 0:
        errors.append("Price must be positive")

    order_value = quantity * price
    if side == "BUY" and order_value > capital:
        errors.append(f"Insufficient capital. Need {order_value:,.0f}, have {capital:,.0f}")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "order_value": order_value,
    }


def process_foreign_flow_data(
    flow_data: List[Dict]
) -> pd.DataFrame:
    """Process foreign flow data for visualization.

    Args:
        flow_data: List of foreign flow dicts

    Returns:
        DataFrame with processed flow data
    """
    if not flow_data:
        return pd.DataFrame()

    df = pd.DataFrame(flow_data)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])

    # Calculate net flow if not present
    if 'foreign_net' not in df.columns:
        if 'foreign_buy' in df.columns and 'foreign_sell' in df.columns:
            df['foreign_net'] = df['foreign_buy'] - df['foreign_sell']

    return df


def calculate_flow_summary(flow_df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate summary metrics for foreign flow.

    Args:
        flow_df: DataFrame with foreign flow data

    Returns:
        Dict with flow summary metrics
    """
    if flow_df.empty or 'foreign_net' not in flow_df.columns:
        return {"net_flow": 0, "streak": 0, "direction": "neutral"}

    net_flow = flow_df['foreign_net'].sum()

    # Calculate consecutive days in same direction
    streak = 0
    if len(flow_df) > 0:
        last_direction = 1 if flow_df['foreign_net'].iloc[-1] > 0 else -1
        for val in flow_df['foreign_net'].iloc[::-1]:
            current_direction = 1 if val > 0 else -1
            if current_direction == last_direction:
                streak += 1
            else:
                break

    direction = "inflow" if net_flow > 0 else "outflow" if net_flow < 0 else "neutral"

    return {
        "net_flow": net_flow,
        "net_flow_billions": round(net_flow / 1e9, 2),
        "streak": streak,
        "direction": direction,
    }


def format_sentiment_score(score: float) -> Dict[str, str]:
    """Format sentiment score with label and color.

    Args:
        score: Sentiment score (0-100)

    Returns:
        Dict with label, emoji, and color
    """
    if score >= 70:
        return {"label": "Bullish", "emoji": "🟢", "color": "#26a69a"}
    elif score >= 55:
        return {"label": "Slightly Bullish", "emoji": "🟢", "color": "#66bb6a"}
    elif score >= 45:
        return {"label": "Neutral", "emoji": "🟡", "color": "#ffeb3b"}
    elif score >= 30:
        return {"label": "Slightly Bearish", "emoji": "🟠", "color": "#ff9800"}
    else:
        return {"label": "Bearish", "emoji": "🔴", "color": "#ef5350"}
