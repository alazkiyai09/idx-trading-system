"""
Helper functions for Stock Screener page - extracted for testability.
"""
import pandas as pd
from typing import Dict, Any, List, Optional


def apply_classification_filters(
    df: pd.DataFrame,
    filters: Dict[str, Any]
) -> pd.DataFrame:
    """Apply classification filters to dataframe.

    Args:
        df: DataFrame with stock data
        filters: Dict with keys: lq45, idx30, sector, sub_sector, board

    Returns:
        Filtered DataFrame
    """
    filtered = df.copy()

    if filters.get("lq45"):
        if 'is_lq45' in filtered.columns:
            filtered = filtered[filtered['is_lq45'] == True]

    if filters.get("idx30"):
        if 'is_idx30' in filtered.columns:
            filtered = filtered[filtered['is_idx30'] == True]

    if filters.get("sector") and filters["sector"] != "All":
        if 'sector' in filtered.columns:
            filtered = filtered[filtered['sector'] == filters["sector"]]

    if filters.get("sub_sector") and filters["sub_sector"] != "All":
        if 'sub_sector' in filtered.columns:
            filtered = filtered[filtered['sub_sector'] == filters["sub_sector"]]

    return filtered


def apply_price_filters(
    df: pd.DataFrame,
    filters: Dict[str, Any]
) -> pd.DataFrame:
    """Apply price-related filters to dataframe.

    Args:
        df: DataFrame with stock data
        filters: Dict with keys: min_price, max_price, min_market_cap

    Returns:
        Filtered DataFrame
    """
    filtered = df.copy()

    if filters.get("min_market_cap", 0) > 0:
        if 'market_cap' in filtered.columns:
            filtered = filtered[filtered['market_cap'] >= filters["min_market_cap"] * 1e9]

    if filters.get("min_price", 0) > 0:
        if 'latest_price' in filtered.columns:
            # Handle nested price data
            pass  # Complex handling needed based on data structure

    if filters.get("max_price", 0) > 0:
        if 'latest_price' in filtered.columns:
            pass

    return filtered


def apply_technical_filters(
    df: pd.DataFrame,
    filters: Dict[str, Any],
    technical_data: Optional[Dict[str, Dict]] = None
) -> pd.DataFrame:
    """Apply technical indicator filters.

    Args:
        df: DataFrame with stock data
        filters: Dict with keys: rsi_range, macd_signal, ma_position
        technical_data: Optional dict mapping symbol to technical indicators

    Returns:
        Filtered DataFrame
    """
    if technical_data is None:
        return df.copy()

    filtered = df.copy()
    rsi_min, rsi_max = filters.get("rsi_range", (0, 100))

    symbols_to_keep = []
    for symbol in filtered['symbol']:
        tech = technical_data.get(symbol, {})
        rsi = tech.get('rsi', 50)

        if rsi_min <= rsi <= rsi_max:
            symbols_to_keep.append(symbol)

    return filtered[filtered['symbol'].isin(symbols_to_keep)]


def calculate_filter_stats(
    original_count: int,
    filtered_count: int
) -> Dict[str, Any]:
    """Calculate filter statistics.

    Args:
        original_count: Number of stocks before filtering
        filtered_count: Number of stocks after filtering

    Returns:
        Dict with filter statistics
    """
    reduction_pct = 0.0
    if original_count > 0:
        reduction_pct = ((original_count - filtered_count) / original_count) * 100

    return {
        "original_count": original_count,
        "filtered_count": filtered_count,
        "removed_count": original_count - filtered_count,
        "reduction_pct": round(reduction_pct, 1),
    }


def get_unique_sectors(df: pd.DataFrame) -> List[str]:
    """Get list of unique sectors from dataframe.

    Args:
        df: DataFrame with stock data

    Returns:
        Sorted list of unique sectors
    """
    if 'sector' not in df.columns:
        return []
    return sorted([s for s in df['sector'].dropna().unique() if s])


def get_unique_sub_sectors(df: pd.DataFrame, sector: Optional[str] = None) -> List[str]:
    """Get list of unique sub-sectors, optionally filtered by sector.

    Args:
        df: DataFrame with stock data
        sector: Optional sector to filter by

    Returns:
        Sorted list of unique sub-sectors
    """
    if 'sub_sector' not in df.columns:
        return []

    if sector and sector != "All":
        df = df[df['sector'] == sector]

    return sorted([s for s in df['sub_sector'].dropna().unique() if s])


def sort_analysis_results(
    df: pd.DataFrame,
    sort_by: str,
    ascending: bool = True
) -> pd.DataFrame:
    """Sort analysis results dataframe.

    Args:
        df: DataFrame with analysis results
        sort_by: Column name to sort by
        ascending: Sort direction

    Returns:
        Sorted DataFrame
    """
    if sort_by not in df.columns:
        return df

    return df.sort_values(sort_by, ascending=ascending, na_position='last')


def format_stock_display(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """Format stock dataframe for display.

    Args:
        df: DataFrame with stock data
        columns: Optional list of columns to include

    Returns:
        Formatted DataFrame
    """
    if columns is None:
        columns = ['symbol', 'name', 'sector', 'sub_sector', 'is_lq45']

    available_cols = [c for c in columns if c in df.columns]
    return df[available_cols].copy()
