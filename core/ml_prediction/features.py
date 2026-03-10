"""
Feature engineering for ML models
"""
import numpy as np
import pandas as pd
from typing import Dict, List

from config.settings import MLConfig
import logging
from core.ml_prediction.utils import safe_calculation

logger = logging.getLogger(__name__)


@safe_calculation(default_return=None)
def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index

    Args:
        series: Price series
        period: RSI period

    Returns:
        RSI series
    """
    try:
        import talib
        return talib.RSI(series.values, timeperiod=period)
    except ImportError:
        # Fallback pandas implementation
        delta = series.diff()
        up = delta.clip(lower=0).rolling(period).mean()
        down = (-delta.clip(upper=0)).rolling(period).mean()
        rs = up / down.replace(0, 1e-9)
        return 100 - (100 / (1 + rs))


@safe_calculation(default_return=None)
def calculate_adx(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """
    Calculate Average Directional Index

    Args:
        df: DataFrame with high, low, close
        period: ADX period

    Returns:
        ADX series
    """
    try:
        import talib
        return talib.ADX(df['high'].values, df['low'].values, df['close'].values, timeperiod=period)
    except ImportError:
        # Fallback implementation
        high, low, close = df['high'], df['low'], df['close']

        tr1 = (high - low).abs()
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        plus_dm = (high - high.shift(1)).clip(lower=0)
        minus_dm = (low.shift(1) - low).clip(lower=0)

        tr_n = tr.ewm(alpha=1/period, adjust=False).mean()
        pdi = 100 * (plus_dm.ewm(alpha=1/period, adjust=False).mean() / tr_n)
        mdi = 100 * (minus_dm.ewm(alpha=1/period, adjust=False).mean() / tr_n)

        dx = (100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, 1e-9))
        adx = dx.ewm(alpha=1/period, adjust=False).mean()

        return adx


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators to DataFrame

    Args:
        df: DataFrame with OHLCV data

    Returns:
        DataFrame with added indicators
    """
    df = df.copy()

    # RSI
    df['RSI'] = calculate_rsi(df['close'], MLConfig.RSI_PERIOD)

    # ADX
    df['ADX'] = calculate_adx(df, MLConfig.ADX_PERIOD)

    # Log returns
    df['log_r'] = np.log(df['close'] / df['close'].shift(1))

    # Price change percentage
    df['price_change'] = df['close'].pct_change() * 100

    # GARCH volatility proxy
    df['GARCH_vol'] = df['log_r'].rolling(MLConfig.VOL_PROXY_WIN).std() * np.sqrt(252)

    logger.debug("Added technical indicators")
    return df


def add_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add market regime features

    Args:
        df: DataFrame with price data

    Returns:
        DataFrame with regime features
    """
    df = df.copy()

    # Trend regime (Golden/Death Cross simplified)
    fast_ma = df['close'].rolling(MLConfig.TREND_REGIME_WIN).mean()
    slow_ma = df['close'].rolling(MLConfig.TREND_REGIME_WIN * 2).mean()
    df['trend_regime'] = np.sign(fast_ma - slow_ma).fillna(0)

    # Volatility regime
    vol = df['log_r'].rolling(MLConfig.VOL_REGIME_WIN).std()
    vol_mean = vol.rolling(MLConfig.VOL_REGIME_WIN * 2).mean()
    df['vol_regime'] = (
        (vol > vol_mean * 1.5).astype(int) - (vol < vol_mean * 0.75).astype(int)
    )

    logger.debug("Added regime features")
    return df


def add_lag_features(df: pd.DataFrame, lags: List[int]) -> pd.DataFrame:
    """
    Add lagged features

    Args:
        df: DataFrame with features
        lags: List of lag periods

    Returns:
        DataFrame with lagged features
    """
    df = df.copy()

    for lag in lags:
        df[f'log_r_lag_{lag}'] = df['log_r'].shift(lag)

    logger.debug(f"Added {len(lags)} lag features")
    return df


def build_exogenous_features(
    exo_dict: Dict[str, pd.DataFrame],
    target_index: pd.DatetimeIndex,
    shift_by: int = 1,
    lags: List[int] = None,
    roll_windows: List[int] = None
) -> pd.DataFrame:
    """
    Build exogenous feature matrix

    Args:
        exo_dict: Dictionary of exogenous DataFrames (name -> df)
        target_index: Target DatetimeIndex to align to
        shift_by: Shift amount to prevent leakage
        lags: Lag periods for features
        roll_windows: Rolling window sizes

    Returns:
        DataFrame with exogenous features
    """
    if lags is None:
        lags = MLConfig.EXO_LAGS
    if roll_windows is None:
        roll_windows = MLConfig.EXO_ROLL_WINDOWS

    pieces = []

    for name, df_exo in exo_dict.items():
        if df_exo.empty:
            logger.warning(f"Exogenous source '{name}' is empty, skipping")
            continue

        # Ensure datetime index
        if 'date' in df_exo.columns:
            df_exo = df_exo.copy()
            df_exo['date'] = pd.to_datetime(df_exo['date'], errors='coerce')
            df_exo = df_exo.dropna(subset=['date']).set_index('date')

        df_exo.index = pd.to_datetime(df_exo.index)

        # Find price column
        price_col = next(
            (c for c in ['Adj Close', 'Close', 'close', 'Price'] if c in df_exo.columns),
            df_exo.columns[0]
        )

        series = df_exo[[price_col]].rename(columns={price_col: name}).sort_index()
        pieces.append(series)

    if not pieces:
        return pd.DataFrame(index=target_index)

    # Merge all exogenous series
    exo_merged = pd.concat(pieces, axis=1).sort_index().ffill()
    exo_reindexed = exo_merged.reindex(target_index).ffill()

    # Build features
    feats = pd.DataFrame(index=exo_reindexed.index)

    for col in exo_reindexed.columns:
        pct_change = exo_reindexed[col].pct_change().fillna(0)

        # Base percentage change
        feats[f"{col}_pct"] = pct_change

        # Lags
        for lag in lags:
            feats[f"{col}_pct_lag{lag}"] = pct_change.shift(lag)

        # Rolling statistics
        for window in roll_windows:
            feats[f"{col}_roll_mean_{window}"] = pct_change.rolling(window).mean()
            feats[f"{col}_roll_std_{window}"] = pct_change.rolling(window).std()
            feats[f"{col}_roll_p90_{window}"] = pct_change.rolling(window).quantile(0.9)

    # Shift to prevent data leakage
    feats = feats.shift(shift_by)

    logger.info(f"Built {feats.shape[1]} exogenous features")
    return feats


def preprocess_stock_data(
    df: pd.DataFrame,
    exo_dict: Dict[str, pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Full preprocessing pipeline for stock data

    Args:
        df: Raw stock DataFrame with OHLCV
        exo_dict: Optional exogenous data dictionary

    Returns:
        Preprocessed DataFrame with all features
    """
    # Ensure proper datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
        else:
            raise ValueError("DataFrame must have datetime index or 'date' column")

    df = df.sort_index()

    # Add technical indicators
    df = add_technical_indicators(df)

    # Add regime features
    df = add_regime_features(df)

    # Add lag features
    df = add_lag_features(df, MLConfig.EXTENDED_LAGS)

    # Add exogenous features if provided
    if exo_dict:
        exo_feats = build_exogenous_features(exo_dict, df.index)
        df = df.join(exo_feats, how='left')

    # Clean up
    initial_rows = len(df)
    df = df.replace([np.inf, -np.inf], np.nan)

    # Drop rows with NaN in critical columns
    core_cols = ['close', 'high', 'low', 'log_r']
    df = df.dropna(subset=core_cols)

    # Forward fill remaining NaNs in feature columns
    feature_cols = [c for c in df.columns if c not in core_cols]
    df[feature_cols] = df[feature_cols].ffill().fillna(0)

    final_rows = len(df)
    logger.info(
        f"Preprocessing complete: {initial_rows} -> {final_rows} rows "
        f"({initial_rows - final_rows} dropped)"
    )

    if final_rows < MLConfig.MIN_TOTAL_ROWS:
        logger.warning(
            f"Final rows ({final_rows}) < minimum required ({MLConfig.MIN_TOTAL_ROWS})"
        )

    return df
