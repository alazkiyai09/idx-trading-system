"""
Model prediction and forecasting
"""
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Literal

from core.ml_prediction.models import summarize_sequence
import logging

logger = logging.getLogger(__name__)


def safe_array(arr, fill_val: float = 0.0) -> np.ndarray:
    """Convert array to safe format with no NaN/inf"""
    return np.nan_to_num(np.asarray(arr, dtype=float), nan=fill_val, posinf=fill_val, neginf=fill_val)


def get_last_valid_sigma(data: pd.DataFrame, window: int = 20) -> float:
    """
    Get last valid volatility estimate

    Args:
        data: DataFrame with log returns
        window: Rolling window for volatility

    Returns:
        Last valid sigma value
    """
    sigma = data['log_r'].rolling(window).std().shift(1)
    return float(sigma.dropna().iloc[-1])


class Predictor:
    """Make predictions with trained ensemble"""

    def __init__(self, artifacts: Dict):
        """
        Initialize predictor

        Args:
            artifacts: Trained model artifacts
        """
        self.artifacts = artifacts
        self.config = artifacts['config']
        self.n_prev = self.config['n_prev']
        self.horizon = self.config['horizon']
        self.base_models = artifacts['base_models']
        self.scaler = artifacts['scaler']
        self.meta_models = artifacts['meta_models']

    def predict_single(
        self,
        data: pd.DataFrame,
        return_components: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray] | Tuple[np.ndarray, np.ndarray, Dict[str, Dict[str, np.ndarray]]]:
        """
        Generate single forecast from one artifact set

        Args:
            data: Preprocessed DataFrame

        Returns:
            Tuple of (returns, prices)
        """
        # Prepare input sequence
        feature_cols = self.artifacts['feature_cols']
        X_df = data[feature_cols].tail(self.n_prev)

        if len(X_df) < self.n_prev:
            raise ValueError(f"Insufficient data: {len(X_df)} < {self.n_prev}")

        X_window = safe_array(X_df.values)
        X_scaled = self.scaler.transform(X_window).reshape(1, self.n_prev, -1)

        # Get base model predictions
        base_preds = {}
        for model_name in self.base_models:
            model = self.artifacts[model_name]

            if model_name in ['lstm', 'cnn_lstm']:
                preds = model.predict(X_scaled, verbose=0)
            else:  # SVR
                preds = model.predict(X_scaled)

            base_preds[model_name] = safe_array(preds.flatten())

        # Stack base predictions
        base_preds_matrix = np.stack(
            [base_preds[name] for name in self.base_models],
            axis=-1
        )

        # Meta-learner predictions
        scaled_returns = np.zeros(self.horizon, dtype=float)
        for h in range(self.horizon):
            meta_input = base_preds_matrix[h, :].reshape(1, -1)
            scaled_returns[h] = self.meta_models[h].predict(meta_input)[0]

        # Unscale returns
        last_sigma = get_last_valid_sigma(data)
        final_returns = scaled_returns * last_sigma

        # Calculate price path
        last_close = data['close'].iloc[-1]
        final_prices = last_close * np.exp(np.cumsum(final_returns))

        if return_components:
            base_components: Dict[str, Dict[str, np.ndarray]] = {}
            for model_name, scaled_pred in base_preds.items():
                model_returns = scaled_pred * last_sigma
                model_prices = last_close * np.exp(np.cumsum(model_returns))
                base_components[model_name] = {
                    "returns": model_returns,
                    "prices": model_prices,
                }
            return final_returns, final_prices, base_components

        return final_returns, final_prices


class MultiSeedPredictor:
    """Aggregate predictions from multiple seeds"""

    def __init__(self, multiseed_artifacts: Dict):
        """
        Initialize multi-seed predictor

        Args:
            multiseed_artifacts: Multi-seed artifacts dictionary
        """
        if multiseed_artifacts.get('type') != 'multiseed':
            raise ValueError("Expected multiseed artifacts")

        self.artifacts = multiseed_artifacts
        self.predictors = [
            Predictor(art) for art in multiseed_artifacts['artifacts_list']
        ]

    def predict(
        self,
        data: pd.DataFrame,
        aggregation: Literal['weighted', 'mean', 'median'] = 'weighted',
        return_components: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray] | Tuple[np.ndarray, np.ndarray, Dict[str, Dict[str, np.ndarray]]]:
        """
        Aggregate predictions from all seeds

        Args:
            data: Preprocessed DataFrame
            aggregation: Aggregation method

        Returns:
            Tuple of (aggregated_returns, aggregated_prices)
        """
        all_returns = []
        all_weights = []
        base_model_returns: Dict[str, list[np.ndarray]] = {}

        # Collect predictions from each seed
        for i, predictor in enumerate(self.predictors):
            if return_components:
                returns, _, components = predictor.predict_single(data, return_components=True)
                for model_name, component in components.items():
                    base_model_returns.setdefault(model_name, []).append(component["returns"])
            else:
                returns, _ = predictor.predict_single(data)
            all_returns.append(returns)

            # Calculate weight based on validation error
            artifact = self.artifacts['artifacts_list'][i]
            total_error = sum(artifact['val_errors'].values())
            weight = 1.0 / (total_error + 1e-9)
            all_weights.append(weight)

        all_returns = np.array(all_returns)

        # Aggregate
        if aggregation == 'weighted':
            weights = np.array(all_weights)
            weights /= weights.sum()
            agg_returns = np.sum(all_returns * weights[:, np.newaxis], axis=0)
            logger.info("Using weighted aggregation")

        elif aggregation == 'mean':
            agg_returns = np.nanmean(all_returns, axis=0)
            logger.info("Using mean aggregation")

        else:  # median
            agg_returns = np.nanmedian(all_returns, axis=0)
            logger.info("Using median aggregation")

        # Calculate price path
        last_close = data['close'].iloc[-1]
        agg_prices = last_close * np.exp(np.cumsum(agg_returns))

        if return_components:
            component_predictions: Dict[str, Dict[str, np.ndarray]] = {}
            weights = np.array(all_weights, dtype=float)
            if weights.sum() == 0:
                weights = np.ones(len(all_weights), dtype=float)
            weights = weights / weights.sum()
            for model_name, returns_list in base_model_returns.items():
                stacked = np.array(returns_list)
                model_returns = np.sum(stacked * weights[:, np.newaxis], axis=0)
                model_prices = last_close * np.exp(np.cumsum(model_returns))
                component_predictions[model_name] = {
                    "returns": model_returns,
                    "prices": model_prices,
                }
            return agg_returns, agg_prices, component_predictions

        return agg_returns, agg_prices


def next_business_day(date: pd.Timestamp) -> pd.Timestamp:
    """Get next business day (Mon-Fri)"""
    date += pd.Timedelta(days=1)
    while date.weekday() >= 5:
        date += pd.Timedelta(days=1)
    return date


def walk_forward_forecast(
    data: pd.DataFrame,
    multiseed_artifacts: Dict,
    n_days: int = 30,
    exo_mode: str = 'hold'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate long-term forecast by iteratively predicting

    Args:
        data: Preprocessed DataFrame
        multiseed_artifacts: Multi-seed artifacts
        n_days: Number of days to forecast
        exo_mode: How to handle exogenous features ('hold' or 'zero')

    Returns:
        Tuple of (returns, prices) arrays
    """
    predictor = MultiSeedPredictor(multiseed_artifacts)
    df = data.copy()

    future_prices = []
    future_returns = []

    logger.info(f"Starting {n_days}-day walk-forward forecast")

    for i in range(n_days):
        # Predict next horizon, use only first step
        ret_hat, price_path = predictor.predict(df)
        next_ret = ret_hat[0]
        next_price = price_path[0]

        future_returns.append(next_ret)
        future_prices.append(next_price)

        # Create new row for next day
        next_date = next_business_day(df.index[-1])
        new_row = df.iloc[-1:].copy()
        new_row.index = [next_date]
        new_row['close'] = next_price
        new_row['log_r'] = np.log(next_price / df['close'].iloc[-1])

        # Handle exogenous features
        if exo_mode == 'zero':
            exo_cols = [c for c in df.columns if 'pct' in c or 'roll' in c]
            new_row[exo_cols] = 0.0

        # Append and trim
        df = pd.concat([df, new_row])
        df = df.iloc[-200:]  # Keep last 200 rows to manage memory

    return np.array(future_returns), np.array(future_prices)
