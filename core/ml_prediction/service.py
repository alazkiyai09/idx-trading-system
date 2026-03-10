"""
Service wrapper for ML Prediction pipeline.
Provides high-level methods for training and inference, integrated with the app's database.
"""
import logging
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
import os
import json

from sqlalchemy.orm import Session
from sqlalchemy import text
from datetime import datetime, timedelta

from config.settings import settings
from core.data.database import DatabaseManager
from core.ml_prediction.features import preprocess_stock_data
from core.ml_prediction.trainer import load_artifacts
from core.ml_prediction.predictor import Predictor, MultiSeedPredictor

logger = logging.getLogger(__name__)

class PredictionService:
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
        self.artifacts_dir = settings.get_data_path("ml_artifacts")

    def _get_artifact_path(self, symbol: str) -> str:
        # Simplify by using a single main ensemble file per symbol
        return os.path.join(self.artifacts_dir, f"{symbol}_ensemble.pkl")

    def get_model_metadata(self, symbol: str) -> Dict[str, Any]:
        """Extract basic model metadata from the stored artifact."""
        artifact_path = self._get_artifact_path(symbol)
        if not os.path.exists(artifact_path):
            raise ValueError(f"No trained model found for {symbol}.")

        artifacts = load_artifacts(artifact_path)
        config = artifacts.get("config", {})
        training_meta = artifacts.get("training_metadata", {})
        trained_at = training_meta.get("trained_at")
        if not trained_at and os.path.exists(artifact_path):
            trained_at = datetime.utcfromtimestamp(os.path.getmtime(artifact_path)).isoformat() + "Z"
        validation_mae = self._extract_validation_mae(artifacts)

        return {
            "artifact_path": artifact_path,
            "artifact_version": artifacts.get("version", "unknown"),
            "artifact_type": artifacts.get("type", "single"),
            "trained_horizon": int(config.get("horizon", 0) or 0),
            "lookback_window": int(config.get("n_prev", 0) or 0),
            "feature_count": len(artifacts.get("feature_cols", [])),
            "feature_set": artifacts.get("feature_cols", []),
            "trained_at": trained_at,
            "source_latest_date": training_meta.get("source_latest_date"),
            "source_row_count": training_meta.get("source_row_count"),
            "training_overwrite": training_meta.get("overwrite"),
            "validation_mae": validation_mae,
            "uses_exogenous": any(
                not col.startswith(("open", "high", "low", "close", "volume", "log_r"))
                for col in artifacts.get("feature_cols", [])
            ),
        }

    def _extract_validation_mae(self, artifacts: Dict[str, Any]) -> Dict[str, float]:
        """Summarize validation MAE by base model from the stored artifact."""
        summary: Dict[str, float] = {}
        if artifacts.get("type") == "multiseed":
            buckets: Dict[str, list[float]] = {}
            for item in artifacts.get("artifacts_list", []):
                for model_name, value in item.get("val_errors", {}).items():
                    if value is None:
                        continue
                    buckets.setdefault(model_name, []).append(float(value))
            for model_name, values in buckets.items():
                if values:
                    summary[model_name] = float(np.mean(values))
            return summary

        for model_name, value in artifacts.get("val_errors", {}).items():
            if value is None:
                continue
            summary[model_name] = float(value)
        return summary

    def _compute_model_contributions(self, validation_mae: Dict[str, float]) -> Dict[str, float]:
        """Convert validation MAE into normalized contribution weights."""
        if not validation_mae:
            return {}
        raw = {
            model_name: 1.0 / max(value, 1e-9)
            for model_name, value in validation_mae.items()
            if value is not None
        }
        total = sum(raw.values())
        if total <= 0:
            return {}
        return {model_name: weight / total for model_name, weight in raw.items()}

    def has_model(self, symbol: str) -> bool:
        return os.path.exists(self._get_artifact_path(symbol))

    def _get_stock_data(self, symbol: str, lookback_days: int = 400) -> pd.DataFrame:
        """Fetch stock data formatted for the ML pipeline"""
        # We need historical data. The ML pipeline expects 'open', 'high', 'low', 'close', 'volume', 'date' (as index)
        # Using the same database logic as our /stocks/{symbol}/chart endpoint
        with self.db.get_session() as session:
            # Query standard OHLCV from price_history table
            query = text("""
                SELECT date, open, high, low, close, volume
                FROM price_history
                WHERE symbol = :symbol
                ORDER BY date DESC
                LIMIT :limit
            """)
            result = session.execute(query, {"symbol": symbol, "limit": lookback_days}).fetchall()

            if not result:
                raise ValueError(f"No historical data found for {symbol}")

            df = pd.DataFrame(result, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            df.sort_index(ascending=True, inplace=True) # Must be ascending
            return df

    def predict(self, symbol: str, n_days: int = 7) -> Dict[str, Any]:
        """
        Run inference using the trained ensemble model.
        Returns a dictionary with dates and predicted prices.
        """
        if not self.has_model(symbol):
            raise ValueError(f"No trained model found for {symbol}. Train the model first.")

        try:
            logger.info(f"Loading artifacts for {symbol}")
            artifacts = load_artifacts(self._get_artifact_path(symbol))
            metadata = self.get_model_metadata(symbol)
            trained_horizon = metadata["trained_horizon"]

            if trained_horizon and n_days != trained_horizon:
                raise ValueError(
                    f"Unsupported horizon {n_days} for {symbol}. "
                    f"Model artifact is trained for {trained_horizon} business days."
                )
            
            logger.info(f"Fetching recent data for {symbol}")
            # Need at least MIN_TOTAL_ROWS + lags. Let's pull 300 days.
            df = self._get_stock_data(symbol, lookback_days=300)
            
            # Preprocess
            # Note: For now, we skip exogenous features in inference to simplify dependencies.
            # The models might expect them if trained with them, so we'll need to handle that if needed.
            # Assuming we train without exo for MVP integration, or handle empty exo.
            processed_df = preprocess_stock_data(df, exo_dict=None)

            logger.info(f"Running prediction for {symbol}")
            
            if artifacts.get('type') == 'multiseed':
                predictor = MultiSeedPredictor(artifacts)
            else:
                predictor = Predictor(artifacts)

            # Predict returns and prices
            # The predictor expects just the generic predict call. But MultiSeedPredictor returns tuple
            # If we just want 7 days, predict() (if using predictor.py) just predicts 1 step ahead by default?
            # Actually, predict_single returns (returns_horizon, prices_horizon) for the trained horizon.
            if hasattr(predictor, 'predict_single'):
                # Single artifact set
                returns, prices, component_predictions = predictor.predict_single(
                    processed_df,
                    return_components=True,
                )
            else:
                # MultiSeed
                returns, prices, component_predictions = predictor.predict(
                    processed_df,
                    return_components=True,
                )
            
            # Format output
            last_date = df.index[-1]
            future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, len(prices)+1)]
            # Skip weekends in dates
            bus_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=len(prices), freq='B')

            model_contributions = self._compute_model_contributions(metadata.get("validation_mae", {}))
            base_model_predictions = {
                model_name: [
                    {
                        "date": d.strftime("%Y-%m-%d"),
                        "predicted_price": float(p),
                        "predicted_return": float(r),
                    }
                    for d, p, r in zip(bus_dates, component["prices"], component["returns"])
                ]
                for model_name, component in component_predictions.items()
            }

            return {
                "symbol": symbol,
                "current_price": float(df['close'].iloc[-1]),
                "artifact_metadata": metadata,
                "model_contributions": model_contributions,
                "base_model_predictions": base_model_predictions,
                "predictions": [
                    {
                        "date": d.strftime("%Y-%m-%d"),
                        "predicted_price": float(p),
                        "predicted_return": float(r)
                    } for d, p, r in zip(bus_dates, prices, returns)
                ]
            }

        except Exception as e:
            logger.error(f"Prediction failed for {symbol}: {e}")
            raise

    # Training method would go here, omitting for brevity or mock it if needed 
    # to avoid blowing up the system during inference test. Let's add a placeholder.
    def train(self, symbol: str):
        """Train standard ML ensemble. Not implemented in MVP to save local resources."""
        raise NotImplementedError("Training requires high compute and full historical exo-data sync.")
