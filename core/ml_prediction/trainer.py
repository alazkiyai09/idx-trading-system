"""
Model training orchestration
"""
import numpy as np
import pickle
from typing import Dict, List, Tuple
try:
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_absolute_error
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
except ImportError:
    pass

from core.ml_prediction.models import build_model, create_sequences, TENSORFLOW_AVAILABLE
from config.settings import MLConfig
import logging
from core.ml_prediction.utils import set_random_seed

logger = logging.getLogger(__name__)


def safe_finite_mask(*arrays) -> np.ndarray:
    """
    Create mask for finite values across multiple arrays

    Args:
        *arrays: Variable number of numpy arrays

    Returns:
        Boolean mask array
    """
    mask = np.ones(len(arrays[0]), dtype=bool)

    for arr in arrays:
        if arr.ndim == 1:
            mask &= np.isfinite(arr)
        elif arr.ndim == 2:
            mask &= np.isfinite(arr).all(axis=1)
        elif arr.ndim == 3:
            mask &= np.isfinite(arr).all(axis=(1, 2))

    return mask


class EnsembleTrainer:
    """Train ensemble of base models with meta-learner"""

    def __init__(
        self,
        base_models: List[str] = None,
        n_prev: int = None,
        horizon: int = None
    ):
        """
        Initialize ensemble trainer

        Args:
            base_models: List of base model names
            n_prev: Lookback window
            horizon: Forecast horizon
        """
        if base_models is None:
            self.base_models = ['lstm', 'cnn_lstm', 'svr'] if TENSORFLOW_AVAILABLE else ['svr']
        else:
            self.base_models = base_models
        self.n_prev = n_prev or MLConfig.N_PREV
        self.horizon = horizon or MLConfig.HORIZON

    def prepare_data(
        self,
        df,
        feature_cols: List[str]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for training

        Args:
            df: Preprocessed DataFrame
            feature_cols: List of feature column names

        Returns:
            Tuple of (X_sequences, y_scaled, sigma_seq)
        """
        # Calculate rolling volatility for scaling (shift(1) to avoid data leakage)
        sigma = df['log_r'].rolling(20).std().shift(1).ffill()
        df = df.assign(_sigma=sigma).dropna(subset=['_sigma'])

        # Create sequences
        X_df = df[feature_cols]
        y_prices = df['close'].values

        X_seq, y_seq = create_sequences(
            X_df.values,
            y_prices,
            self.n_prev,
            self.horizon
        )

        # Get corresponding sigma values for scaling
        end_indices = np.arange(self.n_prev - 1, self.n_prev - 1 + len(y_seq))
        sigma_seq = df['_sigma'].iloc[end_indices].values

        # Scale targets by volatility
        y_scaled = y_seq / (sigma_seq[:, np.newaxis] + 1e-9)

        # Remove non-finite values
        finite_mask = safe_finite_mask(X_seq, y_scaled)
        X_seq = X_seq[finite_mask]
        y_scaled = y_scaled[finite_mask]

        logger.info(
            f"Prepared sequences: X_seq shape={X_seq.shape}, "
            f"y_scaled shape={y_scaled.shape}"
        )

        return X_seq, y_scaled, sigma_seq

    def train_single_fold(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        model_name: str
    ):
        """
        Train a single base model

        Args:
            X_train: Training sequences
            y_train: Training targets
            X_val: Validation sequences
            y_val: Validation targets
            model_name: Name of model to train

        Returns:
            Tuple of (trained_model, validation_predictions)
        """
        input_shape = X_train.shape[1:]
        model = build_model(model_name, input_shape, self.horizon)

        if model_name in ['lstm', 'cnn_lstm']:
            # Setup callbacks
            early_stop = EarlyStopping(
                monitor='val_loss',
                patience=MLConfig.PATIENCE,
                restore_best_weights=True,
                verbose=0
            )

            reduce_lr = ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=0
            )

            model.fit(
                X_train, y_train,
                X_val, y_val,
                epochs=MLConfig.EPOCHS,
                batch_size=MLConfig.BATCH_SIZE,
                callbacks=[early_stop, reduce_lr]
            )

            val_preds = model.predict(X_val)

        else:  # SVR
            model.fit(X_train, y_train)
            val_preds = model.predict(X_val)

        return model, val_preds

    def train_ensemble(
        self,
        data,
        feature_cols: List[str],
        n_splits: int = None
    ) -> Dict:
        """
        Train full ensemble with time series cross-validation

        Args:
            data: Preprocessed DataFrame
            feature_cols: List of feature columns
            n_splits: Number of CV splits

        Returns:
            Dictionary of trained artifacts
        """
        n_splits = n_splits or MLConfig.N_SPLITS

        # Prepare data
        X_seq, y_scaled, _ = self.prepare_data(data, feature_cols)
        N, _, n_features = X_seq.shape

        # Setup time series CV
        test_size = max(
            40,
            int(np.ceil(MLConfig.TARGET_OOF_FRAC * N / n_splits))
        )
        tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)

        # Initialize OOF predictions and error tracking
        oof_preds = {name: np.zeros_like(y_scaled) for name in self.base_models}
        val_errors = {name: [] for name in self.base_models}
        oof_mask = np.zeros(N, dtype=bool)

        # Cross-validation loop
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_seq)):
            logger.info(f"Training fold {fold + 1}/{n_splits}")

            X_train_raw, X_val_raw = X_seq[train_idx], X_seq[val_idx]
            y_train, y_val = y_scaled[train_idx], y_scaled[val_idx]

            # Scale features
            scaler = StandardScaler()
            scaler.fit(X_train_raw.reshape(-1, n_features))

            X_train = scaler.transform(X_train_raw.reshape(-1, n_features))
            X_train = X_train.reshape(X_train_raw.shape)

            X_val = scaler.transform(X_val_raw.reshape(-1, n_features))
            X_val = X_val.reshape(X_val_raw.shape)

            # Train each base model
            for model_name in self.base_models:
                logger.info(f"  Training {model_name}")

                _, val_preds = self.train_single_fold(
                    X_train, y_train,
                    X_val, y_val,
                    model_name
                )

                oof_preds[model_name][val_idx] = val_preds
                val_errors[model_name].append(mean_absolute_error(y_val, val_preds))

            oof_mask[val_idx] = True

        # Train meta-learners (one for each forecast horizon)
        logger.info("Training meta-learners")
        oof_features = np.stack(
            [oof_preds[name][oof_mask] for name in self.base_models],
            axis=-1
        )

        meta_models = []
        for h in range(self.horizon):
            meta = build_model('meta')
            meta.fit(oof_features[:, h, :], y_scaled[oof_mask, h])
            meta_models.append(meta)

        # Train final models on all data
        logger.info("Training final models on full dataset")
        final_scaler = StandardScaler()
        final_scaler.fit(X_seq.reshape(-1, n_features))
        X_all_scaled = final_scaler.transform(X_seq.reshape(-1, n_features))
        X_all_scaled = X_all_scaled.reshape(X_seq.shape)

        final_models = {}
        for model_name in self.base_models:
            logger.info(f"  Training final {model_name}")
            model, _ = self.train_single_fold(
                X_all_scaled, y_scaled,
                X_all_scaled, y_scaled,
                model_name
            )
            final_models[model_name] = model

        # Package artifacts
        artifacts = {
            'feature_cols': feature_cols,
            'base_models': self.base_models,
            'meta_models': meta_models,
            'scaler': final_scaler,
            'val_errors': {name: np.mean(errors) for name, errors in val_errors.items()},
            'config': {
                'n_prev': self.n_prev,
                'horizon': self.horizon
            }
        }

        # Add final base models
        artifacts.update(final_models)

        logger.info("Ensemble training complete")
        return artifacts


def train_multiseed_ensemble(
    data,
    feature_cols: List[str],
    seeds: List[int] = None,
    base_models: List[str] = None
) -> Dict:
    """
    Train ensemble with multiple random seeds for robustness

    Args:
        data: Preprocessed DataFrame
        feature_cols: Feature column names
        seeds: List of random seeds
        base_models: List of base model names

    Returns:
        Multi-seed artifacts dictionary
    """
    seeds = seeds or MLConfig.SEEDS
    if base_models is None:
        base_models = ['lstm', 'cnn_lstm', 'svr'] if TENSORFLOW_AVAILABLE else ['svr']

    all_artifacts = []

    for seed in seeds:
        logger.info(f"\n{'='*60}")
        logger.info(f"Training with seed: {seed}")
        logger.info(f"{'='*60}")

        set_random_seed(seed)

        trainer = EnsembleTrainer(base_models=base_models)
        artifacts = trainer.train_ensemble(data, feature_cols)
        artifacts['seed'] = seed

        all_artifacts.append(artifacts)

    # Package multi-seed artifacts
    multiseed_artifacts = {
        'type': 'multiseed',
        'seeds': seeds,
        'base_models': base_models,
        'artifacts_list': all_artifacts,
        'feature_cols': feature_cols,
        'config': all_artifacts[0]['config']
    }

    logger.info(f"\nMulti-seed training complete with {len(seeds)} seeds")
    return multiseed_artifacts


def save_artifacts(artifacts: Dict, filepath: str) -> None:
    """
    Save artifacts to pickle file

    Args:
        artifacts: Artifacts dictionary
        filepath: Output file path
    """
    with open(filepath, 'wb') as f:
        pickle.dump(artifacts, f)
    logger.info(f"Artifacts saved to {filepath}")


def load_artifacts(filepath: str) -> Dict:
    """
    Load artifacts from pickle file

    Args:
        filepath: Input file path

    Returns:
        Artifacts dictionary
    """
    with open(filepath, 'rb') as f:
        artifacts = pickle.load(f)
    logger.info(f"Artifacts loaded from {filepath}")
    return artifacts
