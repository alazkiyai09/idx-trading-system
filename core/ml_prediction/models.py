"""
Machine Learning model architectures
"""
import numpy as np
from typing import Tuple, Any

try:
    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.svm import SVR

    import tensorflow as tf
    from tensorflow.keras import Input, Model, regularizers
    from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout, Conv1D, MaxPooling1D
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.losses import Huber
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

from config.settings import MLConfig
import logging

logger = logging.getLogger(__name__)


def create_sequences(
    X: np.ndarray,
    y_prices: np.ndarray,
    n_prev: int,
    horizon: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences for time series prediction

    Args:
        X: Feature array (n_samples, n_features)
        y_prices: Price array
        n_prev: Number of previous timesteps
        horizon: Forecast horizon

    Returns:
        Tuple of (X_sequences, y_targets)
    """
    if len(X) < n_prev + horizon:
        raise ValueError(f"Insufficient data: {len(X)} < {n_prev + horizon}")

    sequences, targets = [], []

    for t in range(n_prev, len(X) - horizon + 1):
        sequences.append(X[t - n_prev:t, :])
        # Target: log returns for next H periods
        targets.append(np.log(y_prices[t:t + horizon] / y_prices[t - 1]))

    return np.array(sequences), np.array(targets)


def summarize_sequence(X_sequence: np.ndarray) -> np.ndarray:
    """
    Summarize 3D sequence to 2D for non-sequential models (SVR)

    Args:
        X_sequence: 3D array (batch, timesteps, features)

    Returns:
        2D array (batch, summary_features)
    """
    last_step = X_sequence[:, -1, :]
    mean_step = X_sequence.mean(axis=1)
    std_step = X_sequence.std(axis=1)

    return np.concatenate([last_step, mean_step, std_step], axis=1)


class LSTMModel:
    """Bidirectional LSTM model"""

    def __init__(self, input_shape: Tuple[int, int], horizon: int):
        """
        Initialize LSTM model

        Args:
            input_shape: (timesteps, features)
            horizon: Forecast horizon
        """
        self.input_shape = input_shape
        self.horizon = horizon
        self.model = self._build()

    def _build(self) -> Any:
        """Build LSTM architecture"""
        cfg = MLConfig.LSTM_CONFIG

        inputs = Input(shape=self.input_shape)

        # First Bidirectional LSTM layer
        x = Bidirectional(
            LSTM(
                cfg['units'][0],
                return_sequences=True,
                kernel_regularizer=regularizers.l2(cfg['l2_reg'])
            )
        )(inputs)
        x = Dropout(cfg['dropout'])(x)

        # Second Bidirectional LSTM layer
        x = Bidirectional(
            LSTM(
                cfg['units'][1],
                kernel_regularizer=regularizers.l2(cfg['l2_reg'])
            )
        )(x)
        x = Dropout(cfg['dropout'])(x)

        # Output layer
        outputs = Dense(self.horizon, activation='linear')(x)

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=Adam(learning_rate=cfg['lr']),
            loss=Huber(delta=cfg['loss_delta'])
        )

        logger.info(f"Built LSTM model with shape {self.input_shape}")
        return model

    def fit(self, X_train, y_train, X_val, y_val, epochs: int, batch_size: int, callbacks):
        """Train the model"""
        return self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=0
        )

    def predict(self, X, verbose=0):
        """Make predictions"""
        return self.model.predict(X, verbose=verbose)


class CNNLSTMModel:
    """CNN-LSTM hybrid model"""

    def __init__(self, input_shape: Tuple[int, int], horizon: int):
        """
        Initialize CNN-LSTM model

        Args:
            input_shape: (timesteps, features)
            horizon: Forecast horizon
        """
        self.input_shape = input_shape
        self.horizon = horizon
        self.model = self._build()

    def _build(self) -> Any:
        """Build CNN-LSTM architecture"""
        cfg = MLConfig.CNN_LSTM_CONFIG

        inputs = Input(shape=self.input_shape)

        # Conv1D for feature extraction
        x = Conv1D(
            filters=cfg['filters'],
            kernel_size=cfg['kernel_size'],
            activation='relu'
        )(inputs)
        x = MaxPooling1D(pool_size=cfg['pool_size'])(x)

        # LSTM layer
        x = LSTM(cfg['lstm_units'])(x)
        x = Dropout(cfg['dropout'])(x)

        # Output
        outputs = Dense(self.horizon)(x)

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=Adam(learning_rate=cfg['lr']),
            loss=Huber(delta=cfg['loss_delta'])
        )

        logger.info(f"Built CNN-LSTM model with shape {self.input_shape}")
        return model

    def fit(self, X_train, y_train, X_val, y_val, epochs: int, batch_size: int, callbacks):
        """Train the model"""
        return self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=0
        )

    def predict(self, X, verbose=0):
        """Make predictions"""
        return self.model.predict(X, verbose=verbose)


class SVRModel:
    """Support Vector Regression model"""

    def __init__(self, horizon: int):
        """
        Initialize SVR model

        Args:
            horizon: Forecast horizon
        """
        self.horizon = horizon
        cfg = MLConfig.SVR_CONFIG
        self.model = MultiOutputRegressor(SVR(**cfg))
        logger.info("Built SVR model")

    def fit(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        """Train the model (X should be 3D, will be summarized)"""
        X_train_summary = summarize_sequence(X_train)
        self.model.fit(X_train_summary, y_train)
        return self

    def predict(self, X, **kwargs):
        """Make predictions (X should be 3D, will be summarized)"""
        X_summary = summarize_sequence(X)
        return self.model.predict(X_summary)


class MetaLearner:
    """Gradient Boosting meta-learner"""

    def __init__(self):
        """Initialize meta-learner"""
        cfg = MLConfig.META_GBR_CONFIG
        self.model = GradientBoostingRegressor(**cfg)
        logger.debug("Built meta-learner")

    def fit(self, X, y):
        """Train meta-learner"""
        return self.model.fit(X, y)

    def predict(self, X):
        """Predict with meta-learner"""
        return self.model.predict(X)


def build_model(model_type: str, input_shape: Tuple[int, int] = None, horizon: int = None):
    """
    Factory function to build models

    Args:
        model_type: 'lstm', 'cnn_lstm', 'svr', or 'meta'
        input_shape: Required for LSTM and CNN-LSTM
        horizon: Required for all except meta

    Returns:
        Model instance
    """
    if model_type == 'lstm':
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is not installed; LSTM model is unavailable")
        return LSTMModel(input_shape, horizon)
    elif model_type == 'cnn_lstm':
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is not installed; CNN-LSTM model is unavailable")
        return CNNLSTMModel(input_shape, horizon)
    elif model_type == 'svr':
        return SVRModel(horizon)
    elif model_type == 'meta':
        return MetaLearner()
    else:
        raise ValueError(f"Unknown model type: {model_type}")
