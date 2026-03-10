"""
Global settings for IDX Trading System.

All configurable parameters should be here.
Use environment variables for secrets.
"""

from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment.

    This class contains all configurable parameters for the trading system.
    Settings can be overridden via environment variables or .env file.

    Attributes:
        anthropic_api_key: API key for Claude/Anthropic services.
        glm_api_key: API key for GLM services.
        telegram_bot_token: Telegram bot token for notifications.
        telegram_chat_id: Telegram chat ID for notifications.
        database_url: Database connection URL.
        initial_capital: Starting capital in IDR.
        default_mode: Default trading mode (intraday, swing, position, investor).
        paper_trading: Whether to use paper trading mode.
        max_risk_per_trade: Maximum risk per trade as decimal.
        max_daily_loss: Maximum daily loss as decimal.
        max_drawdown: Maximum drawdown as decimal.
        max_positions: Maximum number of concurrent positions.
        max_position_pct: Maximum position size as decimal of capital.
        lot_size: IDX lot size (shares per lot).
        buy_fee_pct: Buy fee percentage including all charges.
        sell_fee_pct: Sell fee percentage including tax.
        data_dir: Directory for data storage.
        logs_dir: Directory for log files.
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR).
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # API Keys
    anthropic_api_key: str = Field(default="", description="Anthropic API key")
    glm_api_key: str = Field(default="", description="GLM API key")
    telegram_bot_token: Optional[str] = Field(
        default=None, description="Telegram bot token"
    )
    telegram_chat_id: Optional[str] = Field(
        default=None, description="Telegram chat ID"
    )

    # Database
    database_url: str = Field(
        default="sqlite:///data/trading.db",
        description="Database connection URL",
    )

    # Trading
    initial_capital: float = Field(
        default=100_000_000,  # 100M IDR
        description="Initial capital in IDR",
    )
    default_mode: str = Field(
        default="swing",
        description="Default trading mode",
    )
    paper_trading: bool = Field(
        default=True,
        description="Use paper trading mode",
    )

    # Risk limits
    max_risk_per_trade: float = Field(
        default=0.01,
        ge=0.0,
        le=0.05,
        description="Maximum risk per trade (default 1%)",
    )
    max_daily_loss: float = Field(
        default=0.02,
        ge=0.0,
        le=0.10,
        description="Maximum daily loss (default 2%)",
    )
    max_drawdown: float = Field(
        default=0.10,
        ge=0.0,
        le=0.30,
        description="Maximum drawdown (default 10%)",
    )
    max_positions: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum concurrent positions",
    )
    max_position_pct: float = Field(
        default=0.25,
        ge=0.05,
        le=0.50,
        description="Maximum position as % of capital (default 25%)",
    )

    # IDX specific
    lot_size: int = Field(
        default=100,
        description="IDX lot size (shares per lot)",
    )
    buy_fee_pct: float = Field(
        default=0.0015,
        description="Buy fee percentage (0.15%)",
    )
    sell_fee_pct: float = Field(
        default=0.0025,
        description="Sell fee percentage including tax (0.25%)",
    )

    # Paths
    data_dir: Path = Field(
        default=Path("data"),
        description="Directory for data storage",
    )
    logs_dir: Path = Field(
        default=Path("logs"),
        description="Directory for log files",
    )

    # API
    api_url: str = Field(
        default="http://localhost:8000",
        description="API server URL",
    )

    # Data refresh policy
    stock_list_file: str = Field(
        default="/home/ubuntu/Downloads/Stock_list.txt",
        description="Path to the IDX stock list file used by batch ingestion",
    )
    daily_refresh_policy: str = Field(
        default="Daily after market close or by midnight Asia/Jakarta",
        description="Human-readable expectation for price data refresh cadence",
    )
    model_training_batch_limit: int = Field(
        default=10,
        ge=1,
        le=25,
        description="Maximum number of stock models allowed in a single manual training batch",
    )

    # Logging
    log_level: str = Field(
        default="INFO",
        description="Logging level",
    )

    def get_data_path(self, subdir: str) -> Path:
        """Get path to a data subdirectory.

        Args:
            subdir: Name of the subdirectory (e.g., 'market', 'trades').

        Returns:
            Path to the subdirectory.
        """
        path = self.data_dir / subdir
        path.mkdir(parents=True, exist_ok=True)
        return path

    def ensure_directories(self) -> None:
        """Ensure all required directories exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        # Create data subdirectories
        for subdir in ["market", "trades", "backtest", "fundamental"]:
            (self.data_dir / subdir).mkdir(parents=True, exist_ok=True)


class TimesFMSettings(BaseSettings):
    """Settings for TimesFM forecasting integration.

    TimesFM is Google's Time Series Foundation Model for price forecasting.
    These settings control the model loading, forecasting behavior, and
    integration with signal generation and risk management.

    Attributes:
        enabled: Whether TimesFM forecasting is enabled.
        model_name: HuggingFace model name for TimesFM.
        max_context_len: Maximum context length for the model.
        horizon_len: Forecast horizon (number of steps ahead).
        device: Device to use ('auto', 'cpu', 'cuda').
        forecast_weight: Weight of forecast in composite score (0-1).
        min_risk_reward: Minimum R:R ratio from forecast.
        max_uncertainty: Maximum uncertainty allowed (as decimal).
        cache_ttl_minutes: Cache TTL in minutes.
    """

    model_config = SettingsConfigDict(
        env_prefix="TIMESFM_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    enabled: bool = Field(
        default=False,
        description="Enable TimesFM forecasting"
    )
    model_name: str = Field(
        default="google/timesfm-2.5-200m",
        description="HuggingFace model name"
    )
    max_context_len: int = Field(
        default=512,
        ge=64,
        le=1024,
        description="Maximum context length"
    )
    horizon_len: int = Field(
        default=32,
        ge=1,
        le=128,
        description="Forecast horizon"
    )
    device: str = Field(
        default="auto",
        description="Device to use (auto/cpu/cuda)"
    )
    forecast_weight: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        description="Forecast weight in composite score"
    )
    min_risk_reward: float = Field(
        default=1.5,
        ge=0.5,
        description="Minimum forecast R:R ratio"
    )
    max_uncertainty: float = Field(
        default=0.20,
        ge=0.0,
        le=1.0,
        description="Maximum forecast uncertainty"
    )
    cache_ttl_minutes: int = Field(
        default=60,
        ge=1,
        description="Forecast cache TTL in minutes"
    )


class LLMSettings(BaseSettings):
    """Settings for LLM provider configuration.

    Controls which LLM provider to use, fallback behavior,
    and daily budget limits.

    Attributes:
        default_provider: Primary LLM provider (claude or glm).
        fallback_provider: Fallback provider if primary fails.
        daily_budget_usd: Daily budget for LLM API calls in USD.
        max_retries: Maximum retries per API call.
        retry_base_delay: Base delay between retries in seconds.
        enable_fallback: Whether to enable automatic fallback.
        enable_cost_tracking: Whether to track API costs.
    """

    model_config = SettingsConfigDict(
        env_prefix="LLM_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    default_provider: str = Field(
        default="claude",
        description="Primary LLM provider (claude or glm)",
    )
    fallback_provider: str = Field(
        default="glm",
        description="Fallback LLM provider",
    )
    daily_budget_usd: float = Field(
        default=10.0,
        ge=0.0,
        description="Daily budget for LLM API calls in USD",
    )
    max_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum retries per API call",
    )
    retry_base_delay: float = Field(
        default=1.0,
        ge=0.1,
        description="Base delay between retries in seconds",
    )
    enable_fallback: bool = Field(
        default=True,
        description="Enable automatic provider fallback",
    )
    enable_cost_tracking: bool = Field(
        default=True,
        description="Track API costs per call",
    )


class EmailSettings(BaseSettings):
    """Settings for email notifications.

    Attributes:
        smtp_host: SMTP server hostname.
        smtp_port: SMTP server port.
        smtp_user: SMTP username.
        smtp_password: SMTP password.
        from_address: From email address.
        recipients: Comma-separated list of recipient emails.
        enabled: Whether email notifications are enabled.
    """

    model_config = SettingsConfigDict(
        env_prefix="EMAIL_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    smtp_host: str = Field(default="smtp.gmail.com", description="SMTP host")
    smtp_port: int = Field(default=587, description="SMTP port")
    smtp_user: str = Field(default="", description="SMTP username")
    smtp_password: str = Field(default="", description="SMTP password")
    from_address: str = Field(default="", description="From email address")
    recipients: str = Field(default="", description="Comma-separated recipients")
    enabled: bool = Field(default=False, description="Enable email notifications")


class MLConfig:
    """Configuration for Machine Learning prediction models."""
    N_PREV = 60
    HORIZON = 7
    N_SPLITS = 3
    TARGET_OOF_FRAC = 0.25
    PATIENCE = 10
    EPOCHS = 30
    BATCH_SIZE = 32
    SEEDS = [7, 21, 42]
    RSI_PERIOD = 14
    ADX_PERIOD = 14
    VOL_PROXY_WIN = 20
    TREND_REGIME_WIN = 50
    VOL_REGIME_WIN = 20
    EXTENDED_LAGS = [1, 2, 3, 5, 10, 20]
    EXO_LAGS = [1, 5, 20]
    EXO_ROLL_WINDOWS = [5, 20, 60]
    MIN_TOTAL_ROWS = 200
    
    LSTM_CONFIG = {
        'units': [64, 32],
        'l2_reg': 1e-4,
        'dropout': 0.2,
        'lr': 0.001,
        'loss_delta': 1.0
    }
    
    CNN_LSTM_CONFIG = {
        'filters': 64,
        'kernel_size': 3,
        'pool_size': 2,
        'lstm_units': 64,
        'dropout': 0.2,
        'lr': 0.001,
        'loss_delta': 1.0
    }
    
    SVR_CONFIG = {
        'C': 1.0,
        'epsilon': 0.1,
        'kernel': 'rbf'
    }
    
    META_GBR_CONFIG = {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 3,
        'random_state': 42
    }


# Global settings instances
settings = Settings()
timesfm_settings = TimesFMSettings()
llm_settings = LLMSettings()
email_settings = EmailSettings()
