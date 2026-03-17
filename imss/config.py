"""IMSS configuration — environment settings and simulation config."""

from __future__ import annotations

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class IMSSSettings(BaseSettings):
    """Environment-driven settings for IMSS infrastructure."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM
    glm_api_key: str = Field(alias="GLM_API_KEY")
    glm_base_url: str = Field(
        default="https://api.z.ai/api/paas/v4/", alias="IMSS_GLM_BASE_URL"
    )
    glm_model: str = Field(default="glm-5", alias="IMSS_GLM_MODEL")

    # Embedding
    embedding_model: str = Field(default="embedding-3", alias="IMSS_EMBEDDING_MODEL")
    embedding_dimension: int = Field(default=1024, alias="IMSS_EMBEDDING_DIMENSION")

    # Database
    database_url: str = Field(
        default="sqlite+aiosqlite:///./data/imss.db", alias="IMSS_DATABASE_URL"
    )

    # ChromaDB
    chroma_persist_dir: str = Field(
        default="./data/chroma", alias="IMSS_CHROMA_PERSIST_DIR"
    )

    # Concurrency
    max_concurrent_llm_calls: int = Field(
        default=5, alias="IMSS_MAX_CONCURRENT_LLM_CALLS"
    )
    llm_request_timeout: int = Field(default=30, alias="IMSS_LLM_REQUEST_TIMEOUT")

    # Cost
    cost_alert_threshold_usd: float = Field(
        default=5.0, alias="IMSS_COST_ALERT_THRESHOLD_USD"
    )


class SimulationConfig(BaseModel):
    """Per-run simulation configuration."""

    # Target
    target_stocks: list[str] = ["BBRI"]

    # Mode
    mode: str = "BACKTEST"  # BACKTEST or PREDICT

    # Time
    backtest_start: str = "2024-07-01"
    backtest_end: str = "2024-09-30"
    prediction_horizon_days: int = 20
    steps_per_day: int = 1

    # Agent Population
    tier1_personas: list[str] = ["pak_budi", "sarah", "andi"]
    tier2_per_archetype: int = 4
    tier2_archetypes: list[str] = [
        "momentum_chaser",
        "panic_seller",
        "news_reactive",
    ]
    tier3_total: int = 50
    tier3_distribution: dict[str, float] = {
        "momentum_follower": 0.30,
        "mean_reversion": 0.25,
        "random_walk": 0.30,
        "volume_follower": 0.15,
    }

    # Multi-Run
    num_parallel_runs: int = 1
    runs_batch_size: int = 5

    # LLM
    tier1_temperature: float = 0.7
    tier1_max_tokens: int = 1024
    tier2_temperature: float = 0.5
    tier2_max_tokens: int = 512

    # Memory (all disabled for Phase 1)
    enable_episodic_memory: bool = False
    enable_social_memory: bool = False
    enable_causal_retrieval: bool = False


def get_settings() -> IMSSSettings:
    """Return cached IMSS settings instance."""
    return IMSSSettings()
