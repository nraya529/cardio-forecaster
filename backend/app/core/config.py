from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="CARDIO_", extra="ignore")

    app_name: str = "CardioForecaster"
    api_v1_prefix: str = "/api/v1"
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:5173", "http://localhost:3000"])

    model_path: Path = Path("artifacts/transformer_best.pt")
    model_arch: str = "transformer"
    history_window: int = 360
    forecast_horizon: int = 120
    sampling_rate_hz: float = 1.0
    n_channels: int = 6
    channel_names: list[str] = Field(default_factory=lambda: ["hr", "sbp", "dbp", "spo2", "resp", "temp"])

    d_model: int = 128
    n_heads: int = 8
    n_encoder_layers: int = 4
    n_decoder_layers: int = 2
    dim_feedforward: int = 256
    dropout: float = 0.1

    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()
