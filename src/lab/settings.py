"""Pydantic-managed lab settings loaded from env / .env."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Lab runtime configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="LAB_",
        extra="ignore",
    )

    home: Path = Field(default=Path("/data/lab"), description="Lab artifacts root")
    pg_dsn: str = Field(default="postgresql://m@/lab")
    redis_url: str = Field(default="redis://localhost:6379/0")

    s3_endpoint: str = Field(default="http://localhost:9000")
    s3_bucket: str = Field(default="lab")
    s3_access_key: str = Field(default="labadmin")
    s3_secret_key: str = Field(default="")

    mlflow_url: str = Field(default="http://localhost:5000")
    litellm_url: str = Field(default="http://localhost:4000")
    litellm_key: str = Field(default="")

    ollama_local_url: str = Field(default="http://localhost:11434")
    ollama_cloud_url: str = Field(default="https://ollama.com")


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
