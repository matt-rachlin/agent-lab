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
    kb_root: Path = Field(
        default=Path("~/db/kb").expanduser(),
        description="Knowledge-base root directory (lab.rag).",
    )
    hf_cache_root: Path = Field(
        default=Path("/data/lab/services/hf-cache"),
        description=(
            "Host directory bind-mounted into the sandbox at /hf-cache for the "
            "Phase 7 cross-encoder reranker. Shared rw across cells so the "
            "~1.5 GB Qwen3-Reranker weights download only once."
        ),
    )
    pg_dsn: str = Field(default="postgresql://m@/lab")
    redis_url: str = Field(default="redis://localhost:6379/0")

    s3_endpoint: str = Field(default="http://localhost:9000")
    s3_bucket: str = Field(default="lab")
    s3_access_key: str = Field(default="labadmin")
    s3_secret_key: str = Field(default="")

    mlflow_url: str = Field(default="http://localhost:5050")
    litellm_url: str = Field(default="http://localhost:4000")
    litellm_key: str = Field(default="")

    ollama_local_url: str = Field(default="http://localhost:11434")
    ollama_cloud_url: str = Field(default="https://ollama.com")

    # Phase 19b/19c — llama-swap multi-model orchestrator. The
    # :class:`lab.core.model_pool.ModelPool` defaults to this URL for
    # pre-flight / predictive-load / teardown calls.
    llama_swap_url: str = Field(default="http://localhost:8080")


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings  # noqa: PLW0603  # reason: module-level singleton, by design
    if _settings is None:
        _settings = Settings()
    return _settings
